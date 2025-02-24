import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from transformer_model import MusicTransformer


class DualTransformer(nn.Module):
    """Dual transformer architecture for hierarchical music generation.

    Implements a two-level transformer architecture where:
    1. Event Transformer: Processes high-level musical events
    2. Token Transformer: Handles token-level details within each event

    """

    def __init__(self,
                 vocab_size: int,
                 event_context_size: int = 128,
                 token_context_size: int = 8,
                 n_embd: int = 512,
                 event_head: int = 8,
                 token_head: int = 2,
                 event_depth: int = 6,
                 token_depth: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        # Store configuration
        self.vocab_size = vocab_size
        self.event_context_size = event_context_size
        self.token_context_size = token_context_size

        # Event-level transformer
        self.event_transformer = MusicTransformer(
            num_tokens=vocab_size,
            dim=n_embd,
            depth=event_depth,
            heads=event_head,
            dim_head=n_embd // event_head,
            max_seq_len=event_context_size,
            dropout=dropout
        )

        # Token-level transformer
        self.token_transformer = MusicTransformer(
            num_tokens=vocab_size,
            dim=n_embd,
            depth=token_depth,
            heads=token_head,
            dim_head=n_embd // token_head,
            max_seq_len=token_context_size,
            dropout=dropout
        )

        # Output projection
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, cache=None):
        # Create proper attention mask
        b, n, _ = x.shape
        attention_mask = torch.ones((b, n, n), device=x.device)
        attention_mask = torch.triu(attention_mask, diagonal=1).bool()
        attention_mask = ~attention_mask

        # Process through event transformer
        x = self.event_transformer.token_emb(x)
        x = x.sum(dim=-2)

        for i, layer in enumerate(self.event_transformer.layers):
            x, k, v = layer(x, mask=attention_mask)
            if cache is not None:
                cache.update(k, v, i)

        return self.event_transformer.norm(x)

    def forward_token(self, hidden_state=None, x=None, cache=None):
        if hidden_state is not None:
            hidden_state = hidden_state.unsqueeze(1)

        if x is not None:
            x = self.token_transformer.token_emb(x)
            if hidden_state is not None:
                x = torch.cat([hidden_state, x], dim=1)
            hidden_state = x

        b, n, _ = hidden_state.shape
        attention_mask = torch.ones((b, n, n), device=hidden_state.device)
        attention_mask = torch.triu(attention_mask, diagonal=1).bool()
        attention_mask = ~attention_mask

        x = hidden_state
        for i, layer in enumerate(self.token_transformer.layers):
            x, k, v = layer(x, mask=attention_mask)
            if cache is not None:
                cache.update(k, v, i)

        x = self.token_transformer.norm(x)
        return self.lm_head(x)

    def sample_top_p_k(self, probs, p=0.9, k=20, generator=None):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0

        mask = torch.zeros(probs_sort.shape[-1], device=probs_sort.device)
        mask[:k] = 1
        probs_sort = probs_sort * mask

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        shape = probs_sort.shape
        next_token = torch.multinomial(
            probs_sort.reshape(-1, shape[-1]),
            num_samples=1,
            generator=generator
        ).reshape(*shape[:-1], 1)

        next_token = torch.gather(probs_idx, -1, next_token).reshape(*shape[:-1])
        return next_token

    @torch.inference_mode()
    def generate(self, prompt=None, tokenizer=None, batch_size=1, max_len=512,
                 temp=1.0, top_p=0.98, top_k=20, generator=None):
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for generation")

        max_token_seq = tokenizer.max_token_seq
        device = next(self.parameters()).device

        # Initialize input tensor
        if prompt is None:
            # Create initial tensor with padding and BOS token
            input_tensor = torch.full(
                (1, max_token_seq),
                tokenizer.pad_id,
                dtype=torch.long,
                device=device
            )
            input_tensor[0, 0] = tokenizer.bos_id
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = torch.cat([input_tensor] * batch_size, dim=0)
        else:
            # Process provided prompt
            if len(prompt.shape) == 2:
                prompt = prompt[None, :]
                prompt = np.repeat(prompt, repeats=batch_size, axis=0)
            elif prompt.shape[0] == 1:
                prompt = np.repeat(prompt, repeats=batch_size, axis=0)
            elif len(prompt.shape) != 3 or prompt.shape[0] != batch_size:
                raise ValueError(f"Invalid shape for prompt: {prompt.shape}")

            # Ensure prompt fits within max token sequence length
            prompt = prompt[..., :max_token_seq]
            if prompt.shape[-1] < max_token_seq:
                prompt = np.pad(
                    prompt,
                    ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
                    mode="constant",
                    constant_values=tokenizer.pad_id
                )
            input_tensor = torch.from_numpy(prompt).to(dtype=torch.long, device=device)

        # Initialize generation
        cur_len = input_tensor.shape[1]
        cache1 = DynamicCache()  # Event transformer cache
        past_len = 0

        # Main generation loop
        while cur_len < max_len:
            end = [False] * batch_size
            # Process through event transformer
            hidden = self.forward(input_tensor[:, past_len:], cache=cache1)[:, -1]
            next_token_seq = None
            event_names = [""] * batch_size
            cache2 = DynamicCache()  # Token transformer cache

            # Generate tokens for each position in the sequence
            for i in range(max_token_seq):
                # Create mask for valid tokens at current position
                mask = torch.zeros(
                    (batch_size, tokenizer.vocab_size),
                    dtype=torch.int64,
                    device=device
                )
                # Apply token validation masks based on current position
                for b in range(batch_size):
                    if end[b]:
                        mask[b, tokenizer.pad_id] = 1
                        continue

                    if i == 0:
                        # First position: allow events and EOS token
                        mask[b, list(tokenizer.event_ids.values()) + [tokenizer.eos_id]] = 1
                    else:
                        # Other positions: allow valid parameters for current event
                        param_names = tokenizer.events[event_names[b]]
                        if i > len(param_names):
                            mask[b, tokenizer.pad_id] = 1
                            continue
                        mask[b, tokenizer.parameter_ids[param_names[i - 1]]] = 1

                mask = mask.unsqueeze(1)
                x = next_token_seq

                if i != 0:
                    # Use cached values for non-first positions
                    hidden = None
                    x = x[:, -1:]

                # Generate next token
                logits = self.forward_token(hidden, x, cache=cache2)[:, -1:]
                scores = torch.softmax(logits / temp, dim=-1) * mask
                samples = self.sample_top_p_k(scores, top_p, top_k, generator=generator)

                if i == 0:
                    next_token_seq = samples
                    for b in range(batch_size):
                        if end[b]:
                            continue
                        eid = samples[b].item()
                        if eid == tokenizer.eos_id:
                            end[b] = True
                        else:
                            event_names[b] = tokenizer.id_events[eid]
                else:
                    next_token_seq = torch.cat([next_token_seq, samples], dim=1)
                    if all([len(tokenizer.events[event_names[b]]) == i
                            for b in range(batch_size) if not end[b]]):
                        break

            # Pad sequence if necessary
            if next_token_seq.shape[1] < max_token_seq:
                next_token_seq = F.pad(
                    next_token_seq,
                    (0, max_token_seq - next_token_seq.shape[1]),
                    "constant",
                    value=tokenizer.pad_id
                )

            # Add generated tokens to input sequence
            next_token_seq = next_token_seq.unsqueeze(1)
            input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
            past_len = cur_len
            cur_len += 1

            # Check if all sequences are complete
            if all(end):
                break

        return input_tensor.cpu().numpy()