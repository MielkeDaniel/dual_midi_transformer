import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
import tqdm
from transformers import DynamicCache


def rotate_half(x):
    # Rotates half of the dimensions of a tensor by splitting it into two
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    # Create a new tensor with rotated dimensions
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # For caching sin and cos values
        self.register_buffer('cache', None)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]  # Get sequence length from input

        if self.cache is not None and self.cache.shape[0] >= seq_len:
            # Use cached values if available
            return self.cache[:seq_len]

        # Calculate angles based on positions
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)

        # Calculate sin and cos for embedding
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cache = emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb


def apply_rotary_pos_emb(t, freqs):
    rot_dim = freqs.shape[-1] // 2

    # Split t into two halves along last dimension
    t_left, t_pass = t[..., :rot_dim], t[..., rot_dim:]  # t_left becomes [4, 8, 8192, 32]

    # Split freqs into sin and cos
    sin, cos = freqs[..., :rot_dim], freqs[..., rot_dim:]

    # Reshape sin & cos for broadcasting
    sin = sin[..., :rot_dim].unsqueeze(0).unsqueeze(0)
    cos = cos[..., :rot_dim].unsqueeze(0).unsqueeze(0)

    t_left = (t_left * cos) + (rotate_half(t_left) * sin)

    return torch.cat([t_left, t_pass], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, heads, d_head, dropout, disable_causal=False):
        super().__init__()
        self.heads = heads
        self.d_head = d_head
        self.combined_heads = heads * d_head
        self.disable_causal = disable_causal

        self.to_q = nn.Linear(n_embd, self.combined_heads)
        self.to_k = nn.Linear(n_embd, self.combined_heads)
        self.to_v = nn.Linear(n_embd, self.combined_heads)

        # Output projection and dropout
        self.to_out = nn.Linear(self.combined_heads, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(d_head)

    def forward(self, x, mask=None):
        b, n, _ = x.size()
        device = x.device

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=self.heads)

        # Apply RoPE
        freqs = self.rope(x, seq_len=n)
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)

        # Try using flash attention first
        try:
            # Handle attention mask
            if mask is not None:
                # Ensure mask is in correct format (b, n, n)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                elif mask.dim() != 3:
                    raise ValueError(f"Mask must be 2D or 3D, got {mask.dim()}D")

                # Expand mask for all heads
                # True in mask means "attend to this position"
                attention_mask = mask.unsqueeze(1).expand(-1, self.heads, -1, -1)
                attention_mask = attention_mask.to(dtype=torch.bool)
            else:
                attention_mask = None

            # Use flash attention
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=(mask is None and not self.disable_causal)
            )

        except (RuntimeError, NotImplementedError) as e:
            # Fallback to regular attention
            scale = (self.d_head ** 0.5)
            attn_scores = (q @ k.transpose(-2, -1)) / scale

            if mask is not None:
                # Ensure mask is in correct format
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
            elif not self.disable_causal:
                # Add causal mask only if no explicit mask and causal not disabled
                causal_mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            output = attn_probs @ v

        # Reshape and project output
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.resid_dropout(self.to_out(output))
        return output

class FeedForward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, disable_causal=False):
        super().__init__()
        d_head = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, d_head, dropout, disable_causal)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        # Handle attention with cache and mask
        attn_output = self.sa(
            self.ln1(x),
            mask=mask,
        )
        x = x + attn_output

        # Feed forward
        x = x + self.ffwd(self.ln2(x))
        return x


class DualMusicTransformer(nn.Module):
    def __init__(self, vocab_size, event_context_size=96, token_context_size=8,
                 n_embd=512, event_head=8, token_head=2, event_depth=6,
                 token_depth=2, dropout=0.1, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.event_context_size = event_context_size
        self.token_context_size = token_context_size
        self.vocab_size = vocab_size

        # Event transformer (primary)
        self.event_embedding = nn.Embedding(vocab_size, n_embd)
        self.event_blocks = nn.ModuleList([
            Block(n_embd, event_head, dropout)
            for _ in range(event_depth)
        ])
        self.event_ln_f = nn.LayerNorm(n_embd)

        # Token transformer (secondary)
        self.token_blocks = nn.ModuleList([
            Block(n_embd, token_head, dropout, disable_causal=False)
            for _ in range(token_depth)
        ])
        self.token_ln_f = nn.LayerNorm(n_embd)

        # Output projection
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_token(self, hidden_state=None, x=None, cache=None):
        """
        Forward pass through token transformer

        Args:
            hidden_state: (batch_size, n_embd) - Event embedding
            x: (batch_size, token_sequence_length) - Token sequence
            cache: DynamicCache - Cache for token transformer

        Returns:
            (batch_size, 1 + token_sequence_length, vocab_size)
        """
        if hidden_state is not None:
            # Add sequence dimension for hidden state
            hidden_state = hidden_state.unsqueeze(1)  # (batch_size, 1, n_embd)

        if x is not None:
            # Embed token sequence
            x = self.event_embedding(x)
            # Concatenate with hidden state if provided
            if hidden_state is not None:
                x = torch.cat([hidden_state, x], dim=1)
            hidden_state = x

        # Create attention mask for the sequence length
        seq_length = hidden_state.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, device=hidden_state.device),
            diagonal=1
        ).bool()
        attention_mask = ~causal_mask  # True means "attend to this position"
        attention_mask = attention_mask.unsqueeze(0)  # Add batch dimension

        # Process through token transformer blocks
        for block in self.token_blocks:
            hidden_state = block(
                hidden_state,
                mask=attention_mask
            )
            if cache:
                # Update cache with current key/value states
                cache.update(block, hidden_state)

        hidden_state = self.token_ln_f(hidden_state)
        return self.lm_head(hidden_state)

    def forward(self, x, cache=None):
        """
        Forward pass through event transformer

        Args:
            x: (batch_size, midi_sequence_length, token_sequence_length)
            cache: Cache for event transformer

        Returns:
            hidden: (batch_size, midi_sequence_length, n_embd)
        """
        # Embed and sum token sequence
        x = self.event_embedding(x)
        x = x.sum(dim=-2)
        # Process through event transformer
        for block in self.event_blocks:
            x = block(x)

        return self.event_ln_f(x)

    def sample_top_p_k(self, probs, p=0.9, k=20, generator=None):
        """
        Sample from the distribution
        """
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
    def generate(self, prompt=None, batch_size=1, max_len=128, temp=1.0,
                 top_p=0.9, top_k=20, generator=None, tokenizer=None):

        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for generation")

        max_token_seq = tokenizer.max_token_seq

        # Initialize input tensor
        if prompt is None:
            input_tensor = torch.full(
                (1, max_token_seq),
                tokenizer.pad_id,
                dtype=torch.long,
                device=self.event_embedding.weight.device
            )
            input_tensor[0, 0] = tokenizer.bos_id
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = torch.cat([input_tensor] * batch_size, dim=0)
        else:
            if len(prompt.shape) == 2:
                prompt = prompt[None, :]
                prompt = np.repeat(prompt, repeats=batch_size, axis=0)
            elif prompt.shape[0] == 1:
                prompt = np.repeat(prompt, repeats=batch_size, axis=0)
            elif len(prompt.shape) != 3 or prompt.shape[0] != batch_size:
                raise ValueError(f"Invalid shape for prompt: {prompt.shape}")

            prompt = prompt[..., :max_token_seq]
            if prompt.shape[-1] < max_token_seq:
                prompt = np.pad(
                    prompt,
                    ((0, 0), (0, 0), (0, max_token_seq - prompt.shape[-1])),
                    mode="constant",
                    constant_values=tokenizer.pad_id
                )
            input_tensor = torch.from_numpy(prompt).to(
                dtype=torch.long,
                device=self.event_embedding.weight.device
            )

        cur_len = input_tensor.shape[1]
        bar = tqdm.tqdm(desc="generating", total=max_len - cur_len, disable=True)
        cache1 = DynamicCache()  # Event transformer cache
        past_len = 0

        with bar:
            while cur_len < max_len:
                end = [False] * batch_size
                hidden = self.forward(input_tensor[:, past_len:], cache=cache1)[:, -1]
                next_token_seq = None
                event_names = [""] * batch_size
                cache2 = DynamicCache()  # Token transformer cache

                # Generate tokens for each position
                for i in range(max_token_seq):
                    # print generated sequence
                    mask = torch.zeros(
                        (batch_size, tokenizer.vocab_size),
                        dtype=torch.int64,
                        device=input_tensor.device
                    )

                    # Apply token validation masks
                    for b in range(batch_size):
                        if end[b]:
                            mask[b, tokenizer.pad_id] = 1
                            continue

                        if i == 0:
                            mask[b, list(tokenizer.event_ids.values()) + [tokenizer.eos_id]] = 1
                        else:
                            param_names = tokenizer.events[event_names[b]]
                            if i > len(param_names):
                                mask[b, tokenizer.pad_id] = 1
                                continue
                            mask[b, tokenizer.parameter_ids[param_names[i - 1]]] = 1

                    mask = mask.unsqueeze(1)
                    x = next_token_seq

                    if i != 0:
                        hidden = None
                        x = x[:, -1:]

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

                if next_token_seq.shape[1] < max_token_seq:
                    next_token_seq = F.pad(
                        next_token_seq,
                        (0, max_token_seq - next_token_seq.shape[1]),
                        "constant",
                        value=tokenizer.pad_id
                    )

                next_token_seq = next_token_seq.unsqueeze(1)
                input_tensor = torch.cat([input_tensor, next_token_seq], dim=1)
                past_len = cur_len
                cur_len += 1
                bar.update(1)

                if all(end):
                    break

        return input_tensor.cpu().numpy()