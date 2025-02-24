import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange  # For easier tensor manipulations


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
    t_left, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    sin, cos = freqs[..., :rot_dim], freqs[..., rot_dim:]

    sin = sin[..., :rot_dim].unsqueeze(0).unsqueeze(0)
    cos = cos[..., :rot_dim].unsqueeze(0).unsqueeze(0)

    t_left = (t_left * cos.clone()) + (rotate_half(t_left) * sin.clone())

    return torch.cat([t_left, t_pass], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.flash = True

        # Key, Query, Value projections
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = dropout

        # Add RoPE
        self.rope = RotaryEmbedding(dim_head)

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        h = self.heads
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=h)

        freqs = self.rope(x, seq_len=n)
        q = apply_rotary_pos_emb(q, freqs)
        k = apply_rotary_pos_emb(k, freqs)

        # Create mask with correct dimensions for multi-head attention
        if mask is not None:
            # Ensure mask has correct shape [batch, heads, seq_len, seq_len]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask = mask.expand(b, h, n, n)

        # In MultiHeadAttention, add the return statement:
        if self.flash and torch.cuda.is_available():
            with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                                enable_math=False,
                                                enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.,
                    is_causal=mask is None
                )
        else:
            scale = (self.d_head ** 0.5)
            attn_scores = (q @ k.transpose(-2, -1)) / scale

            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            out = attn_probs @ v

        return self.to_out(rearrange(out, 'b h n d -> b n (h d)')), k, v


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        # followed by GELU activation (modern alternative to ReLU)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # Each token embedding is separately passed through the FFN
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, hidden_dim=hidden_dim if hidden_dim is not None else 4 * dim, dropout=dropout)

        # Layer normalizations
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention with residual connection and layer norm
        attn_output, k, v = self.attention(self.norm1(x), mask=mask)
        x = x + self.dropout(attn_output)
        # Feed forward with residual connection and layer norm
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, k, v


class MusicTransformer(nn.Module):
    def __init__(self,
                 num_tokens=836,
                 dim=512,  # Model dimension
                 depth=6,  # Number of decoder blocks
                 heads=8,  # Number of attention heads
                 dim_head=64,  # Dimension of each head
                 max_seq_len=128,  # Maximum sequence length
                 dropout=0.1):  # Dropout rate
        super().__init__()

        # Token embedding layer (converts token IDs to vectors)
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

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

    def forward(self, x, mask=None):
        b, n = x.shape
        x = self.token_emb(x)
        x = self.dropout(x)

        # Create causal mask with right shape here
        if mask is None:
            mask = torch.triu(torch.ones((b, 1, n, n), device=x.device), diagonal=1).bool()
            mask = ~mask

        for layer in self.layers:
            x = layer(x, mask)
            print(x.shape)

        x = self.norm(x)
        return self.to_logits(x)


def exists(val):
    return val is not None