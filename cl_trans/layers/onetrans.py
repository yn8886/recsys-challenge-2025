import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from fuxictr.pytorch.layers import MLP_Block

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization used for Pre-Norm[cite: 202]."""

    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class MixLinear(nn.Module):
    """
    Implements the Mixed Parameterization strategy[cite: 166].
    Shares weights for S-tokens and uses token-specific weights for NS-tokens[cite: 199].
    """

    def __init__(self, d_in, d_out, L_ns):
        super().__init__()
        self.L_ns = L_ns
        # Shared projection for sequential (S) tokens
        self.shared_proj = nn.Linear(d_in, d_out)

        # Token-specific projections for non-sequential (NS) tokens
        self.ns_proj_weight = nn.Parameter(torch.empty(L_ns, d_in, d_out))
        self.ns_proj_bias = nn.Parameter(torch.zeros(L_ns, d_out))

        nn.init.xavier_uniform_(self.ns_proj_weight)

    def forward(self, x, L_s):
        # x shape: (Batch, L_s + L_ns, d_in)
        x_s = x[:, :L_s, :]
        x_ns = x[:, L_s:, :]

        # 1. Shared projection
        out_s = self.shared_proj(x_s)

        # 2. Token-specific projection via Einstein summation
        out_ns = torch.einsum('b n i, n i o -> b n o', x_ns, self.ns_proj_weight) + self.ns_proj_bias

        return torch.cat([out_s, out_ns], dim=1)


class MixMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, L_ns):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Mixed Q, K, V projections [cite: 205, 208]
        self.W_q = MixLinear(d_model, d_model, L_ns)
        self.W_k = MixLinear(d_model, d_model, L_ns)
        self.W_v = MixLinear(d_model, d_model, L_ns)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, L_s, L_s_out, padding_mask):
        B, L, _ = x.shape
        L_ns = self.W_q.L_ns

        # Full Keys and Values over the entire sequence [cite: 247]
        K = self.W_k(x, L_s).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x, L_s).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if L_s_out < L_s:
            q_input = torch.cat([x[:, L_s - L_s_out: L_s, :], x[:, L_s:, :]], dim=1)
        else:
            q_input = x

        L_q = L_s_out + L_ns
        L_k = L_s + L_ns
        Q = self.W_q(q_input, L_s_out).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal Attention Mask [cite: 168]
        # Q length is (L_s_out + L_ns), K/V length is (L_s + L_ns)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Generate causal mask aligned with the retained tail
        causal_mask = torch.ones(L_q, L_k, dtype=torch.bool, device=x.device)
        causal_mask = torch.tril(causal_mask, diagonal=L_k-L_q)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        ns_mask = torch.ones(B, L_ns, dtype=torch.bool, device=x.device)
        full_padding_mask = torch.cat([padding_mask.bool(), ns_mask], dim=1)
        full_padding_mask = full_padding_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = causal_mask & full_padding_mask

        t = attn_mask[0,0]
        # attn_mask = causal_mask

        attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L_s_out + L_ns, self.d_model)

        return self.out_proj(out)


class MixFFN(nn.Module):
    def __init__(self, d_model, d_ff, L_ns):
        super().__init__()
        self.fc1 = MixLinear(d_model, d_ff, L_ns)
        self.act = nn.GELU()
        self.fc2 = MixLinear(d_ff, d_model, L_ns)

    def forward(self, x, L_s):
        return self.fc2(self.act(self.fc1(x, L_s)), L_s)


class OneTransBlock(nn.Module):
    """A pre-norm causal Transformer block with mixed parameterization[cite: 197]."""

    def __init__(self, d_model, num_heads, d_ff, L_ns):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MixMultiHeadAttention(d_model, num_heads, L_ns)
        self.norm2 = RMSNorm(d_model)
        self.ffn = MixFFN(d_model, d_ff, L_ns)

    def forward(self, x, L_s, L_s_out, padding_mask):
        # Apply attention with pyramid truncation
        attn_out = self.attn(self.norm1(x), L_s, L_s_out, padding_mask)

        # Truncate residual connection to match pyramid output
        if L_s_out < L_s:
            residual = torch.cat([x[:, L_s - L_s_out: L_s, :], x[:, L_s:, :]], dim=1)
        else:
            residual = x

        z = residual + attn_out
        y = z + self.ffn(self.norm2(z), L_s_out)
        return y


class OneTransModel(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        num_layers,
        L_ns,
        initial_L_s,
        final_L_s,
        ns_raw_dim,
        last_embed_dim=256,
        mlp_hidden_units=[256],
        dropout=0.1,
    ):

        super().__init__()
        self.L_ns = L_ns
        self.pyramid_schedule = [int(x) for x in np.linspace(initial_L_s, final_L_s, num_layers + 1)[1:]]

        self.auto_split_mlp = nn.Linear(ns_raw_dim, L_ns * d_model)

        self.layers = nn.ModuleList([
            OneTransBlock(d_model, num_heads, d_model * 4, L_ns)
            for _ in range(num_layers)
        ])

        self.final_mlp = MLP_Block(
            input_dim=d_model * (L_ns + 1) if final_L_s > 0 else d_model * L_ns,
            output_dim=last_embed_dim,
            hidden_units=mlp_hidden_units,
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rates=dropout
        )

    def forward(self, s_tokens, raw_ns_features, padding_mask):
        B, L_s, d_model = s_tokens.shape

        # 1. Tokenize NS Features via Auto-Split [cite: 210]
        ns_tokens = self.auto_split_mlp(raw_ns_features).view(B, self.L_ns, d_model)

        # 2. Create Unified Token Sequence [cite: 159, 160]
        x = torch.cat([s_tokens, ns_tokens], dim=1)

        # 3. Pass through Pyramid Stack [cite: 170]
        current_L_s = L_s
        for idx, layer in enumerate(self.layers):
            target_L_s = self.pyramid_schedule[idx]
            x = layer(x, current_L_s, target_L_s, padding_mask)
            padding_mask = padding_mask[:,-target_L_s:]
            current_L_s = target_L_s

        # 4. Extract final distilled NS-tokens for prediction [cite: 170, 171]
        # final_ns_tokens = x[:, current_L_s:, :]
        if current_L_s == 0:
            final_ns_tokens = x
        else:
            final_ns_tokens = torch.concat([x[:, :current_L_s, :].mean(dim=1).unsqueeze(1), x[:, current_L_s:, :]], dim=1)
        out = self.final_mlp(final_ns_tokens.flatten(start_dim=1))

        return out