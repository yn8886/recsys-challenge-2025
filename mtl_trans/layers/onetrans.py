import torch
import torch.nn as nn
from fuxictr.pytorch.layers import MLP_Block
import math
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


class FFNLayer(nn.Module):
    def __init__(self, d_model, expansion_factor=2, dropout=0.1):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        self.dense_1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.SiLU()
        self.dense_2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.dense_2(self.activation(self.dense_1(x))))


class MixedCausalAttention(nn.Module):
    def __init__(self, ns_len, seq_len, d_model, num_heads, dropout=0.1, rope_fraction=1.0,
                 rope_base=10000.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.ns_len = ns_len

        self.q_shared = nn.Linear(d_model, d_model, bias=False)
        self.k_shared = nn.Linear(d_model, d_model, bias=False)
        self.v_shared = nn.Linear(d_model, d_model, bias=False)

        self.q_ns = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(ns_len)])
        self.k_ns = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(ns_len)])
        self.v_ns = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(ns_len)])

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _apply_mixed_projection(self, x, shared_layer, ns_layers_list):
        B, L, D = x.shape
        L_s = L - self.ns_len

        if L_s > 0:
            s_part = x[:, :L_s, :]
            s_proj = shared_layer(s_part)
        else:
            s_proj = torch.empty(B, 0, D, device=x.device)

        ns_projs = []
        for i in range(self.ns_len):
            idx = L_s + i
            if idx >= 0:
                ns_projs.append(ns_layers_list[i](x[:, idx: idx + 1, :]))

        if len(ns_projs) > 0:
            ns_proj = torch.cat(ns_projs, dim=1)
            return torch.cat([s_proj, ns_proj], dim=1)
        return s_proj

    def forward(self, x, query_len=None, key_padding_mask=None):
        B, L_in, _ = x.shape
        k = self._apply_mixed_projection(x, self.k_shared, self.k_ns)
        v = self._apply_mixed_projection(x, self.v_shared, self.v_ns)

        q_full = self._apply_mixed_projection(x, self.q_shared, self.q_ns)

        if query_len is not None and query_len < L_in:
            q = q_full[:, -query_len:, :]  # 只取最后 query_len 个
        else:
            q = q_full

        L_q = q.shape[1]
        L_k = k.shape[1]  # L_k == L_in

        # Multi-head reshape
        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        attention_mask_tril = torch.tril(torch.ones((L_k, L_k), dtype=torch.bool, device=x.device))
        attn_mask = attention_mask_tril[-L_q:, :]
        # key_padding_mask = (1.0 - key_padding_mask).bool().to(x.device)
        key_padding_mask = key_padding_mask.bool().to(x.device)
        attention_mask = attn_mask.unsqueeze(0) & key_padding_mask.unsqueeze(1)


        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).logical_not(), -1e9)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        return self.out_proj(output)


class MixedFFN(nn.Module):
    def __init__(self, ns_len, d_model):
        super().__init__()
        self.ns_len = ns_len
        self.ffn_shared = FFNLayer(d_model)
        self.ffn_ns = nn.ModuleList([FFNLayer(d_model) for _ in range(ns_len)])

    def forward(self, x):
        B, L, D = x.shape
        L_s = L - self.ns_len

        if L_s > 0:
            s_out = self.ffn_shared(x[:, :L_s, :])
        else:
            s_out = torch.empty(B, 0, D, device=x.device)

        # Specific Part
        ns_outs = []
        for i in range(self.ns_len):
            idx = L_s + i
            if idx >= 0:
                ns_outs.append(self.ffn_ns[i](x[:, idx: idx + 1, :]))

        if len(ns_outs) > 0:
            ns_out = torch.cat(ns_outs, dim=1)
            return torch.cat([s_out, ns_out], dim=1)
        return s_out


class OneTransBlock(nn.Module):
    def __init__(self, ns_len, seq_len, d_model, num_heads):
        super().__init__()
        self.rms_attn = RMSNorm(d_model)
        self.attn = MixedCausalAttention(ns_len, seq_len, d_model, num_heads)
        self.rms_ffn = RMSNorm(d_model)
        self.ffn = MixedFFN(ns_len, d_model)

    def forward(self, x, keep_len=None, mask=None):
        norm_x = self.rms_attn(x)
        attn_out = self.attn(norm_x, query_len=keep_len, key_padding_mask=mask)

        if keep_len is not None and keep_len < x.shape[1]:
            x_residual = x[:, -keep_len:, :]
        else:
            x_residual = x

        x = x_residual + attn_out

        norm_x = self.rms_ffn(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out

        return x

class OneTransModel(nn.Module):
    def __init__(self,
                 num_layers=6,
                 final_seq_len=12,
                 d_model=256,
                 num_heads=4,
                 ns_len=10,
                 seq_len=64,
                 ns_input_dim=64,
                 last_embed_dim=256,
                 mlp_hidden_units=[256],
                 dropout=0.1,
    ):
        super().__init__()

        self.ns_len = ns_len
        self.d_model = d_model
        self.final_seq_len = max(final_seq_len, ns_len)
        self.num_layers = num_layers

        self.ns_tokenizer = nn.Sequential(
            nn.Linear(ns_input_dim, ns_len * d_model),
            nn.SiLU()
        )

        self.blocks = nn.ModuleList([
            OneTransBlock(ns_len, seq_len, d_model, num_heads)
            for _ in range(num_layers)
        ])

        self.final_mlp = MLP_Block(
            input_dim=d_model,
            output_dim=last_embed_dim,
            hidden_units=mlp_hidden_units,
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rates=dropout
        )

    def forward(self, s_tokens, ns_features, s_padding_mask):
        B = s_tokens.shape[0]

        ns_flat = self.ns_tokenizer(ns_features)
        ns_tokens = ns_flat.view(B, self.ns_len, self.d_model)

        x = torch.cat([s_tokens, ns_tokens], dim=1)

        ns_mask = torch.ones((B, self.ns_len), device=s_tokens.device, dtype=s_padding_mask.dtype)
        current_mask = torch.cat([s_padding_mask, ns_mask], dim=1)  # [B, L_initial]

        L_in = x.shape[1]

        schedule = []
        total_drop = L_in - self.final_seq_len
        for i in range(self.num_layers):
            drop_amount = int(total_drop * (i + 1) / self.num_layers)
            target_len = L_in - drop_amount
            target_len = max(target_len, self.final_seq_len)
            schedule.append(target_len)

        for i, block in enumerate(self.blocks):
            target_len = schedule[i]
            x = block(x, keep_len=target_len, mask=current_mask)

            if target_len < current_mask.shape[1]:
                current_mask = current_mask[:, -target_len:]

        ns_out = x[:, -self.ns_len:, :]
        ns_out = torch.mean(ns_out, dim=1)
        final_emb = self.final_mlp(ns_out)

        return final_emb