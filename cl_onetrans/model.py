import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from fuxictr.pytorch.layers import MLP_Block

# -----------------------------------------------------------------------------
# 1. 基础组件 (保持不变或微调)
# -----------------------------------------------------------------------------

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

        # 1. Shared Weights (for S-tokens)
        self.q_shared = nn.Linear(d_model, d_model, bias=False)
        self.k_shared = nn.Linear(d_model, d_model, bias=False)
        self.v_shared = nn.Linear(d_model, d_model, bias=False)

        # 2. Specific Weights (for NS-tokens)
        self.q_ns = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(ns_len)])
        self.k_ns = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(ns_len)])
        self.v_ns = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(ns_len)])

        # max_seq_len = seq_len + ns_len
        # self.rel_pos_bias = torch.nn.Embedding(2 * max_seq_len - 1, self.num_heads)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _apply_mixed_projection(self, x, shared_layer, ns_layers_list):
        B, L, D = x.shape
        L_s = L - self.ns_len

        # 投影 S-tokens
        if L_s > 0:
            s_part = x[:, :L_s, :]
            s_proj = shared_layer(s_part)
        else:
            s_proj = torch.empty(B, 0, D, device=x.device)

        # 投影 NS-tokens (为了性能，实际工程中通常使用 Grouped Linear 或 vmap，这里保持逻辑清晰)
        ns_projs = []
        for i in range(self.ns_len):
            idx = L_s + i  # NS token 始终位于序列末尾
            if idx >= 0:  # 确保索引有效
                # 注意：这里我们取 idx:idx+1 保持维度，方便后续 cat
                ns_projs.append(ns_layers_list[i](x[:, idx: idx + 1, :]))

        if len(ns_projs) > 0:
            ns_proj = torch.cat(ns_projs, dim=1)
            return torch.cat([s_proj, ns_proj], dim=1)
        return s_proj

    def forward(self, x, query_len=None, key_padding_mask=None):
        """
        x: [B, L_in, D] - 当前层的完整输入
        query_len: int - 本层输出序列的长度 (即 Query 的数量)
        key_padding_mask: [B, L_in] - 对应输入 x 的 mask
        """
        B, L_in, _ = x.shape

        # 1. Key & Value: 基于完整输入计算
        #    这意味着即使上一层传下来 500 个 token，我们虽然只输出 400 个，
        #    但这 400 个 Query 依然会 attend 到这 500 个 Key 上。
        k = self._apply_mixed_projection(x, self.k_shared, self.k_ns)
        v = self._apply_mixed_projection(x, self.v_shared, self.v_ns)

        # 2. Query: 进行剪枝 (Pruning)
        #    根据 Pyramid Schedule，我们只取尾部的 query_len 个 Token 作为 Query
        #    因为 NS Tokens 在尾部，且 query_len >= ns_len，所以 NS Tokens 总是被保留
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

        # positions = torch.arange(L_k, device=x.device).view(-1, 1) - torch.arange(L_k, device=x.device).view(1,-1)
        # rel_pos_indices = positions + L_k - 1
        # rel_bias = self.rel_pos_bias(rel_pos_indices).permute(2, 0, 1).unsqueeze(0)
        # rel_bias = rel_bias[:, :, -L_q:, :]

        attention_mask_tril = torch.tril(torch.ones((L_k, L_k), dtype=torch.bool, device=x.device))
        attn_mask = attention_mask_tril[-L_q:, :]
        # key_padding_mask = (1.0 - key_padding_mask).bool().to(x.device)
        key_padding_mask = key_padding_mask.bool().to(x.device)
        attention_mask = attn_mask.unsqueeze(0) & key_padding_mask.unsqueeze(1)

        t = attention_mask[0, :, :]
        # 计算 Attention Score
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # attn_weights = attn_weights + rel_bias
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
        # x 的长度已经是被裁剪过的 L_q
        B, L, D = x.shape
        L_s = L - self.ns_len

        # Shared Part
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


# -----------------------------------------------------------------------------
# 2. OneTrans Block (单层)
# -----------------------------------------------------------------------------

class OneTransBlock(nn.Module):
    def __init__(self, ns_len, seq_len, d_model, num_heads):
        super().__init__()
        self.rms_attn = RMSNorm(d_model)
        self.attn = MixedCausalAttention(ns_len, seq_len, d_model, num_heads)
        self.rms_ffn = RMSNorm(d_model)
        self.ffn = MixedFFN(ns_len, d_model)

    def forward(self, x, keep_len=None, mask=None):
        """
        输入:
            x: [B, L_prev, D] (上一层的输出)
            keep_len: int (本层想要保留的长度)
            mask: [B, L_prev] (上一层对应的 mask)
        输出:
            x_new: [B, keep_len, D]
        """
        # 1. Attention (输入 L_prev -> 输出 keep_len)
        norm_x = self.rms_attn(x)
        attn_out = self.attn(norm_x, query_len=keep_len, key_padding_mask=mask)

        # 2. Residual Connection (关键点)
        # 残差连接必须把 Input 对齐到 Output 的形状。
        # 因为 Output 是 Input 的尾部生成的，所以 Input 也要取尾部。
        if keep_len is not None and keep_len < x.shape[1]:
            x_residual = x[:, -keep_len:, :]
        else:
            x_residual = x

        x = x_residual + attn_out

        # 3. FFN (输入 keep_len -> 输出 keep_len)
        # 此时序列长度已经变短了
        norm_x = self.rms_ffn(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out

        return x


# -----------------------------------------------------------------------------
# 3. OneTrans Model (扁平化架构 + 线性调度)
# -----------------------------------------------------------------------------

class OneTransModel(nn.Module):
    def __init__(self,
                 num_layers=6,  # 总层数 (Depth)
                 final_seq_len=12,  # 最终保留的 Token 数 (通常 >= ns_len)
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
        self.final_seq_len = max(final_seq_len, ns_len)  # 确保至少保留 NS tokens
        self.num_layers = num_layers

        # NS Tokenizer (这里简化处理)
        self.ns_tokenizer = nn.Sequential(
            nn.Linear(ns_input_dim, ns_len * d_model),
            nn.SiLU()
        )

        # Pyramid Stack: 直接是一个 ModuleList
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
        """
        s_tokens: [B, L_s, D]
        ns_features: [B, D_feat]
        s_padding_mask: [B, L_s] (True for padding)
        """
        B = s_tokens.shape[0]

        # 1. Prepare Inputs
        ns_flat = self.ns_tokenizer(ns_features)
        ns_tokens = ns_flat.view(B, self.ns_len, self.d_model)

        # 拼接: [S-Tokens, NS-Tokens]
        x = torch.cat([s_tokens, ns_tokens], dim=1)

        # 构造初始 Mask (NS Tokens 假设不是 Padding)
        ns_mask = torch.ones((B, self.ns_len), device=s_tokens.device, dtype=s_padding_mask.dtype)
        current_mask = torch.cat([s_padding_mask, ns_mask], dim=1)  # [B, L_initial]

        L_in = x.shape[1]

        # 2. 计算 Linear Pyramid Schedule (线性金字塔调度)
        # 每一层我们要把长度从 L_in 减少到 final_seq_len
        # 公式: Target_i = L_in - (L_in - L_final) * (i+1) / N
        schedule = []
        total_drop = L_in - self.final_seq_len
        for i in range(self.num_layers):
            drop_amount = int(total_drop * (i + 1) / self.num_layers)
            target_len = L_in - drop_amount
            # 保证不低于最小长度
            target_len = max(target_len, self.final_seq_len)
            schedule.append(target_len)

        # 3. 逐层前向传播
        for i, block in enumerate(self.blocks):
            target_len = schedule[i]

            # Block Forward:
            # 输入: x (长度 L_prev)
            # 输出: x (长度 target_len)
            x = block(x, keep_len=target_len, mask=current_mask)

            # 更新 Mask 以匹配下一层的输入
            # 下一层的 Input 就是这一层的 Output，是当前 Mask 的尾部
            if target_len < current_mask.shape[1]:
                current_mask = current_mask[:, -target_len:]

        # 4. Final Output
        # 取 NS 部分进行最终预测 (通常 Pooling 或者取特定 Token)
        # 此时 x 的长度为 final_seq_len, 其中最后 ns_len 个是 NS tokens
        ns_out = x[:, -self.ns_len:, :]
        ns_out= torch.mean(ns_out, dim=1)

        final_emb = self.final_mlp(ns_out)
        return final_emb


# 测试代码
if __name__ == "__main__":
    # 模拟数据
    B, L_s, D = 2, 100, 32
    ns_len = 5
    s_tokens = torch.randn(B, L_s, D)
    ns_features = torch.randn(B, 64)
    mask = torch.zeros(B, L_s).bool()  # 全 False，无 Padding

    model = OneTransModel(num_layers=4, final_seq_len=10, d_model=D, ns_len=ns_len, seq_len=L_s, ns_input_dim=64)
    out = model(s_tokens, ns_features, mask)

    print("Output shape:", out.shape)  # [B, D]

    # 打印 schedule 验证
    L_total = L_s + ns_len  # 105
    L_final = 10
    print(f"Start Len: {L_total}, Final Len: {L_final}")
    # 模拟 Schedule 计算
    total_drop = L_total - L_final
    for i in range(4):
        drop = int(total_drop * (i + 1) / 4)
        print(f"Layer {i + 1} Target: {L_total - drop}")