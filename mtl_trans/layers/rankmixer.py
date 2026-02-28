import torch
import torch.nn as nn
import torch.nn.functional as F

class ParameterFreeTokenMixer(nn.Module):
    def __init__(self, num_tokens: int, d_model: int, num_heads: int = None, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads) if num_heads is not None else int(num_tokens)
        self.dropout = nn.Dropout(float(dropout))

        if self.num_heads != self.num_tokens:
            raise ValueError("Parameter-free token mixing requires num_heads == num_tokens.")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model must be divisible by num_heads, got d_model={self.d_model} num_heads={self.num_heads}"
            )

        self.d_head = self.d_model // self.num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, T, D]
        batch_size = x.size(0)
        t_count = self.num_tokens
        h_count = self.num_heads
        d_head = self.d_head

        split = x.view(batch_size, t_count, h_count, d_head)
        shuffled = split.transpose(1, 2).contiguous()
        mixed = shuffled.view(batch_size, t_count, self.d_model)

        return self.dropout(mixed)


class PerTokenSparseMoE(nn.Module):
    def __init__(
            self,
            num_tokens: int,
            d_model: int,
            mult: int = 4,
            num_experts: int = 4,
            dropout: float = 0.0,
            l1_coef: float = 0.0,
            sparsity_ratio: float = 1.0,
            use_dtsi: bool = True,
            routing_type: str = "relu_dtsi"
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.num_experts = int(num_experts)
        self.dropout_prob = float(dropout)
        self.l1_coef = float(l1_coef)
        self.sparsity_ratio = float(sparsity_ratio) if sparsity_ratio else 1.0
        self.use_dtsi = bool(use_dtsi)
        self.routing_type = str(routing_type).lower()

        hidden_dim = self.d_model * self.mult

        # 专家网络权重
        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, self.d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, self.num_experts, hidden_dim, self.d_model))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts, self.d_model))

        # 训练路由权重
        self.gate_w_train = nn.Parameter(torch.empty(self.num_tokens, self.d_model, self.num_experts))
        self.gate_b_train = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts))

        # 推理路由权重 (DTSI)
        if self.use_dtsi:
            self.gate_w_infer = nn.Parameter(torch.empty(self.num_tokens, self.d_model, self.num_experts))
            self.gate_b_infer = nn.Parameter(torch.zeros(self.num_tokens, self.num_experts))

        self._init_weights()

    def _init_weights(self):
        # 对应 TF: variance_scaling_initializer(scale=2.0)
        nn.init.kaiming_normal_(self.W1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W2, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.gate_w_train, mode='fan_in', nonlinearity='linear')
        if self.use_dtsi:
            nn.init.kaiming_normal_(self.gate_w_infer, mode='fan_in', nonlinearity='linear')

    def _router_logits(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # btd (batch, token, dim) * tde (token, dim, expert) -> bte
        return torch.einsum("btd,tde->bte", x, w) + b

    def forward(self, x: torch.Tensor):
        # x 形状: [B, T, D]

        # 1. 计算所有专家的输出
        h = torch.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        h = F.gelu(h)
        if self.dropout_prob > 0 and self.training:
            h = F.dropout(h, p=self.dropout_prob, training=self.training)

        expert_out = torch.einsum("bteh,tehd->bted", h, self.W2) + self.b2
        if self.dropout_prob > 0 and self.training:
            expert_out = F.dropout(expert_out, p=self.dropout_prob, training=self.training)

        # 2. 训练阶段门控
        gate_train_logits = self._router_logits(x, self.gate_w_train, self.gate_b_train)
        if self.routing_type == "relu_dtsi":
            gate_train = F.softmax(gate_train_logits, dim=-1)
        elif self.routing_type == "relu":
            gate_train = F.relu(gate_train_logits)
        else:
            raise ValueError(f"Unsupported routing_type: {self.routing_type}")

        # 3. 推理阶段门控
        if self.use_dtsi:
            gate_infer_logits = self._router_logits(x, self.gate_w_infer, self.gate_b_infer)
            gate_infer = F.relu(gate_infer_logits)
        else:
            gate_infer = gate_train

        # 4. 根据所处模式选择 gate，并进行专家特征加权求和
        gate = gate_train if self.training else gate_infer
        # expert_out: [B, T, E, D], gate: [B, T, E, 1] -> [B, T, D]
        y = torch.sum(expert_out * gate.unsqueeze(-1), dim=2)

        # 5. L1 稀疏正则化计算
        if self.l1_coef > 0.0:
            scale = 1.0 / max(self.sparsity_ratio, 1e-6)
            # tf.reduce_mean(tf.reduce_sum(gate_infer, axis=-1))
            l1_loss = self.l1_coef * scale * torch.mean(torch.sum(gate_infer, dim=-1))
        else:
            l1_loss = torch.tensor(0.0, device=x.device)

        return y, l1_loss

class PerTokenFFN(nn.Module):
    def __init__(self, num_tokens: int, d_model: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.dropout = nn.Dropout(float(dropout))

        hidden_dim = self.d_model * self.mult

        self.W1 = nn.Parameter(torch.empty(self.num_tokens, self.d_model, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(self.num_tokens, hidden_dim))
        self.W2 = nn.Parameter(torch.empty(self.num_tokens, hidden_dim, self.d_model))
        self.b2 = nn.Parameter(torch.zeros(self.num_tokens, self.d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W1, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.W2, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("btd,tdh->bth", x, self.W1) + self.b1
        h = F.gelu(h)
        h = self.dropout(h)

        y = torch.einsum("bth,thd->btd", h, self.W2) + self.b2
        y = self.dropout(y)

        return y

class RankMixerBlock(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        ffn_mult=4,
        token_dp=0.0,
        ffn_dp=0.0,
        ln_style="pre",
        use_moe=False,
        moe_experts=4,
        moe_l1_coef=0.0,
        moe_sparsity_ratio=1.0,
        moe_use_dtsi=True,
        moe_routing_type="relu_dtsi"
    ):
        super().__init__()
        self.ln_style = str(ln_style).lower()
        self.use_moe = bool(use_moe)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.token_mixer = ParameterFreeTokenMixer(num_tokens, d_model=d_model, dropout=token_dp)

        if self.use_moe:
            self.per_token_ffn = PerTokenSparseMoE(
                num_tokens=num_tokens,
                d_model=d_model,
                mult=ffn_mult,
                num_experts=moe_experts,
                dropout=ffn_dp,
                l1_coef=moe_l1_coef,
                sparsity_ratio=moe_sparsity_ratio,
                use_dtsi=moe_use_dtsi,
                routing_type=moe_routing_type
            )
        else:
            self.per_token_ffn = PerTokenFFN(
                num_tokens=num_tokens, d_model=d_model, mult=ffn_mult, dropout=ffn_dp
            )

    def forward(self, x):
        moe_loss = torch.tensor(0.0, device=x.device)

        if self.ln_style == "post":
            y = self.token_mixer(x)
            x = self.ln1(x + y)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(x)
            else:
                z = self.per_token_ffn(x)
            out = self.ln2(x + z)
        else:  # pre
            y = self.ln1(x)
            y = self.token_mixer(y)
            x = x + y
            z = self.ln2(x)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(z)
            else:
                z = self.per_token_ffn(z)
            out = x + z

        return out, moe_loss


class RankMixerEncoder(nn.Module):
    def __init__(self, num_layers, num_tokens, d_model, **kwargs):
        super().__init__()
        use_final_ln = kwargs.pop("use_final_ln", True)
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens=num_tokens, d_model=d_model, **kwargs)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model) if use_final_ln else nn.Identity()

    def forward(self, x):
        total_moe_loss = torch.tensor(0.0, device=x.device)
        for blk in self.blocks:
            x, blk_moe_loss = blk(x)
            total_moe_loss += blk_moe_loss

        return self.final_ln(x), total_moe_loss