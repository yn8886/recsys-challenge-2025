import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import abc
from typing import Callable

class RelativeAttentionBiasModule(torch.nn.Module):
    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
            self,
            maxlen: int,
            num_buckets: int,
            bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._maxlen: int = maxlen
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * maxlen - 1).normal_(mean=0, std=0.02),
        )

        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
            self,
            all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._maxlen
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1: N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()

        indices = bucketed_timestamps.view(-1).to(self._ts_w.device)
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=indices
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class WordEmbedding(nn.Module):
    def __init__(self, num_word, word_emb_dim, dropout):
        super().__init__()

        self.word_embedding = nn.Embedding(num_word, word_emb_dim, padding_idx=0)
        self.word_ln = nn.LayerNorm(word_emb_dim)
        self.word_dropout = nn.Dropout(dropout)


    def forward(self, word_ids):
        word_emb = self.word_embedding(word_ids)
        avg_word_emb = torch.mean(word_emb, dim=-2)
        avg_word_emb = self.word_ln(avg_word_emb)
        avg_word_emb = self.word_dropout(avg_word_emb)
        return avg_word_emb


class EnhancedFeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 特征重要性评估 - 使用更稳定的结构
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, 64),
            nn.LayerNorm(64),  # 添加归一化层
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 特征融合 - 更新融合维度以包含新增特征
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 6, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Tanh()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用更保守的初始化方法
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)  # 减小初始权重方差

    def forward(self,
                event_types: torch.Tensor,
                sku_ids: torch.Tensor,
                url_ids: torch.Tensor,
                cat_ids: torch.Tensor,
                price_ids: torch.Tensor,
                word_ids: torch.Tensor) -> torch.Tensor:  # 新增参数

        batch_size, seq_len, _ = word_ids.shape

        event_emb = self.event_embedding(event_types)

        cat_emb = self.category_embedding(cat_ids)
        price_emb = self.price_embedding(price_ids)

        word_emb = self.word_embedding(word_ids)
        word_emb = self.word_mlp(word_emb)

        item_emb = self.item_embedding(sku_ids)
        item_emb = self.item_projection(item_emb)

        url_emb = self.url_embedding(url_ids)
        url_emb = self.url_projection(url_emb)

        # 计算特征重要性 - 添加数值稳定性 - 现在包含Item ID和URL特征
        features = [event_emb, cat_emb, price_emb, word_emb, item_emb, url_emb]
        importance_scores = []
        for feat in features:
            # 添加特征归一化
            norm_feat = F.normalize(feat, p=2, dim=-1)
            score = self.importance_net(norm_feat)
            importance_scores.append(score)

        # 加权特征融合 - 使用softmax确保权重和为1
        all_scores = torch.cat(importance_scores, dim=-1)
        normalized_scores = F.softmax(all_scores, dim=-1).unsqueeze(-1)

        weighted_features = []
        for i, feat in enumerate(features):
            weighted_features.append(feat * normalized_scores[:, :, i])

        # 特征融合
        features = torch.cat(weighted_features, dim=-1)
        fused_features = self.feature_fusion(features)

        # 检查并处理NaN值
        if torch.isnan(fused_features).any():
            logger.warning("NaN values detected in feature fusion output. Replacing with zeros.")
            fused_features = torch.where(
                torch.isnan(fused_features),
                torch.zeros_like(fused_features),
                fused_features
            )

        # 添加额外的梯度裁剪
        fused_features = torch.clamp(fused_features, min=-5.0, max=5.0)

        return fused_features


class HSTUBlock(torch.nn.Module):
    """
    HSTU (Hierarchical Sequential Transduction Unit) 模块的实现。
    该模块同时替代了标准 Transformer 中的多头注意力和前馈网络层。
    """

    def __init__(self, hidden_units, num_heads, dropout_rate, max_seq_len):
        super(HSTUBlock, self).__init__()
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.max_seq_len = max_seq_len

        # 根据论文，f1 和 f2 是简单的线性层
        # 这个单一的投影层用于一次性生成 Q, K, V 和门控向量 U
        self.f1_linear = torch.nn.Linear(hidden_units, hidden_units * 4)

        # 最终的输出投影层 f2
        self.f2_linear = torch.nn.Linear(hidden_units, hidden_units)

        # 论文中提到激活函数 φ1 和 φ2 均为 SiLU
        self.activation = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

        # 相对位置偏置 (rab)，类似于 T5 的位置偏置实现
        # self.rel_pos_bias = torch.nn.Embedding(2 * max_seq_len - 1, self.num_heads)
        relative_attention_bias_module = (
            RelativeBucketedTimeAndPositionBasedBias(
                maxlen=max_seq_len,  # accounts for next item.
                num_buckets=64,
                bucketization_fn=lambda x: (
                        torch.log(torch.abs(x).clamp(min=1)) / 0.301  # 相对时间需要更细致的分桶
                ).long(),
            )
        )
        self._rel_attn_bias = relative_attention_bias_module

    def forward(self, x, timestamp, attn_mask=None):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch, seq_len, hidden_units]。假定输入已经经过了归一化 (Pre-LN)。
            attn_mask (torch.Tensor, optional): 注意力掩码，形状为 [batch, seq_len, seq_len]。默认为 None。
        """
        batch_size, seq_len, _ = x.shape
        # --- 1. 逐点投影 (Pointwise Projection)，对应 f1 和 φ1 ---
        projected = self.f1_linear(x)
        # 在分割前应用 φ1 (SiLU) 激活函数
        activated = self.activation(projected)

        # 分割成 U, Q, K, V 四个部分
        U, Q_proj, K_proj, V_proj = torch.chunk(activated, 4, dim=-1)

        # 为多头注意力重塑 Q, K, V 的形状
        Q = Q_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V_proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # --- 2. 空间聚合 (Spatial Aggregation)，即修改后的注意力机制 ---
        scale = (self.head_dim) ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # 添加相对位置偏置 (rab)
        # positions = torch.arange(seq_len, device=x.device).view(-1, 1) - torch.arange(seq_len, device=x.device).view(
        #     1, -1
        # )
        # rel_pos_indices = positions + self.max_seq_len - 1
        # 为了支持“扩长序列”（长度可能大于 max_seq_len），这里对索引进行裁剪，避免越界
        # rel_pos_indices = torch.clamp(rel_pos_indices, min=0, max=2 * self.max_seq_len - 2)
        # rel_bias = self.rel_pos_bias(rel_pos_indices).permute(2, 0, 1).unsqueeze(0)
        timestamp = timestamp / 1e9
        rel_bias = self._rel_attn_bias(timestamp).unsqueeze(1)

        scores += rel_bias

        # 先激活
        attn_weights = self.activation(scores)
        # 再应用注意力掩码
        if attn_mask is not None:
            device = attn_weights.device
            attn_mask = attn_mask.to(device)
            attn_weights = attn_weights.masked_fill(attn_mask.unsqueeze(1).logical_not(), 0.0)

        # 应用 φ2 (SiLU) 激活函数，替代 Softmax

        attn_weights = self.dropout(attn_weights)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        # --- 3. 逐点变换 (Pointwise Transformation) ---
        # 使用 U 进行门控 (逐元素相乘)，然后通过 f2 进行最终投影
        # 论文公式为 f2(Norm(attn_output) * U)，由于我们采用 Pre-LN 架构，直接应用门控
        gated_output = attn_output * U
        final_output = self.f2_linear(gated_output)

        return final_output