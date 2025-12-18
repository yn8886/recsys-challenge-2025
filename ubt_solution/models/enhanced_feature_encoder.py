import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init # Added init for _initialize_weights
from typing import Dict, List, Optional, Tuple # Though not directly used by this class, good practice if it were
from config import Config
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 事件类型编码
        self.event_embedding = nn.Sequential(
            nn.Embedding(5 + 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )

        self.category_embedding = nn.Sequential(
            nn.Embedding(config.num_cat, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        self.price_embedding = nn.Sequential(
            nn.Embedding(102, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )

        self.word_embedding = WordEmbedding(
            num_word=config.num_word,
            word_emb_dim=config.hidden_size,
            dropout=config.dropout,
        )
        
        # 新增：Item ID编码
        self.item_embedding = nn.Sequential(
            nn.Embedding(config.num_sku, config.item_emb_dim, padding_idx=0),
            nn.LayerNorm(config.item_emb_dim),
            nn.Dropout(config.dropout)
        )
        
        # URL编码
        self.url_embedding = nn.Sequential(
            nn.Embedding(config.num_url, config.url_emb_dim, padding_idx=0),
            nn.LayerNorm(config.url_emb_dim),
            nn.Dropout(config.dropout)
        )
        
        # URL和Item ID特征映射到隐藏层大小
        self.item_projection = nn.Sequential(
            nn.Linear(config.item_emb_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.url_projection = nn.Sequential(
            nn.Linear(config.url_emb_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
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
        
        # 新增：name 嵌入的 MLP 映射
        self.word_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
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
            weighted_features.append(feat * normalized_scores[:,:,i])
        
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