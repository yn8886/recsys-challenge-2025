import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Dict, List, Optional, Tuple
from config import Config
from .enhanced_feature_encoder import EnhancedFeatureEncoder
from .hstu_modules import (
    RelativeBucketedTimeAndPositionBasedBias,
    SequentialTransductionUnitJagged,
    HSTUJagged,
    HSTUCacheState
)
import numpy as np
import lightning as L
import math

class EventType(Enum):
    # 0はpad_idx、1はmask
    PAD_IDX = 0
    MASK = 1
    PRODUCT_BUY = 2
    ADD_TO_CART = 3
    REMOVE_FROM_CART = 4
    PAGE_VISIT = 5
    SEARCH_QUERY = 6

class FusionModule(nn.Module):
    def __init__(self, input_dim, hidden1_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, output_dim)

        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class UniversalBehavioralTransformer(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lr = config.learning_rate
        self.padding_idx = config.padding_idx
        self.feature_encoder = EnhancedFeatureEncoder(config)

        # Define Relative Attention Bias module once
        self.relative_attention_bias = RelativeBucketedTimeAndPositionBasedBias(
            max_seq_len=config.max_seq_length,
            num_buckets=config.time_buckets,
            bucketization_fn=lambda x: (
                    torch.log(torch.abs(x).clamp(min=1)) / 0.69314718056  # Using ln(2)
            ).long(),
        )

        self.model = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=config.hidden_size,
                    linear_hidden_dim=config.linear_dim,  # dv per head
                    attention_dim=config.attention_dim,  # dqk per head
                    num_heads=config.num_heads,
                    relative_attention_bias_module=self.relative_attention_bias,  # Pass the shared module
                    dropout_ratio=config.dropout,  # General dropout
                    attn_dropout_ratio=config.attention_dropout,  # Specific attention dropout
                    linear_activation=config.linear_activation,
                    # normalization and linear_config are less critical now
                )
                for _ in range(config.num_layers)
            ],
            max_length=config.max_seq_length,
            autocast_dtype=None,  # Consider config.mixed_precision_dtype if you add it
        )

        # self.model = SASRec(config.max_seq_length, config.num_heads, self.d_model, config.dropout, config.item_emb_dim, config.num_layers)

        self.fusion_mlp_input_dim = config.hidden_size + config.static_features_dim
        self.fusion_mlp = FusionModule(
            input_dim=self.fusion_mlp_input_dim,
            hidden1_dim=config.fusion_mlp_hidden_dim,
            output_dim=config.hidden_size,
            dropout=config.fusion_mlp_dropout,
        )

        self.category_embeddings = self.feature_encoder.category_embedding[0]

        self.sku_embeddings = nn.Sequential(
            self.feature_encoder.item_embedding[0],
            self.feature_encoder.item_projection
        )


        self.register_buffer('task_weights', torch.tensor([
            config.task_weights['category_propensity'],
            config.task_weights['product_propensity']
        ]))

        self.loss_scale = config.loss_scale if hasattr(config, 'loss_scale') else 0.1
        self.pos_weight = config.pos_weight if hasattr(config, 'pos_weight') else 5.0
        self.use_dynamic_task_weights = config.use_dynamic_task_weights if hasattr(config,
                                                                                   'use_dynamic_task_weights') else False

        self.propensity_positive_sample_weight_boost = config.propensity_positive_sample_weight_boost


    def forward(self, batch: Dict[str, torch.Tensor], only_infer=False) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        mask = batch['event_type'] == self.padding_idx

        feature_embeddings = self.feature_encoder(
            batch['event_type'], batch['sku'], batch['url'], batch['category'], batch['price'], batch['word']
        )

        if batch['timestamp'].dtype != torch.int64:
            all_timestamps = (batch['timestamp'] * 86400).to(torch.int64)
        else:
            all_timestamps = batch['timestamps']

        seq_feat_emb, cache_states = self.model(
            x=feature_embeddings,
            all_timestamps=all_timestamps,
            mask=mask,
            device=device
        )

        # seq_feat_emb = self.model(feature_embeddings, mask)

        concat_feat = torch.concat(
            [seq_feat_emb, batch["statistical_feature"]],
            dim=-1,
        )
        user_embeddings = self.fusion_mlp(concat_feat)

        if only_infer:
            return user_embeddings

        losses = {}

        self.temperature = 0.07
        # 多正样本 NCE 损失：类别
        category_losses = []
        for i, user_emb in enumerate(user_embeddings):
            pos_ids = batch['pos_cat_ids'][i]
            if len(pos_ids) == 0:
                continue
            pos_tensor = torch.tensor(pos_ids, dtype=torch.long, device=device)
            neg_ids = batch['neg_cat_ids'][i]
            sampled_ids = torch.cat([pos_tensor, neg_ids], dim=0)
            emb = self.category_embeddings(sampled_ids)
            emb = F.normalize(emb, p=2, dim=-1)
            logits = (emb * user_emb.unsqueeze(0)).sum(-1) / self.temperature
            log_probs = torch.log_softmax(logits, dim=0)
            category_losses.append(-log_probs[:pos_tensor.size(0)].mean())
        if category_losses:
            category_loss = torch.stack(category_losses).mean()
        else:
            category_loss = torch.tensor(0.0, device=device)
        losses['category_loss'] = category_loss
        # 多正样本 NCE 损失：商品
        product_losses = []
        for i, user_emb in enumerate(user_embeddings):
            pos_ids = batch['pos_sku_ids'][i]
            if len(pos_ids) == 0:
                continue
            pos_tensor = torch.tensor(pos_ids, dtype=torch.long, device=device)
            neg_ids = batch['neg_sku_ids'][i]
            sampled_ids = torch.cat([pos_tensor, neg_ids], dim=0)
            emb = self.sku_embeddings(sampled_ids)
            emb = F.normalize(emb, p=2, dim=-1)
            logits = (emb * user_emb.unsqueeze(0)).sum(-1) / self.temperature
            log_probs = torch.log_softmax(logits, dim=0)
            product_losses.append(-log_probs[:pos_tensor.size(0)].mean())
        if product_losses:
            product_loss = torch.stack(product_losses).mean()
        else:
            product_loss = torch.tensor(0.0, device=device)
        losses['product_loss'] = product_loss

        catw, pw = self.task_weights
        weighted_category = catw * category_loss
        weighted_product = pw * product_loss
        losses['weighted_category_loss'] = weighted_category
        losses['weighted_product_loss'] = weighted_product
        total_loss = weighted_category + weighted_product
        total_loss = total_loss
        losses['loss'] = total_loss * self.loss_scale

        return {
            'user_embedding': user_embeddings,
            **losses
        }

    def clip_gradients(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

UBTModel = UniversalBehavioralTransformer