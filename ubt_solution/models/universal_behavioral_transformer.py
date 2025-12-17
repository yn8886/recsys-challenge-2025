import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Dict, List, Optional, Tuple
from config import Config
from .hstu_modules import (
    RelativeBucketedTimeAndPositionBasedBias,
    SequentialTransductionUnitJagged,
    HSTUJagged,
    HSTUCacheState
)
from .trm_modules import SASRec
from .task_specific_encoder import TaskSpecificEncoder
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


class SkuEmbedding(nn.Module):
    def __init__(
        self,
        config,
        num_sku,
        num_cat,
        num_price,
        word_emb_layer,
        item_emb_dim,
        padding_idx=0,
    ):
        super().__init__()
        self.sku_emb_layer = nn.Sequential(
            nn.Embedding(
                num_embeddings=num_sku,
                embedding_dim=config.sku_emb_dim,
                padding_idx=padding_idx,
            ),
            nn.LayerNorm(config.sku_emb_dim),
            nn.Dropout(config.dropout)
        )

        self.sku_projection = nn.Sequential(
            nn.Linear(config.sku_emb_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.cat_emb_layer = nn.Sequential(
            nn.Embedding(
                num_embeddings=num_cat,
                embedding_dim=config.hidden_size,
                padding_idx=padding_idx,
            ),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )

        self.price_emb_layer = nn.Sequential(
            nn.Embedding(
                num_embeddings=num_price,
                embedding_dim=config.hidden_size,
                padding_idx=padding_idx,
            ),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout)
        )

        self.word_emb_layer = word_emb_layer

        self.fc1 = nn.Linear(
            3 * config.hidden_size + item_emb_dim,
            item_emb_dim,
        )
        self.relu = nn.ReLU()

    def forward(self, sku_id, cat_id, price_id, word_ids):
        sku_emb = self.sku_emb_layer(sku_id)
        sku_emb = self.sku_projection(sku_emb)
        cat_emb = self.cat_emb_layer(cat_id)
        price_emb = self.price_emb_layer(price_id)
        word_emb = self.word_emb_layer(word_ids)
        concat_emb = torch.cat([sku_emb, cat_emb, price_emb, word_emb], dim=-1)
        item_emb = self.fc1(concat_emb)
        item_emb = self.relu(item_emb)
        return item_emb


class StaticFeatureMLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        return x


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


class EnhancedFeatureEncoder(nn.Module):
    def __init__(self, config, dtype=torch.float):
        super().__init__()
        self.config = config
        self.padding_idx = config.padding_idx
        self.device = config.device
        self.d_model = config.event_emb_dim + config.item_emb_dim
        self.dtype = dtype

        self.event_emb_layer = nn.Sequential(
            nn.Embedding(
                num_embeddings=config.num_event,
                embedding_dim=config.event_emb_dim,
                padding_idx=self.padding_idx,
            ),
            nn.LayerNorm(config.event_emb_dim),
            nn.Dropout(config.dropout)
        )

        self.word_emb_layer = WordEmbedding(
            num_word=config.num_word,
            word_emb_dim=config.item_emb_dim,
            dropout=config.dropout,
        )

        self.item_emb_layer = SkuEmbedding(
            config=config,
            num_sku=config.num_sku,
            num_cat=config.num_cat,
            num_price=config.num_price,
            word_emb_layer=self.word_emb_layer,
            item_emb_dim=config.item_emb_dim,
            padding_idx=self.padding_idx,
        )

        self.url_emb_layer = nn.Sequential(
            nn.Embedding(config.num_url, config.url_emb_dim, padding_idx=0),
            nn.LayerNorm(config.url_emb_dim),
            nn.Dropout(config.dropout)
        )

        self.url_projection = nn.Sequential(
            nn.Linear(config.url_emb_dim, config.item_emb_dim),
            nn.LayerNorm(config.item_emb_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        B = config.batch_size
        S = config.max_seq_length
        self.template_agg_embeddings = torch.zeros(
            (B, S, config.item_emb_dim), dtype=self.dtype, device=config.device
        )

    def _generate_padding_mask(self, seq):
        mask = seq == self.padding_idx
        return mask  # (batch_size, seq_len)

    def _aggregate_embeddings(
        self,
        event_type,
        sku_id,
        url_id,
        cat_id,
        price_id,
        word_id,
    ):

        B, _ = event_type.shape
        agg_embeddings = self.template_agg_embeddings.clone().detach()
        agg_embeddings = agg_embeddings[:B, :, :]

        sku_pos_idx = (
            (event_type == EventType.ADD_TO_CART.value)
            | (event_type == EventType.PRODUCT_BUY.value)
            | (event_type == EventType.REMOVE_FROM_CART.value)
        )
        sku_id = sku_id[sku_pos_idx]
        cat_id = cat_id[sku_pos_idx]
        price_id = price_id[sku_pos_idx]
        sku_word_id = word_id[sku_pos_idx]
        x = self.item_emb_layer(sku_id, cat_id, price_id, sku_word_id)

        agg_embeddings[sku_pos_idx, :] = x

        url_pos_idx = event_type == EventType.PAGE_VISIT.value
        url_id = url_id[url_pos_idx]
        url_emb = self.url_emb_layer(url_id)
        url_emb = self.url_projection(url_emb)
        agg_embeddings[url_pos_idx, :] = url_emb

        query_pos_idx = event_type == EventType.SEARCH_QUERY.value
        query_word_id = word_id[query_pos_idx]
        agg_embeddings[query_pos_idx, :] = self.word_emb_layer(query_word_id)

        return agg_embeddings

    def compute_user_embedding(
        self,
        event_type,
        sku_id,
        url_id,
        cat_id,
        price_id,
        word_id,
    ):
        event_type_seq_emb = self.event_emb_layer(event_type)
        event_content_seq_emb = self._aggregate_embeddings(
            event_type,
            sku_id,
            url_id,
            cat_id,
            price_id,
            word_id,
        )

        seq_emb = torch.concat(
            [event_type_seq_emb, event_content_seq_emb],
            dim=-1,
        )
        return seq_emb

    def forward(
        self,
        event_type,
        sku_id,
        url_id,
        cat_id,
        price_id,
        word_id,
    ):
        mask = self._generate_padding_mask(event_type)
        user_emb = self.compute_user_embedding(
            event_type,
            sku_id,
            url_id,
            cat_id,
            price_id,
            word_id,
        )

        return user_emb, mask


class UniversalBehavioralTransformer(L.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.d_model = config.event_emb_dim + config.item_emb_dim
        self.user_emb_dim = config.fusion_mlp_output_dim
        self.fusion_mlp_input_dim = self.d_model + config.static_features_dim
        self.lr = config.learning_rate


        self.encoder = EnhancedFeatureEncoder(config, self.dtype)

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
                    embedding_dim=self.d_model,
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

        self.fusion_mlp_input_dim = self.d_model + config.static_features_dim

        self.fusion_mlp = FusionModule(
            input_dim=self.fusion_mlp_input_dim,
            hidden1_dim=config.fusion_mlp_hidden_dim,
            output_dim=config.hidden_size,
            dropout=config.fusion_mlp_dropout,
        )

        self.category_embeddings = self.encoder.item_emb_layer.cat_emb_layer[0]
        self.sku_embeddings = nn.Sequential(
            self.encoder.item_emb_layer.sku_emb_layer[0],
            self.encoder.item_emb_layer.sku_projection
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

        feature_embeddings, mask = self.encoder(
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