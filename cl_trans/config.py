from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import logging
import os
import argparse

logger = logging.getLogger(__name__)

@dataclass
class Config:
    # 模型架构参数
    ns_len: int = 1
    num_layers: int = 2
    hidden_dim: int = 512
    max_len: int = 64
    num_heads: int = 4
    pooling_strategy: str = "max"

    # 特征维度
    num_event = 5 + 1
    num_sku = 1260365 + 1
    num_cat = 6774 + 1
    num_price = 100 + 1
    num_url = 373220 + 2
    num_word = 256 + 1
    num_day = 365
    num_week = 52
    num_buy_categories = 16
    num_buy_skus = 3
    static_features_dim = 46
    item_emb_dim = 384
    url_emb_dim = 128
    event_emb_dim = 8
    sku_emb_dim = 384
    cat_emb_dim = 96
    price_emb_dim = 16
    word_emb_dim = 384
    day_emb_dim = 16
    week_emb_dim = 4
    embed_dim = 512

    # 行为类型配置
    dropout= 0.2
    num_heads = 4
    churn_loss_weight = 0.025
    activation = "gelu"
    dim_feedforward = 1024
    num_encoder_layers = 2
    num_decoder_layers = 1
    last_embed_dim = 256
    dcn_cross_layers = 3
    dcn_hidden_units = [1024, 512]
    mlp_hidden_units = [512]

    use_semantic: bool = True
    semantic_groups = [5, 38, 12, 22, 13]
    total_ns_dim = 90
    rankmixer_layers = 2
    rankmixer_ffn_mult = 4
    rankmixer_token_dp = 0.0
    rankmixer_ffn_dp = 0.2
    use_moe: bool = True
    moe_experts = 4
    moe_l1_coef = 0.01
    moe_use_dtsi: bool = True

    # 训练配置
    batch_size: int = 128
    num_epochs: int = 3
    learning_rate: float = 1e-3
    mask_rate = 0.2
    padding_idx = 0
    temperature = 0.02

    # 设备配置
    accelerator: str = "cuda"
    devices: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 0
    output_dir: str = "../submit"
    device: str = "cuda:0"


SEMANTIC_GROUPS = {
    "group_lifecycle": [
        "all_duration_days", "min_all_from_first_to_end_days", "unique_all_timestamp", "global_gap_ratio", "ratio_active_visit"
    ],
    "group_recency": [
        "all_from_last_to_end_days", "add_from_last_to_end_days", "buy_from_last_to_end_days",
        "remove_from_last_to_end_days", "visit_from_last_to_end_days", "search_from_last_to_end_days",
        # 窗口统计
        "add_count_7d", "buy_count_7d", "remove_count_7d", "visit_count_7d",
        "add_count_30d", "buy_count_30d", "remove_count_30d", "visit_count_30d",
        # EWA
        "buy_ewa_1d", "buy_ewa_7d", "buy_ewa_14d", "buy_ewa_30d", "buy_ewa_90d", "buy_ewa_180d",
        "add_ewa_1d", "add_ewa_7d", "add_ewa_14d", "add_ewa_30d", "add_ewa_90d", "add_ewa_180d",
        "remove_ewa_1d", "remove_ewa_7d", "remove_ewa_14d", "remove_ewa_30d", "remove_ewa_90d", "remove_ewa_180d",
        "visit_ewa_1d", "visit_ewa_7d", "visit_ewa_14d", "visit_ewa_30d", "visit_ewa_90d", "visit_ewa_180d",
    ],
    "group_purchase": [
        "buy_flag", "unique_buy_sku", "unique_buy_category", "unique_buy_price", "unique_buy_name", "buy_price_mean",
        "ratio_buy_visit", "ratio_buy_add", "buy_from_last_to_end_days", "min_buy_from_first_to_end_days", "buy_duration_days", "unique_buy_timestamp"
    ],
    "group_cart_intent": [
        "add_flag", "unique_add_sku", "unique_add_category", "unique_add_price", "unique_add_name", "add_price_mean",
        "add_from_last_to_end_days", "min_add_from_first_to_end_days", "add_duration_days", "unique_add_timestamp",
        "remove_flag", "unique_remove_sku", "unique_remove_category", "unique_remove_price", "unique_remove_name", "remove_price_mean",
        "remove_from_last_to_end_days", "min_remove_from_first_to_end_days", "remove_duration_days", "unique_remove_timestamp",
        "ratio_add_visit", "ratio_remove_add",
    ],
    "group_exploration": [
        "page_visit_flag", "unique_url", "visit_from_last_to_end_days", "min_visit_from_first_to_end_days", "visit_duration_days", "unique_visit_timestamp",
        "search_query_flag", "unique_search_query", "search_from_last_to_end_days", "min_search_from_first_to_end_days", "search_duration_days", "unique_search_timestamp",
        "category_entropy_30d"
    ]
}

# 创建默认配置实例
config = Config()

