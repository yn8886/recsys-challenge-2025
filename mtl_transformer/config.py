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
    num_heads: int = 1
    num_layers: int = 2
    dropout: int = 0.2
    
    # 特征维度
    num_event = 5 + 1
    num_sku = 1260365 + 1
    num_cat = 6774 + 1
    num_price = 100 + 1
    num_url = 373220 + 2
    num_word = 256 + 1
    static_features_dim = 46
    item_emb_dim = 384
    url_emb_dim = 128
    event_emb_dim = 8
    sku_emb_dim = 384
    cat_emb_dim = 96
    price_emb_dim = 16
    
    # 行为类型配置
    fusion_mlp_hidden_dim = 256
    fusion_mlp_output_dim = 256
    fusion_mlp_dropout = 0.01
    num_shared_experts = 4
    num_task_experts = 4
    num_tasks = 3
    expert_hidden_dims = [256]
    expert_output_dim = 128
    task_tower_hidden_dims = [128]
    task_tower_dropout = 0.1
    churn_loss_weight = 0.025

    # 训练配置
    max_len = 64
    batch_size: int = 128
    num_epochs: int = 3
    learning_rate: float = 1e-3
    mask_rate = 0.2
    temperature = 0.1
    padding_idx = 0

    # 设备配置
    accelerator: str = "cuda"
    devices: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 0
    output_dir: str = "../submit"
    device: str = "cuda:0"


# 创建默认配置实例
config = Config()

