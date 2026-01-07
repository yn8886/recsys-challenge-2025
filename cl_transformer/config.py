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
    hidden_size: int = 256  # 增加隐藏层大小以提升模型表达能力
    num_heads: int = 4  # 注意力头数量
    num_layers: int = 3  # HSTU层数
    dropout: int = 0.2  # dropout比例

    # HSTU特定参数
    use_hstu: bool = False
    attention_dim: int = 128  # 注意力维度
    linear_dim: int = 256  # 线性层维度
    linear_activation: str = "silu"  # 线性层激活函数
    normalization: str = "rel_bias"  # 归一化方法
    linear_config: str = "uvqk"  # 线性层配置
    concat_ua: bool = False  # 是否连接u和a
    time_buckets: int = 128  # 时间桶数量
    
    # 特征维度
    num_event = 5 + 1
    num_sku = 1_260_370
    num_cat = 6_995
    num_price = 100 + 1
    num_url = 373_500
    num_word = 256 + 1
    static_features_dim = 46
    item_emb_dim = 128
    url_emb_dim = 128
    event_emb_dim = 8
    sku_emb_dim = 384
    cat_emb_dim = 96
    price_emb_dim = 16
    
    # 行为类型配置
    max_seq_length: int = 150  # 序列长度

    fusion_mlp_hidden_dim = 256
    fusion_mlp_output_dim = 256
    fusion_mlp_dropout = 0.01

    mask_rate = 0.2
    temperature = 0.1
    num_shared_experts = 4
    num_task_experts = 4
    num_tasks = 3
    expert_hidden_dims = [256]
    expert_output_dim = 128
    task_tower_hidden_dims = [128]
    task_tower_dropout = 0.1
    churn_loss_weight = 0.025

    # 训练配置
    batch_size: int = 128  # 批次大小
    learning_rate: float = 5e-5  # 学习率
    weight_decay: float = 1e-3  # 权重衰减
    num_epochs: int = 150  # 训练轮数
    warmup_steps: int = 2000  # 预热步数
    patience: int = 10  # 早停耐心
    
    # 设备配置
    accelerator: str = "cuda"
    devices: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 0
    # 负采样数量，用于负采样softmax-ce损失
    num_negative_samples: int = 400
    output_dir: str = "../submit"
    device: str = "cuda:0"
    use_cpu: bool = False  # 添加CPU选项

    
    padding_idx = 0

# 创建默认配置实例
config = Config()

