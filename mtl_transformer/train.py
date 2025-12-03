import argparse
import math
import os
from datetime import datetime
from enum import Enum

import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
# import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
# from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import AUROC, MultilabelAccuracy

NUM_CANDIDATES_SKU = 100
NUM_CANDIDATES_CAT = 100
NUM_CANDIDATES_PRICE = 100

BASE_PATH = "../dataset/mtl"

exp_name = os.path.splitext(os.path.basename(__file__))[0]
device = "cpu"
torch.backends.cudnn.benchmark = True

np.random.seed(42)


class EventType(Enum):
    # 0はpad_idx、1はmask
    PAD_IDX = 0
    MASK = 1
    PRODUCT_BUY = 2
    ADD_TO_CART = 3
    REMOVE_FROM_CART = 4
    PAGE_VISIT = 5
    SEARCH_QUERY = 6


class CFG:
    n_epochs = 3
    batch_size = 128
    num_workers = 12
    num_event = 5 + 2
    num_sku = 1_260_370
    num_cat = 6_995
    num_price = 100 + 2
    num_url = 373_500
    num_word = 256 + 3
    num_day = 365
    num_week = 52
    static_features_dim = 46
    item_emb_dim = 384
    event_emb_dim = 8
    sku_emb_dim = 384
    cat_emb_dim = 96
    price_emb_dim = 16
    day_emb_dim = 16
    week_emb_dim = 4
    fusion_mlp_hidden_dim = 256
    fusion_mlp_output_dim = 256
    fusion_mlp_dropout = 0.01
    num_shared_experts = 4
    num_task_experts = 4
    num_tasks = 8
    expert_hidden_dims = [256]
    expert_output_dim = 128
    task_tower_hidden_dims = [128]
    task_tower_dropout = 0.1
    num_heads = 1
    num_layers = 2
    dropout = 0.2
    max_len = 64
    learning_rate = 1e-3
    mask_rate = 0.2
    temperature = 0.1
    device = "cpu"
    padding_idx = 0
    churn_loss_weight = 0.025
    add_loss_weight = 0.025


# def login_wandb():
#     load_dotenv()
#     WANDB_API_KEY = os.getenv("WANDB_API_KEY")
#     wandb.login(key=WANDB_API_KEY)


class RecsysDatasetV12(Dataset):
    def __init__(self, dataset_dir, max_len=128, mask_rate=0.2):
        self.dataset_dir = dataset_dir
        self.max_len = max_len
        self.mask_rate = mask_rate

        self.indexes_template = np.arange(self.max_len)
        self.item_mask_template = 1
        self.word_mask_template = np.array([1] * 16)

        print("Loading client_id.npy")
        self.client_ids = np.load(
            os.path.join(self.dataset_dir, "client_id.npy"), allow_pickle=True
        )
        print("Loading event_type.npy")
        self.event_types = np.load(
            os.path.join(self.dataset_dir, "event_type.npy"), allow_pickle=True
        )

        print("Loading sku_id.npy")
        self.sku_ids = np.load(
            os.path.join(self.dataset_dir, "sku_id.npy"), allow_pickle=True
        )
        print("Loading url_id.npy")
        self.url_ids = np.load(
            os.path.join(self.dataset_dir, "url_id.npy"), allow_pickle=True
        )
        print("Loading word_id.npy")
        self.word_ids = np.load(
            os.path.join(self.dataset_dir, "word_ids.npy"), allow_pickle=True
        )

        print("Loading category_id.npy")
        self.category_ids = np.load(
            os.path.join(self.dataset_dir, "category_id.npy"), allow_pickle=True
        )

        print("Loading price.npy")
        self.price_ids = np.load(
            os.path.join(self.dataset_dir, "price_id.npy"), allow_pickle=True
        )

        print("Loading static_features.npy")
        self.statistical_features = np.load(
            os.path.join(self.dataset_dir, "stats_features.npy"), allow_pickle=True
        )

        print(self.statistical_features.shape)
        print(f"Min: {self.statistical_features.min()}")
        print(f"Max: {self.statistical_features.max()}")

        print("Loading labels")
        self.is_churn = np.load(
            os.path.join(self.dataset_dir, "is_churn.npy"), allow_pickle=True
        )
        self.pos_sku_ids = np.load(
            os.path.join(self.dataset_dir, "pos_sku_ids.npy"), allow_pickle=True
        )
        self.neg_sku_ids = np.load(
            os.path.join(self.dataset_dir, "neg_sku_ids.npy"), allow_pickle=True
        )
        self.pos_cat_ids = np.load(
            os.path.join(self.dataset_dir, "pos_sku_ids.npy"), allow_pickle=True
        )
        self.neg_cat_ids = np.load(
            os.path.join(self.dataset_dir, "neg_sku_ids.npy"), allow_pickle=True
        )

    def __len__(self):
        return len(self.client_ids)

    def _pad_sequence(self, seq):
        seq = seq.tolist()
        sliced_seq = seq[-self.max_len :]
        padding_length = self.max_len - len(sliced_seq)
        padded_seq = [0] * padding_length + sliced_seq
        padded_seq = np.array(padded_seq)
        return padded_seq, padding_length

    def _pad_word_sequence(self, seq):
        seq = seq.tolist()
        sliced_seq = seq[-self.max_len :]
        padding_length = self.max_len - len(sliced_seq)
        padded_seq = [[0] * 16] * padding_length + sliced_seq
        padded_seq = np.array(padded_seq)

        return padded_seq, padding_length

    def _mask_sequence(self, seq, padding_length):
        non_pad_indices = self.indexes_template[padding_length:]

        n_replace = int(len(non_pad_indices) * self.mask_rate)

        replace_indices = np.random.choice(
            non_pad_indices, size=n_replace, replace=False
        )

        seq[replace_indices] = self.item_mask_template
        return seq

    def _mask_word_sequence(self, seq, padding_length):
        non_pad_indices = self.indexes_template[padding_length:]

        n_replace = int(len(non_pad_indices) * self.mask_rate)

        replace_indices = np.random.choice(
            non_pad_indices, size=n_replace, replace=False
        )
        seq[replace_indices] = self.word_mask_template
        return seq

    def __getitem__(self, idx):
        # sequence features
        client_id = self.client_ids[idx]
        event_type = self.event_types[idx]
        sku_id = self.sku_ids[idx]
        url_id = self.url_ids[idx]
        word_id = self.word_ids[idx]
        category_id = self.category_ids[idx]
        price_id = self.price_ids[idx]
        diff_days = self.diff_days[idx]
        diff_weeks = self.diff_weeks[idx]

        # statistical features
        statistical_features = self.statistical_features[idx]

        # labels
        buy_sku_label = self.buy_sku_labels[idx]
        buy_cat_label = self.buy_cat_labels[idx]
        buy_price_label = self.buy_price_labels[idx]
        add_sku_label = self.add_sku_labels[idx]
        add_cat_label = self.add_cat_labels[idx]
        add_price_label = self.add_price_labels[idx]

        is_contain_buy_sku = self.contain_buy_sku_labels[idx]
        is_contain_buy_cat = self.contain_buy_cat_labels[idx]
        is_contain_buy_price = self.contain_buy_price_labels[idx]
        is_contain_add_sku = self.contain_add_sku_labels[idx]
        is_contain_add_cat = self.contain_add_cat_labels[idx]
        is_contain_add_price = self.contain_add_price_labels[idx]

        is_churn = self.is_churn[idx]
        is_add = self.is_add[idx]

        # padding and masking sequence
        event_type, _ = self._pad_sequence(event_type)
        sku_id, _ = self._pad_sequence(sku_id)
        url_id, _ = self._pad_sequence(url_id)
        category_id, _ = self._pad_sequence(category_id)
        price_id, _ = self._pad_sequence(price_id)
        word_id, _ = self._pad_word_sequence(word_id)
        diff_days, _ = self._pad_sequence(diff_days)
        diff_weeks, _ = self._pad_sequence(diff_weeks)

        # statistical features
        statistical_features = torch.tensor(statistical_features, dtype=torch.float)

        buy_sku_label = torch.tensor(buy_sku_label, dtype=torch.long)
        buy_cat_label = torch.tensor(buy_cat_label, dtype=torch.long)
        buy_price_label = torch.tensor(buy_price_label, dtype=torch.long)
        add_sku_label = torch.tensor(add_sku_label, dtype=torch.long)
        add_cat_label = torch.tensor(add_cat_label, dtype=torch.long)
        add_price_label = torch.tensor(add_price_label, dtype=torch.long)

        return_dict = {
            "client_id": client_id,
            "original_seq": {
                "event_type": event_type,
                "sku": sku_id,
                "url": url_id,
                "category": category_id,
                "price": price_id,
                "word": word_id,
                "diff_days": diff_days,
                "diff_weeks": diff_weeks,
            },
            "statistical_feature": statistical_features,
            "labels": {
                "is_churn": is_churn,
                "is_add": is_add,
                "buy_sku_label": buy_sku_label,
                "buy_cat_label": buy_cat_label,
                "buy_price_label": buy_price_label,
                "add_sku_label": add_sku_label,
                "add_cat_label": add_cat_label,
                "add_price_label": add_price_label,
                "is_contain_buy_sku": is_contain_buy_sku,
                "is_contain_buy_cat": is_contain_buy_cat,
                "is_contain_buy_price": is_contain_buy_price,
                "is_contain_add_sku": is_contain_add_sku,
                "is_contain_add_cat": is_contain_add_cat,
                "is_contain_add_price": is_contain_add_price,
            },
        }
        return return_dict


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
    def __init__(self, num_word, word_emb_dim, padding_idx=0):
        super().__init__()

        self.word_emb_layer = nn.Embedding(
            num_embeddings=num_word,
            embedding_dim=word_emb_dim,
            padding_idx=padding_idx,
        )

    def forward(self, word_ids):
        word_emb = self.word_emb_layer(word_ids)
        avg_word_emb = torch.mean(word_emb, dim=-2)
        return avg_word_emb


class SkuEmbedding(nn.Module):
    def __init__(
        self,
        num_sku,
        sku_emb_dim,
        num_cat,
        cat_emb_dim,
        num_price,
        price_emb_dim,
        word_emb_dim,
        word_emb_layer,
        item_emb_dim,
        padding_idx=0,
    ):
        super().__init__()
        self.sku_emb_layer = nn.Embedding(
            num_embeddings=num_sku,
            embedding_dim=sku_emb_dim,
            padding_idx=padding_idx,
        )
        self.cat_emb_layer = nn.Embedding(
            num_embeddings=num_cat,
            embedding_dim=cat_emb_dim,
            padding_idx=padding_idx,
        )
        self.price_emb_layer = nn.Embedding(
            num_embeddings=num_price,
            embedding_dim=price_emb_dim,
            padding_idx=padding_idx,
        )
        self.word_emb_layer = word_emb_layer

        self.fc1 = nn.Linear(
            sku_emb_dim + cat_emb_dim + price_emb_dim + word_emb_dim,
            item_emb_dim,
        )
        self.relu = nn.ReLU()

    def forward(self, sku_id, cat_id, price_id, word_ids):
        sku_emb = self.sku_emb_layer(sku_id)
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


class BehaviorSequenceTransformer(nn.Module):
    def __init__(self, cfg=CFG, dtype=torch.float):
        super().__init__()
        self.cfg = cfg
        self.padding_idx = cfg.padding_idx
        self.device = device
        self.d_model = (
            cfg.event_emb_dim + cfg.item_emb_dim + cfg.day_emb_dim + cfg.week_emb_dim
        )
        self.dtype = dtype

        self.event_emb_layer = nn.Embedding(
            num_embeddings=cfg.num_event,
            embedding_dim=cfg.event_emb_dim,
            padding_idx=self.padding_idx,
        )

        self.word_emb_layer = WordEmbedding(
            num_word=cfg.num_word,
            word_emb_dim=cfg.item_emb_dim,
            padding_idx=self.padding_idx,
        )

        self.sku_emb_layer = SkuEmbedding(
            num_sku=cfg.num_sku,
            sku_emb_dim=cfg.sku_emb_dim,
            num_cat=cfg.num_cat,
            cat_emb_dim=cfg.cat_emb_dim,
            num_price=cfg.num_price,
            price_emb_dim=cfg.price_emb_dim,
            word_emb_dim=cfg.item_emb_dim,
            word_emb_layer=self.word_emb_layer,
            item_emb_dim=cfg.item_emb_dim,
            padding_idx=self.padding_idx,
        )

        self.url_emb_layer = nn.Embedding(
            num_embeddings=cfg.num_url,
            embedding_dim=cfg.item_emb_dim,
            padding_idx=self.padding_idx,
        )

        self.day_emb_layer = nn.Embedding(
            num_embeddings=cfg.num_day,
            embedding_dim=cfg.day_emb_dim,
            padding_idx=self.padding_idx,
        )
        self.week_emb_layer = nn.Embedding(
            num_embeddings=cfg.num_week,
            embedding_dim=cfg.week_emb_dim,
            padding_idx=self.padding_idx,
        )

        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
        )
        self.trm_enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.item_emb_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.trm_enc = nn.TransformerEncoder(
            self.trm_enc_layer, num_layers=cfg.num_layers
        )

        B = cfg.batch_size
        S = cfg.max_len

        self.template_agg_embeddings = torch.zeros(
            (B, S, self.cfg.item_emb_dim), dtype=self.dtype, device=device
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
        x = self.sku_emb_layer(sku_id, cat_id, price_id, sku_word_id)

        agg_embeddings[sku_pos_idx, :] = x

        url_pos_idx = event_type == EventType.PAGE_VISIT.value
        url_id = url_id[url_pos_idx]
        agg_embeddings[url_pos_idx, :] = self.url_emb_layer(url_id)

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
        diff_days,
        diff_weeks,
    ):
        src_padding_mask = self._generate_padding_mask(event_type)
        event_type_seq_emb = self.event_emb_layer(event_type)
        event_content_seq_emb = self._aggregate_embeddings(
            event_type,
            sku_id,
            url_id,
            cat_id,
            price_id,
            word_id,
        )

        day_seq_emb = self.day_emb_layer(diff_days)
        week_seq_emb = self.week_emb_layer(diff_weeks)

        seq_emb = torch.concat(
            [event_type_seq_emb, event_content_seq_emb, day_seq_emb, week_seq_emb],
            dim=-1,
        )
        seq_emb = self.pos_encoder(seq_emb)
        trm_enc_out = self.trm_enc(seq_emb, src_key_padding_mask=src_padding_mask)

        user_emb = trm_enc_out[:, -1, :]
        return user_emb

    def forward(
        self,
        event_type,
        sku_id,
        url_id,
        cat_id,
        price_id,
        word_id,
        diff_days,
        diff_weeks,
    ):
        user_emb = self.compute_user_embedding(
            event_type,
            sku_id,
            url_id,
            cat_id,
            price_id,
            word_id,
            diff_days,
            diff_weeks,
        )

        return user_emb


class MultiLayerPerceptoron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super().__init__()
        layers = list()
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PLE(nn.Module):
    def __init__(
        self,
        user_emb_dim,
        num_shared_experts,
        num_task_experts,
        num_tasks,
        expert_hidden_dims,
        expert_output_dim,
        task_tower_hidden_dims,
    ):
        super().__init__()

        self.user_emb_dim = user_emb_dim
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.num_tasks = num_tasks

        self.shared_expert_1 = nn.ModuleList(
            [
                MultiLayerPerceptoron(
                    user_emb_dim, expert_hidden_dims, expert_output_dim, dropout=0.01
                )
                for i in range(num_shared_experts)
            ]
        )

        self.task_expert_1 = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MultiLayerPerceptoron(
                            user_emb_dim,
                            expert_hidden_dims,
                            expert_output_dim,
                            dropout=0.01,
                        )
                        for i in range(num_task_experts)
                    ]
                )
                for j in range(self.num_tasks)
            ]
        )

        self.gate_1 = nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        user_emb_dim,
                        (
                            self.num_shared_experts + self.num_task_experts
                            if i < self.num_tasks
                            else self.num_shared_experts
                            + self.num_tasks * self.num_task_experts
                        ),
                    ),
                    torch.nn.Softmax(dim=1),
                )
                for i in range(self.num_tasks + 1)
            ]
        )

        self.shared_expert_2 = nn.ModuleList(
            [
                MultiLayerPerceptoron(
                    expert_output_dim,
                    expert_hidden_dims,
                    expert_output_dim,
                    dropout=0.01,
                )
                for i in range(num_shared_experts)
            ]
        )

        self.task_expert_2 = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        MultiLayerPerceptoron(
                            expert_output_dim,
                            expert_hidden_dims,
                            expert_output_dim,
                            dropout=0.01,
                        )
                        for i in range(num_task_experts)
                    ]
                )
                for j in range(self.num_tasks)
            ]
        )
        self.gate_2 = nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        expert_output_dim,
                        self.num_shared_experts + self.num_task_experts,
                    ),
                    torch.nn.Softmax(dim=1),
                )
                for i in range(self.num_tasks)
            ]
        )
        self.churn_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=1, dropout=0.1
        )
        self.add_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=1, dropout=0.1
        )
        self.buy_sku_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.buy_cat_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.buy_price_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.add_sku_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.add_cat_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )
        self.add_price_tower = MultiLayerPerceptoron(
            expert_output_dim, task_tower_hidden_dims, output_dim=100, dropout=0.1
        )

    def forward(self, user_emb):
        # [(B, NUM_EXPERTS), ...]
        shared_expert_outputs = torch.cat(
            [
                self.shared_expert_1[i](user_emb).unsqueeze(1)
                for i in range(self.num_shared_experts)
            ],
            dim=1,
        )

        concat_feats = []
        all_task_expert_outputs = []
        for i in range(self.num_tasks):
            task_expert_outputs = torch.cat(
                [
                    self.task_expert_1[i][j](user_emb).unsqueeze(1)
                    for j in range(self.num_task_experts)
                ],
                dim=1,
            )
            concat_feats.append(
                torch.cat([shared_expert_outputs, task_expert_outputs], dim=1)
            )
            all_task_expert_outputs.append(task_expert_outputs)

        all_task_expert_outputs.append(shared_expert_outputs)
        concat_feats.append(torch.cat(all_task_expert_outputs, dim=1))

        gate_value = [self.gate_1[i](user_emb) for i in range(self.num_tasks + 1)]

        task_feats = [
            torch.bmm(gate_value[i].unsqueeze(1), concat_feats[i]).squeeze(1)
            for i in range(self.num_tasks + 1)
        ]

        shared_experts_outputs = torch.cat(
            [
                self.shared_expert_2[i](task_feats[-1]).unsqueeze(1)
                for i in range(self.num_shared_experts)
            ],
            dim=1,
        )
        expert_outputs = []
        for i in range(self.num_tasks):
            task_expert_outputs = torch.cat(
                [
                    self.task_expert_2[i][j](task_feats[i]).unsqueeze(1)
                    for j in range(self.num_task_experts)
                ],
                dim=1,
            )
            expert_outputs.append(
                torch.cat([shared_experts_outputs, task_expert_outputs], dim=1)
            )
        gate_value = [self.gate_2[i](task_feats[i]) for i in range(self.num_tasks)]
        tower_inputs = [
            torch.bmm(gate_value[i].unsqueeze(1), expert_outputs[i]).squeeze(1)
            for i in range(self.num_tasks)
        ]

        churn_logits = self.churn_tower(tower_inputs[0])
        add_logits = self.add_tower(tower_inputs[1])
        buy_sku_logits = self.buy_sku_tower(tower_inputs[2])
        buy_cat_logits = self.buy_cat_tower(tower_inputs[3])
        buy_price_logits = self.buy_price_tower(tower_inputs[4])
        add_sku_logits = self.add_sku_tower(tower_inputs[5])
        add_cat_logits = self.add_cat_tower(tower_inputs[6])
        add_price_logits = self.add_price_tower(tower_inputs[7])

        return (
            churn_logits,
            add_logits,
            buy_sku_logits,
            buy_cat_logits,
            buy_price_logits,
            add_sku_logits,
            add_cat_logits,
            add_price_logits,
        )


class LightningRecsysModel(L.LightningModule):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        self.d_model = (
            cfg.event_emb_dim + cfg.item_emb_dim + cfg.day_emb_dim + cfg.week_emb_dim
        )

        self.user_emb_dim = cfg.fusion_mlp_output_dim
        self.fusion_mlp_input_dim = self.d_model + cfg.static_features_dim

        self.encoder = BehaviorSequenceTransformer(cfg, self.dtype)

        self.fusion_mlp = FusionModule(
            input_dim=self.fusion_mlp_input_dim,
            hidden1_dim=cfg.fusion_mlp_hidden_dim,
            output_dim=cfg.fusion_mlp_output_dim,
            dropout=cfg.fusion_mlp_dropout,
        )

        self.ple = PLE(
            user_emb_dim=self.user_emb_dim,
            num_shared_experts=cfg.num_shared_experts,
            num_task_experts=cfg.num_task_experts,
            num_tasks=cfg.num_tasks,
            expert_hidden_dims=cfg.expert_hidden_dims,
            expert_output_dim=cfg.expert_output_dim,
            task_tower_hidden_dims=cfg.task_tower_hidden_dims,
        )

        self.lr = cfg.learning_rate

        self.bce_loss = nn.BCEWithLogitsLoss()

        self.valid_auroc_churn = AUROC(task="binary")
        self.valid_auroc_add = AUROC(task="binary")
        self.valid_auroc_buy_sku = AUROC(
            task="multilabel", num_labels=NUM_CANDIDATES_SKU
        )
        self.valid_auroc_buy_cat = AUROC(
            task="multilabel", num_labels=NUM_CANDIDATES_CAT
        )
        self.valid_auroc_buy_price = AUROC(
            task="multilabel", num_labels=NUM_CANDIDATES_PRICE
        )
        self.valid_auroc_add_sku = AUROC(
            task="multilabel", num_labels=NUM_CANDIDATES_SKU
        )
        self.valid_auroc_add_cat = AUROC(
            task="multilabel", num_labels=NUM_CANDIDATES_CAT
        )
        self.valid_auroc_add_price = AUROC(
            task="multilabel", num_labels=NUM_CANDIDATES_PRICE
        )

        self.save_hyperparameters()

    def compute_user_embedding(
        self,
        event_type,
        sku_id,
        url_id,
        cat_id,
        price_id,
        word_id,
        diff_days,
        diff_weeks,
        statistical_features,
    ):
        seq_feat_emb = self.encoder(
            event_type, sku_id, url_id, cat_id, price_id, word_id, diff_days, diff_weeks
        )

        concat_feat = torch.concat(
            [seq_feat_emb, statistical_features],
            dim=-1,
        )

        user_emb = self.fusion_mlp(concat_feat)
        return user_emb

    def calc_logits(self, user_emb):
        return self.ple(user_emb)

    def forward(
        self,
        event_type,
        sku_id,
        url_id,
        cat_id,
        price_id,
        word_id,
        diff_days,
        diff_weeks,
        statistical_features,
    ):

        user_emb = self.compute_user_embedding(
            event_type,
            sku_id,
            url_id,
            cat_id,
            price_id,
            word_id,
            diff_days,
            diff_weeks,
            statistical_features,
        )
        (
            logits_churn,
            logits_add,
            logits_buy_sku,
            logits_buy_cat,
            logits_buy_price,
            logits_add_sku,
            logits_add_cat,
            logits_add_price,
        ) = self.calc_logits(user_emb)

        return (
            logits_churn,
            logits_add,
            logits_buy_sku,
            logits_buy_cat,
            logits_buy_price,
            logits_add_sku,
            logits_add_cat,
            logits_add_price,
        )

    def training_step(self, batch, batch_idx):
        original_seq = batch["original_seq"]
        labels = batch["labels"]
        # statistical feat
        statistical_feature = batch["statistical_feature"]

        # sequence feat
        event_type = original_seq["event_type"]
        sku = original_seq["sku"]
        url = original_seq["url"]
        category = original_seq["category"]
        price = original_seq["price"]
        word = original_seq["word"]
        diff_days = original_seq["diff_days"]
        diff_weeks = original_seq["diff_weeks"]

        # label
        label_churn = labels["is_churn"]
        label_add = labels["is_add"]
        label_buy_sku = labels["buy_sku_label"]
        label_buy_cat = labels["buy_cat_label"]
        label_buy_price = labels["buy_price_label"]

        label_add_sku = labels["add_sku_label"]
        label_add_cat = labels["add_cat_label"]
        label_add_price = labels["add_price_label"]

        is_contain_buy_sku = labels["is_contain_buy_sku"]
        is_contain_buy_cat = labels["is_contain_buy_cat"]
        is_contain_buy_price = labels["is_contain_buy_price"]

        is_contain_add_sku = labels["is_contain_add_sku"]
        is_contain_add_cat = labels["is_contain_add_cat"]
        is_contain_add_price = labels["is_contain_add_price"]

        user_emb = self.compute_user_embedding(
            event_type,
            sku,
            url,
            category,
            price,
            word,
            diff_days,
            diff_weeks,
            statistical_feature,
        )

        (
            logits_churn,
            logits_add,
            logits_buy_sku,
            logits_buy_cat,
            logits_buy_price,
            logits_add_sku,
            logits_add_cat,
            logits_add_price,
        ) = self.calc_logits(user_emb)

        # Logging user embedding statistics
        zero_ratio = (user_emb == 0).float().mean()

        self.log("train/user_embedding/mean", user_emb.mean())
        self.log("train/user_embedding/std", user_emb.std())
        self.log("train/user_embedding/zero_ratio", zero_ratio)

        logits_churn = logits_churn.squeeze(-1)
        loss_churn = self.bce_loss(logits_churn, label_churn.float())
        loss_churn = loss_churn * self.cfg.churn_loss_weight

        logits_add = logits_add.squeeze(-1)
        loss_add = self.bce_loss(logits_add, label_add.float())
        loss_add = loss_add * self.cfg.add_loss_weight

        # Auxiliary Task
        # calculate loss for 100 types of sku/cat/price for buy
        loss_buy_sku = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_sku) > 0:
            mask = is_contain_buy_sku == 1
            loss_buy_sku = self.bce_loss(
                logits_buy_sku[mask], label_buy_sku[mask].float()
            )

        loss_buy_cat = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_cat) > 0:
            mask = is_contain_buy_cat == 1
            loss_buy_cat = self.bce_loss(
                logits_buy_cat[mask], label_buy_cat[mask].float()
            )

        loss_buy_price = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_price) > 0:
            mask = is_contain_buy_price == 1
            loss_buy_price = self.bce_loss(
                logits_buy_price[mask], label_buy_price[mask].float()
            )
        # calculate loss for 100 types of sku/cat/price for add
        loss_add_sku = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_add_sku) > 0:
            mask = is_contain_add_sku == 1
            loss_add_sku = self.bce_loss(
                logits_add_sku[mask], label_add_sku[mask].float()
            )

        loss_add_cat = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_add_cat) > 0:
            mask = is_contain_add_cat == 1
            loss_add_cat = self.bce_loss(
                logits_add_cat[mask], label_add_cat[mask].float()
            )

        loss_add_price = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_add_price) > 0:
            mask = is_contain_add_price == 1
            loss_add_price = self.bce_loss(
                logits_add_price[mask], label_add_price[mask].float()
            )

        self.log("train/loss_churn", loss_churn)
        self.log("train/loss_add", loss_add)
        self.log("train/loss_buy_sku", loss_buy_sku)
        self.log("train/loss_buy_cat", loss_buy_cat)
        self.log("train/loss_buy_price", loss_buy_price)
        self.log("train/loss_add_sku", loss_add_sku)
        self.log("train/loss_add_cat", loss_add_cat)
        self.log("train/loss_add_price", loss_add_price)

        sum_loss = (
            loss_churn
            + loss_add
            + loss_buy_sku
            + loss_buy_cat
            + loss_buy_price
            + loss_add_sku
            + loss_add_cat
            + loss_add_price
        )

        self.log("train/sum_loss", sum_loss)
        return sum_loss

    def validation_step(self, batch, batch_idx):
        original_seq = batch["original_seq"]
        labels = batch["labels"]
        # statistical feat
        statistical_feature = batch["statistical_feature"]

        # sequence feat
        event_type = original_seq["event_type"]
        sku = original_seq["sku"]
        url = original_seq["url"]
        category = original_seq["category"]
        price = original_seq["price"]
        word = original_seq["word"]
        diff_days = original_seq["diff_days"]
        diff_weeks = original_seq["diff_weeks"]

        # label
        label_churn = labels["is_churn"]
        label_add = labels["is_add"]
        label_buy_sku = labels["buy_sku_label"]
        label_buy_cat = labels["buy_cat_label"]
        label_buy_price = labels["buy_price_label"]

        label_add_sku = labels["add_sku_label"]
        label_add_cat = labels["add_cat_label"]
        label_add_price = labels["add_price_label"]

        is_contain_buy_sku = labels["is_contain_buy_sku"]
        is_contain_buy_cat = labels["is_contain_buy_cat"]
        is_contain_buy_price = labels["is_contain_buy_price"]

        is_contain_add_sku = labels["is_contain_add_sku"]
        is_contain_add_cat = labels["is_contain_add_cat"]
        is_contain_add_price = labels["is_contain_add_price"]
        user_emb = self.compute_user_embedding(
            event_type,
            sku,
            url,
            category,
            price,
            word,
            diff_days,
            diff_weeks,
            statistical_feature,
        )

        (
            logits_churn,
            logits_add,
            logits_buy_sku,
            logits_buy_cat,
            logits_buy_price,
            logits_add_sku,
            logits_add_cat,
            logits_add_price,
        ) = self.calc_logits(user_emb)

        zero_ratio = (user_emb == 0).float().mean()

        self.log("valid/user_embedding/mean", user_emb.mean())
        self.log("valid/user_embedding/std", user_emb.std())
        self.log("valid/user_embedding/zero_ratio", zero_ratio)

        logits_churn = logits_churn.squeeze(-1)
        loss_churn = self.bce_loss(logits_churn, label_churn.float())
        loss_churn = loss_churn * self.cfg.churn_loss_weight

        logits_add = logits_add.squeeze(-1)
        loss_add = self.bce_loss(logits_add, label_add.float())
        loss_add = loss_add * self.cfg.add_loss_weight

        # Auxiliary Task
        loss_buy_sku = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_sku) > 0:
            mask = is_contain_buy_sku == 1
            loss_buy_sku = self.bce_loss(
                logits_buy_sku[mask], label_buy_sku[mask].float()
            )

        loss_buy_cat = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_cat) > 0:
            mask = is_contain_buy_cat == 1
            loss_buy_cat = self.bce_loss(
                logits_buy_cat[mask], label_buy_cat[mask].float()
            )

        loss_buy_price = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_price) > 0:
            mask = is_contain_buy_price == 1
            loss_buy_price = self.bce_loss(
                logits_buy_price[mask], label_buy_price[mask].float()
            )
        # add
        loss_add_sku = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_add_sku) > 0:
            mask = is_contain_add_sku == 1
            loss_add_sku = self.bce_loss(
                logits_add_sku[mask], label_add_sku[mask].float()
            )

        loss_add_cat = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_add_cat) > 0:
            mask = is_contain_add_cat == 1
            loss_add_cat = self.bce_loss(
                logits_add_cat[mask], label_add_cat[mask].float()
            )

        loss_add_price = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_add_price) > 0:
            mask = is_contain_add_price == 1
            loss_add_price = self.bce_loss(
                logits_add_price[mask], label_add_price[mask].float()
            )

        self.log("valid/loss_churn", loss_churn)
        self.log("valid/loss_add", loss_add)
        self.log("valid/loss_buy_sku", loss_buy_sku)
        self.log("valid/loss_buy_cat", loss_buy_cat)
        self.log("valid/loss_buy_price", loss_buy_price)
        self.log("valid/loss_add_sku", loss_add_sku)
        self.log("valid/loss_add_cat", loss_add_cat)
        self.log("valid/loss_add_price", loss_add_price)

        sum_loss = (
            loss_churn
            + loss_add
            + loss_buy_sku
            + loss_buy_cat
            + loss_buy_price
            + loss_add_sku
            + loss_add_cat
            + loss_add_price
        )

        self.log("valid/sum_loss", sum_loss)

        # Update AUROC metrics
        self.valid_auroc_churn.update(logits_churn, label_churn)
        self.valid_auroc_add.update(logits_add, label_add)

        if torch.sum(is_contain_buy_sku) > 0:
            mask = is_contain_buy_sku == 1
            self.valid_auroc_buy_sku.update(
                logits_buy_sku[mask], label_buy_sku[mask].int()
            )

        if torch.sum(is_contain_buy_cat) > 0:
            mask = is_contain_buy_cat == 1
            self.valid_auroc_buy_cat.update(
                logits_buy_cat[mask], label_buy_cat[mask].int()
            )

        if torch.sum(is_contain_buy_price) > 0:
            mask = is_contain_buy_price == 1
            self.valid_auroc_buy_price.update(
                logits_buy_price[mask], label_buy_price[mask].int()
            )
        if torch.sum(is_contain_add_sku) > 0:
            mask = is_contain_add_sku == 1
            self.valid_auroc_add_sku.update(
                logits_add_sku[mask], label_add_sku[mask].int()
            )

        if torch.sum(is_contain_add_cat) > 0:
            mask = is_contain_add_cat == 1
            self.valid_auroc_add_cat.update(
                logits_add_cat[mask], label_add_cat[mask].int()
            )

        if torch.sum(is_contain_add_price) > 0:
            mask = is_contain_add_price == 1
            self.valid_auroc_add_price.update(
                logits_add_price[mask], label_add_price[mask].int()
            )

    def on_validation_epoch_end(self):
        self.log("valid/AUROC_churn", self.valid_auroc_churn.compute())
        self.log("valid/AUROC_add", self.valid_auroc_add.compute())
        # self.log("valid/AUROC_buy_sku", self.valid_auroc_buy_sku.compute())
        self.log("valid/AUROC_buy_cat", self.valid_auroc_buy_cat.compute())
        self.log("valid/AUROC_buy_price", self.valid_auroc_buy_price.compute())
        self.log("valid/AUROC_add_sku", self.valid_auroc_add_sku.compute())
        self.log("valid/AUROC_add_cat", self.valid_auroc_add_cat.compute())
        self.log("valid/AUROC_add_price", self.valid_auroc_add_price.compute())
        self.log(
            "valid/sum_score",
            self.valid_auroc_churn.compute()
            # + self.valid_auroc_buy_sku.compute()
            + self.valid_auroc_buy_cat.compute(),
        )
        self.valid_auroc_churn.reset()
        self.valid_auroc_add.reset()
        # self.valid_auroc_buy_sku.reset()
        self.valid_auroc_buy_cat.reset()
        self.valid_auroc_buy_price.reset()
        self.valid_auroc_add_sku.reset()
        self.valid_auroc_add_cat.reset()
        self.valid_auroc_add_price.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def main():
    # login_wandb()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="train",
        choices=["train", "valid"],
    )
    # args = parser.parse_args()
    # dataset_type = args.dataset_type
    # dataset_dir = os.path.join(BASE_PATH, dataset_type)

    train_dataset_dir = os.path.join(BASE_PATH, "train")
    valid_dataset_dir = os.path.join(BASE_PATH, "valid")

    train_dataset = RecsysDatasetV12(dataset_dir=train_dataset_dir, max_len=CFG.max_len)
    valid_dataset = RecsysDatasetV12(dataset_dir=valid_dataset_dir, max_len=CFG.max_len)

    print(f"train_size : {len(train_dataset)}")
    print(f"valid_size : {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    model = LightningRecsysModel(CFG)

    save_path = "./results/weights/"
    os.makedirs(save_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        monitor="valid/sum_score",
        save_top_k=3,
        mode="max",
        save_weights_only=True,
    )
    model_summary = ModelSummary(max_depth=3)

    # wandb_logger = WandbLogger(project="Recsys", name=exp_name, log_model=True)

    trainer = L.Trainer(
        callbacks=[checkpoint_callback, model_summary],
        # logger=wandb_logger,
        accelerator="cpu",
        max_epochs=CFG.n_epochs,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model,
        train_dataloader,
        valid_dataloader,
    )


if __name__ == "__main__":
    main()
