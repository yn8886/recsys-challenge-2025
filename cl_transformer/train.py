import argparse
import math
import os
from datetime import datetime
from enum import Enum
import logging
import lightning as L
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from sympy.utilities.timeutils import timethis
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import AUROC
from config import Config
from model import WordEmbedding, PositionalEncoding
# from models.hstu_modules import (
#     RelativeBucketedTimeAndPositionBasedBias,
#     SequentialTransductionUnitJagged,
#     HSTUJagged,
# )

NUM_CANDIDATES_SKU = 100
NUM_CANDIDATES_CAT = 100
NUM_CANDIDATES_PRICE = 100

exp_name = os.path.splitext(os.path.basename(__file__))[0]
torch.backends.cudnn.benchmark = True

np.random.seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    # 0はpad_idx、1はmask
    PAD_IDX = 0
    MASK = 1
    PRODUCT_BUY = 2
    ADD_TO_CART = 3
    REMOVE_FROM_CART = 4
    PAGE_VISIT = 5
    SEARCH_QUERY = 6


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

        self.timestamps = np.load(
            os.path.join(self.dataset_dir, "timestamp.npy"), allow_pickle=True
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
        self.buy_sku_labels = np.load(
            os.path.join(self.dataset_dir, "buy_sku_label.npy"), allow_pickle=True
        )
        self.buy_cat_labels = np.load(
            os.path.join(self.dataset_dir, "buy_cat_label.npy"), allow_pickle=True
        )
        self.contain_buy_sku_labels = np.load(
            os.path.join(self.dataset_dir, "contain_buy_sku_label.npy"),
            allow_pickle=True,
        )
        self.contain_buy_cat_labels = np.load(
            os.path.join(self.dataset_dir, "contain_buy_cat_label.npy"),
            allow_pickle=True,
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
        timestamp = self.timestamps[idx]

        # statistical features
        statistical_features = self.statistical_features[idx]

        # labels
        buy_sku_label = self.buy_sku_labels[idx]
        buy_cat_label = self.buy_cat_labels[idx]

        is_contain_buy_sku = self.contain_buy_sku_labels[idx]
        is_contain_buy_cat = self.contain_buy_cat_labels[idx]
        is_churn = self.is_churn[idx]

        # padding and masking sequence
        event_type, _ = self._pad_sequence(event_type)
        sku_id, _ = self._pad_sequence(sku_id)
        url_id, _ = self._pad_sequence(url_id)
        category_id, _ = self._pad_sequence(category_id)
        price_id, _ = self._pad_sequence(price_id)
        word_id, _ = self._pad_word_sequence(word_id)
        timestamp, _ = self._pad_sequence(timestamp)
        timestamp = torch.tensor(timestamp, dtype=torch.float)

        # statistical features
        statistical_features = torch.tensor(statistical_features, dtype=torch.float)

        buy_sku_label = torch.tensor(buy_sku_label, dtype=torch.long)
        buy_cat_label = torch.tensor(buy_cat_label, dtype=torch.long)

        return_dict = {
            "client_id": torch.tensor(client_id),
            "event_type": event_type,
            "sku": sku_id,
            "url": url_id,
            "category": category_id,
            "price": price_id,
            "word": word_id,
            "timestamp": timestamp,
            "statistical_feature": statistical_features,
            "is_churn": is_churn,
            "buy_sku_label": buy_sku_label,
            "buy_cat_label": buy_cat_label,
            "is_contain_buy_sku": is_contain_buy_sku,
            "is_contain_buy_cat": is_contain_buy_cat,
        }

        return return_dict


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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.padding_idx
        self.device = config.device
        self.d_model = config.item_emb_dim + config.event_emb_dim + config.sku_emb_dim + config.url_emb_dim + config.cat_emb_dim + config.price_emb_dim

        self.event_embedding = nn.Embedding(config.num_event, config.event_emb_dim, padding_idx=0)
        self.sku_embedding = nn.Embedding(config.num_sku, config.sku_emb_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(config.num_cat, config.cat_emb_dim, padding_idx=0)
        self.price_embedding = nn.Embedding(config.num_price, config.price_emb_dim, padding_idx=0)
        self.url_embedding = nn.Embedding(config.num_url, config.url_emb_dim, padding_idx=0)

        self.word_embedding = WordEmbedding(
            num_word=config.num_word,
            word_emb_dim=config.item_emb_dim,
            dropout=config.dropout,
        )

        self.feature_encoder = nn.Linear(self.d_model, config.hidden_size)

        # self.relative_attention_bias = RelativeBucketedTimeAndPositionBasedBias(
        #     max_seq_len=config.max_seq_length,
        #     num_buckets=config.time_buckets,
        #     bucketization_fn=lambda x: (
        #             torch.log(torch.abs(x).clamp(min=1)) / 0.69314718056  # Using ln(2)
        #     ).long(),
        # )

        if not config.use_hstu:
            self.pos_encoder = PositionalEncoding(
                d_model=config.hidden_size,
                dropout=config.dropout,
                max_len=config.max_seq_length,
            )
            self.trm_enc_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                batch_first=True,
            )
            self.model = nn.TransformerEncoder(
                self.trm_enc_layer, num_layers=config.num_layers
            )

    def forward(
        self,
        event_types,
        sku_ids,
        url_ids,
        cat_ids,
        price_ids,
        word_ids,
        timestamp,
    ):
        mask = event_types == self.padding_idx

        event_emb = self.event_embedding(event_types)
        cat_emb = self.category_embedding(cat_ids)
        price_emb = self.price_embedding(price_ids)
        word_emb = self.word_embedding(word_ids)
        sku_emb = self.sku_embedding(sku_ids)
        url_emb = self.url_embedding(url_ids)

        # 计算特征重要性 - 添加数值稳定性 - 现在包含Item ID和URL特征
        features = [event_emb, cat_emb, price_emb, word_emb, sku_emb, url_emb]
        features = torch.cat(features, dim=2)
        feature_embeddings = self.feature_encoder(features)

        if not self.config.use_hstu:
            feature_embeddings = self.pos_encoder(feature_embeddings)
            trm_enc_out = self.model(feature_embeddings, src_key_padding_mask=mask)
            user_emb = trm_enc_out[:, -1, :]

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
        buy_sku_logits = self.buy_sku_tower(tower_inputs[1])
        buy_cat_logits = self.buy_cat_tower(tower_inputs[2])

        return (
            churn_logits,
            buy_sku_logits,
            buy_cat_logits,
        )


class LightningRecsysModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = BehaviorSequenceTransformer(cfg)

        self.fusion_mlp_input_dim = cfg.hidden_size + cfg.static_features_dim
        self.fusion_mlp = FusionModule(
            input_dim=self.fusion_mlp_input_dim,
            hidden1_dim=cfg.fusion_mlp_hidden_dim,
            output_dim=cfg.fusion_mlp_output_dim,
            dropout=cfg.fusion_mlp_dropout,
        )

        self.ple = PLE(
            user_emb_dim=cfg.hidden_size,
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
        self.valid_auroc_buy_sku = AUROC(
            task="multilabel", num_labels=NUM_CANDIDATES_SKU
        )
        self.valid_auroc_buy_cat = AUROC(
            task="multilabel", num_labels=NUM_CANDIDATES_CAT
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
        timestamp,
        statistical_features,
    ):
        seq_feat_emb = self.encoder(
            event_type, sku_id, url_id, cat_id, price_id, word_id, timestamp
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
        timestamp,
        statistical_features,
    ):

        user_emb = self.compute_user_embedding(
            event_type,
            sku_id,
            url_id,
            cat_id,
            price_id,
            word_id,
            timestamp,
            statistical_features,
        )
        (
            logits_churn,
            logits_buy_sku,
            logits_buy_cat,
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
        # statistical feat
        statistical_feature = batch["statistical_feature"]

        # sequence feat
        event_type = batch["event_type"]
        sku = batch["sku"]
        url = batch["url"]
        category = batch["category"]
        price = batch["price"]
        word = batch["word"]
        timestamp = batch["timestamp"]
        # label
        label_churn = batch["is_churn"]
        label_buy_sku = batch["buy_sku_label"]
        label_buy_cat = batch["buy_cat_label"]
        is_contain_buy_sku = batch["is_contain_buy_sku"]
        is_contain_buy_cat = batch["is_contain_buy_cat"]

        user_emb = self.compute_user_embedding(
            event_type,
            sku,
            url,
            category,
            price,
            word,
            timestamp,
            statistical_feature,
        )

        (
            logits_churn,
            logits_buy_sku,
            logits_buy_cat,
        ) = self.calc_logits(user_emb)

        # Logging user embedding statistics
        zero_ratio = (user_emb == 0).float().mean()

        self.log("train/user_embedding/mean", user_emb.mean())
        self.log("train/user_embedding/std", user_emb.std())
        self.log("train/user_embedding/zero_ratio", zero_ratio)

        logits_churn = logits_churn.squeeze(-1)
        loss_churn = self.bce_loss(logits_churn, label_churn.float())
        loss_churn = loss_churn * self.cfg.churn_loss_weight

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

        self.log("train/loss_churn", loss_churn)
        self.log("train/loss_buy_sku", loss_buy_sku)
        self.log("train/loss_buy_cat", loss_buy_cat)

        sum_loss = (
            loss_churn
            + loss_buy_sku
            + loss_buy_cat
        )

        self.log("train/sum_loss", sum_loss)
        return sum_loss

    def validation_step(self, batch, batch_idx):
        statistical_feature = batch["statistical_feature"]

        # sequence feat
        event_type = batch["event_type"]
        sku = batch["sku"]
        url = batch["url"]
        category = batch["category"]
        price = batch["price"]
        word = batch["word"]
        timestamp = batch["timestamp"]
        # label
        label_churn = batch["is_churn"]
        label_buy_sku = batch["buy_sku_label"]
        label_buy_cat = batch["buy_cat_label"]
        is_contain_buy_sku = batch["is_contain_buy_sku"]
        is_contain_buy_cat = batch["is_contain_buy_cat"]

        user_emb = self.compute_user_embedding(
            event_type,
            sku,
            url,
            category,
            price,
            word,
            timestamp,
            statistical_feature,
        )

        (
            logits_churn,
            logits_buy_sku,
            logits_buy_cat,
        ) = self.calc_logits(user_emb)

        zero_ratio = (user_emb == 0).float().mean()

        self.log("valid/user_embedding/mean", user_emb.mean())
        self.log("valid/user_embedding/std", user_emb.std())
        self.log("valid/user_embedding/zero_ratio", zero_ratio)

        logits_churn = logits_churn.squeeze(-1)
        loss_churn = self.bce_loss(logits_churn, label_churn.float())
        loss_churn = loss_churn * self.cfg.churn_loss_weight

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

        self.log('valid/loss_churn', loss_churn)

        sum_loss = (
            loss_churn
            + loss_buy_sku
            + loss_buy_cat
        )

        self.log("valid/sum_loss", sum_loss)

        val_metrics = {}
        val_metrics['churn_loss'] = loss_churn.item()
        val_metrics['cat_loss'] = loss_buy_cat.item()
        val_metrics['sku_loss'] = loss_buy_sku.item()
        val_metrics['sum_loss'] = sum_loss.item()
        logger.info(f"Validation metrics: {val_metrics}")

        # Update AUROC metrics
        self.valid_auroc_churn.update(logits_churn, label_churn)

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


    def on_validation_epoch_end(self):
        valid_auroc_churn_score = self.valid_auroc_churn.compute().item()
        valid_auroc_sku_score = self.valid_auroc_buy_sku.compute().item()
        valid_auroc_cat_score = self.valid_auroc_buy_cat.compute().item()
        sum_aucroc_score = valid_auroc_churn_score + valid_auroc_sku_score + valid_auroc_cat_score

        self.log("valid/AUROC_churn", valid_auroc_churn_score)
        self.log("valid/AUROC_buy_sku", valid_auroc_sku_score)
        self.log("valid/AUROC_buy_cat", valid_auroc_cat_score)
        self.log(
            "valid/sum_score",
            self.valid_auroc_churn.compute()
            + self.valid_auroc_buy_sku.compute()
            + self.valid_auroc_buy_cat.compute(),
        )

        val_metrics = {}
        val_metrics['churn_auc'] = valid_auroc_churn_score
        val_metrics['cat_lo'] = valid_auroc_cat_score
        val_metrics['sku_auc'] = valid_auroc_sku_score
        val_metrics['sum_auc'] = sum_aucroc_score
        logger.info(f"Validation metrics: {val_metrics}")

        self.valid_auroc_churn.reset()
        self.valid_auroc_buy_sku.reset()
        self.valid_auroc_buy_cat.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default='../dataset/mtl',
        help="Directory where target and input data are stored",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        help="Accelerator type (cuda or cpu)",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="Device ID",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    args = parser.parse_args()
    device = f"cuda:{args.devices}" if args.accelerator == "cuda" else "cpu"
    config = Config(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        accelerator=args.accelerator,
        device=device,
        num_workers=args.num_workers,
        devices=[int(args.devices)] if args.accelerator == "cuda" else [],
    )

    train_dataset_dir = os.path.join(args.data_dir, "train")
    valid_dataset_dir = os.path.join(args.data_dir, "valid")

    train_dataset = RecsysDatasetV12(dataset_dir=train_dataset_dir, max_len=config.max_seq_length)
    valid_dataset = RecsysDatasetV12(dataset_dir=valid_dataset_dir, max_len=config.max_seq_length)

    print(f"train_size : {len(train_dataset)}")
    print(f"valid_size : {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    model = LightningRecsysModel(config)

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
        accelerator=args.accelerator,
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model,
        train_dataloader,
        valid_dataloader,
    )


if __name__ == "__main__":
    main()
