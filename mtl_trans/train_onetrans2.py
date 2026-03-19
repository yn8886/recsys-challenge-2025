import argparse
import math
import os
from datetime import datetime
from enum import Enum
import logging
import lightning as L
import torch.nn.functional as F
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import AUROC
from config import Config
from layers.ple import PLE
from data_collator import RecsysDatasetV12
from embed import SkuEmbedding, WordEmbedding, UrlEmbedding, QueryEmbedding, PositionalEncoding
from layers.onetrans import OneTransModel


NUM_CANDIDATES_SKU = 100
NUM_CANDIDATES_CAT = 100

exp_name = os.path.splitext(os.path.basename(__file__))[0]
torch.backends.cudnn.benchmark = True

np.random.seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    PAD_IDX = 0
    PRODUCT_BUY = 1
    ADD_TO_CART = 2
    REMOVE_FROM_CART = 3
    PAGE_VISIT = 4
    SEARCH_QUERY = 5




class BehaviorSequenceTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        final_seq_len=12,
        d_model=256,
        num_heads=4,
        ns_len=10,
        seq_len=64,
        ns_input_dim=64,
        last_embed_dim=256,
        mlp_hidden_units=[256],
        dropout=0.1,
    ):
        super().__init__()

        self.model = OneTransModel(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            L_ns=ns_len,
            initial_L_s=seq_len,
            final_L_s=final_seq_len,
            ns_raw_dim=ns_input_dim,
            last_embed_dim=last_embed_dim,
            mlp_hidden_units=mlp_hidden_units,
            dropout=dropout
        )

    def forward(self, seq_emb, statistical_features, s_padding_mask):
        user_emb = self.model(seq_emb, statistical_features, s_padding_mask)

        return user_emb


class LightningRecsysModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.hidden_dim

        self.event_emb_layer = nn.Embedding(
            num_embeddings=cfg.num_event,
            embedding_dim=cfg.event_emb_dim,
            padding_idx=cfg.padding_idx,
        )

        self.word_emb_layer = WordEmbedding(
            num_word=cfg.num_word,
            word_emb_dim=cfg.item_emb_dim,
            padding_idx=cfg.padding_idx,
        )

        self.day_emb_layer = nn.Embedding(
            num_embeddings=cfg.num_day,
            embedding_dim=cfg.day_emb_dim,
            padding_idx=cfg.padding_idx,
        )
        self.week_emb_layer = nn.Embedding(
            num_embeddings=cfg.num_week,
            embedding_dim=cfg.week_emb_dim,
            padding_idx=cfg.padding_idx,
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
            event_emb_layer=self.event_emb_layer,
            event_emb_dim=cfg.event_emb_dim,
            item_emb_dim=cfg.hidden_dim,
            padding_idx=cfg.padding_idx,
        )

        self.url_emb_layer = UrlEmbedding(
            num_url=cfg.num_url,
            url_emb_dim=cfg.url_emb_dim,
            event_emb_layer=self.event_emb_layer,
            event_emb_dim=cfg.event_emb_dim,
            item_emb_dim=cfg.hidden_dim,
            padding_idx=cfg.padding_idx,
        )

        self.query_emb_layer = QueryEmbedding(
            word_emb_dim=cfg.word_emb_dim,
            word_emb_layer=self.word_emb_layer,
            event_emb_dim=cfg.event_emb_dim,
            event_emb_layer=self.event_emb_layer,
            item_emb_dim=cfg.hidden_dim,
            padding_idx=cfg.padding_idx,
        )

        self.num_seps = 2
        self.sep_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim))

        self.time_fusion_layer = nn.Linear(
            cfg.hidden_dim + cfg.day_emb_dim + cfg.week_emb_dim,
            cfg.hidden_dim
        )

        self.model = BehaviorSequenceTransformer(
            num_layers=cfg.num_layers,
            final_seq_len=cfg.final_l_s,
            d_model=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            ns_len=cfg.ns_len,
            seq_len=cfg.max_len + self.num_seps,
            ns_input_dim=cfg.total_ns_dim,
            last_embed_dim=cfg.last_embed_dim,
            mlp_hidden_units=cfg.mlp_hidden_units,
            dropout=cfg.dropout
        )

        self.ple = PLE(
            user_emb_dim=cfg.last_embed_dim,
            num_shared_experts=cfg.num_shared_experts,
            num_task_experts=cfg.num_task_experts,
            num_tasks=cfg.num_tasks,
            expert_hidden_dims=cfg.expert_hidden_dims,
            expert_output_dim=cfg.expert_output_dim,
            task_tower_hidden_dims=cfg.task_tower_hidden_dims,
        )

        self.churn_head = nn.Linear(cfg.last_embed_dim, 1)
        self.buy_category_head = nn.Linear(cfg.last_embed_dim, 100)
        self.buy_sku_head = nn.Linear(cfg.last_embed_dim, 100)

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

    def _aggregate_embeddings(
        self,
        event_type,
        sku_id,
        url_id,
        query_ids,
        cat_id,
        price_id,
        name_ids,
        diff_days=None,
        diff_weeks=None,
    ):
        B, S = event_type.shape

        agg_embeddings = torch.zeros(
            (B, S, self.hidden_dim),
            device=event_type.device
        )

        sku_pos_idx = (
            (event_type == EventType.ADD_TO_CART.value)
            | (event_type == EventType.PRODUCT_BUY.value)
            | (event_type == EventType.REMOVE_FROM_CART.value)
        )


        sku_id = sku_id[sku_pos_idx]
        cat_id = cat_id[sku_pos_idx]
        price_id = price_id[sku_pos_idx]
        sku_word_id = name_ids[sku_pos_idx]
        event_id = event_type[sku_pos_idx]
        x = self.sku_emb_layer(event_id, sku_id, cat_id, price_id, sku_word_id)
        agg_embeddings[sku_pos_idx, :] = x

        url_pos_idx = event_type == EventType.PAGE_VISIT.value
        url_id = url_id[url_pos_idx]
        event_id = event_type[url_pos_idx]
        agg_embeddings[url_pos_idx, :] = self.url_emb_layer(event_id, url_id)

        query_pos_idx = event_type == EventType.SEARCH_QUERY.value
        query_word_id = query_ids[query_pos_idx]
        event_id = event_type[query_pos_idx]
        agg_embeddings[query_pos_idx, :] = self.query_emb_layer(event_id, query_word_id)

        return agg_embeddings

    def compute_user_embedding(
        self,
        event_type,
        sku_id,
        url_id,
        query_ids,
        cat_id,
        price_id,
        name_ids,
        diff_days,
        diff_weeks,
        feature_list,
    ):
        # 1. 获取原始 Embedding (形状: B, S, D)
        agg_embeddings = self._aggregate_embeddings(
            event_type, sku_id, url_id, query_ids, cat_id, price_id, name_ids
        )

        day_emb = self.day_emb_layer(diff_days)  # (B, S, day_emb_dim)
        week_emb = self.week_emb_layer(diff_weeks)  # (B, S, week_emb_dim)

        fused_embeddings = torch.cat([agg_embeddings, day_emb, week_emb], dim=-1)
        seq_feat_emb = self.time_fusion_layer(fused_embeddings)

        B, S, D = seq_feat_emb.shape

        # ---------------------------------------------------------
        # 优化版：左填充 + 意图右对齐 (Timestamp-agnostic)
        # ---------------------------------------------------------
        # 步骤 A：定义意图层级 (数值越大，意图越强，排序越靠右)
        sort_keys = torch.zeros_like(event_type, dtype=torch.float32)  # PAD 默认 0.0, 排在最左侧

        # 弱意图 -> 1.0
        sort_keys[event_type == EventType.PAGE_VISIT.value] = 1.0
        sort_keys[event_type == EventType.SEARCH_QUERY.value] = 1.0

        # 中意图 -> 2.0
        sort_keys[event_type == EventType.ADD_TO_CART.value] = 2.0
        sort_keys[event_type == EventType.REMOVE_FROM_CART.value] = 2.0

        # 强意图 -> 3.0
        sort_keys[event_type == EventType.PRODUCT_BUY.value] = 3.0

        # 步骤 B：添加微小的时序偏移量，确保同等意图下，时间较晚的行为依然靠右
        # 偏移量必须极小，不能跨越意图层级(不能大于 1.0)
        pos_offset = torch.arange(S, device=event_type.device, dtype=torch.float32).unsqueeze(0) * 0.0001
        # PAD 不加偏移量，让它们完美扎堆在最左侧
        sort_keys = torch.where(event_type == EventType.PAD_IDX.value, 0.0, sort_keys + pos_offset)

        # 步骤 C：准备 SEP 标记并赋予夹心 sort_key
        sep_embs = self.sep_token.expand(B, self.num_seps, D)
        sep_event_types = torch.full((B, self.num_seps), -1, device=event_type.device, dtype=event_type.dtype)
        # SEP 穿插在 1.0, 2.0, 3.0 之间
        sep_sort_keys = torch.tensor([1.5, 2.5], device=event_type.device, dtype=torch.float32).unsqueeze(0).expand(B,
                                                                                                                    self.num_seps)

        # 步骤 D：拼接并执行一次向量化排序
        concat_embs = torch.cat([seq_feat_emb, sep_embs], dim=1)  # (B, S+2, D)
        concat_event_types = torch.cat([event_type, sep_event_types], dim=1)  # (B, S+2)
        concat_sort_keys = torch.cat([sort_keys, sep_sort_keys], dim=1)  # (B, S+2)

        # argsort 将按照从小到大的顺序排列，因此：PAD -> 浏览 -> SEP -> 加购 -> SEP -> 购买
        sorted_indices = torch.argsort(concat_sort_keys, dim=1)

        # 根据排序索引收集特征
        sorted_embs = torch.gather(concat_embs, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, D))
        sorted_event_types = torch.gather(concat_event_types, 1, sorted_indices)

        # 重新生成 Padding Mask。PAD_IDX(0) 因为 sort_key 是 0.0，全部被挤到了矩阵的最左边。
        # 这里提取出来的 Padding Mask 会自动呈现左侧全 0，右侧全 1 (包含有效特征和 SEP) 的形态
        src_padding_mask = (sorted_event_types != EventType.PAD_IDX.value).float()

        # 2. 统计特征处理及模型推断
        statistical_features = torch.cat(feature_list, dim=-1)
        user_emb = self.model(sorted_embs, statistical_features, src_padding_mask)

        return user_emb

    def calc_logits(self, user_emb):
        return self.ple(user_emb)

    def training_step(self, batch, batch_idx):
        # statistical feat
        group_lifecycle = batch["group_lifecycle"]
        group_recency = batch["group_recency"]
        group_purchase = batch["group_purchase"]
        group_cart_intent = batch["group_cart_intent"]
        group_exploration = batch["group_exploration"]

        # sequence feat
        event_type = batch["event_type"]
        sku = batch["sku"]
        url = batch["url"]
        query = batch["query"]
        category = batch["category"]
        price = batch["price"]
        name = batch["name"]
        diff_days = batch["diff_days"]
        diff_weeks = batch["diff_weeks"]

        # label
        label_churn = batch["churn"]
        label_buy_sku = batch["buy_sku_label"]
        label_buy_cat = batch["buy_cat_label"]

        user_emb = self.compute_user_embedding(
            event_type,
            sku,
            url,
            query,
            category,
            price,
            name,
            diff_days,
            diff_weeks,
            [group_lifecycle, group_recency, group_purchase, group_cart_intent, group_exploration],
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
        is_contain_buy_sku = (torch.sum(label_buy_sku, dim=1) > 0).long()
        loss_buy_sku = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_sku) > 0:
            mask = is_contain_buy_sku == 1
            loss_buy_sku = self.bce_loss(
                logits_buy_sku[mask], label_buy_sku[mask].float()
            ) * self.cfg.buy_weight

        is_contain_buy_cat = (torch.sum(label_buy_cat, dim=1) > 0).long()
        loss_buy_cat = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_cat) > 0:
            mask = is_contain_buy_cat == 1
            loss_buy_cat = self.bce_loss(
                logits_buy_cat[mask], label_buy_cat[mask].float()
            ) * self.cfg.buy_weight

        self.log("train/loss_churn", loss_churn, prog_bar=True)
        self.log("train/loss_buy_sku", loss_buy_sku, prog_bar=True)
        self.log("train/loss_buy_cat", loss_buy_cat, prog_bar=True)

        sum_loss = (
            loss_churn
            + loss_buy_sku
            + loss_buy_cat
        )

        emb_detached = user_emb.detach()
        logits_churn = self.churn_head(emb_detached).squeeze(dim=1)
        logits_buy_category = self.buy_category_head(emb_detached)
        logits_buy_sku = self.buy_sku_head(emb_detached)

        def _get_bce_loss(
                logits: torch.Tensor,
                target: torch.Tensor,
        ) -> torch.Tensor:
            assert logits.ndim == target.ndim
            assert logits.ndim <= 2
            _loss = F.binary_cross_entropy_with_logits(logits, target.to(dtype=torch.float32), reduction="none")
            if logits.ndim == 2:
                _loss = _loss.mean(dim=1)
            _loss = _loss.mean()
            return _loss

        sum_loss += _get_bce_loss(logits_churn, label_churn)
        sum_loss += _get_bce_loss(logits_buy_category, label_buy_cat)
        sum_loss += _get_bce_loss(logits_buy_sku, label_buy_sku)

        self.log("train/sum_loss", sum_loss)
        return sum_loss

    def validation_step(self, batch, batch_idx):
        # statistical feat
        group_lifecycle = batch["group_lifecycle"]
        group_recency = batch["group_recency"]
        group_purchase = batch["group_purchase"]
        group_cart_intent = batch["group_cart_intent"]
        group_exploration = batch["group_exploration"]

        # sequence feat
        event_type = batch["event_type"]
        sku = batch["sku"]
        url = batch["url"]
        query = batch["query"]
        category = batch["category"]
        price = batch["price"]
        name = batch["name"]
        diff_days = batch["diff_days"]
        diff_weeks = batch["diff_weeks"]

        # label
        label_churn = batch["churn"]
        label_buy_sku = batch["buy_sku_label"]
        label_buy_cat = batch["buy_cat_label"]

        user_emb = self.compute_user_embedding(
            event_type,
            sku,
            url,
            query,
            category,
            price,
            name,
            diff_days,
            diff_weeks,
            [group_lifecycle, group_recency, group_purchase, group_cart_intent, group_exploration],
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
        is_contain_buy_sku = (torch.sum(label_buy_sku, dim=1) > 0).long()
        loss_buy_sku = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_sku) > 0:
            mask = is_contain_buy_sku == 1
            loss_buy_sku = self.bce_loss(
                logits_buy_sku[mask], label_buy_sku[mask].float()
            )

        is_contain_buy_cat = (torch.sum(label_buy_cat, dim=1) > 0).long()
        loss_buy_cat = torch.tensor(0.0, device=self.device)
        if torch.sum(is_contain_buy_cat) > 0:
            mask = is_contain_buy_cat == 1
            loss_buy_cat = self.bce_loss(
                logits_buy_cat[mask], label_buy_cat[mask].float()
            )

        self.log("valid/loss_churn", loss_churn, logger=True, on_step=False, on_epoch=True)
        self.log("valid/loss_cat", loss_buy_cat, logger=True, on_step=False, on_epoch=True)
        self.log("valid/loss_sku", loss_buy_sku, logger=True, on_step=False, on_epoch=True)

        emb_detached = user_emb
        logits_churn2 = self.churn_head(emb_detached).squeeze(dim=1)
        logits_buy_category2 = self.buy_category_head(emb_detached)
        logits_buy_sku2 = self.buy_sku_head(emb_detached)

        def _get_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            _loss = F.binary_cross_entropy_with_logits(logits, target.to(dtype=torch.float32), reduction="none")
            if logits.ndim == 2:
                _loss = _loss.mean(dim=1)
            return _loss.mean()

        loss_churn2 = _get_bce_loss(logits_churn2, label_churn)
        loss_buy_cat2 = _get_bce_loss(logits_buy_category2, label_buy_cat)
        loss_buy_sku2 = _get_bce_loss(logits_buy_sku2, label_buy_sku)


        self.log("valid/loss_churn2", loss_churn2, logger=True, on_step=False, on_epoch=True)
        self.log("valid/loss_cat2", loss_buy_cat2, logger=True, on_step=False, on_epoch=True)
        self.log("valid/loss_sku2", loss_buy_sku2, logger=True, on_step=False, on_epoch=True)

        self.valid_auroc_churn.update(
            logits_churn2,
            label_churn.to(dtype=torch.uint8),
        )
        self.valid_auroc_buy_cat.update(
            logits_buy_category2,
            label_buy_cat.int(),
        )
        self.valid_auroc_buy_sku.update(
            logits_buy_sku2,
            label_buy_sku.int()
        )

        # boolean_indices = labels_empty.to(dtype=torch.bool).logical_not()
        # if boolean_indices.any():
        #     self.valid_acc.update(sim[boolean_indices].detach().cpu(), sim_labels[boolean_indices].cpu())


        # val_metrics = {}
        # val_metrics['churn_loss'] = loss_churn.item()
        # val_metrics['cat_loss'] = loss_buy_cat.item()
        # val_metrics['sku_loss'] = loss_buy_sku.item()
        # val_metrics['sum_loss'] = sum_loss.item()
        # # logger.info(f"Validation metrics: {val_metrics}")



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
        val_metrics['cat_auc'] = valid_auroc_cat_score
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
        "--hidden-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--ns-len",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--final-l-s",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--churn-loss-weight",
        type=float,
        default=0.3,
        help="Learning rate",
    )
    parser.add_argument(
        "--buy-weight",
        type=float,
        default=0.35,
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
        ns_len=args.ns_len,
        num_layers=args.num_layers,
        num_workers=args.num_workers,
        hidden_dim=args.hidden_dim,
        devices=[int(args.devices)] if args.accelerator == "cuda" else [],
        final_l_s=args.final_l_s,
        churn_loss_weight=args.churn_loss_weight,
        buy_weight=args.buy_weight,
    )

    train_dataset_dir = os.path.join(args.data_dir, "train")
    valid_dataset_dir = os.path.join(args.data_dir, "valid")

    train_dataset = RecsysDatasetV12(dataset_dir=train_dataset_dir, max_len=config.max_len)
    valid_dataset = RecsysDatasetV12(dataset_dir=valid_dataset_dir, max_len=config.max_len)

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
        strategy="ddp_find_unused_parameters_true",
        # logger=wandb_logger,
        accelerator=args.accelerator,
        max_epochs=config.num_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=50
    )
    trainer.fit(
        model,
        train_dataloader,
        valid_dataloader,
    )


if __name__ == "__main__":
    main()
