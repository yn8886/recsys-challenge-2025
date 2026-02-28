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
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import AUROC
from config import Config
from layers.ple import PLE
from data_collator import RecsysDatasetV12
from embed import SkuEmbedding, WordEmbedding, UrlEmbedding, QueryEmbedding, PositionalEncoding
from fuxictr.pytorch.layers import MLP_Block
from layers.rankmixer import RankMixerEncoder

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


class SemanticTokenizer(nn.Module):
    def __init__(self, d_model: int, target_tokens: int, use_semantic: bool, total_ns_dim:int, semantic_groups=None):
        super().__init__()
        self.d_model = d_model
        self.target_tokens = target_tokens
        self.use_semantic = use_semantic
        self.semantic_groups = semantic_groups
        self.projections = nn.ModuleList()

        if use_semantic:
            for i, in_dim in enumerate(self.semantic_groups):
                self.projections.append(nn.Linear(in_dim, d_model))

            self.actual_tokens = len(self.projections)
        else:
            self.shared_proj = nn.Linear(total_ns_dim, target_tokens * d_model)

    def forward(self, feature_list):
        batch_size = feature_list[0].size(0)

        if self.use_semantic:
            tokens = []
            for i in range(self.actual_tokens):
                token = self.projections[i](feature_list[i])
                tokens.append(token)
            stacked_tokens = torch.stack(tokens, dim=1)

            return stacked_tokens

        else:
            concat_tensor = torch.cat(feature_list, dim=-1)
            flat_tokens = self.shared_proj(concat_tensor)
            return flat_tokens.view(batch_size, self.target_tokens, self.d_model)

class BehaviorSequenceTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        static_features_dim,
        num_heads,
        num_layers,
        dropout,
        max_len,
        last_embed_dim,
        ns_len: int = 1,
        use_semantic=True,
        semantic_groups=[5,38,12,22,13],
        total_ns_dim: int = 90,
        rankmixer_layers: int = 2,
        rankmixer_ffn_mult: int = 4,
        rankmixer_token_dp: float = 0.0,
        rankmixer_ffn_dp: float = 0.2,
        use_moe: bool = True,
        moe_experts: int = 4,
        moe_l1_coef: float = 0.01,
        moe_use_dtsi: bool = True,
        mlp_hidden_units=[512],
        dcn_dropout=0.2,
        pooling_strategy='last'
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
        )
        self.trm_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.trm_enc = nn.TransformerEncoder(
            self.trm_enc_layer, num_layers=num_layers
        )

        self.ns_len = ns_len
        self.tokenizer = SemanticTokenizer(
            d_model=d_model,
            target_tokens=ns_len,
            use_semantic=use_semantic,
            total_ns_dim=total_ns_dim,
            semantic_groups=semantic_groups,
        )

        self.num_rankmixer_tokens = ns_len + 1

        self.rankmixer = RankMixerEncoder(
            num_layers=rankmixer_layers,
            num_tokens=self.num_rankmixer_tokens,
            d_model=d_model,
            ffn_mult=rankmixer_ffn_mult,
            token_dp=rankmixer_token_dp,
            ffn_dp=rankmixer_ffn_dp,
            ln_style="pre",
            use_moe=use_moe,
            moe_experts=moe_experts,
            moe_l1_coef=moe_l1_coef,
            moe_use_dtsi=moe_use_dtsi
        )

        final_in_dim = self.num_rankmixer_tokens * d_model
        self.final_mlp = MLP_Block(
            input_dim=final_in_dim,
            output_dim=last_embed_dim,
            hidden_units=mlp_hidden_units,
            hidden_activations="ReLU",
            output_activation=None,
            dropout_rates=dcn_dropout
        )

    def forward(self, seq_emb, statistical_features, attention_mask):
        lengths = attention_mask.sum(dim=1)
        src_key_padding_mask = attention_mask.to(dtype=torch.bool).logical_not()
        seq_emb = self.pos_encoder(seq_emb)
        z = self.trm_enc(seq_emb, src_key_padding_mask=src_key_padding_mask)

        if self.pooling_strategy == "max":
            z_masked_for_pool = z.masked_fill(
                src_key_padding_mask.unsqueeze(-1), -1e9
            )
            pooled_features = z_masked_for_pool.max(dim=1).values

        elif self.pooling_strategy == "mean":
            sum_features = z.sum(dim=1)
            pooled_features = sum_features / lengths.unsqueeze(-1).clamp(min=1e-9)

        elif self.pooling_strategy == "last":
            batch_size = seq_emb.size(0)
            last_indices = (lengths - 1).long().clamp(min=0)
            pooled_features = z[torch.arange(batch_size, device=z.device), last_indices]

        # static_token = self.static_token_proj(static_feats).unsqueeze(1)
        ns_tokens = self.tokenizer(statistical_features)

        seq_token = pooled_features.unsqueeze(1)

        rm_input = torch.cat([seq_token, ns_tokens], dim=1)

        rm_output, moe_loss = self.rankmixer(rm_input)

        rm_output_flat = rm_output.view(rm_output.size(0), -1)
        user_emb = self.final_mlp(rm_output_flat)

        return user_emb, moe_loss


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

        self.model = BehaviorSequenceTransformer(
            d_model=cfg.hidden_dim,
            static_features_dim=cfg.static_features_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            last_embed_dim=cfg.last_embed_dim,
            ns_len=cfg.ns_len,
            use_semantic=cfg.use_semantic,
            semantic_groups=cfg.semantic_groups,
            total_ns_dim=cfg.total_ns_dim,
            rankmixer_layers=cfg.rankmixer_layers,
            rankmixer_ffn_mult=cfg.rankmixer_ffn_mult,
            rankmixer_token_dp=cfg.rankmixer_token_dp,
            rankmixer_ffn_dp=cfg.rankmixer_ffn_dp,
            use_moe=cfg.use_moe,
            moe_experts=cfg.moe_experts,
            moe_l1_coef=cfg.moe_l1_coef,
            moe_use_dtsi=cfg.moe_use_dtsi,
            mlp_hidden_units=cfg.mlp_hidden_units,
            pooling_strategy=cfg.pooling_strategy,

        )

        self.ple = PLE(
            user_emb_dim=cfg.fusion_mlp_output_dim,
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

    def _aggregate_embeddings(
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
        statistical_features,
    ):
        seq_feat_emb = self._aggregate_embeddings(
            event_type, sku_id, url_id, query_ids, cat_id, price_id, name_ids, diff_days, diff_weeks
        )
        src_padding_mask = (event_type != 0).float()
        user_emb, moe_loss = self.model(seq_feat_emb, statistical_features, src_padding_mask)

        return user_emb, moe_loss

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

        user_emb, moe_loss = self.compute_user_embedding(
            event_type,
            sku,
            url,
            query,
            category,
            price,
            name,
            diff_days,
            diff_weeks,
            [group_lifecycle, group_recency, group_purchase, group_cart_intent, group_exploration]
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
            )

        is_contain_buy_cat = (torch.sum(label_buy_cat, dim=1) > 0).long()
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
            + moe_loss
        )

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

        user_emb, moe_loss = self.compute_user_embedding(
            event_type,
            sku,
            url,
            query,
            category,
            price,
            name,
            diff_days,
            diff_weeks,
            [group_lifecycle, group_recency, group_purchase, group_cart_intent, group_exploration]
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
        # logger.info(f"Validation metrics: {val_metrics}")

        # Update AUROC metrics
        self.valid_auroc_churn.update(logits_churn, label_churn)

        is_contain_buy_sku = (torch.sum(label_buy_sku, dim=1) > 0).long()
        if torch.sum(is_contain_buy_sku) > 0:
            mask = is_contain_buy_sku == 1
            self.valid_auroc_buy_sku.update(
                logits_buy_sku[mask], label_buy_sku[mask].int()
            )

        is_contain_buy_cat = (torch.sum(label_buy_cat, dim=1) > 0).long()
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
        "--pooling-strategy",
        type=str,
        default='last',
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
        "--hidden-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--ns-len",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--use-semantic",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use-moe",
        type=bool,
        default=False,
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
        ns_len=args.ns_len,
        use_moe=args.use_moe,
        use_semantic=args.use_semantic,
        hidden_dim=args.hidden_dim,
        pooling_strategy=args.pooling_strategy,
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
        # strategy="ddp_find_unused_parameters_true",
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
