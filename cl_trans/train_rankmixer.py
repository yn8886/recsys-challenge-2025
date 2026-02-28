import argparse
import math
import os
from dataclasses import dataclass
from enum import Enum
import logging
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
import torch.nn.functional as F
import torchmetrics
from schedulefree import RAdamScheduleFree
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import AUROC
from data_collator import RecsysDatasetV12, EventDataCollatorContrastive
from embed import PositionalEncoding, WordEmbedding, SkuEmbedding
from config import Config
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


@dataclass
class EventTransformerBoneOutputs:
    pooled_output: torch.Tensor
    last_hidden_state: torch.Tensor
    attention_mask: torch.Tensor
    moe_loss: torch.Tensor


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



class EventTransformerBone(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        d_model: int = 512,
        static_features_dim: int = 46,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation = "relu",
        num_encoder_layers: int = 1,
        last_embed_dim: int = 512,
        max_len: int = 100,
        ns_len: int = 1,
        use_semantic=True,
        semantic_groups=[5, 38, 12, 22, 13],
        total_ns_dim: int = 90,
        rankmixer_layers: int = 2,
        rankmixer_ffn_mult: int = 4,
        rankmixer_token_dp: float = 0.0,
        rankmixer_ffn_dp: float = 0.2,
        use_moe: bool = True,
        moe_experts: int = 4,
        moe_l1_coef: float = 0.01,
        moe_use_dtsi: bool = True,
        mlp_hidden_units: list = [256],
        pooling_strategy: str = "max"
    ):
        super().__init__()
        self.last_embed_dim = last_embed_dim
        self.static_features_dim = static_features_dim

        if pooling_strategy not in ["max", "mean", "last"]:
            raise ValueError(f"pooling_strategy must be one of ['max', 'mean', 'last'], got {pooling_strategy}")
        self.pooling_strategy = pooling_strategy

        self.input_linear = nn.Linear(input_dim, d_model, bias=True)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        # self.static_token_proj = nn.Linear(static_features_dim, embed_dim)
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
            dropout_rates=rankmixer_ffn_dp
        )

    def forward(
        self,
        x,
        statistical_features,
        attention_mask,
    ):
        src_key_padding_mask = attention_mask.to(dtype=torch.bool).logical_not()
        lengths = attention_mask.sum(dim=1)
        x = self.input_linear(x)
        x = self.pos_encoder(x)

        z = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        z = z.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        if self.pooling_strategy == "max":
            z_masked_for_pool = z.masked_fill(
                src_key_padding_mask.unsqueeze(-1), -1e9
            )
            pooled_features = z_masked_for_pool.max(dim=1).values

        elif self.pooling_strategy == "mean":
            sum_features = z.sum(dim=1)
            pooled_features = sum_features / lengths.unsqueeze(-1).clamp(min=1e-9)

        elif self.pooling_strategy == "last":
            batch_size = x.size(0)
            last_indices = (lengths - 1).long().clamp(min=0)
            pooled_features = z[torch.arange(batch_size, device=z.device), last_indices]

        # static_token = self.static_token_proj(static_feats).unsqueeze(1)
        ns_tokens = self.tokenizer(statistical_features)

        seq_token = pooled_features.unsqueeze(1)

        rm_input = torch.cat([seq_token, ns_tokens], dim=1)

        rm_output, moe_loss = self.rankmixer(rm_input)

        rm_output_flat = rm_output.view(rm_output.size(0), -1)
        # rm_output_flat = rm_output.mean(dim=1)

        pooled_output = self.final_mlp(rm_output_flat)

        return EventTransformerBoneOutputs(
            pooled_output=pooled_output,
            last_hidden_state=z,
            attention_mask=attention_mask,
            moe_loss=moe_loss
        )


class EventTransformerTarget(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        embed_dim: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation = "relu",
        num_decoder_layers: int = 1,
        last_embed_dim: int = 512,
    ):
        super().__init__()

        self.last_embed_dim = last_embed_dim

        self.input_linear = nn.Linear(input_dim, embed_dim, bias=True)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.last_linear = nn.Linear(embed_dim, last_embed_dim, bias=True)
        self.last_embed_dim = last_embed_dim

    def forward(
        self,
        x: dict[str, torch.Tensor],
        attention_mask
    ) -> torch.Tensor:

        src_key_padding_mask = attention_mask.to(dtype=torch.bool).logical_not()
        x = self.input_linear(x)
        z = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        z = z.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)
        # average pooling
        pooling_mask = attention_mask.to(dtype=torch.float32)
        pooled_trm_z = z.sum(dim=1) / pooling_mask.sum(dim=1, keepdim=True)

        # last linear
        pooled_trm_z = self.last_linear(pooled_trm_z)

        return EventTransformerBoneOutputs(
            pooled_output=pooled_trm_z,
            last_hidden_state=z,
            attention_mask=attention_mask,
            moe_loss=None,
        )

class LightningRecsysModel(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.padding_idx = cfg.padding_idx

        self.event_emb_layer = nn.Embedding(
            num_embeddings=cfg.num_event,
            embedding_dim=cfg.event_emb_dim,
            padding_idx=self.padding_idx,
        )

        self.word_emb_layer = WordEmbedding(
            num_word=cfg.num_word,
            word_emb_dim=cfg.word_emb_dim,
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
            embedding_dim=cfg.url_emb_dim,
            padding_idx=self.padding_idx,
        )

        self.model = EventTransformerBone(
            input_dim=cfg.event_emb_dim + cfg.item_emb_dim + cfg.url_emb_dim + cfg.word_emb_dim + cfg.day_emb_dim + cfg.week_emb_dim,
            d_model=cfg.hidden_dim,
            static_features_dim=cfg.static_features_dim,
            num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation,
            num_encoder_layers=cfg.num_encoder_layers,
            last_embed_dim=cfg.last_embed_dim,
            max_len=cfg.max_len,
            ns_len=cfg.ns_len,
            use_semantic=cfg.use_semantic,
            semantic_groups=cfg.semantic_groups,
            total_ns_dim=cfg.total_ns_dim,
            rankmixer_layers=cfg.rankmixer_layers,
            rankmixer_ffn_mult=cfg.rankmixer_ffn_mult,
            rankmixer_token_dp=cfg.rankmixer_token_dp,
            rankmixer_ffn_dp=cfg.rankmixer_ffn_dp,
            mlp_hidden_units=cfg.mlp_hidden_units,
            pooling_strategy=cfg.pooling_strategy
        )

        self.model_target = EventTransformerTarget(
            input_dim=cfg.event_emb_dim + cfg.item_emb_dim + cfg.url_emb_dim + cfg.word_emb_dim,
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation,
            num_decoder_layers=cfg.num_decoder_layers,
            last_embed_dim=cfg.last_embed_dim,
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

        self.lr = cfg.learning_rate
        self.temperature = cfg.temperature

        self.save_hyperparameters()

        assert self.model.last_embed_dim == self.model_target.last_embed_dim
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=cfg.batch_size)

        # auxiliary task for embedding model
        self.empty_head = nn.Linear(self.model.last_embed_dim, 1, bias=True)
        self.train_empty_auc = torchmetrics.AUROC(task="binary")

        # linear probe to evaluate embedding
        self.churn_head = nn.Linear(self.model.last_embed_dim, 1)
        self.buy_category_head = nn.Linear(self.model.last_embed_dim, cfg.num_buy_categories)
        self.buy_sku_head = nn.Linear(self.model.last_embed_dim, cfg.num_buy_skus)

        self.valid_empty_auc = torchmetrics.AUROC(task="binary")
        self.valid_churn_auc = torchmetrics.AUROC(task="binary")
        self.valid_buy_category_auc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=cfg.num_buy_categories,
            average="macro",
        )
        self.valid_buy_sku_auc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=cfg.num_buy_skus,
            average="macro",
        )

    def compute_user_embedding(
        self,
        event_type,
        sku_id,
        url_id,
        query_id,
        cat_id,
        price_id,
        name_id,
        statistical_features=None,
        diff_days=None,
        diff_weeks=None,
        is_target=False,
    ):

        event_emb = self.event_emb_layer(event_type)
        sku_emb = self.sku_emb_layer(sku_id, cat_id, price_id, name_id)
        url_emb = self.url_emb_layer(url_id)
        query_emb = self.word_emb_layer(query_id)

        if is_target:
            seq_emb = torch.concat([event_emb, sku_emb, url_emb, query_emb], dim=-1, )
            return seq_emb
        else:
            day_seq_emb = self.day_emb_layer(diff_days)
            week_seq_emb = self.week_emb_layer(diff_weeks)

            seq_emb = torch.concat(
                [event_emb, sku_emb, url_emb, query_emb, day_seq_emb, week_seq_emb],
                dim=-1,
            )

        s_padding_mask = (event_type != 0).float()
        outputs1: EventTransformerBoneOutputs = self.model(seq_emb, statistical_features, s_padding_mask)

        return outputs1

    def training_step(self, batch, batch_idx):
        input_features, target_features, labels = batch
        labels_empty = labels.pop("empty")

        group_lifecycle = input_features["group_lifecycle"]
        group_recency = input_features["group_recency"]
        group_purchase = input_features["group_purchase"]
        group_cart_intent = input_features["group_cart_intent"]
        group_exploration = input_features["group_exploration"]

        outputs1 = self.compute_user_embedding(
            input_features['event_id'],
            input_features['sku'],
            input_features['url'],
            input_features['query_id'],
            input_features['category'],
            input_features['price'],
            input_features['name_id'],
            [group_lifecycle, group_recency, group_purchase, group_cart_intent, group_exploration],
            diff_days=input_features['diff_days'],
            diff_weeks=input_features['diff_weeks'],
        )

        targets_seq = self.compute_user_embedding(
            target_features['event_id'],
            target_features['sku'],
            target_features['url'],
            target_features['query_id'],
            target_features['category'],
            target_features['price'],
            target_features['name_id'],
            is_target=True
        )
        attention_mask = (target_features['event_id'] != 0).float()
        attention_mask[:, -1] = 1
        outputs2 = self.model_target(targets_seq, attention_mask=attention_mask)

        # contrastive learning
        sim = F.cosine_similarity(
            outputs1.pooled_output.unsqueeze(0),  # (1, B, D)
            outputs2.pooled_output.unsqueeze(1),  # (B, 1, D)
            dim=-1,
        )
        sim_labels = torch.arange(len(sim), dtype=torch.long, device=sim.device)
        # mask for empty
        _loss = F.cross_entropy(sim / self.temperature, sim_labels, reduction="none")
        loss = _loss.mean()

        total_loss = loss
        if hasattr(outputs1, 'moe_loss') and outputs1.moe_loss is not None:
            total_loss = total_loss + outputs1.moe_loss

        self.log("train/loss", loss.detach().item(), prog_bar=True, logger=True, on_step=True)
        # boolean_indices = labels_empty.to(dtype=torch.bool).logical_not()
        # if boolean_indices.any():
        #     self.log(
        #         "train/acc",
        #         self.train_acc(sim[boolean_indices].detach().cpu(), sim_labels[boolean_indices].cpu()),
        #         prog_bar=True,
        #         logger=True,
        #         on_step=True,
        #         on_epoch=False,
        #     )

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

        # auxiliary task: emptry prediction
        logits_empty = self.empty_head(outputs1.pooled_output).squeeze(dim=1)
        total_loss += _get_bce_loss(logits_empty, labels_empty)
        self.train_empty_auc.update(logits_empty.detach(), labels_empty.to(dtype=torch.uint8))

        # linear probe
        # detach to prevent gradient flow to embedding model
        emb_detached = outputs1.pooled_output.detach()
        logits_churn = self.churn_head(emb_detached).squeeze(dim=1)
        logits_buy_category = self.buy_category_head(emb_detached)
        logits_buy_sku = self.buy_sku_head(emb_detached)
        total_loss += _get_bce_loss(logits_churn, labels["churn"])
        total_loss += _get_bce_loss(logits_buy_category, labels["buy_category"])
        total_loss += _get_bce_loss(logits_buy_sku, labels["buy_sku"])

        return total_loss


    def validation_step(self, batch, batch_idx):
        input_features, target_features, labels = batch
        labels_empty = labels.pop("empty")  # (B,)

        group_lifecycle = input_features["group_lifecycle"]
        group_recency = input_features["group_recency"]
        group_purchase = input_features["group_purchase"]
        group_cart_intent = input_features["group_cart_intent"]
        group_exploration = input_features["group_exploration"]

        outputs1 = self.compute_user_embedding(
            input_features['event_id'],
            input_features['sku'],
            input_features['url'],
            input_features['query_id'],
            input_features['category'],
            input_features['price'],
            input_features['name_id'],
            [group_lifecycle, group_recency, group_purchase, group_cart_intent, group_exploration],
            diff_days=input_features['diff_days'],
            diff_weeks=input_features['diff_weeks'],
        )

        targets_seq = self.compute_user_embedding(
            target_features['event_id'],
            target_features['sku'],
            target_features['url'],
            target_features['query_id'],
            target_features['category'],
            target_features['price'],
            target_features['name_id'],
            is_target=True
        )
        attention_mask = (target_features['event_id'] != 0).float()
        attention_mask[:, -1] = 1
        outputs2 = self.model_target(targets_seq, attention_mask=attention_mask)

        # 3. Contrastive Loss & Accuracy
        sim = F.cosine_similarity(
            outputs1.pooled_output.unsqueeze(0),
            outputs2.pooled_output.unsqueeze(1),
            dim=-1,
        )
        sim_labels = torch.arange(len(sim), dtype=torch.long, device=sim.device)
        _loss = F.cross_entropy(sim / self.temperature, sim_labels, reduction="none")
        loss = _loss.mean()

        # 定义辅助 Loss 函数
        def _get_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            _loss = F.binary_cross_entropy_with_logits(logits, target.to(dtype=torch.float32), reduction="none")
            if logits.ndim == 2:
                _loss = _loss.mean(dim=1)
            return _loss.mean()

        # 4. Auxiliary Task: Empty Prediction
        logits_empty = self.empty_head(outputs1.pooled_output).squeeze(dim=1)
        loss_empty = _get_bce_loss(logits_empty, labels_empty)
        loss += loss_empty

        self.valid_empty_auc.update(logits_empty, labels_empty.to(dtype=torch.uint8))

        # 5. Linear Probes (Downstream Tasks)
        emb_detached = outputs1.pooled_output  # Validation 不需要 detach 用于梯度阻断，但保持一致性无妨
        logits_churn = self.churn_head(emb_detached).squeeze(dim=1)
        logits_buy_category = self.buy_category_head(emb_detached)
        logits_buy_sku = self.buy_sku_head(emb_detached)

        loss_churn = _get_bce_loss(logits_churn, labels["churn"])
        loss_buy_cat = _get_bce_loss(logits_buy_category, labels["buy_category"])
        loss_buy_sku = _get_bce_loss(logits_buy_sku, labels["buy_sku"])

        loss += loss_churn + loss_buy_cat + loss_buy_sku

        self.log("valid/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("valid/loss_churn", loss_churn, logger=True, on_step=False, on_epoch=True)
        self.log("valid/loss_cat", loss_buy_cat, logger=True, on_step=False, on_epoch=True)
        self.log("valid/loss_sku", loss_buy_sku, logger=True, on_step=False, on_epoch=True)

        # boolean_indices = labels_empty.to(dtype=torch.bool).logical_not()
        # if boolean_indices.any():
        #     self.valid_acc.update(sim[boolean_indices].detach().cpu(), sim_labels[boolean_indices].cpu())

        self.valid_churn_auc.update(
            logits_churn,
            labels["churn"].to(dtype=torch.uint8)
        )
        is_contain_buy_sku = (torch.sum(labels["buy_sku"], dim=1) > 0).long()
        if torch.sum(is_contain_buy_sku) > 0:
            mask = is_contain_buy_sku == 1
            self.valid_buy_sku_auc.update(
                logits_buy_sku[mask], labels["buy_sku"][mask].int()
            )

        is_contain_buy_cat = (torch.sum(labels["buy_category"], dim=1) > 0).long()
        if torch.sum(is_contain_buy_cat) > 0:
            mask = is_contain_buy_cat == 1
            self.valid_buy_category_auc.update(
                logits_buy_category[mask], labels["buy_category"][mask].int()
            )

        return loss


    def on_validation_epoch_end(self):
        # valid_acc = self.valid_acc.compute()
        valid_empty_auc = self.valid_empty_auc.compute()
        valid_churn_auc = self.valid_churn_auc.compute()
        valid_cat_auc = self.valid_buy_category_auc.compute()
        valid_sku_auc = self.valid_buy_sku_auc.compute()

        # Log
        # self.log("valid/acc", valid_acc, prog_bar=True, logger=True)
        self.log("valid/empty_auc", valid_empty_auc, prog_bar=False, logger=True)
        self.log("valid/churn_auc", valid_churn_auc, prog_bar=False, logger=True)
        self.log("valid/buy_category_auc", valid_cat_auc, prog_bar=False, logger=True)
        self.log("valid/buy_sku_auc", valid_sku_auc, prog_bar=False, logger=True)

        sum_score = valid_churn_auc + valid_cat_auc + valid_sku_auc
        self.log("valid/sum_score", sum_score, prog_bar=True, logger=True)

        val_metrics = {
            'churn_auc': valid_churn_auc.item(),
            'cat_auc': valid_cat_auc.item(),
            'sku_auc': valid_sku_auc.item(),
            'sum_score': sum_score.item()
        }
        logger.info(f"Validation metrics: {val_metrics}")

        # self.valid_acc.reset()
        self.valid_empty_auc.reset()
        self.valid_churn_auc.reset()
        self.valid_buy_category_auc.reset()
        self.valid_buy_sku_auc.reset()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def _set_state_radam_schedule_free(self, is_train: bool):
        for i in range(len(self.trainer.optimizers)):
            if isinstance(self.trainer.optimizers[i], RAdamScheduleFree):
                if is_train:
                    self.trainer.optimizers[i].train()
                else:
                    self.trainer.optimizers[i].eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default='../dataset/cl',
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
        "--max-len",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--ns-len",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
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
        num_heads=args.num_heads,
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
        collate_fn=EventDataCollatorContrastive(padding='longest', max_length=args.max_len),
        pin_memory=True,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=EventDataCollatorContrastive(padding='longest', max_length=args.max_len),
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
        # strategy="ddp_find_unused_parameters_true",
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
