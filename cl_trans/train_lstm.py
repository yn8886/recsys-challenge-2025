import argparse
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
from torch.nn.utils.rnn import pack_padded_sequence
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torchmetrics.classification import AUROC
from data_collator import RecsysDatasetV12, EventDataCollatorContrastive
from embed import PositionalEncoding, WordEmbedding, SkuEmbedding
from config import Config

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


class EventTransformerBone(nn.Module):
    def __init__(
            self,
            input_dim: int = 512,
            embed_dim: int = 512,
            num_heads: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation="relu",
            num_encoder_layers: int = 1,
            last_embed_dim: int = 512,
            max_len: int = 100,
    ):
        super().__init__()
        self.last_embed_dim = last_embed_dim

        self.input_linear = nn.Linear(input_dim, embed_dim, bias=True)

        self.pos_encoder = PositionalEncoding(
            d_model=embed_dim,
            dropout=dropout,
            max_len=max_len,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
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

        proj_size = 0 if embed_dim == last_embed_dim else last_embed_dim
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            proj_size=proj_size
        )

    def forward(
            self,
            x,
            attention_mask,
    ):
        src_key_padding_mask = attention_mask.to(dtype=torch.bool).logical_not()
        lengths = attention_mask.sum(dim=1)
        x = self.input_linear(x)
        x = self.pos_encoder(x)

        z = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        z = z.masked_fill(src_key_padding_mask.unsqueeze(-1), 0.0)

        z_packed = pack_padded_sequence(
            z,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(z_packed, hx=None)
        pooled_output = h_n.squeeze(0)

        return EventTransformerBoneOutputs(
            pooled_output=pooled_output,
            last_hidden_state=z,
            attention_mask=attention_mask,
        )


class EventTransformerTarget(nn.Module):
    def __init__(
            self,
            input_dim: int = 512,
            embed_dim: int = 512,
            num_heads: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation="relu",
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
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation=cfg.activation,
            num_encoder_layers=cfg.num_encoder_layers,
            last_embed_dim=cfg.last_embed_dim,
            max_len=cfg.max_len,
        )

        self.model_target = EventTransformerTarget(
            input_dim=cfg.event_emb_dim + cfg.item_emb_dim + cfg.url_emb_dim + cfg.word_emb_dim,
            embed_dim=cfg.embed_dim,
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
        self.train_churn_auc = torchmetrics.AUROC(task="binary")
        self.train_buy_category_auc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=cfg.num_buy_categories,
            average="macro",
        )
        self.train_buy_sku_auc = torchmetrics.AUROC(
            task="multilabel",
            num_labels=cfg.num_buy_skus,
            average="macro",
        )

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
        outputs1: EventTransformerBoneOutputs = self.model(seq_emb, s_padding_mask)

        return outputs1

    def training_step(self, batch, batch_idx):
        input_features, target_features, labels = batch
        labels_empty = labels.pop("empty")

        outputs1 = self.compute_user_embedding(
            input_features['event_id'],
            input_features['sku'],
            input_features['url'],
            input_features['query_id'],
            input_features['category'],
            input_features['price'],
            input_features['name_id'],
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
        loss += _get_bce_loss(logits_empty, labels_empty)
        self.train_empty_auc.update(logits_empty.detach(), labels_empty.to(dtype=torch.uint8))

        # linear probe
        # detach to prevent gradient flow to embedding model
        emb_detached = outputs1.pooled_output.detach()
        logits_churn = self.churn_head(emb_detached).squeeze(dim=1)
        logits_buy_category = self.buy_category_head(emb_detached)
        logits_buy_sku = self.buy_sku_head(emb_detached)
        loss += _get_bce_loss(logits_churn, labels["churn"])
        loss += _get_bce_loss(logits_buy_category, labels["buy_category"])
        loss += _get_bce_loss(logits_buy_sku, labels["buy_sku"])
        # evaluate only on relevant examples
        self.train_churn_auc.update(
            logits_churn.detach(),
            labels["churn"].to(dtype=torch.uint8),
        )
        self.train_buy_category_auc.update(
            logits_buy_category.detach(),
            labels["buy_category"].to(dtype=torch.uint8),
        )
        self.train_buy_sku_auc.update(
            logits_buy_sku.detach(),
            labels["buy_sku"].to(dtype=torch.uint8),
        )

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/empty_auc", self.train_empty_auc.compute(), prog_bar=False, logger=True)
        self.log("train/churn_auc", self.train_churn_auc.compute(), prog_bar=False, logger=True)
        self.log("train/buy_category_auc", self.train_buy_category_auc.compute(), prog_bar=False, logger=True)
        self.log("train/buy_sku_auc", self.train_buy_sku_auc.compute(), prog_bar=False, logger=True)
        self.train_empty_auc.reset()
        self.train_churn_auc.reset()
        self.train_buy_category_auc.reset()
        self.train_buy_sku_auc.reset()
        self._set_state_radam_schedule_free(is_train=False)

    def validation_step(self, batch, batch_idx):
        input_features, target_features, labels = batch
        labels_empty = labels.pop("empty")  # (B,)

        outputs1 = self.compute_user_embedding(
            input_features['event_id'],
            input_features['sku'],
            input_features['url'],
            input_features['query_id'],
            input_features['category'],
            input_features['price'],
            input_features['name_id'],
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
        self.valid_buy_category_auc.update(
            logits_buy_category,
            labels["buy_category"].to(dtype=torch.uint8)
        )
        self.valid_buy_sku_auc.update(
            logits_buy_sku,
            labels["buy_sku"].to(dtype=torch.uint8)
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
        "--num-workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
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
