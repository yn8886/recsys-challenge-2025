import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from schedulefree import RAdamScheduleFree
from torchmetrics.functional import auroc
from typing import Optional
from ensembles.torch.model import StackingModel


class StackingModelModule(L.LightningModule):
    def __init__(
        self,
        model: StackingModel,
        tasks: list[str],
        lr: float,
        num_propensity_sku: Optional[int] = None,
        num_propensity_category: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.tasks = tasks
        self.lr = lr

        self.train_loss = torchmetrics.MeanMetric()
        self.valid_loss = torchmetrics.MeanMetric()

        if "churn" in tasks:
            self.train_churn_auc = torchmetrics.AUROC(task="binary")
            self.valid_churn_auc = torchmetrics.AUROC(task="binary")

        if "churn_cart" in tasks:
            self.train_churn_cart_auc = torchmetrics.AUROC(task="binary")
            self.valid_churn_cart_auc = torchmetrics.AUROC(task="binary")

        if "propensity_sku" in tasks:
            assert num_propensity_sku is not None
            self.train_propensity_sku_auc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=num_propensity_sku,
                average="macro",
            )
            self.valid_propensity_sku_auc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=num_propensity_sku,
                average="macro",
            )

        if "propensity_category" in tasks:
            assert num_propensity_category is not None
            self.train_propensity_category_auc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=num_propensity_category,
                average="macro",
            )
            self.valid_propensity_category_auc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=num_propensity_category,
                average="macro",
            )

        if "cart_sku" in tasks:
            assert num_propensity_sku is not None
            self.train_cart_sku_auc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=num_propensity_sku,
                average="macro",
            )
            self.valid_cart_sku_auc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=num_propensity_sku,
                average="macro",
            )

        if "cart_category" in tasks:
            assert num_propensity_category is not None
            self.train_cart_category_auc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=num_propensity_category,
                average="macro",
            )
            self.valid_cart_category_auc = torchmetrics.AUROC(
                task="multilabel",
                num_labels=num_propensity_category,
                average="macro",
            )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        embedding = batch["embedding"]
        outputs = self.model(embedding)

        loss = 0

        if "churn" in self.tasks:
            churn_logits = outputs.churn_logits
            churn_labels = batch["churn_label"].unsqueeze(-1)
            churn_loss = F.binary_cross_entropy_with_logits(
                churn_logits,
                churn_labels.to(torch.float32),
                reduction="mean",
            )
            loss += churn_loss
            self.train_churn_auc.update(
                churn_logits.detach().cpu(),
                churn_labels.cpu().to(torch.int64),
            )
            batch_churn_auc = auroc(
                churn_logits.detach().cpu(),
                churn_labels.cpu().to(torch.int64),
                task="binary",
            )
            self.log("train_churn_auc", batch_churn_auc, prog_bar=True)

        if "churn_cart" in self.tasks:
            churn_cart_logits = outputs.churn_cart_logits
            churn_cart_labels = batch["churn_cart_label"].unsqueeze(-1)
            churn_cart_loss = F.binary_cross_entropy_with_logits(
                churn_cart_logits,
                churn_cart_labels.to(torch.float32),
                reduction="mean",
            )
            loss += churn_cart_loss
            self.train_churn_cart_auc.update(
                churn_cart_logits.detach().cpu(),
                churn_cart_labels.cpu().to(torch.int64),
            )
            batch_churn_cart_auc = auroc(
                churn_cart_logits.detach().cpu(),
                churn_cart_labels.cpu().to(torch.int64),
                task="binary",
            )
            self.log("train_churn_cart_auc", batch_churn_cart_auc, prog_bar=True)

        if "propensity_sku" in self.tasks:
            propensity_sku_logits = outputs.propensity_sku_logits
            propensity_sku_labels = batch["propensity_sku_labels"]
            propensity_sku_loss = F.binary_cross_entropy_with_logits(
                propensity_sku_logits,
                propensity_sku_labels.to(torch.float32),
                reduction="mean",
            )
            loss += propensity_sku_loss
            self.train_propensity_sku_auc.update(
                propensity_sku_logits.detach().cpu(),
                propensity_sku_labels.cpu().to(torch.int64),
            )

        if "propensity_category" in self.tasks:
            propensity_category_logits = outputs.propensity_category_logits
            propensity_category_labels = batch["propensity_category_labels"]
            propensity_category_loss = F.binary_cross_entropy_with_logits(
                propensity_category_logits,
                propensity_category_labels.to(torch.float32),
                reduction="mean",
            )
            loss += propensity_category_loss
            self.train_propensity_category_auc.update(
                propensity_category_logits.detach().cpu(),
                propensity_category_labels.cpu().to(torch.int64),
            )

        if "cart_sku" in self.tasks:
            cart_sku_logits = outputs.cart_sku_logits
            cart_sku_labels = batch["cart_sku_labels"]
            cart_sku_loss = F.binary_cross_entropy_with_logits(
                cart_sku_logits,
                cart_sku_labels.to(torch.float32),
                reduction="mean",
            )
            loss += cart_sku_loss
            self.train_cart_sku_auc.update(
                cart_sku_logits.detach().cpu(),
                cart_sku_labels.cpu().to(torch.int64),
            )

        if "cart_category" in self.tasks:
            cart_category_logits = outputs.cart_category_logits
            cart_category_labels = batch["cart_category_labels"]
            cart_category_loss = F.binary_cross_entropy_with_logits(
                cart_category_logits,
                cart_category_labels.to(torch.float32),
                reduction="mean",
            )
            loss += cart_category_loss
            self.train_cart_category_auc.update(
                cart_category_logits.detach().cpu(),
                cart_category_labels.cpu().to(torch.int64),
            )

        self.train_loss.update(loss.item())

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        embedding = batch["embedding"]
        outputs = self.model(embedding)

        loss = 0

        if "churn" in self.tasks:
            churn_logits = outputs.churn_logits
            churn_labels = batch["churn_label"].unsqueeze(-1)
            churn_loss = F.binary_cross_entropy_with_logits(
                churn_logits,
                churn_labels.to(torch.float32),
                reduction="mean",
            )
            loss += churn_loss
            self.valid_churn_auc.update(
                churn_logits.detach().cpu(),
                churn_labels.cpu().to(torch.int64),
            )
            batch_churn_auc = auroc(
                churn_logits.detach().cpu(),
                churn_labels.cpu().to(torch.int64),
                task="binary",
            )
            self.log("valid_churn_auc", batch_churn_auc, prog_bar=True)

        if "churn_cart" in self.tasks:
            churn_cart_logits = outputs.churn_cart_logits
            churn_cart_labels = batch["churn_cart_label"].unsqueeze(-1)
            churn_cart_loss = F.binary_cross_entropy_with_logits(
                churn_cart_logits,
                churn_cart_labels.to(torch.float32),
                reduction="mean",
            )
            loss += churn_cart_loss
            self.valid_churn_cart_auc.update(
                churn_cart_logits.detach().cpu(),
                churn_cart_labels.cpu().to(torch.int64),
            )
            batch_churn_cart_auc = auroc(
                churn_cart_logits.detach().cpu(),
                churn_cart_labels.cpu().to(torch.int64),
                task="binary",
            )
            self.log("valid_churn_cart_auc", batch_churn_cart_auc, prog_bar=True)

        if "propensity_sku" in self.tasks:
            propensity_sku_logits = outputs.propensity_sku_logits
            propensity_sku_labels = batch["propensity_sku_labels"]
            propensity_sku_loss = F.binary_cross_entropy_with_logits(
                propensity_sku_logits,
                propensity_sku_labels.to(torch.float32),
                reduction="mean",
            )
            loss += propensity_sku_loss
            self.valid_propensity_sku_auc.update(
                propensity_sku_logits.detach().cpu(),
                propensity_sku_labels.cpu().to(torch.int64),
            )

        if "propensity_category" in self.tasks:
            propensity_category_logits = outputs.propensity_category_logits
            propensity_category_labels = batch["propensity_category_labels"]
            propensity_category_loss = F.binary_cross_entropy_with_logits(
                propensity_category_logits,
                propensity_category_labels.to(torch.float32),
                reduction="mean",
            )
            loss += propensity_category_loss
            self.valid_propensity_category_auc.update(
                propensity_category_logits.detach().cpu(),
                propensity_category_labels.cpu().to(torch.int64),
            )

        if "cart_sku" in self.tasks:
            cart_sku_logits = outputs.cart_sku_logits
            cart_sku_labels = batch["cart_sku_labels"]
            cart_sku_loss = F.binary_cross_entropy_with_logits(
                cart_sku_logits,
                cart_sku_labels.to(torch.float32),
                reduction="mean",
            )
            loss += cart_sku_loss
            self.valid_cart_sku_auc.update(
                cart_sku_logits.detach().cpu(),
                cart_sku_labels.cpu().to(torch.int64),
            )

        if "cart_category" in self.tasks:
            cart_category_logits = outputs.cart_category_logits
            cart_category_labels = batch["cart_category_labels"]
            cart_category_loss = F.binary_cross_entropy_with_logits(
                cart_category_logits,
                cart_category_labels.to(torch.float32),
                reduction="mean",
            )
            loss += cart_category_loss
            self.valid_cart_category_auc.update(
                cart_category_logits.detach().cpu(),
                cart_category_labels.cpu().to(torch.int64),
            )

        self.valid_loss.update(loss.item())

        self.log("valid_loss", loss, prog_bar=True)

        return loss

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        embedding = batch["embedding"]
        outputs = self.model(embedding)
        return outputs.embedding.cpu().numpy().astype(np.float16)

    def configure_optimizers(self):
        optimizer = RAdamScheduleFree(self.model.parameters(), lr=self.lr)
        return optimizer

    def _assert_state_radam_schedule_free(self, is_train: bool):
        for i in range(len(self.trainer.optimizers)):
            if isinstance(self.trainer.optimizers[i], RAdamScheduleFree):
                assert (
                    self.trainer.optimizers[i].param_groups[0]["train_mode"] == is_train
                )

    def _set_state_radam_schedule_free(self, is_train: bool):
        for i in range(len(self.trainer.optimizers)):
            if isinstance(self.trainer.optimizers[i], RAdamScheduleFree):
                if is_train:
                    self.trainer.optimizers[i].train()
                else:
                    self.trainer.optimizers[i].eval()

    def on_train_epoch_start(self):
        self._set_state_radam_schedule_free(is_train=True)

    def on_validation_epoch_start(self):
        self._set_state_radam_schedule_free(is_train=False)

    def on_save_checkpoint(self, checkpoint):
        self._assert_state_radam_schedule_free(is_train=False)

    def on_train_epoch_end(self):
        self.log("train_loss", self.train_loss.compute(), prog_bar=True, logger=True)
        self.train_loss.reset()

        if "churn" in self.tasks:
            self.log(
                "train_churn_auc",
                self.train_churn_auc.compute(),
                prog_bar=False,
                logger=True,
            )
            self.train_churn_auc.reset()

        if "churn_cart" in self.tasks:
            self.log(
                "train_churn_cart_auc",
                self.train_churn_cart_auc.compute(),
                prog_bar=False,
                logger=True,
            )
            self.train_churn_cart_auc.reset()

        if "propensity_sku" in self.tasks:
            self.log(
                "train_propensity_sku_auc",
                self.train_propensity_sku_auc.compute(),
                prog_bar=False,
                logger=True,
            )
            self.train_propensity_sku_auc.reset()

        if "propensity_category" in self.tasks:
            self.log(
                "train_propensity_category_auc",
                self.train_propensity_category_auc.compute(),
                prog_bar=False,
                logger=True,
            )
            self.train_propensity_category_auc.reset()

        if "cart_sku" in self.tasks:
            self.log(
                "train_cart_sku_auc",
                self.train_cart_sku_auc.compute(),
                prog_bar=False,
                logger=True,
            )
            self.train_cart_sku_auc.reset()

        if "cart_category" in self.tasks:
            self.log(
                "train_cart_category_auc",
                self.train_cart_category_auc.compute(),
                prog_bar=False,
                logger=True,
            )
            self.train_cart_category_auc.reset()

        self._set_state_radam_schedule_free(is_train=False)

    def on_validation_epoch_end(self):
        self.log("valid_loss", self.valid_loss.compute(), prog_bar=True, logger=True)
        self.valid_loss.reset()

        if "churn" in self.tasks:
            self.log(
                "valid_churn_auc",
                self.valid_churn_auc.compute(),
                prog_bar=True,
                logger=True,
            )
            self.valid_churn_auc.reset()

        if "churn_cart" in self.tasks:
            self.log(
                "valid_churn_cart_auc",
                self.valid_churn_cart_auc.compute(),
                prog_bar=True,
                logger=True,
            )
            self.valid_churn_cart_auc.reset()

        if "propensity_sku" in self.tasks:
            self.log(
                "valid_propensity_sku_auc",
                self.valid_propensity_sku_auc.compute(),
                prog_bar=True,
                logger=True,
            )
            self.valid_propensity_sku_auc.reset()

        if "propensity_category" in self.tasks:
            self.log(
                "valid_propensity_category_auc",
                self.valid_propensity_category_auc.compute(),
                prog_bar=True,
                logger=True,
            )
            self.valid_propensity_category_auc.reset()

        if "cart_sku" in self.tasks:
            self.log(
                "valid_cart_sku_auc",
                self.valid_cart_sku_auc.compute(),
                prog_bar=False,
                logger=True,
            )
            self.valid_cart_sku_auc.reset()

        if "cart_category" in self.tasks:
            self.log(
                "valid_cart_category_auc",
                self.valid_cart_category_auc.compute(),
                prog_bar=False,
                logger=True,
            )
            self.valid_cart_category_auc.reset()
