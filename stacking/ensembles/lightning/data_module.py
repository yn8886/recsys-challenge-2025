import pytorch_lightning as L
from torch.utils.data import DataLoader

from ensembles.torch.dataset import StackingDataset


class StackingDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: StackingDataset,
        valid_dataset: StackingDataset,
        pred_dataset: StackingDataset,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.pred_dataset = pred_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
        )
