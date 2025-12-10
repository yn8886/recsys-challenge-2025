import argparse
import math
import os
from datetime import datetime
import logging
from pathlib import Path
from enum import Enum
import random
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from aiohttp.log import client_logger
from tensorflow import timestamp
from torch.utils.data._utils.collate import default_collate
from config import Config
# import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
# from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import AUROC, MultilabelAccuracy
from trainer import UBTTrainer
from model import UniversalBehavioralTransformer

NUM_CANDIDATES_SKU = 100
NUM_CANDIDATES_CAT = 100
NUM_CANDIDATES_PRICE = 100
exp_name = os.path.splitext(os.path.basename(__file__))[0]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def login_wandb():
#     load_dotenv()
#     WANDB_API_KEY = os.getenv("WANDB_API_KEY")
#     wandb.login(key=WANDB_API_KEY)


def custom_collate(batch):
    out = {}
    sample = batch[0]
    for k in sample:
        if k in ('pos_sku_ids', 'pos_cat_ids'):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = default_collate([b[k] for b in batch])
    return out

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

        print("Loading timestamp.npy")
        self.timestamps = np.load(
            os.path.join(self.dataset_dir, "norm_timestamp.npy"), allow_pickle=True
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
            os.path.join(self.dataset_dir, "pos_cat_ids.npy"), allow_pickle=True
        )
        self.neg_cat_ids = np.load(
            os.path.join(self.dataset_dir, "neg_cat_ids.npy"), allow_pickle=True
        )

    def __len__(self):
        return len(self.client_ids)

    def _pad_sequence(self, seq):
        seq = seq.tolist()
        sliced_seq = seq[-self.max_len :]
        padding_length = self.max_len - len(sliced_seq)
        padded_seq = sliced_seq + [0] * padding_length
        padded_seq = np.array(padded_seq)
        return padded_seq, padding_length

    def _pad_word_sequence(self, seq):
        seq = seq.tolist()
        sliced_seq = seq[-self.max_len :]
        padding_length = self.max_len - len(sliced_seq)
        padded_seq = sliced_seq + [[0] * 16] * padding_length
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
        client_id = self.client_ids[idx]

        # sequence features
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
        pos_sku_ids = self.pos_sku_ids[idx]
        neg_sku_ids = torch.tensor(self.neg_sku_ids[idx], dtype=torch.long)
        pos_cat_ids = self.pos_cat_ids[idx]
        neg_cat_ids = torch.tensor(self.neg_cat_ids[idx], dtype=torch.long)
        is_churn = torch.tensor(self.is_churn[idx], dtype=torch.float)


        # padding and masking sequence
        event_type, _ = self._pad_sequence(event_type)
        event_type = torch.tensor(event_type, dtype=torch.long)
        sku_id, _ = self._pad_sequence(sku_id)
        sku_id = torch.tensor(sku_id, dtype=torch.long)
        url_id, _ = self._pad_sequence(url_id)
        url_id = torch.tensor(url_id, dtype=torch.long)
        category_id, _ = self._pad_sequence(category_id)
        category_id = torch.tensor(category_id, dtype=torch.long)
        price_id, _ = self._pad_sequence(price_id)
        price_id = torch.tensor(price_id, dtype=torch.long)
        word_id, _ = self._pad_word_sequence(word_id)
        word_id = torch.tensor(word_id, dtype=torch.long)
        timestamp, _ = self._pad_sequence(timestamp)
        timestamp = torch.tensor(timestamp, dtype=torch.float)

        # statistical features
        statistical_features = torch.tensor(statistical_features, dtype=torch.float)

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
            "pos_sku_ids": pos_sku_ids,
            "neg_sku_ids": neg_sku_ids,
            "pos_cat_ids": pos_cat_ids,
            "neg_cat_ids": neg_cat_ids,
        }
        return return_dict


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default='../dataset/mtl/',
        help="Directory where target and input data are stored",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default='../submit/ubt_emb/',
        help="Directory where to store generated embeddings",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default='./save/train_features_epoch3_category/',
        help="Directory where to store generated embeddings",
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
        "--test-mode",
        action="store_true",
        help="Whether to use test mode (process only the first 100 users)",
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
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--task-weights",
        type=str,
        default="churn:1.0,category_propensity:0.0,product_propensity:0.0",
        help="Task weights in format 'churn:1.0,category_propensity:0.5,product_propensity:0.5'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser



def main():
    # login_wandb()
    parser = get_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Using seed: {args.seed}")
    data_dir = Path(args.data_dir)
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    task_weights = None
    if args.task_weights:
        task_weights = {}
        for pair in args.task_weights.split(','):
            key, value = pair.split(':')
            task_weights[key] = float(value)
        logger.info(f"Using custom task weights: {task_weights}")
    device = f"cuda:{args.devices}" if args.accelerator == "cuda" else "cpu"
    config = Config(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        accelerator=args.accelerator,
        device=device,
        num_workers=args.num_workers,
        output_dir=str(embeddings_dir),
        devices=[int(args.devices)] if args.accelerator == "cuda" else [],
        task_weights=task_weights,
        save_dir=args.save_dir,
    )

    train_dataset_dir = os.path.join(data_dir, "train")
    valid_dataset_dir = os.path.join(data_dir, "valid")

    train_dataset = RecsysDatasetV12(dataset_dir=train_dataset_dir, max_len=config.max_seq_length)
    valid_dataset = RecsysDatasetV12(dataset_dir=valid_dataset_dir, max_len=config.max_seq_length)

    print(f"train_size : {len(train_dataset)}")
    print(f"valid_size : {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=custom_collate
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=custom_collate
    )

    model = UniversalBehavioralTransformer(config)

    # wandb_logger = WandbLogger(project="Recsys", name=exp_name, log_model=True)

    trainer = UBTTrainer(
        model=model,
        config=config,
    )
    trainer.train(train_loader=train_loader, val_loader=valid_loader)


if __name__ == "__main__":
    main()
