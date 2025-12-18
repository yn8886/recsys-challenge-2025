import argparse
import math
import os
import json
import logging
from pathlib import Path
from enum import Enum
import random
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from config import Config
from torch.utils.data import DataLoader, Dataset
from trainer import UBTTrainer
from model import UniversalBehavioralTransformer

NUM_CANDIDATES_SKU = 100
NUM_CANDIDATES_CAT = 100
NUM_CANDIDATES_PRICE = 100
exp_name = os.path.splitext(os.path.basename(__file__))[0]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    PAD_IDX = 0
    MASK = 1
    PRODUCT_BUY = 2
    ADD_TO_CART = 3
    REMOVE_FROM_CART = 4
    PAGE_VISIT = 5
    SEARCH_QUERY = 6

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
    def __init__(self, data_dir, mode, test_mode=False, max_len=128, mask_rate=0.2, num_negative_samples=120):
        self.max_len = max_len
        self.mask_rate = mask_rate
        self.mode = mode
        self.test_mode = test_mode
        self.num_negative_samples = num_negative_samples
        self.indexes_template = np.arange(self.max_len)
        self.item_mask_template = 1
        self.word_mask_template = np.array([1] * 16)

        self.dataset_dir = f'./dataset/{mode}' if not test_mode else os.path.join(data_dir, f"mtl/{mode}")
        with open(os.path.join(f"{data_dir}/mtl/mappings", "sku2id_mapping.json"), 'r') as f:
            loaded_sku_map = json.load(f)
            self.available_skus_list = [v for _, v in loaded_sku_map.items()]
            sku2id_mapping = {int(k): v for k, v in loaded_sku_map.items()}

        with open(os.path.join(f"{data_dir}/mtl/mappings", "sku2catid_mapping.json"), 'r') as f:
            loaded_cat_map = json.load(f)
            self.available_categories_list = [v for _, v in loaded_cat_map.items()]

        target_path = f"{data_dir}/ubc_data_tiny/target"
        self.propensity_sku = np.load(f"{target_path}/propensity_sku.npy")
        self.propensity_sku = np.array([sku2id_mapping[sku] for sku in self.propensity_sku])
        self.propensity_category = np.load(f"{target_path}/propensity_category.npy")
        self.propensity_category = self.propensity_category + 2

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

        if not test_mode:
            print("Loading labels")
            self.pos_sku_ids = np.load(
                os.path.join(self.dataset_dir, "pos_sku_ids.npy"), allow_pickle=True
            )
            self.pos_cat_ids = np.load(
                os.path.join(self.dataset_dir, "pos_cat_ids.npy"), allow_pickle=True
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

        if not self.test_mode:
            pos_sku_ids = self.pos_sku_ids[idx]
            pos_cat_ids = self.pos_cat_ids[idx]

            if not pos_sku_ids.any() or not pos_cat_ids.any():
                found_fallback = False
                priority_events = [EventType.PRODUCT_BUY.value, EventType.ADD_TO_CART.value, EventType.REMOVE_FROM_CART.value]
                valid_indices = sorted([i for i, v in enumerate(sku_id) if v != -1], reverse=True)

                for target_evt_type in priority_events:
                    if found_fallback: break

                    for i in valid_indices:
                        evt_type = event_type[i]
                        sku = sku_id[i]
                        cat = category_id[i]

                        if evt_type == target_evt_type and sku != -1:
                            pos_sku_ids = [sku]
                            pos_cat_ids = [cat]
                            found_fallback = True
                            break

            pos_cat_set = set(pos_cat_ids)
            neg_cat_ids = []
            for cat in self.propensity_category.tolist():
                if cat not in category_id and cat not in pos_cat_set:
                    neg_cat_ids.append(cat)
                    if len(neg_cat_ids) >= self.num_negative_samples:
                        break
            while len(neg_cat_ids) < self.num_negative_samples and self.available_categories_list:
                cand = random.choice(self.available_categories_list)
                if cand not in category_id and cand not in pos_cat_set and cand not in neg_cat_ids:
                    neg_cat_ids.append(cand)
            neg_cat_ids = torch.tensor(neg_cat_ids, dtype=torch.long)


            pos_sku_set = set(pos_sku_ids)
            neg_sku_ids = []
            for sku in self.propensity_sku.tolist():
                if sku not in sku_id and sku not in pos_sku_set:
                    neg_sku_ids.append(sku)
                    if len(neg_sku_ids) >= self.num_negative_samples:
                        break
                while len(neg_sku_ids) < self.num_negative_samples and self.available_skus_list:
                    cand = random.choice(self.available_skus_list)
                    if cand not in sku_id and cand not in pos_sku_set and cand not in neg_sku_ids:
                        neg_sku_ids.append(cand)
            neg_sku_ids = torch.tensor(neg_sku_ids, dtype=torch.long)

        # statistical features
        statistical_features = self.statistical_features[idx]

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
        }
        if not self.test_mode:
            return_dict["pos_sku_ids"] = pos_sku_ids
            return_dict["neg_sku_ids"] = neg_sku_ids
            return_dict["pos_cat_ids"] = pos_cat_ids
            return_dict["neg_cat_ids"] = neg_cat_ids

        return return_dict


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default='../dataset/',
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
        default="category_propensity:1.0,product_propensity:0.0",
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

    train_dataset = RecsysDatasetV12(data_dir=data_dir, mode='train', max_len=config.max_seq_length)
    valid_dataset = RecsysDatasetV12(data_dir=data_dir, mode='valid', max_len=config.max_seq_length)

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

    trainer = UBTTrainer(
        model=model,
        config=config,
    )
    trainer.train(train_loader=train_loader, val_loader=valid_loader)


if __name__ == "__main__":
    main()
