import argparse
import logging
import os
from pathlib import Path
import numpy as np
import random
import torch
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from config import Config
from trainer import UBTTrainer
from torch.utils.data import DataLoader
from model import UniversalBehavioralTransformer
from dataclasses import asdict
from train import RecsysDatasetV12
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_collate(batch):
    out = {}
    sample = batch[0]
    for k in sample:
        if k in ('pos_sku_ids', 'pos_cat_ids'):
            out[k] = [b[k] for b in batch]
        else:
            out[k] = default_collate([b[k] for b in batch])
    return out


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate user embeddings using trained UBT model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default='../dataset/',
        help="Directory where input data are stored",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default='../submit/ubt_emb/',
        help="Directory to save generated embeddings",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default='./save/train_features_epoch3_category/',
        help="Path to the saved model checkpoint (e.g., best_model.pt)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='train',
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="0",
        help="GPU device ID (ignored if accelerator is cpu)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Process only first 100 users for debugging",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser


def save_embeddings(embeddings_dir: Path, mode, client_ids: np.ndarray, embeddings: np.ndarray):
    logger.info("Saving embeddings...")
    embeddings_dir = os.path.join(embeddings_dir, mode)
    os.makedirs(embeddings_dir, exist_ok=True)
    np.save(os.path.join(embeddings_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(embeddings_dir, "client_ids.npy"), client_ids)


def main():
    parser = get_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.accelerator == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Using device: {args.accelerator}")
    device = f"cuda:{args.devices}" if args.accelerator == "cuda" else "cpu"
    data_dir = Path(args.data_dir)
    config = Config(
        batch_size=args.batch_size,
        num_epochs=1,
        learning_rate=0.0,
        device=device,
        accelerator=args.accelerator,
        devices=[int(args.devices)] if args.accelerator == "cuda" else [],
        num_workers=args.num_workers,
        output_dir=str(Path(args.embeddings_dir)),
    )

    logger.info(f"Config: {asdict(config)}")

    if args.mode == "train":
        train_dataset = RecsysDatasetV12(data_dir=data_dir, mode='train', test_mode=True, max_len=config.max_seq_length)
        infer_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=custom_collate
        )
    else:
        valid_dataset = RecsysDatasetV12(data_dir=data_dir, mode='valid', test_mode=True, max_len=config.max_seq_length)
        infer_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=custom_collate
        )

    model = UniversalBehavioralTransformer(config)

    logger.info(f"Loading model weights from {args.checkpoint_path}")
    state_dict = torch.load(args.checkpoint_path + 'best_model.pt', map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    trainer = UBTTrainer(model=model, config=config)

    logger.info("Generating user embeddings...")
    client_ids, embeddings = trainer.generate_embeddings(infer_loader)

    if client_ids.size == 0:
        logger.error("No valid embeddings generated!")
        return

    save_embeddings(args.embeddings_dir, args.mode, client_ids, embeddings)
    logger.info("Embedding generation completed successfully.")


if __name__ == "__main__":
    main()