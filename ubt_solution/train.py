import argparse
import logging
import os
from pathlib import Path
import numpy as np
import random
import torch
from tqdm import tqdm
from config import Config
from trainer import UBTTrainer
from data_processor import create_data_loaders
from model import UniversalBehavioralTransformer
from dataclasses import asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default='../dataset/ubc_data_tiny/',
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

def save_embeddings(embeddings_dir: Path, client_ids: np.ndarray, embeddings: np.ndarray):
    """保存嵌入向量和客户端ID"""
    logger.info("Saving embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    np.save(embeddings_dir / "client_ids.npy", client_ids)

def main():
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
        accelerator="cpu",
        device=device,
        num_workers=args.num_workers,
        output_dir=str(embeddings_dir),
        devices=[int(args.devices)] if args.accelerator == "cuda" else [],
        task_weights=task_weights,
        save_dir=args.save_dir,
    )

    logger.info(f"Config parameters: {asdict(config)}")
    train_loader = create_data_loaders(data_dir, config, mode='train', test_mode=args.test_mode)
    val_loader = create_data_loaders(data_dir, config, mode='valid', test_mode=args.test_mode)

    model = UniversalBehavioralTransformer(config)

    trainer = UBTTrainer(
        model=model,
        config=config
    )

    logger.info("开始训练模型...")
    trainer.train(train_loader=train_loader, val_loader=val_loader)

if __name__ == "__main__":
    main()
