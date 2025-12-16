from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import polars as pl
import pytorch_lightning as L
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.callbacks import RichModelSummary, TQDMProgressBar
# from pytorch_lightning.loggers import WandbLogger

from ensembles.embeddings import (
    Embedding,
    Embeddings,
    read_embedding_zip,
    save_embeddings,
)
from ensembles.align_embed import StatisticalAlignment
from ensembles.lightning.data_module import StackingDataModule
from ensembles.lightning.model_module import StackingModelModule
from ensembles.preprocess import preprocess
from ensembles.torch.dataset import StackingDataset
from ensembles.torch.model import StackingModel


@hydra.main(version_base=None, config_path="conf", config_name="stacking")
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    df_train_target = pl.read_parquet(cfg.label_split.train.target_path)
    df_valid_target = pl.read_parquet(cfg.label_split.valid.target_path)
    arr_relevant_clients = np.load(cfg.path_relevant_clients)
    arr_propensity_sku = np.load(cfg.path_propensity_sku)
    arr_propensity_category = np.load(cfg.path_propensity_category)

    logger.info("Preprocessing labels")
    train_preprocess_labels, valid_preprocess_labels = preprocess(
        arr_relevant_clients=arr_relevant_clients,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
        df_train_target=df_train_target,
        df_valid_target=df_valid_target,
    )

    logger.info("Loading embeddings for local")
    for mode in ['train', 'valid']:
        list_embeddings = []
        for file in cfg.mode[mode]:
            logger.info(f"mode: train, file: {file}")
            file_path = Path(f"../submit/valid/{mode}/{file}")
            arr_clients, arr_embeddings = read_embedding_zip(file_path)
            embedding = Embedding(file, arr_clients, arr_embeddings)
            list_embeddings.append(embedding)
        embeddings = Embeddings(list_embeddings)
        if cfg.do_svd:
            logger.info(f"SVD offline, n_components: {cfg.svd_n_components}")
            embeddings.svd(cfg.svd_n_components, cfg.svd_device)
        embeddings.info()
        if mode == 'train':
            arr_clients_train = embeddings.arr_clients
            arr_embeddings_train = embeddings.arr_embeddings
        elif mode == 'valid':
            arr_clients_valid = embeddings.arr_clients
            arr_embeddings_valid = embeddings.arr_embeddings


    train_dataset = StackingDataset(
        df_labels=train_preprocess_labels,
        arr_relevant_clients=arr_clients_train,
        arr_embeddings=arr_embeddings_train,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
        tasks=cfg.tasks,
    )
    valid_dataset = StackingDataset(
        df_labels=valid_preprocess_labels,
        arr_relevant_clients=arr_clients_valid,
        arr_embeddings=arr_embeddings_valid,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
        tasks=cfg.tasks,
    )
    model = StackingModel(
        tasks=cfg.tasks,
        input_dim=arr_embeddings_train.shape[1],
        embedding_dim=cfg.embedding_dim,
        hidden_size_thin=cfg.hidden_size_thin,
        hidden_size_wide=cfg.hidden_size_wide,
        num_propensity_sku=len(arr_propensity_sku),
        num_propensity_category=len(arr_propensity_category),
    )

    data_module = StackingDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model_module = StackingModelModule(
        model=model,
        tasks=cfg.tasks,
        lr=cfg.lr,
        num_propensity_sku=len(arr_propensity_sku),
        num_propensity_category=len(arr_propensity_category),
    )

    trainer = L.Trainer(
        # logger=WandbLogger(
        #     name="stacking",
        #     save_dir=output_dir,
        #     log_model=False,
        #     tags=["stacking"],
        # ),
        callbacks=[
            RichModelSummary(max_depth=0),
            TQDMProgressBar(leave=True),
        ],
        fast_dev_run=cfg.fast_dev_run,
        max_epochs=cfg.max_epochs,
        accumulate_grad_batches=1,
        gradient_clip_val=None,
        deterministic=False,
        benchmark=True,
        inference_mode=True,
        default_root_dir=output_dir,
        accelerator=cfg.accelerator,
        num_sanity_val_steps=0,
    )

    trainer.fit(model_module, data_module)


if __name__ == "__main__":
    main()
