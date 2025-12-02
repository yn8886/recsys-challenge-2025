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


def safe_statistics(arr, name="array"):
    """Compute statistics safely to avoid overflow."""
    try:
        # Use float64 for better numerical stability
        arr_float64 = arr.astype(np.float64)

        # Compute mean and std in chunks to avoid overflow
        chunk_size = 10000
        means = []
        stds = []

        for i in range(0, arr_float64.shape[0], chunk_size):
            chunk = arr_float64[i : i + chunk_size]
            means.append(np.mean(chunk, axis=0))
            stds.append(np.std(chunk, axis=0))

        overall_mean = np.mean(means)
        overall_std = np.mean(stds)

        return overall_mean, overall_std

    except Exception as e:
        logger.warning(f"Error computing statistics for {name}: {e}")
        return 0.0, 1.0


@hydra.main(version_base=None, config_path="conf", config_name="stacking")
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    client_order = np.load(cfg.client_order) if cfg.client_order else None

    df_product_properties = pl.read_parquet(cfg.path_product_properties)
    df_product_buy = pl.read_parquet(cfg.path_product_buy)
    df_add_to_cart = pl.read_parquet(cfg.path_add_to_cart)
    df_remove_from_cart = pl.read_parquet(cfg.path_remove_from_cart)
    df_page_visit = pl.read_parquet(cfg.path_page_visit)
    df_search_query = pl.read_parquet(cfg.path_search_query)

    arr_relevant_clients = np.load(cfg.path_relevant_clients)
    arr_propensity_sku = np.load(cfg.path_propensity_sku)
    arr_propensity_category = np.load(cfg.path_propensity_category)

    logger.info("Preprocessing labels")
    train_preprocess_labels, valid_preprocess_labels = preprocess(
        df_product_properties=df_product_properties,
        df_product_buy=df_product_buy,
        df_add_to_cart=df_add_to_cart,
        df_remove_from_cart=df_remove_from_cart,
        df_page_visit=df_page_visit,
        df_search_query=df_search_query,
        train_start_datetime=datetime.strptime(
            cfg.label_split.train.input_start_datetime, "%Y-%m-%d %H:%M:%S"
        ),
        train_end_datetime=datetime.strptime(
            cfg.label_split.train.input_end_datetime, "%Y-%m-%d %H:%M:%S"
        ),
        valid_start_datetime=datetime.strptime(
            cfg.label_split.valid.input_start_datetime, "%Y-%m-%d %H:%M:%S"
        ),
        valid_end_datetime=datetime.strptime(
            cfg.label_split.valid.input_end_datetime, "%Y-%m-%d %H:%M:%S"
        ),
        arr_relevant_clients=arr_relevant_clients,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
    )

    logger.info("Loading embeddings for local")
    list_embeddings = []
    for file in cfg.mode["local"]:
        logger.info(f"mode: local, file: {file}")
        file_path = Path(f"../data/embeddings/local/{file}")
        arr_clients, arr_embeddings = read_embedding_zip(file_path)
        embedding = Embedding(file, arr_clients, arr_embeddings)
        if cfg.normalize or file.startswith("313038"):  # rule base count feature needs normalization
            logger.info(f"Normalizing {file}")
            embedding.normalize()
        list_embeddings.append(embedding)
    embeddings = Embeddings(list_embeddings)
    if cfg.do_svd:
        logger.info(f"SVD offline, n_components: {cfg.svd_n_components}")
        embeddings.svd(cfg.svd_n_components, cfg.svd_device)
    embeddings.info()

    arr_clients_local = embeddings.arr_clients
    arr_embeddings_local = embeddings.arr_embeddings

    # Apply statistical alignment if enabled
    if cfg.do_align_embed:
        logger.info(
            "Applying statistical alignment between local and submit embeddings"
        )

        # Log statistics before alignment
        # Use safer computation methods to avoid overflow
        local_mean, local_std = safe_statistics(arr_embeddings_local, "local")
        # submit_mean, submit_std = safe_statistics(arr_embeddings_submit, "submit")

        # Add detailed diagnostics
        logger.info(f"Local embeddings shape: {arr_embeddings_local.shape}")
        logger.info(
            f"Local embeddings min: {np.min(arr_embeddings_local):.6f}, max: {np.max(arr_embeddings_local):.6f}"
        )
        logger.info(f"Local embeddings - mean: {local_mean:.6f}, std: {local_std:.6f}")
        # logger.info(f"Submit embeddings shape: {arr_embeddings_submit.shape}")
        # logger.info(
        #     f"Submit embeddings min: {np.min(arr_embeddings_submit):.6f}, max: {np.max(arr_embeddings_submit):.6f}"
        # )
        # logger.info(
        #     f"Submit embeddings - mean: {submit_mean:.6f}, std: {submit_std:.6f}"
        # )

        # Check for extreme values
        # local_extreme = np.sum(
        #     np.abs(arr_embeddings_local.astype(np.float64)) > 1e6, dtype=np.int64
        # )
        # submit_extreme = np.sum(
        #     np.abs(arr_embeddings_submit.astype(np.float64)) > 1e6, dtype=np.int64
        # )
        # logger.info(f"Local embeddings with |value| > 1e6: {local_extreme}")
        # logger.info(f"Submit embeddings with |value| > 1e6: {submit_extreme}")
        #
        try:
        #     # Check for numerical issues before alignment
        #     if np.any(np.isnan(arr_embeddings_local)) or np.any(
        #         np.isinf(arr_embeddings_local)
        #     ):
        #         logger.warning(
        #             "Local embeddings contain NaN or Inf values. Skipping alignment."
        #         )
        #         return
        #
        #     if np.any(np.isnan(arr_embeddings_submit)) or np.any(
        #         np.isinf(arr_embeddings_submit)
        #     ):
        #         logger.warning(
        #             "Submit embeddings contain NaN or Inf values. Skipping alignment."
        #         )
        #         return
        #
        #     # Preprocess embeddings to prevent overflow
        #     # Clip extreme values that could cause numerical issues
            arr_embeddings_local_clipped = np.clip(
                arr_embeddings_local.astype(np.float64), -1e6, 1e6
            )
        #     arr_embeddings_submit_clipped = np.clip(
        #         arr_embeddings_submit.astype(np.float64), -1e6, 1e6
        #     )
        #
        #     # Check if clipping was necessary
        #     if not np.array_equal(arr_embeddings_local, arr_embeddings_local_clipped):
        #         clipped_count = np.sum(
        #             arr_embeddings_local != arr_embeddings_local_clipped
        #         )
        #         logger.warning(f"Clipped {clipped_count} values in local embeddings")
        #
        #     if not np.array_equal(arr_embeddings_submit, arr_embeddings_submit_clipped):
        #         clipped_count = np.sum(
        #             arr_embeddings_submit != arr_embeddings_submit_clipped
        #         )
        #         logger.warning(f"Clipped {clipped_count} values in submit embeddings")
        #
        #     # Create base embedding from local data
            base_embedding = Embedding(
                "local_base", arr_clients_local, arr_embeddings_local_clipped
            )
        #
        #     # Create target embedding from submit data
        #     target_embedding = Embedding(
        #         "submit_target", arr_clients_submit, arr_embeddings_submit_clipped
        #     )

            # Perform statistical alignment
            alignment = StatisticalAlignment(
                embedding_base=base_embedding,
                # embedding_to_align=target_embedding,
                n_components=cfg.align_n_components,
            )

            aligned_embedding = alignment.align()
            arr_clients_submit = aligned_embedding.arr_clients
            arr_embeddings_submit = aligned_embedding.arr_embeddings

            # Log statistics after alignment
            aligned_mean, aligned_std = safe_statistics(
                arr_embeddings_submit, "aligned"
            )

            logger.info(
                f"Aligned submit embeddings - mean: {aligned_mean:.6f}, std: {aligned_std:.6f}"
            )
            logger.info(
                f"Alignment completed. Submit embeddings shape: {arr_embeddings_submit.shape}"
            )

        except Exception as e:
            logger.warning(
                f"Statistical alignment failed: {str(e)}. Using original submit embeddings."
            )
            logger.warning(
                "Continuing without alignment - this may affect model performance."
            )

    train_dataset = StackingDataset(
        df_labels=train_preprocess_labels,
        arr_relevant_clients=arr_clients_local,
        arr_embeddings=arr_embeddings_local,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
        tasks=cfg.tasks,
    )
    valid_dataset = StackingDataset(
        df_labels=valid_preprocess_labels,
        arr_relevant_clients=arr_clients_local,
        arr_embeddings=arr_embeddings_local,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
        tasks=cfg.tasks,
    )
    pred_dataset = StackingDataset(
        df_labels=None,
        arr_relevant_clients=arr_clients_local,
        arr_embeddings=arr_embeddings_local,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
        tasks=cfg.tasks,
    )

    model = StackingModel(
        tasks=cfg.tasks,
        input_dim=arr_embeddings_local.shape[1],
        embedding_dim=cfg.embedding_dim,
        hidden_size_thin=cfg.hidden_size_thin,
        hidden_size_wide=cfg.hidden_size_wide,
        num_propensity_sku=len(arr_propensity_sku),
        num_propensity_category=len(arr_propensity_category),
    )

    data_module = StackingDataModule(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        pred_dataset=pred_dataset,
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
            RichModelSummary(max_depth=2),
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
    )

    trainer.fit(model_module, data_module)

    logger.info("Predicting embeddings for local")

    embeddings = trainer.predict(model_module, data_module)
    embeddings = np.concatenate(embeddings, axis=0)

    logger.info(f"Saving embeddings for local, shape: {embeddings.shape}")

    embedding = Embedding("local", pred_dataset.arr_relevant_clients, embeddings)
    if isinstance(client_order, np.ndarray):
        logger.info("Aligning local to client order")
        embedding.align_to(client_order)

    save_embeddings(
        embedding.arr_clients,
        embedding.arr_embeddings,
        shuffle=False,
        output_dir=output_dir / "local",
    )

    logger.info("Predicting embeddings for submit")
    pred_dataset = StackingDataset(
        df_labels=None,
        arr_relevant_clients=embedding.arr_clients,
        arr_embeddings=embedding.arr_embeddings,
        arr_propensity_sku=arr_propensity_sku,
        arr_propensity_category=arr_propensity_category,
        tasks=cfg.tasks,
    )
    pred_data_module = StackingDataModule(
        train_dataset=None,
        valid_dataset=None,
        pred_dataset=pred_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    embeddings = trainer.predict(model_module, pred_data_module)
    embeddings = np.concatenate(embeddings, axis=0)

    logger.info(f"Saving embeddings for submit, shape: {embeddings.shape}")

    embedding = Embedding("submit", pred_dataset.arr_relevant_clients, embeddings)
    if isinstance(client_order, np.ndarray):
        logger.info("Aligning submit to client order")
        embedding.align_to(client_order)

    save_embeddings(
        embedding.arr_clients,
        embedding.arr_embeddings,
        shuffle=False,
        output_dir=output_dir / "submit",
    )


if __name__ == "__main__":
    main()
