import numpy as np
import polars as pl
import torch
from loguru import logger
from torch.utils.data import Dataset
from typing import Union, Optional

class StackingDataset(Dataset):
    def __init__(
        self,
        df_labels: Union[pl.DataFrame, None],
        arr_relevant_clients: np.ndarray,
        arr_embeddings: np.ndarray,
        arr_propensity_sku: np.ndarray,
        arr_propensity_category: np.ndarray,
        tasks: list[str],
    ):
        if df_labels is not None:
            self.df_labels = df_labels.sort("client_id")
        else:
            self.df_labels = None

        sorted_indices = np.argsort(arr_relevant_clients)
        self.arr_relevant_clients = arr_relevant_clients[sorted_indices]
        self.tns_embeddings = torch.from_numpy(arr_embeddings[sorted_indices]).to(
            torch.float32
        )

        self.arr_propensity_sku = np.sort(arr_propensity_sku)
        self.arr_propensity_category = np.sort(arr_propensity_category)

        self.tasks = tasks

        if df_labels is not None:
            # assert np.array_equal(
            #     self.df_labels["client_id"].to_numpy(), self.arr_relevant_clients
            # )
            if "churn" in tasks:
                self.churn_label = self.df_labels["churn"].to_torch().to(torch.long)
                logger.info(
                    f"Churn label, {self.churn_label.sum()} / {len(self.churn_label)}"
                )
            if "churn_cart" in tasks:
                self.churn_cart_label = self.df_labels["churn_cart"].to_torch().to(torch.long)
                logger.info(
                    f"Churn cart label, {self.churn_cart_label.sum()} / {len(self.churn_cart_label)}"
                )
            if "propensity_sku" in tasks:
                self.tns_propensity_sku_labels = self.init_propensity_sku_labels()
                logger.info(
                    f"Propensity sku labels, {self.tns_propensity_sku_labels.sum()} / {sum(self.tns_propensity_sku_labels.shape)}"
                )
            if "propensity_category" in tasks:
                self.tns_propensity_category_labels = (
                    self.init_propensity_category_labels()
                )
                logger.info(
                    f"Propensity category labels, {self.tns_propensity_category_labels.sum()} / {sum(self.tns_propensity_category_labels.shape)}"
                )
            if "cart_sku" in tasks:
                self.tns_cart_sku_labels = self.init_cart_sku_labels()
                logger.info(
                    f"Cart sku labels, {self.tns_cart_sku_labels.sum()} / {sum(self.tns_cart_sku_labels.shape)}"
                )
            if "cart_category" in tasks:
                self.tns_cart_category_labels = self.init_cart_category_labels()
                logger.info(
                    f"Cart category labels, {self.tns_cart_category_labels.sum()} / {sum(self.tns_cart_category_labels.shape)}"
                )

    def init_propensity_sku_labels(self):
        tns_propensity_sku_labels = torch.zeros(
            len(self.arr_relevant_clients), len(self.arr_propensity_sku)
        ).to(torch.long)
        for i, sku in enumerate(self.arr_propensity_sku):
            tns_propensity_sku_labels[:, i] = self.df_labels[
                f"propensity_sku_{sku}"
            ].to_torch()
        return tns_propensity_sku_labels.to(torch.long)

    def init_propensity_category_labels(self):
        tns_propensity_category_labels = torch.zeros(
            len(self.arr_relevant_clients), len(self.arr_propensity_category)
        ).to(torch.long)
        for i, category in enumerate(self.arr_propensity_category):
            tns_propensity_category_labels[:, i] = self.df_labels[
                f"propensity_category_{category}"
            ].to_torch()
        return tns_propensity_category_labels.to(torch.long)

    def init_cart_sku_labels(self):
        tns_cart_sku_labels = torch.zeros(
            len(self.arr_relevant_clients), len(self.arr_propensity_sku)
        ).to(torch.long)
        for i, sku in enumerate(self.arr_propensity_sku):
            tns_cart_sku_labels[:, i] = self.df_labels[f"cart_sku_{sku}"].to_torch()
        return tns_cart_sku_labels.to(torch.long)

    def init_cart_category_labels(self):
        tns_cart_category_labels = torch.zeros(
            len(self.arr_relevant_clients), len(self.arr_propensity_category)
        ).to(torch.long)
        for i, category in enumerate(self.arr_propensity_category):
            tns_cart_category_labels[:, i] = self.df_labels[
                f"cart_category_{category}"
            ].to_torch()
        return tns_cart_category_labels.to(torch.long)

    def __len__(self):
        return len(self.arr_relevant_clients)

    def __getitem__(self, idx):
        if self.df_labels is not None:
            ret = {"embedding": self.tns_embeddings[idx]}
            if "churn" in self.tasks:
                ret["churn_label"] = self.churn_label[idx]
            if "churn_cart" in self.tasks:
                ret["churn_cart_label"] = self.churn_cart_label[idx]
            if "propensity_sku" in self.tasks:
                ret["propensity_sku_labels"] = self.tns_propensity_sku_labels[idx]
            if "propensity_category" in self.tasks:
                ret["propensity_category_labels"] = self.tns_propensity_category_labels[
                    idx
                ]
            if "cart_sku" in self.tasks:
                ret["cart_sku_labels"] = self.tns_cart_sku_labels[idx]
            if "cart_category" in self.tasks:
                ret["cart_category_labels"] = self.tns_cart_category_labels[idx]
            return ret
        else:
            return {"embedding": self.tns_embeddings[idx]}
