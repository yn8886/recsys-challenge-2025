from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.utils.data import default_collate
import torch.nn as nn
import os
from torch.utils.data import Dataset
import numpy as np
import math


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
        self.query_ids = np.load(
            os.path.join(self.dataset_dir, "query_ids.npy"), allow_pickle=True
        )

        print("Loading category_id.npy")
        self.category_ids = np.load(
            os.path.join(self.dataset_dir, "category_id.npy"), allow_pickle=True
        )

        print("Loading price.npy")
        self.price_ids = np.load(
            os.path.join(self.dataset_dir, "price_id.npy"), allow_pickle=True
        )

        self.name_ids = np.load(
            os.path.join(self.dataset_dir, "name_ids.npy"), allow_pickle=True
        )

        # self.timestamps = np.load(
        #     os.path.join(self.dataset_dir, "timestamp.npy"), allow_pickle=True
        # )

        print("Loading static_features.npy")
        self.group_lifecycle = np.load(
            os.path.join(self.dataset_dir, "group_lifecycle.npy"), allow_pickle=True
        )
        self.group_recency = np.load(
            os.path.join(self.dataset_dir, "group_recency.npy"), allow_pickle=True
        )
        self.group_purchase = np.load(
            os.path.join(self.dataset_dir, "group_purchase.npy"), allow_pickle=True
        )
        self.group_cart_intent = np.load(
            os.path.join(self.dataset_dir, "group_cart_intent.npy"), allow_pickle=True
        )
        self.group_exploration = np.load(
            os.path.join(self.dataset_dir, "group_exploration.npy"), allow_pickle=True
        )
        # self.statistical_features = np.load(
        #     os.path.join(self.dataset_dir, "stats_features.npy"), allow_pickle=True
        # )
        # print(self.statistical_features.shape)
        # print(f"Min: {self.statistical_features.min()}")
        # print(f"Max: {self.statistical_features.max()}")

        self.diff_days = np.load(
            os.path.join(self.dataset_dir, "diff_days.npy"), allow_pickle=True
        )
        self.diff_weeks = np.load(
            os.path.join(self.dataset_dir, "diff_weeks.npy"), allow_pickle=True
        )


        print("Loading labels")
        self.is_churn = np.load(
            os.path.join(self.dataset_dir, "churn.npy"), allow_pickle=True
        )
        self.is_empty = np.load(
            os.path.join(self.dataset_dir, "empty.npy"), allow_pickle=True
        )
        self.target_event_types = np.load(
            os.path.join(self.dataset_dir, "target_event_type.npy"), allow_pickle=True
        )
        self.target_sku_ids = np.load(
            os.path.join(self.dataset_dir, "target_sku_id.npy"), allow_pickle=True
        )
        self.target_url_ids = np.load(
            os.path.join(self.dataset_dir, "target_url_id.npy"), allow_pickle=True
        )
        self.target_query_ids = np.load(
            os.path.join(self.dataset_dir, "target_query_ids.npy"), allow_pickle=True
        )
        self.target_name_ids = np.load(
            os.path.join(self.dataset_dir, "target_name_ids.npy"), allow_pickle=True
        )
        self.target_category_ids = np.load(
            os.path.join(self.dataset_dir, "target_category_id.npy"), allow_pickle=True
        )
        self.target_price_ids = np.load(
            os.path.join(self.dataset_dir, "target_price_id.npy"), allow_pickle=True
        )

        self.labels_buy_category = np.load(
            os.path.join(self.dataset_dir, "labels_buy_category.npy"), allow_pickle=True
        )
        self.labels_buy_sku = np.load(
            os.path.join(self.dataset_dir, "labels_buy_sku.npy"), allow_pickle=True
        )

    def __len__(self):
        return len(self.client_ids)

    def __getitem__(self, idx):
        # client_id = self.client_ids[idx]

        # sequence features
        features1: dict[str, torch.Tensor] = {}
        features1["event_id"] = torch.tensor(self.event_types[idx], dtype=torch.long)
        features1["sku"] = torch.tensor(self.sku_ids[idx], dtype=torch.long)
        features1["url"] = torch.tensor(self.url_ids[idx], dtype=torch.long)
        features1["query_id"] = torch.tensor(np.stack([np.array(w, dtype=np.int32) for w in self.query_ids[idx]]), dtype=torch.long)
        features1["category"] = torch.tensor(self.category_ids[idx], dtype=torch.long)
        features1["price"] = torch.tensor(self.price_ids[idx], dtype=torch.long)
        features1["name_id"] = torch.tensor(np.stack([np.array(w, dtype=np.int32) for w in self.name_ids[idx]]),
                                            dtype=torch.long)
        # features1["timestamp"] = torch.tensor(self.timestamps[idx], dtype=torch.float32)
        # features1["statistical_features"] = torch.tensor(self.statistical_features[idx], dtype=torch.float32)
        features1["diff_days"] = torch.tensor(self.diff_days[idx], dtype=torch.long)
        features1["diff_weeks"] = torch.tensor(self.diff_weeks[idx], dtype=torch.long)
        features1["group_lifecycle"] = torch.tensor(self.group_lifecycle[idx], dtype=torch.float)
        features1["group_recency"] = torch.tensor(self.group_recency[idx], dtype=torch.float)
        features1["group_purchase"] = torch.tensor(self.group_purchase[idx], dtype=torch.float)
        features1["group_cart_intent"] = torch.tensor(self.group_cart_intent[idx], dtype=torch.float)
        features1["group_exploration"] = torch.tensor(self.group_exploration[idx], dtype=torch.float)


        features2: dict[str, torch.Tensor] = {}
        features2["event_id"] = torch.tensor(self.target_event_types[idx], dtype=torch.long)
        features2["sku"] = torch.tensor(self.target_sku_ids[idx], dtype=torch.long)
        features2["url"] = torch.tensor(self.target_url_ids[idx], dtype=torch.long)
        features2["query_id"] = torch.tensor(np.stack([np.array(w, dtype=np.int32) for w in self.target_query_ids[idx]]), dtype=torch.long)
        features2["category"] = torch.tensor(self.target_category_ids[idx], dtype=torch.long)
        features2["price"] = torch.tensor(self.target_price_ids[idx], dtype=torch.long)
        features2["name_id"] = torch.tensor(np.stack([np.array(w, dtype=np.int32) for w in self.target_name_ids[idx]]),
                                            dtype=torch.long)

        # labels
        labels: dict[str, Any] = {}
        labels["empty"] = torch.tensor(self.is_empty[idx], dtype=torch.float32)
        labels["churn"] = torch.tensor(self.is_churn[idx], dtype=torch.float32)
        labels["buy_category"] = torch.tensor(self.labels_buy_category[idx], dtype=torch.float32)
        labels["buy_sku"] = torch.tensor(self.labels_buy_sku[idx], dtype=torch.float32)

        return (features1, features2, labels)

@dataclass
class EventDataCollator:
    key_column: str = "event_id"
    pad_index: int = 0
    pad_value: int = 0

    def __init__(
        self,
        padding: Literal["longest", "max_length"] = "longest",
        max_length: int  = None,
    ):
        self.padding = padding
        self.max_length = max_length

    def __call__(
        self,
        raw_batch: list[tuple[dict[str, torch.Tensor], dict[str, Any] ]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] ]:
        features, labels = zip(*raw_batch)

        feature_keys = features[0].keys()
        if self.padding == "longest":
            max_length = max(feature[self.key_column].size(0) for feature in features)
        elif self.padding == "max_length":
            assert self.max_length is not None
            max_length = self.max_length
        else:
            raise NotImplementedError(f"padding={self.padding} is not supported.")

        features_dict: dict[str, torch.Tensor] = {}

        for key in feature_keys:
            sequences = [feature[key] for feature in features]

            sample_tensor = sequences[0]
            if sample_tensor.ndim == 1:
                padded_sequences = self._pad_1d_sequences(sequences, max_length)
            elif sample_tensor.ndim == 2:
                padded_sequences = self._pad_2d_sequences(sequences, max_length)
            else:
                raise ValueError(f"ndim={sample_tensor.ndim} is not supported.")

            features_dict[key] = padded_sequences

            if "attention_mask" not in features_dict:
                lengths = torch.tensor([len(seq) for seq in sequences])
                features_dict["attention_mask"] = self._create_attention_mask(lengths, max_length)

        if labels[0] is None:
            return features_dict, None

        labels_dict = default_collate(labels)
        return features_dict, labels_dict

    def _pad_1d_sequences(self, sequences: list[torch.Tensor], max_length: int, target:bool=False) -> torch.Tensor:
        batch_size = len(sequences)
        padded = torch.full((batch_size, max_length), self.pad_index, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            length = min(seq.size(0), max_length)
            sliced_seq = seq[-length:] if not target else seq[:length]
            padded[i, -length:] = sliced_seq
        return padded

    def _pad_2d_sequences(self, sequences: list[torch.Tensor], max_length: int, target:bool=False) -> torch.Tensor:
        batch_size = len(sequences)
        embed_dim = sequences[0].size(1)
        if embed_dim == 1:
            print('yes')
        padded = torch.full((batch_size, max_length, embed_dim), self.pad_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            length = min(seq.size(0), max_length)
            sliced_seq = seq[-length:] if not target else seq[:length]
            padded[i, -length:, :] = sliced_seq
        return padded

    def _create_attention_mask(self, lengths: torch.Tensor, max_length: int) -> torch.Tensor:
        batch_size = lengths.size(0)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1
        return attention_mask


class EventDataCollatorContrastive(EventDataCollator):
    def _get_features_dict(self, features, target=False):
        feature_keys = features[0].keys()
        if target:
            max_length = max(feature[self.key_column].size(0) for feature in features)
        else:
            # if self.padding == "longest":
            #     max_length = max(feature[self.key_column].size(0) for feature in features)
            # elif self.padding == "max_length":
            #     assert self.max_length is not None
            #     max_length = self.max_length
            # else:
            #     raise NotImplementedError(f"padding={self.padding} is not supported.")
            max_length = self.max_length

        features_dict: dict[str, torch.Tensor] = {}
        max_length = min(max_length, self.max_length)

        for key in feature_keys:
            sequences = [feature[key] for feature in features]

            if key not in ['group_lifecycle', 'group_recency', 'group_purchase', 'group_cart_intent', 'group_exploration']:
                sample_tensor = sequences[0]
                if sample_tensor.ndim == 1:
                    padded_sequences = self._pad_1d_sequences(sequences, max_length, target)
                elif sample_tensor.ndim == 2:
                    padded_sequences = self._pad_2d_sequences(sequences, max_length, target)
                else:
                    raise ValueError(f"ndim={sample_tensor.ndim} is not supported.")
            else:
                padded_sequences = self._pad_1d_sequences(sequences, len(sequences[0]), target)
            features_dict[key] = padded_sequences

        return features_dict

    def __call__(
        self,
        raw_batch: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] , dict[str, Any] ]],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] , dict[str, torch.Tensor] ]:
        features1, features2, labels = zip(*raw_batch)

        features_dict_1 = self._get_features_dict(features1)

        if features2[0] is None:
            features_dict_2 = None
        else:
            features_dict_2 = self._get_features_dict(features2, target=True)

        if labels[0] is None:
            labels_dict = None
        else:
            labels_dict = default_collate(labels)

        return features_dict_1, features_dict_2, labels_dict
