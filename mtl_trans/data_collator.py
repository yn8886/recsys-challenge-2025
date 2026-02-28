import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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

        self.query_ids = np.load(
            os.path.join(self.dataset_dir, "query_ids.npy"), allow_pickle=True
        )

        self.name_ids = np.load(
            os.path.join(self.dataset_dir, "name_ids.npy"), allow_pickle=True
        )

        print("Loading category_id.npy")
        self.category_ids = np.load(
            os.path.join(self.dataset_dir, "category_id.npy"), allow_pickle=True
        )

        print("Loading price.npy")
        self.price_ids = np.load(
            os.path.join(self.dataset_dir, "price_id.npy"), allow_pickle=True
        )

        self.diff_days = np.load(
            os.path.join(self.dataset_dir, "diff_days.npy"), allow_pickle=True
        )
        self.diff_weeks = np.load(
            os.path.join(self.dataset_dir, "diff_weeks.npy"), allow_pickle=True
        )

        print("Loading static_features.npy")
        # self.statistical_features = np.load(
        #     os.path.join(self.dataset_dir, "stats_features.npy"), allow_pickle=True
        # )
        # print(self.statistical_features.shape)

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

        print("Loading labels")
        self.is_churn = np.load(
            os.path.join(self.dataset_dir, "churn.npy"), allow_pickle=True
        )
        self.labels_buy_category = np.load(
            os.path.join(self.dataset_dir, "label_buy_cat.npy"), allow_pickle=True
        )
        self.labels_buy_sku = np.load(
            os.path.join(self.dataset_dir, "label_buy_sku.npy"), allow_pickle=True
        )

    def __len__(self):
        return len(self.client_ids)

    def _pad_sequence(self, seq):
        seq = seq.tolist()
        sliced_seq = seq[-self.max_len :]
        padding_length = self.max_len - len(sliced_seq)
        padded_seq = [0] * padding_length + sliced_seq
        padded_seq = np.array(padded_seq)
        return padded_seq, padding_length

    def _pad_word_sequence(self, seq):
        seq = seq.tolist()
        sliced_seq = seq[-self.max_len :]
        padding_length = self.max_len - len(sliced_seq)
        padded_seq = [[0] * 16] * padding_length + sliced_seq
        padded_seq = np.array(padded_seq)

        return padded_seq, padding_length

    def __getitem__(self, idx):
        # sequence features
        client_id = self.client_ids[idx]
        event_type = self.event_types[idx]
        sku_id = self.sku_ids[idx]
        url_id = self.url_ids[idx]
        query_ids = self.query_ids[idx]
        category_id = self.category_ids[idx]
        price_id = self.price_ids[idx]
        name_ids = self.name_ids[idx]
        diff_days = self.diff_days[idx]
        diff_weeks = self.diff_weeks[idx]

        # statistical features
        group_lifecycle = self.group_lifecycle[idx]
        group_recency = self.group_recency[idx]
        group_purchase = self.group_purchase[idx]
        group_cart_intent = self.group_cart_intent[idx]
        group_exploration = self.group_exploration[idx]


        # labels
        buy_sku_label = self.labels_buy_sku[idx]
        buy_cat_label = self.labels_buy_category[idx]
        is_churn = self.is_churn[idx]

        # padding and masking sequence
        event_type, _ = self._pad_sequence(event_type)
        sku_id, _ = self._pad_sequence(sku_id)
        url_id, _ = self._pad_sequence(url_id)
        query_ids, _ = self._pad_word_sequence(query_ids)
        category_id, _ = self._pad_sequence(category_id)
        price_id, _ = self._pad_sequence(price_id)
        name_ids, _ = self._pad_word_sequence(name_ids)
        diff_days, _ = self._pad_sequence(diff_days)
        diff_weeks, _ = self._pad_sequence(diff_weeks)

        # statistical features
        group_lifecycle = torch.tensor(group_lifecycle, dtype=torch.float)
        group_recency = torch.tensor(group_recency, dtype=torch.float)
        group_purchase = torch.tensor(group_purchase, dtype=torch.float)
        group_cart_intent = torch.tensor(group_cart_intent, dtype=torch.float)
        group_exploration = torch.tensor(group_exploration, dtype=torch.float)


        buy_sku_label = torch.tensor(buy_sku_label, dtype=torch.long)
        buy_cat_label = torch.tensor(buy_cat_label, dtype=torch.long)

        return_dict = {
            "client_id": torch.tensor(client_id),
            "event_type": event_type,
            "sku": sku_id,
            "url": url_id,
            "query": query_ids,
            "category": category_id,
            "price": price_id,
            "name": name_ids,
            "diff_days": diff_days,
            "diff_weeks": diff_weeks,
            "group_lifecycle": group_lifecycle,
            "group_recency" : group_recency,
            "group_purchase": group_purchase,
            "group_cart_intent": group_cart_intent,
            "group_exploration": group_exploration,
            "churn": is_churn,
            "buy_sku_label": buy_sku_label,
            "buy_cat_label": buy_cat_label,
        }

        return return_dict