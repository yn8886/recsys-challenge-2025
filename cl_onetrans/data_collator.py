from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.utils.data import default_collate


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
        if self.padding == "longest":
            max_length = max(feature[self.key_column].size(0) for feature in features)
        elif self.padding == "max_length":
            assert self.max_length is not None
            max_length = self.max_length
        else:
            raise NotImplementedError(f"padding={self.padding} is not supported.")

        features_dict: dict[str, torch.Tensor] = {}
        max_length = min(max_length, self.max_length)

        for key in feature_keys:
            sequences = [feature[key] for feature in features]

            sample_tensor = sequences[0]
            if sample_tensor.ndim == 1:
                padded_sequences = self._pad_1d_sequences(sequences, max_length, target)
            elif sample_tensor.ndim == 2:
                padded_sequences = self._pad_2d_sequences(sequences, max_length, target)
            else:
                raise ValueError(f"ndim={sample_tensor.ndim} is not supported.")

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
