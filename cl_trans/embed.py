import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class WordEmbedding(nn.Module):
    def __init__(self, num_word, word_emb_dim, padding_idx=0):
        super().__init__()

        self.word_emb_layer = nn.Embedding(
            num_embeddings=num_word,
            embedding_dim=word_emb_dim,
            padding_idx=padding_idx,
        )

    def forward(self, word_ids):
        word_emb = self.word_emb_layer(word_ids)
        avg_word_emb = torch.mean(word_emb, dim=-2)
        return avg_word_emb


class SkuEmbedding(nn.Module):
    def __init__(
        self,
        num_sku,
        sku_emb_dim,
        num_cat,
        cat_emb_dim,
        num_price,
        price_emb_dim,
        word_emb_dim,
        word_emb_layer,
        item_emb_dim,
        padding_idx=0,
    ):
        super().__init__()
        self.sku_emb_layer = nn.Embedding(
            num_embeddings=num_sku,
            embedding_dim=sku_emb_dim,
            padding_idx=padding_idx,
        )
        self.cat_emb_layer = nn.Embedding(
            num_embeddings=num_cat,
            embedding_dim=cat_emb_dim,
            padding_idx=padding_idx,
        )
        self.price_emb_layer = nn.Embedding(
            num_embeddings=num_price,
            embedding_dim=price_emb_dim,
            padding_idx=padding_idx,
        )
        self.word_emb_layer = word_emb_layer

        self.fc = nn.Linear(
            sku_emb_dim + cat_emb_dim + price_emb_dim + word_emb_dim,
            item_emb_dim,
        )
        self.relu = nn.ReLU()

    def forward(self, sku_id, cat_id, price_id, word_ids):
        sku_emb = self.sku_emb_layer(sku_id)
        cat_emb = self.cat_emb_layer(cat_id)
        price_emb = self.price_emb_layer(price_id)
        word_emb = self.word_emb_layer(word_ids)
        concat_emb = torch.cat([sku_emb, cat_emb, price_emb, word_emb], dim=-1)
        item_emb = self.fc(concat_emb)
        item_emb = self.relu(item_emb)
        return item_emb

