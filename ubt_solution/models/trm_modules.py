import torch.nn as nn
import torch
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


class SASRec(nn.Module):
    def __init__(self, max_len, num_heads, d_model, dropout, item_emb_dim, num_layers) -> None:
        super().__init__()

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len,
        )

        self.trm_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=item_emb_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.trm_enc = nn.TransformerEncoder(
            self.trm_enc_layer, num_layers=num_layers
        )

    def forward(self, seq_emb, src_padding_mask):
        seq_emb = self.pos_encoder(seq_emb)
        trm_enc_out = self.trm_enc(seq_emb, src_key_padding_mask=src_padding_mask)
        user_emb = trm_enc_out[:, -1, :]
        return user_emb

