from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, token_embeddings: Tensor) -> Tensor:
        sequence_length = token_embeddings.size(1)
        if sequence_length > self.encoding.size(1):
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"maximum positional encoding length {self.encoding.size(1)}"
            )
        position_encoding = self.encoding[:, :sequence_length].to(
            dtype=token_embeddings.dtype
        )
        return self.dropout(token_embeddings + position_encoding)


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        max_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )
        self.positions = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_length=max_length,
            dropout=dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def make_padding_mask(self, tokens: Tensor) -> Tensor:
        return tokens.eq(self.pad_token_id)

    def forward(
        self,
        input_ids: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length]")
        if key_padding_mask is None:
            key_padding_mask = self.make_padding_mask(input_ids)

        embeddings = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        embeddings = self.positions(embeddings)
        return self.encoder(embeddings, src_key_padding_mask=key_padding_mask)

