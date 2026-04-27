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
        max_length: int = 512,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= pad_token_id < vocab_size:
            raise ValueError("pad_token_id must be inside vocabulary")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.d_model = d_model
        self.init_std = init_std

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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.normal_(parameter, mean=0.0, std=self.init_std)

        with torch.no_grad():
            self.token_embedding.weight[self.pad_token_id].zero_()

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

        token_embeddings = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        hidden_states = self.positions(token_embeddings)

        return self.encoder(
            src=hidden_states,
            src_key_padding_mask=key_padding_mask,
        )

