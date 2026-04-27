from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoder import TransformerTextEncoder


@dataclass
class TransformerSummarizerConfig:
    vocab_size: int
    pad_token_id: int
    max_source_length: int = 512
    max_target_length: int = 128
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5
    share_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside vocabulary")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_source_length <= 0 or self.max_target_length <= 0:
            raise ValueError("max sequence lengths must be positive")
        if self.num_encoder_layers <= 0 or self.num_decoder_layers <= 0:
            raise ValueError("layer counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerSummarizerConfig":
        return cls(**data)


@dataclass
class SummarizerOutput:
    logits: Tensor
    encoder_last_hidden_state: Tensor
    decoder_last_hidden_state: Tensor
    loss: Tensor | None = None


class TransformerTextSummarizer(nn.Module):
    def __init__(self, config: TransformerSummarizerConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = TransformerTextEncoder(
            vocab_size=config.vocab_size,
            pad_token_id=config.pad_token_id,
            max_length=config.max_source_length,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            init_std=config.init_std,
        )
        self.tgt_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.target_positions = self.encoder.positions.__class__(
            d_model=config.d_model,
            max_length=config.max_target_length,
            dropout=config.dropout,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.share_embeddings:
            self.tgt_embedding.weight = self.encoder.token_embedding.weight
            self.lm_head.weight = self.tgt_embedding.weight

        self.reset_decoder_parameters()

    def reset_decoder_parameters(self) -> None:
        for parameter in self.decoder.parameters():
            if parameter.dim() > 1:
                nn.init.normal_(parameter, mean=0.0, std=self.config.init_std)

        if not self.config.share_embeddings:
            nn.init.normal_(
                self.tgt_embedding.weight,
                mean=0.0,
                std=self.config.init_std,
            )
            nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=self.config.init_std,
            )

        with torch.no_grad():
            self.tgt_embedding.weight[self.config.pad_token_id].zero_()

    def make_padding_mask(self, tokens: Tensor) -> Tensor:
        return self.encoder.make_padding_mask(tokens)

    @staticmethod
    def causal_mask(size: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def encode(
        self,
        src_tokens: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        return self.encoder(
            input_ids=src_tokens,
            key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if tgt_tokens.dim() != 2:
            raise ValueError("tgt_tokens must have shape [batch_size, sequence_length]")
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.make_padding_mask(tgt_tokens)

        tgt_embeddings = self.tgt_embedding(tgt_tokens) * math.sqrt(
            self.config.d_model
        )
        tgt_embeddings = self.target_positions(tgt_embeddings)

        return self.decoder(
            tgt=tgt_embeddings,
            memory=memory,
            tgt_mask=self.causal_mask(tgt_tokens.size(1), tgt_tokens.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    def forward(
        self,
        src_tokens: Tensor,
        tgt_tokens: Tensor,
        labels: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
    ) -> SummarizerOutput:
        if src_key_padding_mask is None:
            src_key_padding_mask = self.make_padding_mask(src_tokens)

        memory = self.encode(
            src_tokens=src_tokens,
            src_key_padding_mask=src_key_padding_mask,
        )
        decoder_hidden = self.decode(
            tgt_tokens=tgt_tokens,
            memory=memory,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.lm_head(decoder_hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )

        return SummarizerOutput(
            logits=logits,
            encoder_last_hidden_state=memory,
            decoder_last_hidden_state=decoder_hidden,
            loss=loss,
        )

    @torch.no_grad()
    def generate(
        self,
        src_tokens: Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int | None = None,
    ) -> Tensor:
        max_new_tokens = max_new_tokens or self.config.max_target_length - 1
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if max_new_tokens + 1 > self.config.max_target_length:
            raise ValueError(
                "max_new_tokens plus BOS token exceeds max_target_length"
            )

        was_training = self.training
        self.eval()
        try:
            src_key_padding_mask = self.make_padding_mask(src_tokens)
            memory = self.encode(
                src_tokens,
                src_key_padding_mask=src_key_padding_mask,
            )

            batch_size = src_tokens.size(0)
            generated = torch.full(
                (batch_size, 1),
                fill_value=bos_token_id,
                dtype=src_tokens.dtype,
                device=src_tokens.device,
            )
            finished = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=src_tokens.device,
            )

            for _ in range(max_new_tokens):
                decoder_hidden = self.decode(
                    tgt_tokens=generated,
                    memory=memory,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                next_token = self.lm_head(decoder_hidden[:, -1]).argmax(dim=-1)
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, self.config.pad_token_id),
                    next_token,
                )

                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                finished |= next_token.eq(eos_token_id)

                if finished.all():
                    break
        finally:
            if was_training:
                self.train()

        return generated

    def summarize_token_ids(
        self,
        src_tokens: Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_new_tokens: int | None = None,
    ) -> Tensor:
        return self.generate(
            src_tokens=src_tokens,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), checkpoint_path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "TransformerTextSummarizer":
        checkpoint = torch.load(path, map_location=map_location)
        config = TransformerSummarizerConfig.from_dict(checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

