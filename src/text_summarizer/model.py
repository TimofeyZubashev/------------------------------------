from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoder import SinusoidalPositionalEncoding, TransformerTextEncoder
from .tokenizer import SimpleTokenizer


@dataclass
class TransformerSummarizerConfig:
    vocab_size: int
    pad_token_id: int
    max_source_length: int
    max_target_length: int
    d_model: int
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be inside vocabulary")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_source_length <= 0 or self.max_target_length <= 0:
            raise ValueError("max sequence lengths must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TransformerSummarizerConfig":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


class SummarizerOutput(NamedTuple):
    logits: Tensor
    encoder_last_hidden_state: Tensor
    decoder_last_hidden_state: Tensor
    loss: Tensor | None = None


class TransformerTextSummarizer(nn.Module):
    def __init__(self, model_config: TransformerSummarizerConfig) -> None:
        super().__init__()
        self.config = model_config
        self.encoder = TransformerTextEncoder(
            vocab_size=model_config.vocab_size,
            pad_token_id=model_config.pad_token_id,
            max_length=model_config.max_source_length,
            d_model=model_config.d_model,
            num_heads=model_config.num_heads,
            num_layers=model_config.num_encoder_layers,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
        )
        self.tgt_embedding = nn.Embedding(
            model_config.vocab_size,
            model_config.d_model,
            padding_idx=model_config.pad_token_id,
        )
        self.target_positions = SinusoidalPositionalEncoding(
            model_config.d_model,
            model_config.max_target_length,
            model_config.dropout,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_config.d_model,
            nhead=model_config.num_heads,
            dim_feedforward=model_config.dim_feedforward,
            dropout=model_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=model_config.num_decoder_layers,
            norm=nn.LayerNorm(model_config.d_model),
        )
        self.lm_head = nn.Linear(
            model_config.d_model,
            model_config.vocab_size,
            bias=False,
        )
        self.tgt_embedding.weight = self.encoder.token_embedding.weight
        self.lm_head.weight = self.tgt_embedding.weight

    def make_padding_mask(self, tokens: Tensor) -> Tensor:
        return tokens.eq(self.config.pad_token_id)

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
        return self.encoder(src_tokens, key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        tgt_tokens: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.make_padding_mask(tgt_tokens)

        embeddings = self.tgt_embedding(tgt_tokens) * math.sqrt(self.config.d_model)
        embeddings = self.target_positions(embeddings)
        return self.decoder(
            tgt=embeddings,
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
    ) -> SummarizerOutput:
        src_key_padding_mask = self.make_padding_mask(src_tokens)
        memory = self.encode(src_tokens, src_key_padding_mask=src_key_padding_mask)
        decoder_hidden = self.decode(
            tgt_tokens,
            memory,
            memory_key_padding_mask=src_key_padding_mask,
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
        max_new_tokens: int,
    ) -> Tensor:
        self.eval()
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        src_key_padding_mask = self.make_padding_mask(src_tokens)
        memory = self.encode(src_tokens, src_key_padding_mask=src_key_padding_mask)
        generated = torch.full(
            (src_tokens.size(0), 1),
            bos_token_id,
            dtype=torch.long,
            device=src_tokens.device,
        )
        finished = torch.zeros(
            src_tokens.size(0),
            dtype=torch.bool,
            device=src_tokens.device,
        )

        for _ in range(max_new_tokens):
            decoder_hidden = self.decode(
                generated,
                memory,
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
        return generated

    def checkpoint_payload(
        self,
        tokenizer: SimpleTokenizer,
        history: dict[str, list[float]] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        payload = {
            "model_config": self.config.to_dict(),
            "model_state_dict": self.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "history": history or {"train_loss": [], "val_loss": []},
        }
        payload.update(extra)
        return payload

    def save_checkpoint(
        self,
        path: str | Path,
        tokenizer: SimpleTokenizer,
        history: dict[str, list[float]] | None = None,
        **extra: Any,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(tokenizer, history, **extra), path)


def load_checkpoint_payload(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def model_from_checkpoint_payload(
    checkpoint: dict[str, Any],
) -> tuple[TransformerTextSummarizer, SimpleTokenizer, dict[str, list[float]]]:
    config_data = checkpoint.get("model_config") or checkpoint.get("config")
    if config_data is None:
        raise KeyError("checkpoint does not contain model_config")

    model_config = TransformerSummarizerConfig.from_dict(config_data)
    model = TransformerTextSummarizer(model_config)
    state_dict = checkpoint["model_state_dict"]
    cleaned_state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)

    tokenizer = SimpleTokenizer.from_dict(checkpoint["tokenizer"])
    history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
    return model, tokenizer, history


def load_model_bundle(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[TransformerTextSummarizer, SimpleTokenizer, dict[str, list[float]]]:
    checkpoint = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    return model_from_checkpoint_payload(checkpoint)

