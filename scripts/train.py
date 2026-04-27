from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer

from text_summarizer import TransformerSummarizerConfig, TransformerTextSummarizer


Batch = Mapping[str, Tensor]


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    log_every: int = 50
    output_dir: str = "runs/baseline"
    metrics_file: str = "metrics.jsonl"
    checkpoint_file: str = "checkpoint.pt"


def select_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def append_metrics(path: Path, record: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def move_batch_to_device(batch: Batch, device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def get_batch_tensors(batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
    required_keys = ("src_tokens", "tgt_tokens", "labels")
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise KeyError(f"batch is missing required keys: {', '.join(missing)}")

    return batch["src_tokens"], batch["tgt_tokens"], batch["labels"]


def build_optimizer(
    model: TransformerTextSummarizer,
    training_config: TrainingConfig,
) -> Optimizer:
    return AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )


def training_step(
    model: TransformerTextSummarizer,
    batch: Batch,
    optimizer: Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    batch = move_batch_to_device(batch, device)
    src_tokens, tgt_tokens, labels = get_batch_tensors(batch)

    optimizer.zero_grad(set_to_none=True)
    output = model(
        src_tokens=src_tokens,
        tgt_tokens=tgt_tokens,
        labels=labels,
    )
    if output.loss is None:
        raise RuntimeError("model did not return loss")

    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()

    return float(output.loss.detach().cpu())


@torch.no_grad()
def evaluate(
    model: TransformerTextSummarizer,
    batches: Iterable[Batch],
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in batches:
        batch = move_batch_to_device(batch, device)
        src_tokens, tgt_tokens, labels = get_batch_tensors(batch)
        output = model(
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            labels=labels,
        )
        if output.loss is None:
            raise RuntimeError("model did not return loss")

        total_loss += float(output.loss.detach().cpu())
        total_batches += 1

    if total_batches == 0:
        raise ValueError("evaluation loader is empty")
    return total_loss / total_batches


def save_checkpoint(
    path: Path,
    model: TransformerTextSummarizer,
    optimizer: Optimizer,
    epoch: int,
    training_config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.checkpoint_payload()
    payload.update(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "training_config": asdict(training_config),
        }
    )
    torch.save(payload, path)


def train(
    model: TransformerTextSummarizer,
    train_batches: Iterable[Batch],
    val_batches: Iterable[Batch] | None,
    training_config: TrainingConfig,
    device: torch.device,
) -> None:
    output_dir = Path(training_config.output_dir)
    metrics_path = output_dir / training_config.metrics_file
    checkpoint_path = output_dir / training_config.checkpoint_file

    model.to(device)
    optimizer = build_optimizer(model, training_config)
    global_step = 0

    for epoch in range(1, training_config.epochs + 1):
        train_loss_sum = 0.0
        train_batch_count = 0

        for batch in train_batches:
            global_step += 1
            loss = training_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                grad_clip_norm=training_config.grad_clip_norm,
            )
            train_loss_sum += loss
            train_batch_count += 1

            if global_step % training_config.log_every == 0:
                append_metrics(
                    metrics_path,
                    {
                        "split": "train",
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": loss,
                        "perplexity": math.exp(min(loss, 20.0)),
                        "learning_rate": training_config.learning_rate,
                    },
                )

        if train_batch_count == 0:
            raise ValueError("train loader is empty")

        epoch_train_loss = train_loss_sum / train_batch_count
        append_metrics(
            metrics_path,
            {
                "split": "train_epoch",
                "epoch": epoch,
                "global_step": global_step,
                "loss": epoch_train_loss,
                "perplexity": math.exp(min(epoch_train_loss, 20.0)),
                "learning_rate": training_config.learning_rate,
            },
        )

        if val_batches is not None:
            val_loss = evaluate(model, val_batches, device)
            append_metrics(
                metrics_path,
                {
                    "split": "validation",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": val_loss,
                    "perplexity": math.exp(min(val_loss, 20.0)),
                    "learning_rate": training_config.learning_rate,
                },
            )

        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            training_config=training_config,
        )


def build_dataloaders(_: argparse.Namespace) -> tuple[Iterable[Batch], Iterable[Batch] | None]:
    raise NotImplementedError(
        "Train/Test Datasets and DataLoaders are intentionally not implemented yet. "
        "Future DataLoaders should yield batches with keys: "
        "src_tokens, tgt_tokens, labels."
    )


def build_model_config(args: argparse.Namespace) -> TransformerSummarizerConfig:
    return TransformerSummarizerConfig(
        vocab_size=args.vocab_size,
        pad_token_id=args.pad_token_id,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        share_embeddings=not args.disable_shared_embeddings,
    )


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        output_dir=args.output_dir,
        metrics_file=args.metrics_file,
        checkpoint_file=args.checkpoint_file,
    )


def run_dry_step(args: argparse.Namespace) -> None:
    device = select_device(args.device)
    model_config = build_model_config(args)
    training_config = build_training_config(args)
    model = TransformerTextSummarizer(model_config).to(device)
    optimizer = build_optimizer(model, training_config)

    batch = {
        "src_tokens": torch.randint(
            low=3,
            high=model_config.vocab_size,
            size=(2, min(16, model_config.max_source_length)),
        ),
        "tgt_tokens": torch.randint(
            low=3,
            high=model_config.vocab_size,
            size=(2, min(8, model_config.max_target_length)),
        ),
        "labels": torch.randint(
            low=3,
            high=model_config.vocab_size,
            size=(2, min(8, model_config.max_target_length)),
        ),
    }
    loss = training_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        grad_clip_norm=training_config.grad_clip_norm,
    )
    print(f"dry training step finished, loss={loss:.4f}, device={device}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer text summarizer")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--disable-shared-embeddings", action="store_true")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--output-dir", default="runs/baseline")
    parser.add_argument("--metrics-file", default="metrics.jsonl")
    parser.add_argument("--checkpoint-file", default="checkpoint.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        run_dry_step(args)
        return

    model = TransformerTextSummarizer(build_model_config(args))
    training_config = build_training_config(args)
    device = select_device(args.device)
    train_batches, val_batches = build_dataloaders(args)
    train(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        training_config=training_config,
        device=device,
    )


if __name__ == "__main__":
    main()

