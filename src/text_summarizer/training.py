from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .tokenizer import SimpleTokenizer


def select_device(device_name: str = "auto") -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda devices: {torch.cuda.device_count()}")
        for gpu_idx in range(torch.cuda.device_count()):
            print(f"gpu {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def maybe_wrap_multi_gpu(model: nn.Module, use_multi_gpu: bool) -> nn.Module:
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"using DataParallel on {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    print("using single device training")
    return model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def amp_autocast(device: torch.device, enabled: bool):
    enabled = enabled and device.type == "cuda"
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(device: torch.device, enabled: bool):
    enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def scalar_loss(loss: Tensor | None) -> Tensor:
    if loss is None:
        raise RuntimeError("model did not return loss")
    return loss.mean()


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    use_amp: bool,
    gradient_accumulation_steps: int,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        with amp_autocast(device, use_amp):
            output = model(
                batch["src_tokens"],
                batch["tgt_tokens"],
                labels=batch["labels"],
            )
            loss = scalar_loss(output.loss)
            scaled_loss = loss / gradient_accumulation_steps

        scaler.scale(scaled_loss).backward()
        should_step = step % gradient_accumulation_steps == 0 or step == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                unwrap_model(model).parameters(),
                grad_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with amp_autocast(device, use_amp):
            output = model(
                batch["src_tokens"],
                batch["tgt_tokens"],
                labels=batch["labels"],
            )
        total_loss += float(scalar_loss(output.loss).detach().cpu())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    epochs: int,
    use_amp: bool,
    gradient_accumulation_steps: int,
    grad_clip_norm: float,
    history: dict[str, list[float]] | None = None,
) -> dict[str, list[float]]:
    history = history or {"train_loss": [], "val_loss": []}
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss = evaluate(model, val_loader, device=device, use_amp=use_amp)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_ppl={perplexity(train_loss):.2f} "
            f"val_loss={val_loss:.4f} val_ppl={perplexity(val_loss):.2f}"
        )
    return history


def save_training_artifacts(
    output_dir: str | Path,
    checkpoint_name: str,
    tokenizer_name: str,
    history_name: str,
    model: nn.Module,
    tokenizer: SimpleTokenizer,
    history: dict[str, list[float]],
    **extra: Any,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    checkpoint_path = output_dir / checkpoint_name
    tokenizer_path = output_dir / tokenizer_name
    history_path = output_dir / history_name

    torch.save(
        model_to_save.checkpoint_payload(tokenizer=tokenizer, history=history, **extra),
        checkpoint_path,
    )
    tokenizer.save(tokenizer_path)
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path, tokenizer_path, history_path

