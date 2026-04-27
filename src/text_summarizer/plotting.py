from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .tokenizer import SimpleTokenizer


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))


def plot_history(
    history: dict[str, list[float]],
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [perplexity(loss) for loss in history["train_loss"]],
        marker="o",
        label="train",
    )
    axes[1].plot(
        epochs,
        [perplexity(loss) for loss in history["val_loss"]],
        marker="o",
        label="validation",
    )
    axes[1].set_title("Perplexity")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)


def load_history(path: str | Path) -> dict[str, list[float]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def token_length_stats(
    frame: pd.DataFrame,
    tokenizer: SimpleTokenizer,
    column: str,
    split_name: str,
) -> dict[str, float | int | str]:
    lengths = frame[column].map(lambda text: len(tokenizer.tokenize(text)))
    return {
        "split": split_name,
        "column": column,
        "rows": int(len(lengths)),
        "mean": float(lengths.mean()),
        "p50": float(lengths.quantile(0.50)),
        "p90": float(lengths.quantile(0.90)),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }

