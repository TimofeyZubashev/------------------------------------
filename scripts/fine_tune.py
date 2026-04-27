from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from text_summarizer.config import DatasetConfig, FineTuneConfig
from text_summarizer.data import load_cnn_dailymail_splits, make_dataloader
from text_summarizer.model import load_model_bundle
from text_summarizer.training import (
    make_grad_scaler,
    maybe_wrap_multi_gpu,
    print_device_info,
    save_training_artifacts,
    select_device,
    train_epochs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue training from checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=DatasetConfig.data_dir)
    parser.add_argument("--dataset-subdir", default=DatasetConfig.dataset_subdir)
    parser.add_argument("--max-train-rows", type=int, default=FineTuneConfig.max_train_rows)
    parser.add_argument("--max-validation-rows", type=int, default=FineTuneConfig.max_validation_rows)
    parser.add_argument("--epochs", type=int, default=FineTuneConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=FineTuneConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=FineTuneConfig.weight_decay)
    parser.add_argument("--batch-size", type=int, default=FineTuneConfig.batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=FineTuneConfig.gradient_accumulation_steps)
    parser.add_argument("--num-workers", type=int, default=FineTuneConfig.num_workers)
    parser.add_argument("--grad-clip-norm", type=float, default=FineTuneConfig.grad_clip_norm)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-multi-gpu", action="store_true")
    parser.add_argument("--output-dir", default=FineTuneConfig.output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print_device_info(device)

    model, tokenizer, history = load_model_bundle(args.checkpoint, map_location=device)
    model.to(device)
    model = maybe_wrap_multi_gpu(model, not args.no_multi_gpu)

    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        dataset_subdir=args.dataset_subdir,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        max_test_rows=0,
    )
    train_df, val_df, _ = load_cnn_dailymail_splits(dataset_config)
    train_loader = make_dataloader(
        train_df,
        tokenizer,
        model.module.config.max_source_length if hasattr(model, "module") else model.config.max_source_length,
        model.module.config.max_target_length if hasattr(model, "module") else model.config.max_target_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = make_dataloader(
        val_df,
        tokenizer,
        model.module.config.max_source_length if hasattr(model, "module") else model.config.max_source_length,
        model.module.config.max_target_length if hasattr(model, "module") else model.config.max_target_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(device, not args.no_amp)
    history = train_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        epochs=args.epochs,
        use_amp=not args.no_amp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        grad_clip_norm=args.grad_clip_norm,
        history=history,
    )
    checkpoint_path, tokenizer_path, history_path = save_training_artifacts(
        output_dir=args.output_dir,
        checkpoint_name=FineTuneConfig.checkpoint_name,
        tokenizer_name=FineTuneConfig.tokenizer_name,
        history_name=FineTuneConfig.history_name,
        model=model,
        tokenizer=tokenizer,
        history=history,
        fine_tune_epochs=args.epochs,
        fine_tune_lr=args.learning_rate,
        fine_tune_batch_size=args.batch_size,
    )
    print(f"saved fine-tuned checkpoint: {checkpoint_path}")
    print(f"saved tokenizer: {tokenizer_path}")
    print(f"saved history: {history_path}")


if __name__ == "__main__":
    main()
