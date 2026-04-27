from __future__ import annotations

import argparse

import torch

from text_summarizer.config import DatasetConfig, TrainingConfig
from text_summarizer.data import (
    build_tokenizer,
    load_cnn_dailymail_splits,
    make_dataloader,
)
from text_summarizer.model import TransformerSummarizerConfig, TransformerTextSummarizer
from text_summarizer.training import (
    make_grad_scaler,
    maybe_wrap_multi_gpu,
    print_device_info,
    save_training_artifacts,
    select_device,
    train_epochs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer summarizer")
    parser.add_argument("--data-dir", default=DatasetConfig.data_dir)
    parser.add_argument("--dataset-subdir", default=DatasetConfig.dataset_subdir)
    parser.add_argument("--max-train-rows", type=int, default=DatasetConfig.max_train_rows)
    parser.add_argument("--max-validation-rows", type=int, default=DatasetConfig.max_validation_rows)
    parser.add_argument("--max-test-rows", type=int, default=DatasetConfig.max_test_rows)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--num-decoder-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=TrainingConfig.gradient_accumulation_steps)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--grad-clip-norm", type=float, default=TrainingConfig.grad_clip_norm)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-multi-gpu", action="store_true")
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        dataset_subdir=args.dataset_subdir,
        max_train_rows=args.max_train_rows,
        max_validation_rows=args.max_validation_rows,
        max_test_rows=args.max_test_rows,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_workers=args.num_workers,
        grad_clip_norm=args.grad_clip_norm,
        use_amp=not args.no_amp,
        use_multi_gpu=not args.no_multi_gpu,
        output_dir=args.output_dir,
    )

    device = select_device(args.device)
    print_device_info(device)

    train_df, val_df, _ = load_cnn_dailymail_splits(dataset_config)
    tokenizer = build_tokenizer(
        train_df,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
    )
    print(f"vocab_size: {tokenizer.vocab_size}")

    model_config = TransformerSummarizerConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    train_loader = make_dataloader(
        train_df,
        tokenizer,
        model_config.max_source_length,
        model_config.max_target_length,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        shuffle=True,
    )
    val_loader = make_dataloader(
        val_df,
        tokenizer,
        model_config.max_source_length,
        model_config.max_target_length,
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        shuffle=False,
    )

    model = TransformerTextSummarizer(model_config).to(device)
    model = maybe_wrap_multi_gpu(model, training_config.use_multi_gpu)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scaler = make_grad_scaler(device, training_config.use_amp)
    history = train_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        epochs=training_config.epochs,
        use_amp=training_config.use_amp,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        grad_clip_norm=training_config.grad_clip_norm,
    )
    checkpoint_path, tokenizer_path, history_path = save_training_artifacts(
        output_dir=training_config.output_dir,
        checkpoint_name=training_config.checkpoint_name,
        tokenizer_name=training_config.tokenizer_name,
        history_name=training_config.history_name,
        model=model,
        tokenizer=tokenizer,
        history=history,
    )
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved tokenizer: {tokenizer_path}")
    print(f"saved history: {history_path}")


if __name__ == "__main__":
    main()

