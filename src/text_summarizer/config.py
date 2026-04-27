from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    data_dir: str = "/kaggle/input/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail"
    dataset_subdir: str = "cnn_dailymail"
    train_file: str = "train.csv"
    validation_file: str = "validation.csv"
    test_file: str = "test.csv"
    text_column: str = "article"
    summary_column: str = "highlights"
    max_train_rows: int | None = 50000
    max_validation_rows: int | None = 5000
    max_test_rows: int | None = 5000


@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 2
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    use_multi_gpu: bool = True
    output_dir: str = "/kaggle/working/summarizer"
    checkpoint_name: str = "summarizer_checkpoint.pt"
    history_name: str = "history.json"
    tokenizer_name: str = "tokenizer_vocab.json"


@dataclass
class FineTuneConfig:
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_workers: int = 2
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    use_multi_gpu: bool = True
    max_train_rows: int | None = None
    max_validation_rows: int | None = 2000
    output_dir: str = "/kaggle/working"
    checkpoint_name: str = "fine_tuned_summarizer_checkpoint.pt"
    history_name: str = "fine_tuned_history.json"
    tokenizer_name: str = "fine_tuned_tokenizer_vocab.json"

