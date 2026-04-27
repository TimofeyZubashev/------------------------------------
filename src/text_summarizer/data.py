from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig
from .tokenizer import SimpleTokenizer


def make_demo_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = [
        {
            "text": "Neural networks learn patterns from data and can summarize text.",
            "summary": "Neural networks can summarize text.",
        },
        {
            "text": "Transformers use attention layers to model sequence dependencies.",
            "summary": "Transformers model sequences with attention.",
        },
        {
            "text": "A dataloader groups tokenized examples into batches.",
            "summary": "Dataloaders batch tokenized examples.",
        },
        {
            "text": "Validation loss helps detect overfitting during training.",
            "summary": "Validation loss tracks overfitting.",
        },
        {
            "text": "The decoder generates a summary token by token.",
            "summary": "The decoder generates summaries autoregressively.",
        },
    ]
    demo = pd.DataFrame(samples * 200)
    train_end = int(len(demo) * 0.8)
    val_end = int(len(demo) * 0.9)
    return (
        demo.iloc[:train_end].reset_index(drop=True),
        demo.iloc[train_end:val_end].reset_index(drop=True),
        demo.iloc[val_end:].reset_index(drop=True),
    )


def candidate_dataset_roots(config: DatasetConfig) -> list[Path]:
    roots = [
        Path(config.data_dir),
        Path("/kaggle/input/newspaper-text-summarization-cnn-dailymail"),
        Path("/kaggle/input/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail"),
    ]
    return list(dict.fromkeys(roots))


def find_split_file(config: DatasetConfig, split_file: str) -> Path | None:
    for root in candidate_dataset_roots(config):
        candidates = [
            root / config.dataset_subdir / split_file,
            root / split_file,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        if root.exists():
            matches = sorted(root.rglob(split_file))
            if matches:
                return matches[0]
    return None


def find_file_in_tree(root: str | Path, filename: str) -> Path:
    root = Path(root)
    matches = sorted(root.rglob(filename), key=lambda path: len(str(path)))
    if not matches:
        raise FileNotFoundError(f"{filename} was not found under {root}")
    return matches[0]


def normalize_split(
    frame: pd.DataFrame,
    config: DatasetConfig,
    limit: int | None,
) -> pd.DataFrame:
    required = [config.text_column, config.summary_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Available columns: {list(frame.columns)}"
        )

    frame = frame[required].rename(
        columns={
            config.text_column: "text",
            config.summary_column: "summary",
        }
    )
    frame = frame.dropna().astype(str)
    frame = frame[(frame["text"].str.len() > 0) & (frame["summary"].str.len() > 0)]
    if limit is not None:
        frame = frame.head(limit)
    return frame.reset_index(drop=True)


def load_cnn_dailymail_splits(
    config: DatasetConfig,
    fallback_to_demo: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_specs = [
        ("train", config.train_file, config.max_train_rows),
        ("validation", config.validation_file, config.max_validation_rows),
        ("test", config.test_file, config.max_test_rows),
    ]

    loaded: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []
    for split_name, split_file, row_limit in split_specs:
        split_path = find_split_file(config, split_file)
        if split_path is None:
            missing_files.append(split_file)
            continue

        frame = pd.read_csv(split_path)
        loaded[split_name] = normalize_split(frame, config, row_limit)
        print(f"{split_name}: {split_path} -> {len(loaded[split_name])} rows")

    if missing_files:
        if not fallback_to_demo:
            raise FileNotFoundError(f"Could not find split files: {missing_files}")
        print(f"Could not find split files: {missing_files}. Using demo dataset.")
        return make_demo_split()

    return loaded["train"], loaded["validation"], loaded["test"]


class SummarizationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: SimpleTokenizer,
        max_source_length: int,
        max_target_length: int,
        include_raw_text: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.include_raw_text = include_raw_text

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        row = self.frame.iloc[index]
        src_tokens = self.tokenizer.encode_source(row["text"], self.max_source_length)
        tgt_tokens, labels = self.tokenizer.encode_target_pair(
            row["summary"],
            self.max_target_length,
        )
        item: dict[str, Tensor | str] = {
            "src_tokens": torch.tensor(src_tokens, dtype=torch.long),
            "tgt_tokens": torch.tensor(tgt_tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if self.include_raw_text:
            item["raw_input"] = row["text"]
            item["raw_target"] = row["summary"]
        return item


def make_dataloader(
    frame: pd.DataFrame,
    tokenizer: SimpleTokenizer,
    max_source_length: int,
    max_target_length: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    include_raw_text: bool = False,
) -> DataLoader:
    dataset = SummarizationDataset(
        frame=frame,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        include_raw_text=include_raw_text,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_tokenizer(
    train_df: pd.DataFrame,
    min_freq: int,
    max_vocab_size: int,
) -> SimpleTokenizer:
    tokenizer = SimpleTokenizer(min_freq=min_freq, max_vocab_size=max_vocab_size)
    tokenizer.fit(train_df["text"].tolist() + train_df["summary"].tolist())
    return tokenizer

