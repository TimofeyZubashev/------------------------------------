from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_summarizer.config import DatasetConfig
from text_summarizer.data import load_cnn_dailymail_splits
from text_summarizer.inference import load_for_inference, summarize
from text_summarizer.training import print_device_info, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summarizer inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default=None)
    parser.add_argument("--data-dir", default=DatasetConfig.data_dir)
    parser.add_argument("--dataset-subdir", default=DatasetConfig.dataset_subdir)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    print_device_info(device)
    model, tokenizer, _ = load_for_inference(args.checkpoint, device=device)

    if args.text:
        print(summarize(model, tokenizer, args.text, device, max_new_tokens=args.max_new_tokens))
        return

    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        dataset_subdir=args.dataset_subdir,
        max_train_rows=0,
        max_validation_rows=args.num_samples,
        max_test_rows=0,
    )
    _, val_df, _ = load_cnn_dailymail_splits(dataset_config)
    for _, row in val_df.head(args.num_samples).iterrows():
        generated = summarize(
            model,
            tokenizer,
            row["text"],
            device,
            max_new_tokens=args.max_new_tokens,
        )
        print("=" * 100)
        print("INPUT")
        print(row["text"])
        print("\nTARGET")
        print(row["summary"])
        print("\nGENERATION")
        print(generated)


if __name__ == "__main__":
    main()
