from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text_summarizer.plotting import load_history, plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training history JSON")
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = load_history(args.history)
    plot_history(history, output_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
