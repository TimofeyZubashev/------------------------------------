from __future__ import annotations

import argparse

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

