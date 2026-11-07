from __future__ import annotations

import argparse
from pathlib import Path

from .model import DEFAULT_CHECKPOINT, train_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the synthetic-data Residual U-Net.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    report = train_checkpoint(args.output, epochs=args.epochs, patience=args.patience)
    print(
        f"Saved Residual U-Net at {report['checkpoint']} "
        f"(best epoch {report['best_epoch']}, validation loss {report['validation_loss']:.6f})."
    )


if __name__ == "__main__":
    main()
