"""Evaluation utilities for trained model."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from data_loader import create_generators
from gradcam import save_gradcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", type=Path, default=Path("models/final_model.h5"))
    parser.add_argument("--out", type=Path, default=Path("data/eval_gradcam.png"))
    parser.add_argument("--sample", type=str, required=False, help="path to sample image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    if args.sample:
        save_gradcam(model, Path(args.sample), args.out)
        print(f"Grad-CAM saved to {args.out}")


if __name__ == "__main__":
    main()
