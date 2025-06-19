"""Data loading utilities for the Chest X-ray Pneumonia dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_URL = "paultimothymooney/chest-xray-pneumonia"
DATA_DIR = Path("data/chest_xray")


def download_dataset() -> None:
    """Download dataset using Kaggle API."""
    if DATA_DIR.exists():
        print("Dataset already downloaded.")
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        import kaggle
    except ImportError as exc:
        raise RuntimeError("Kaggle package is required to download data") from exc

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATA_URL, path=str(DATA_DIR), unzip=True)
    print("Download complete")


def create_generators(img_size: Tuple[int, int] = (224, 224), batch_size: int = 32):
    """Create train, validation, and test generators with augmentation."""
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    test_dir = DATA_DIR / "test"

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.1,
    )

    val_test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
    )
    val_flow = train_gen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
    )
    test_flow = val_test_gen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )
    return train_flow, val_flow, test_flow


__all__ = ["download_dataset", "create_generators", "DATA_DIR"]
