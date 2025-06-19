"""Training script for pneumonia detection."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from data_loader import create_generators, download_dataset, DATA_DIR
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    return parser.parse_args()


def plot_history(history, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(out_dir / "acc_curve.png")
    plt.close()


def main() -> None:
    args = parse_args()
    download_dataset()
    train_gen, val_gen, test_gen = create_generators(
        img_size=(args.img_size, args.img_size), batch_size=args.batch_size
    )

    model = build_model(input_shape=(args.img_size, args.img_size, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint("models/best_model.h5", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    plot_history(history, Path("data"))

    preds = model.predict(test_gen, verbose=1)
    y_pred = (preds.ravel() > 0.5).astype(int)
    cm = confusion_matrix(test_gen.classes, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=test_gen.class_indices.keys())
    disp.plot(cmap="Blues")
    plt.savefig("data/confusion_matrix.png")
    plt.close()

    model.save("models/final_model.h5")


if __name__ == "__main__":
    main()
