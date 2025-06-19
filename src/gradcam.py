"""Grad-CAM visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def compute_gradcam(model: tf.keras.Model, img_array: np.ndarray, layer_name: Optional[str] = None):
    """Compute Grad-CAM for a single image array."""
    if layer_name is None:
        layer_name = next(l.name for l in reversed(model.layers) if hasattr(l, "activation"))

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=2)
    heatmap = tf.image.resize(heatmap, (img.shape[0], img.shape[1])).numpy()
    heatmap = np.uint8(plt.cm.jet(heatmap[..., 0]) * 255)
    overlay = heatmap[..., :3] * alpha + img
    overlay = np.uint8(overlay)
    return overlay


def save_gradcam(model: tf.keras.Model, img_path: Path, out_path: Path, layer: Optional[str] = None):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=model.input_shape[1:3])
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    input_array = np.expand_dims(img_array / 255.0, axis=0)
    heatmap = compute_gradcam(model, input_array, layer)
    overlay = overlay_heatmap(img_array, heatmap)
    plt.imsave(out_path, overlay)


__all__ = ["save_gradcam", "compute_gradcam", "overlay_heatmap"]
