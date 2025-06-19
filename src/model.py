"""Model architecture definition using Keras."""
from __future__ import annotations

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D,
                                     Input)
from tensorflow.keras.models import Model


def build_model(input_shape=(224, 224, 3)) -> Model:
    """Create transfer learning model based on ResNet50."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

__all__ = ["build_model"]
