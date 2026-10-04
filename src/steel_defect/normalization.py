"""Refresh inference moments using only unaugmented training images."""
import numpy as np
import tensorflow as tf


class TrainingBatchNormCalibration(tf.keras.callbacks.Callback):
    """Calibrate BN before fit's validation loop, without changing learned weights.

    Moving averages lag rapidly changing convolution features in this small
    dataset. Compute population moments from training inputs at the current
    weights instead. Validation images never contribute to these statistics.
    Earlier BN layers are calibrated first so each later probe sees exactly
    the activations that inference will use. The dataset must be unaugmented.
    """

    def __init__(self, training_dataset):
        super().__init__()
        self.training_dataset = training_dataset

    def set_model(self, model):
        super().set_model(model)
        self.probes = [(layer, tf.keras.Model(model.inputs, layer.input))
                       for layer in model.layers
                       if isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable]

    def on_test_begin(self, logs=None):
        # Keras uses this hook immediately before validation within model.fit.
        # It receives no validation inputs. All probe calls use inference mode
        # to disable augmentation/dropout and preserve previously calibrated BN.
        for layer, probe in self.probes:
            sums = np.zeros(layer.moving_mean.shape, dtype=np.float64)
            squares = np.zeros_like(sums)
            count = 0
            for pixels, _ in self.training_dataset:
                values = tf.cast(probe(pixels, training=False), tf.float64)
                axes = tuple(range(len(values.shape) - 1))
                count += int(tf.reduce_prod(tf.shape(values)[:-1]).numpy())
                sums += tf.reduce_sum(values, axis=axes).numpy()
                squares += tf.reduce_sum(tf.square(values), axis=axes).numpy()
            if count == 0:
                raise ValueError("BatchNormalization calibration requires training images.")
            mean = sums / count
            variance = np.maximum(squares / count - mean ** 2, 0)
            if not np.isfinite(mean).all() or not np.isfinite(variance).all():
                raise ValueError("Non-finite training BatchNormalization moments.")
            layer.moving_mean.assign(mean)
            layer.moving_variance.assign(variance)
