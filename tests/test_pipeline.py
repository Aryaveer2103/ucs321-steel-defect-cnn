"""Verify model serialization, consistent preprocessing and metric calculations."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from steel_defect.config import CLASSES, TrainConfig
from steel_defect.data import load_image, make_dataset


class PipelineTests(unittest.TestCase):
    def test_validation_batch_norm_calibration_uses_training_population_only(self):
        import tensorflow as tf
        from steel_defect.normalization import TrainingBatchNormCalibration

        inputs = tf.keras.Input((2, 2, 1))
        # An extreme training-only transform would spoil calibration if active.
        x = tf.keras.layers.RandomTranslation(0.5, 0.5, fill_mode="constant", fill_value=100, seed=42)(inputs)
        first = tf.keras.layers.BatchNormalization()
        second = tf.keras.layers.BatchNormalization()
        model = tf.keras.Model(inputs, second(first(x)))
        pixels = np.stack([np.zeros((2, 2, 1)), np.ones((2, 2, 1)) * 2,
                           np.ones((2, 2, 1)) * 4]).astype(np.float32)
        # The last batch is smaller: moments must be sample-weighted, not an
        # average of batch means. Population mean=2, variance=8/3.
        dataset = tf.data.Dataset.from_tensor_slices((pixels, [0, 1, 2])).batch(2)
        callback = TrainingBatchNormCalibration(dataset)
        callback.set_model(model)
        weights = [w.numpy().copy() for w in model.trainable_weights]
        callback.on_test_begin()  # fit's validation hook; no validation pixels supplied.
        np.testing.assert_allclose(first.moving_mean, [2], atol=1e-6)
        np.testing.assert_allclose(first.moving_variance, [8 / 3], atol=1e-6)
        np.testing.assert_allclose(second.moving_mean, [0], atol=1e-6)
        np.testing.assert_allclose(second.moving_variance, [(8 / 3) / (8 / 3 + first.epsilon)], atol=1e-6)
        for before, after in zip(weights, model.trainable_weights):
            np.testing.assert_array_equal(before, after.numpy())
        # The consumer must receive the same inference result across validation
        # batch sizes, including when batches contain only one class.
        together = model(pixels, training=False).numpy()
        separate = np.concatenate([model(p[None], training=False).numpy() for p in pixels])
        np.testing.assert_allclose(together, separate, atol=1e-6)
        tf.keras.backend.clear_session()

    def test_training_shuffle_mixes_classes_preserves_pairs_and_reshuffles(self):
        import tensorflow as tf

        config = TrainConfig(image_size=32, batch_size=16, seed=42)
        with tempfile.TemporaryDirectory() as folder:
            rows = []
            for index in range(192):
                path = f"{index}.bmp"
                Image.new("L", (32, 32), index).save(Path(folder) / path)
                rows.append({"path": path, "label": index // 32})

            def epoch(dataset):
                batches = list(dataset.as_numpy_iterator())
                ids = np.concatenate([x[:, 0, 0, 0].astype(int) for x, _ in batches])
                labels = np.concatenate([y for _, y in batches])
                np.testing.assert_array_equal(labels, ids // 32)
                np.testing.assert_array_equal(np.sort(ids), np.arange(192))
                for _, batch_labels in batches:
                    self.assertGreaterEqual(len(np.unique(batch_labels)), 3)
                return ids

            tf.keras.utils.set_random_seed(config.seed)
            dataset = make_dataset(rows, folder, config, training=True)
            first, second = epoch(dataset), epoch(dataset)
            self.assertFalse(np.array_equal(first, np.arange(192)))
            self.assertFalse(np.array_equal(first, second))
            tf.keras.utils.set_random_seed(config.seed)
            repeated = make_dataset(rows, folder, config, training=True)
            np.testing.assert_array_equal(first, epoch(repeated))
            np.testing.assert_array_equal(second, epoch(repeated))

    def test_nontraining_dataset_keeps_image_label_order_across_iterations(self):
        config = TrainConfig(image_size=32, batch_size=4)
        with tempfile.TemporaryDirectory() as folder:
            rows = []
            for index in range(12):
                path = f"{index}.bmp"
                Image.new("L", (32, 32), index).save(Path(folder) / path)
                rows.append({"path": path, "label": index // 2})
            dataset = make_dataset(rows, folder, config)
            for _ in range(2):
                batches = list(dataset.as_numpy_iterator())
                np.testing.assert_array_equal(np.concatenate([x[:, 0, 0, 0] for x, _ in batches]), np.arange(12))
                np.testing.assert_array_equal(np.concatenate([y for _, y in batches]), np.repeat(np.arange(6), 2))

    def test_model_serialization_and_prediction_use_same_pixels(self):
        from steel_defect.model import build_model
        from steel_defect.predict import predict_image
        import tensorflow as tf

        config = TrainConfig(image_size=32, batch_size=2)
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.bmp"
            Image.fromarray(np.arange(1024, dtype=np.uint8).reshape(32, 32)).save(image)
            model = build_model(config)
            pixels = load_image(image, 32)
            dataset = make_dataset([{"path": "image.bmp", "label": 0}], folder, config)
            np.testing.assert_array_equal(next(iter(dataset))[0].numpy()[0], pixels)
            expected = model(pixels[None], training=False).numpy()[0]
            np.testing.assert_allclose(expected.sum(), 1, atol=1e-6)
            saved = Path(folder) / "best.keras"
            model.save(saved)
            result = predict_image(saved, image)
            np.testing.assert_allclose(list(result["probabilities"].values()), expected, atol=1e-6)
            self.assertEqual(result["predicted_class"], CLASSES[int(expected.argmax())])
            # Random augmentation must be inactive in inference mode.
            np.testing.assert_array_equal(model(pixels[None], training=False), model(pixels[None], training=False))
            tf.keras.backend.clear_session()

    def test_metrics_are_computed_from_predictions(self):
        from steel_defect.evaluate import compute_metrics
        true = np.array([0, 0, 1, 1, 2, 3, 4, 5])
        predicted = np.array([0, 1, 1, 1, 2, 3, 4, 5])
        metrics, report, matrix = compute_metrics(true, predicted)
        self.assertEqual(metrics["accuracy"], 7 / 8)
        self.assertEqual(matrix.sum(), len(true))
        self.assertEqual(matrix[0, 1], 1)
        self.assertEqual(report["crazing"]["recall"], 0.5)
        self.assertGreater(metrics["macro_f1"], 0)


if __name__ == "__main__":
    unittest.main()
