"""Verify model serialization, consistent preprocessing and metric calculations."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from steel_defect.config import CLASSES, TrainConfig
from steel_defect.data import load_image, make_dataset


class PipelineTests(unittest.TestCase):
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
