"""Load the saved CNN and classify one image without retraining."""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import CLASSES, DEFAULT_RUN_DIR
from .data import load_image
from .utils import cli_guard, save_json


def load_trained_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model missing: {path}. Run python -m steel_defect.train first, "
                                "or pass --model with an existing best.keras.")
    import tensorflow as tf
    model = tf.keras.models.load_model(path, compile=False)
    shape = model.input_shape
    if len(shape) != 4 or shape[-1] != 1 or shape[1] != shape[2] or model.output_shape[-1] != len(CLASSES):
        raise ValueError("Expected a square grayscale CNN with six output classes.")
    metadata = path.parent / "metadata.json"
    if metadata.exists() and json.loads(metadata.read_text(encoding="utf-8"))["classes"] != list(CLASSES):
        raise ValueError("Saved class order does not match this project.")
    return model, int(shape[1])


def predict_image(model_path, image_path):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image missing: {image_path}")
    model, image_size = load_trained_model(model_path)
    try:
        pixels = load_image(image_path, image_size)
    except OSError as error:
        raise ValueError(f"Cannot read image {image_path}: {error}") from error
    probabilities = model(pixels[None], training=False).numpy()[0]
    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned non-finite probabilities.")
    index = int(probabilities.argmax())
    return {"image": str(image_path), "predicted_class": CLASSES[index],
            "confidence": float(probabilities[index]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASSES)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to one steel image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_RUN_DIR / "models" / "best.keras")
    parser.add_argument("--output", type=Path, help="Optional JSON prediction file.")
    args = parser.parse_args()
    result = predict_image(args.model, args.image)
    print(json.dumps(result, indent=2))
    if args.output:
        save_json(args.output, result)


if __name__ == "__main__":
    cli_guard(main)
