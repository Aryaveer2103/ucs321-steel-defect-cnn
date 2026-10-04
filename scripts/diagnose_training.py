"""Run a short fresh training diagnostic with a held-out image access guard."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from steel_defect.config import CLASSES, DEFAULT_DATA_DIR, DEFAULT_MANIFEST, TrainConfig
from steel_defect.data import make_dataset, read_manifest
from steel_defect.train import main as train_main
from steel_defect.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/validation_diagnostic"))
    args = parser.parse_args()
    if not 5 <= args.epochs <= 10:
        parser.error("Use 5-10 epochs for this diagnostic; final training is a separate command.")
    root = DEFAULT_DATA_DIR.resolve()
    rows = [r for r in read_manifest(DEFAULT_MANIFEST) if r["split"] in ("train", "val")]
    allowed = {(root / r["path"]).resolve(): r["split"] for r in rows}
    opened = set()
    original_open, original_argv = Image.open, sys.argv

    def guarded_open(path, *positional, **keywords):
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            if resolved.is_relative_to(root):
                if resolved not in allowed:
                    raise AssertionError("Diagnostic attempted to open a held-out/unlisted image")
                opened.add(resolved)
        return original_open(path, *positional, **keywords)

    Image.open = guarded_open
    try:
        sys.argv = ["steel_defect.train", "--epochs", str(args.epochs), "--run-dir", str(args.run_dir)]
        train_main()
        import tensorflow as tf
        model = tf.keras.models.load_model(args.run_dir / "models/best.keras")
        validation = [r for r in rows if r["split"] == "val"]
        probabilities = model.predict(make_dataset(validation, root, TrainConfig()), verbose=0)
        predicted = probabilities.argmax(axis=1)
        true = np.array([r["label"] for r in validation])
        report = {"purpose": "Short validation diagnostic; not final academic training",
                  "test_images_opened": 0, "unique_images_opened": dict(Counter(allowed[p] for p in opened)),
                  "validation_accuracy": float((predicted == true).mean()),
                  "validation_loss": float(-np.log(np.maximum(probabilities[np.arange(len(true)), true], 1e-7)).mean()),
                  "validation_predicted_class_counts": dict(zip(CLASSES, np.bincount(predicted, minlength=6).tolist()))}
        save_json(args.run_dir / "metrics/diagnostic.json", report)
        run_path = args.run_dir / "metrics/run_config.json"
        run_config = json.loads(run_path.read_text())
        run_config.update(status="diagnostic_only", diagnostic_reason=report["purpose"])
        save_json(run_path, run_config)
        print(json.dumps(report, indent=2), flush=True)
    finally:
        Image.open, sys.argv = original_open, original_argv


if __name__ == "__main__":
    main()
