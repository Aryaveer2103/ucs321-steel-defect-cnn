"""Audit training/validation only; never resolve or open held-out test images."""
import itertools
import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

from steel_defect.config import CLASSES, TrainConfig
from steel_defect.data import canonical_class, image_info, make_dataset, read_manifest, resolve_image
from steel_defect.utils import save_json


def save_validation_examples(rows, root, output):
    """Display two validation examples per class alongside their source names."""
    sheet = Image.new("RGB", (6 * 220, 2 * 220), "white")
    draw = ImageDraw.Draw(sheet)
    for column, name in enumerate(CLASSES):
        examples = [r for r in rows if r["split"] == "val" and r["class"] == name][:2]
        for index, row in enumerate(examples):
            x, y = column * 220, index * 220
            with Image.open(resolve_image(root, row["path"])) as image:
                sheet.paste(image.convert("RGB").resize((200, 180)), (x, y))
            draw.text((x + 2, y + 183), f"label {row['label']}: {name}\n{Path(row['path']).name}", fill="black")
    sheet.save(output / "validation_examples.png")


def main():
    output = Path("outputs/validation_audit")
    output.mkdir(parents=True, exist_ok=True)
    run = json.loads(Path("outputs/metrics/run_config.json").read_text())
    config = TrainConfig(**{key: run[key] for key in TrainConfig.__dataclass_fields__})
    root = Path(run["data_dir"])
    rows = [r for r in read_manifest("data/processed/manifest.csv") if r["split"] in ("train", "val")]
    allowed = {resolve_image(root, r["path"]) for r in rows}
    original_open = Image.open

    def guarded_open(path, *args, **kwargs):
        if Path(path).resolve() not in allowed:
            raise AssertionError(f"Audit attempted to open a non-training/validation image: {path}")
        return original_open(path, *args, **kwargs)

    Image.open = guarded_open
    save_validation_examples(rows, root, output)
    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    report = {"test_images_opened": 0, "classes": list(CLASSES)}
    datasets = {}
    for split in ("train", "val"):
        subset = [r for r in rows if r["split"] == split]
        for r in subset:
            assert CLASSES[r["label"]] == r["class"]
            assert canonical_class(re.sub(r"[_\-]?\d+$", "", Path(r["path"]).stem)) == r["class"]
            assert image_info(resolve_image(root, r["path"])) == (r["width"], r["height"], r["sha256"])
        dataset = make_dataset(subset, root, config, training=split == "train")
        datasets[split] = dataset
        batches = list(dataset.as_numpy_iterator())
        labels = np.concatenate([y for _, y in batches])
        pixels = np.concatenate([x for x, _ in batches])
        info = {
            "count": len(subset), "manifest_first_64_labels": [r["label"] for r in subset[:64]],
            "manifest_class_runs": [(int(label), len(list(group))) for label, group in itertools.groupby(r["label"] for r in subset)],
            "first_8_batch_class_counts": [np.bincount(y, minlength=6).tolist() for _, y in batches[:8]],
            "shape": list(pixels.shape), "dtype": str(pixels.dtype),
            "range": [float(pixels.min()), float(pixels.max())], "mean": float(pixels.mean()),
            "finite": bool(np.isfinite(pixels).all()), "labels_match_filenames_and_hashes": True,
        }
        second_labels = np.concatenate([y.numpy() for _, y in dataset])
        info["next_epoch_same_label_order"] = bool(np.array_equal(labels, second_labels))
        report[split] = info
        print(split, json.dumps(info), flush=True)
    model = tf.keras.models.load_model("outputs/models/best.keras")
    augmentation = model.get_layer("training_only_augmentation")
    x, _ = next(iter(datasets["val"]))
    scaled = model.get_layer("pixel_scaling")(x)
    assert np.array_equal(augmentation(scaled, training=False).numpy(), scaled.numpy())
    report["validation_augmentation_is_identity"] = True
    y_true = np.array([r["label"] for r in rows if r["split"] == "val"])

    def score():
        probabilities = model.predict(datasets["val"], verbose=0)
        predictions = probabilities.argmax(axis=1)
        return {"accuracy": float((predictions == y_true).mean()),
                "loss": float(-np.log(np.maximum(probabilities[np.arange(len(y_true)), y_true], 1e-7)).mean()),
                "predicted_class_counts": dict(zip(CLASSES, np.bincount(predictions, minlength=6).tolist()))}

    report["saved_checkpoint_validation"] = score()
    print("Saved checkpoint validation:", report["saved_checkpoint_validation"], flush=True)
    # A causal diagnostic: hold every learned weight fixed and replace only BN
    # moving moments with population moments from TRAINING images. No augmentation,
    # dropout, gradients, or validation statistics are involved in calibration.
    learned_weights = [w.numpy().copy() for w in model.trainable_weights]
    training_rows = [r for r in rows if r["split"] == "train"]
    calibration = make_dataset(training_rows, root, config)
    report["batch_normalization"] = []
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            continue
        probe = tf.keras.Model(model.inputs, layer.input)
        total = 0
        sums = np.zeros(layer.moving_mean.shape, dtype=np.float64)
        squares = np.zeros_like(sums)
        for pixels, _ in calibration:
            activations = probe(pixels, training=False).numpy().astype(np.float64)
            total += np.prod(activations.shape[:-1])
            sums += activations.sum(axis=(0, 1, 2))
            squares += np.square(activations).sum(axis=(0, 1, 2))
        mean = sums / total
        variance = np.maximum(squares / total - mean ** 2, 0)
        info = {"name": layer.name, "momentum": layer.momentum,
                "old_mean_range": [float(layer.moving_mean.numpy().min()), float(layer.moving_mean.numpy().max())],
                "old_variance_range": [float(layer.moving_variance.numpy().min()), float(layer.moving_variance.numpy().max())],
                "training_mean_range": [float(mean.min()), float(mean.max())],
                "training_variance_range": [float(variance.min()), float(variance.max())]}
        layer.moving_mean.assign(mean)
        layer.moving_variance.assign(variance)
        report["batch_normalization"].append(info)
        print("BN diagnostic:", info, flush=True)
    for before, after in zip(learned_weights, model.trainable_weights):
        np.testing.assert_array_equal(before, after.numpy())
    report["training_moments_only_validation"] = score()
    print("After training-only BN moments (learned weights unchanged):", report["training_moments_only_validation"], flush=True)
    report["original_checkpoint_modified"] = False
    save_json(output / "audit.json", report)


if __name__ == "__main__":
    main()
