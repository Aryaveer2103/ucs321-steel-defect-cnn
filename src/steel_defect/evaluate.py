"""Evaluate the selected model on the frozen held-out test partition."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from .config import CLASSES, DEFAULT_RUN_DIR, TrainConfig
from .data import make_dataset, read_manifest, validate_manifest
from .predict import load_trained_model
from .utils import cli_guard, save_json


def compute_metrics(true, predicted):
    labels = list(range(len(CLASSES)))
    precision, recall, f1, _ = precision_recall_fscore_support(true, predicted, labels=labels,
                                                             average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(true, predicted, labels=labels,
                                                         average="weighted", zero_division=0)
    metrics = {"accuracy": float(accuracy_score(true, predicted)), "macro_precision": float(precision),
               "macro_recall": float(recall), "macro_f1": float(f1), "weighted_f1": float(weighted_f1),
               "test_samples": int(len(true))}
    report = classification_report(true, predicted, labels=labels, target_names=CLASSES,
                                   output_dict=True, zero_division=0)
    return metrics, report, confusion_matrix(true, predicted, labels=labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-dir", type=Path, help="Override the saved root if the dataset was moved.")
    args = parser.parse_args()
    metrics_dir = args.run_dir / "metrics"
    config_path = metrics_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config missing: {config_path}. Train this run first.")
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    if saved["status"] != "trained":
        raise ValueError("This run did not finish training; choose a completed run.")
    manifest = metrics_dir / "manifest.csv"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != saved["manifest_sha256"]:
        raise ValueError("Frozen run manifest changed. Restore the original before evaluation.")
    root = args.data_dir or Path(saved["data_dir"])
    rows = read_manifest(manifest)
    validate_manifest(rows, root)
    test_rows = [row for row in rows if row["split"] == "test"]
    config = TrainConfig(**{key: saved[key] for key in TrainConfig.__dataclass_fields__})
    model_path = args.run_dir / "models" / "best.keras"
    model, image_size = load_trained_model(model_path)
    if image_size != config.image_size:
        raise ValueError("Model input resolution disagrees with saved training config.")
    probabilities = model.predict(make_dataset(test_rows, root, config), verbose=0)
    if probabilities.shape != (len(test_rows), len(CLASSES)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid model probabilities; evaluation aborted.")
    true = np.array([row["label"] for row in test_rows])
    predicted = probabilities.argmax(axis=1)
    metrics, report, matrix = compute_metrics(true, predicted)
    metrics.update(test_cross_entropy=float(-np.log(np.clip(probabilities[np.arange(len(true)), true], 1e-7, 1)).mean()),
                   model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                   manifest_sha256=saved["manifest_sha256"], selected_epoch=saved["best_epoch"])
    save_json(metrics_dir / "test_metrics.json", metrics)
    save_json(metrics_dir / "classification_report.json", report)
    (metrics_dir / "classification_report.txt").write_text(
        classification_report(true, predicted, labels=range(6), target_names=CLASSES, digits=4, zero_division=0),
        encoding="utf-8")
    with (metrics_dir / "classification_report.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])
        for name, values in report.items():
            if isinstance(values, dict):
                writer.writerow([name] + [values[key] for key in ("precision", "recall", "f1-score", "support")])
    with (metrics_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_class / predicted_class", *CLASSES])
        for name, values in zip(CLASSES, matrix):
            writer.writerow([name, *values])
    with (metrics_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "true_class", "predicted_class", "confidence", *[f"prob_{n}" for n in CLASSES]])
        for row, prediction, probs in zip(test_rows, predicted, probabilities):
            writer.writerow([row["path"], row["class"], CLASSES[prediction], float(probs[prediction]), *probs.tolist()])
    from .visualize import plot_confusion, plot_predictions
    figures = args.run_dir / "figures"
    plot_confusion(matrix, figures / "confusion_matrix.png")
    plot_confusion(matrix, figures / "confusion_matrix_normalized.png", normalized=True)
    plot_predictions(test_rows, root, probabilities, figures / "test_predictions.png", config.seed)
    print(json.dumps(metrics, indent=2))
    print(f"Held-out results and figures saved to {args.run_dir}. Do not tune against these results.")


if __name__ == "__main__":
    cli_guard(main)
