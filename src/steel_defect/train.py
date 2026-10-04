"""Fit only training images; use validation loss for all training decisions."""
import argparse
import hashlib
import importlib.metadata
import platform
import shutil
from pathlib import Path

from .config import CLASSES, DEFAULT_DATA_DIR, DEFAULT_MANIFEST, DEFAULT_RUN_DIR, TrainConfig
from .data import make_dataset, read_manifest, split_counts, validate_manifest
from .utils import cli_guard, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Use a new directory per experiment.")
    for key in ("seed", "image_size", "batch_size", "epochs", "patience"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=getattr(TrainConfig(), key))
    for key in ("learning_rate", "dropout"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=getattr(TrainConfig(), key))
    args = parser.parse_args()
    config = TrainConfig(**{key: getattr(args, key) for key in
                           ("seed", "image_size", "batch_size", "epochs", "patience", "learning_rate", "dropout")})
    if (args.run_dir / "metrics" / "run_config.json").exists() or (args.run_dir / "models" / "best.keras").exists():
        raise FileExistsError(f"Training run exists: {args.run_dir}. Choose a new --run-dir; runs are not overwritten.")
    rows = read_manifest(args.manifest)
    validate_manifest(rows, args.data_dir)
    # Checking integrity is not fitting: test pixels are never supplied to model.fit.
    training_rows = [r for r in rows if r["split"] == "train"]
    validation_rows = [r for r in rows if r["split"] == "val"]

    import tensorflow as tf
    from .model import build_model
    from .visualize import plot_eda, plot_history

    tf.keras.utils.set_random_seed(config.seed)
    tf.config.experimental.enable_op_determinism()
    for name in ("figures", "metrics", "models"):
        (args.run_dir / name).mkdir(parents=True, exist_ok=True)
    metrics_dir, model_dir = args.run_dir / "metrics", args.run_dir / "models"
    manifest_copy = metrics_dir / "manifest.csv"
    shutil.copyfile(args.manifest, manifest_copy)
    summary = args.manifest.with_suffix(".json")
    if summary.exists():
        shutil.copyfile(summary, metrics_dir / "dataset_summary.json")
    run_config = {**config.to_dict(), "data_dir": str(args.data_dir.resolve()), "classes": CLASSES,
                  "manifest_sha256": hashlib.sha256(manifest_copy.read_bytes()).hexdigest(),
                  "counts": split_counts(rows), "status": "training"}
    save_json(metrics_dir / "run_config.json", run_config)
    save_json(metrics_dir / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "devices": [str(device) for device in tf.config.list_physical_devices()],
    })
    model = build_model(config)
    model.summary()
    summary_lines = []
    model.summary(print_fn=lambda line: summary_lines.append(line))
    (model_dir / "architecture.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    save_json(model_dir / "metadata.json", {"classes": CLASSES, "image_size": config.image_size,
                                            "input_range": [0, 255], "channels": 1})
    plot_eda(rows, args.data_dir, args.run_dir / "figures", config.seed)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_dir / "best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(metrics_dir / "history.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(make_dataset(training_rows, args.data_dir, config, training=True),
                        validation_data=make_dataset(validation_rows, args.data_dir, config),
                        epochs=config.epochs, callbacks=callbacks, verbose=2, shuffle=False)
    numeric_history = {key: [float(value) for value in values] for key, values in history.history.items()}
    save_json(metrics_dir / "history.json", numeric_history)
    plot_history(numeric_history, args.run_dir / "figures")
    run_config.update(status="trained", epochs_completed=len(history.epoch),
                      best_epoch=min(range(len(history.epoch)), key=lambda i: numeric_history["val_loss"][i]) + 1)
    save_json(metrics_dir / "run_config.json", run_config)
    print(f"Best validation model: {model_dir / 'best.keras'}. Test set has not been evaluated.")


if __name__ == "__main__":
    cli_guard(main)
