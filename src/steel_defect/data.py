"""Inspect images and freeze leakage-aware partitions before model fitting.

Support class folders, original NEU filename prefixes (Cr/In/Pa/PS/RS/Sc),
and packaged mirrors. Pool unique images regardless of source split folders.
Paths in manifests are relative to the data root.
"""
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from .config import CLASSES, DEFAULT_SPLIT_SEED, DEFAULT_VAL_FRACTION, DEFAULT_TEST_FRACTION, SPLIT_METHOD
from .utils import save_json

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
FIELDS = ("path", "class", "label", "sha256", "width", "height", "split", "split_method")
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}


def canonical_class(value):
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    aliases = {"cr": "crazing", "in": "inclusion", "pa": "patches",
               "ps": "pitted_surface", "rs": "rolled-in_scale", "sc": "scratches",
               "patch": "patches", "scratch": "scratches"}
    aliases.update({re.sub(r"[^a-z0-9]", "", name): name for name in CLASSES})
    return aliases.get(key)


def image_info(path):
    try:
        with Image.open(path) as image:
            image.load()  # Force decoding; verify() alone does not decode all pixels.
            gray = image.convert("L")
            width, height = gray.size
            digest = hashlib.sha256(f"{width}x{height}:".encode() + gray.tobytes()).hexdigest()
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise ValueError(f"Unreadable image {path}: {error}") from error
    return width, height, digest


def resolve_image(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Manifest path points outside dataset root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Image missing: {path}. Check --data-dir and the manifest.")
    return path


def inspect_dataset(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory missing: {root}. Run python scripts/prepare_data.py "
                                "--download or follow data/README.md and pass --data-dir.")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        relative = path.relative_to(root)
        folders = {canonical_class(part) for part in relative.parts[:-1]} - {None}
        prefix = canonical_class(re.sub(r"[_\-]?\d+$", "", path.stem))
        names = folders | ({prefix} if prefix else set())
        if len(names) != 1:
            raise ValueError(f"Cannot determine one class for {relative}. Use six class folders "
                             "or NEU filenames, e.g. crazing_1.jpg / Cr_1.bmp.")
        name = names.pop()
        splits = {SPLIT_NAMES[p.lower()] for p in relative.parts[:-1] if p.lower() in SPLIT_NAMES}
        if len(splits) > 1:
            raise ValueError(f"Conflicting split folders: {relative}")
        width, height, digest = image_info(path)
        rows.append(dict(path=relative.as_posix(), **{"class": name}, label=CLASSES.index(name),
                         sha256=digest, width=width, height=height,
                         split=next(iter(splits), "")))
    counts = Counter(r["class"] for r in rows)
    if set(counts) != set(CLASSES):
        raise ValueError(f"Expected all six classes. Found {dict(counts)} in {root}.")
    if any(counts[name] < 7 for name in CLASSES):
        raise ValueError("At least 7 distinct images per class are needed for three stratified splits.")
    return rows


def _partition(rows, count, seed):
    try:
        left, right = train_test_split(rows, test_size=count, random_state=seed,
                                       stratify=[r["label"] for r in rows])
    except ValueError as error:
        raise ValueError(f"Not enough images for stratified splits: {error}") from error
    return left, right


def assign_splits(rows, seed=DEFAULT_SPLIT_SEED, val_fraction=DEFAULT_VAL_FRACTION,
                  test_fraction=DEFAULT_TEST_FRACTION):
    """Stratify the complete unique pool; folder names are source metadata only.

    Round each held-out target against the full pool, then take validation from
    the remainder. sklearn's stratification apportions counts by class; the
    remainder is training. No training/model results influence this operation.
    """
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1.")
    pool = list(rows)
    for split, fraction in (("test", test_fraction), ("val", val_fraction)):
        pool, subset = _partition(pool, max(len(CLASSES), round(len(rows) * fraction)), seed)
        for row in subset:
            row["split"] = split
    for row in pool:
        row["split"] = "train"
    for row in rows:
        row["split_method"] = SPLIT_METHOD
    counts = Counter((r["split"], r["class"]) for r in rows)
    for split in ("train", "val", "test"):
        for name in CLASSES:
            if counts[split, name] == 0:
                raise ValueError(f"Split {split} has no {name} samples; use more images or larger split fractions.")


def split_counts(rows):
    return {split: {name: sum(r["split"] == split and r["class"] == name for r in rows)
                    for name in CLASSES} for split in ("train", "val", "test")}


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_manifest(root, manifest, seed=DEFAULT_SPLIT_SEED, val_fraction=DEFAULT_VAL_FRACTION,
                     test_fraction=DEFAULT_TEST_FRACTION, deduplicate=False):
    manifest = Path(manifest)
    if manifest.exists():
        raise ValueError(f"Manifest already exists: {manifest}. Reuse it, or choose a new --manifest "
                         "for a deliberately different experiment.")
    rows = inspect_dataset(root)
    raw_count = len(rows)
    source_counts = {split or "unsplit": {name: sum(r["split"] == split and r["class"] == name for r in rows)
                                         for name in CLASSES} for split in sorted({r["split"] for r in rows})}
    # Resolve same-class duplicates globally BEFORE assigning final partitions.
    # The lexicographically first path is a deterministic representative; no
    # source folder has priority because its packaging split is not retained.
    retained, excluded = {}, []
    for row in sorted(rows, key=lambda r: r["path"]):
        prior = retained.get(row["sha256"])
        if prior is None:
            retained[row["sha256"]] = row
            continue
        if prior["class"] != row["class"]:
            raise ValueError(f"Duplicate pixels have conflicting class labels: {prior['path']} and {row['path']}.")
        if not deduplicate:
            raise ValueError(f"Duplicate decoded pixels: {prior['path']} and {row['path']}. "
                             "Use --deduplicate to explicitly exclude copies and record them, "
                             "or resolve duplicates manually.")
        excluded.append({"excluded_path": row["path"], "retained_path": prior["path"], "sha256": row["sha256"]})
    rows = sorted(retained.values(), key=lambda r: r["path"])
    assign_splits(rows, seed, val_fraction, test_fraction)
    write_manifest(manifest, rows)
    save_json(manifest.with_suffix(".json"), {
        "data_dir": str(Path(root).resolve()), "seed": seed, "classes": CLASSES,
        "split_method": SPLIT_METHOD, "requested_train_fraction": 1 - (val_fraction + test_fraction),
        "requested_val_fraction": val_fraction, "requested_test_fraction": test_fraction,
        "packaging_boundaries_retained": False, "source_packaging_counts": source_counts,
        "split_rationale": "Pool unique images for classification; source train/valid packaging "
                           "is not a required classification benchmark split.",
        "counts": split_counts(rows),
        "image_sizes": dict(Counter(f'{r["width"]}x{r["height"]}' for r in rows)),
        "raw_image_count": raw_count, "excluded_duplicates": excluded,
        "duplicate_policy": "Exclude exact same-class copies globally before splitting; retain first sorted path."
                            if deduplicate else "Reject identical decoded grayscale pixels.",
    })
    return rows


def read_manifest(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest missing: {path}. Run python scripts/prepare_data.py first.")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Invalid manifest columns or obsolete split methodology in {path}; "
                             "recreate using prepare_data.py --deduplicate --regenerate.")
        rows = list(reader)
    for row in rows:
        if row["split_method"] != SPLIT_METHOD:
            raise ValueError(f"Manifest uses obsolete split methodology: {path}. Regenerate it.")
        for key in ("label", "width", "height"):
            row[key] = int(row[key])
    return rows


def validate_manifest(rows, root, content_splits=("train", "val", "test")):
    """Check global metadata; open pixels only for explicitly selected splits.

    Training selects train/val to keep held-out images untouched. The default
    full content check remains available for preparation and final evaluation.
    """
    paths, hashes = set(), set()
    for row in rows:
        if row.get("split_method") != SPLIT_METHOD:
            raise ValueError("Manifest uses obsolete split methodology; regenerate it before use.")
        if row["class"] not in CLASSES or row["label"] != CLASSES.index(row["class"]):
            raise ValueError("Manifest class/label mapping is inconsistent.")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {row['split']}")
        path = (Path(root).resolve() / row["path"]).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            raise ValueError(f"Manifest path points outside dataset root: {row['path']}")
        if str(path) in paths or row["sha256"] in hashes:
            raise ValueError("Duplicate path or pixels in manifest; possible split leakage.")
        paths.add(str(path))
        hashes.add(row["sha256"])
        if row["split"] in content_splits:
            path = resolve_image(root, row["path"])
            if image_info(path) != (row["width"], row["height"], row["sha256"]):
                raise ValueError(f"Image content changed since preparation: {row['path']}")
    counts = split_counts(rows)
    if any(value == 0 for group in counts.values() for value in group.values()):
        raise ValueError("Every split must contain every class.")


def load_image(path, image_size):
    """Shared training/inference preprocessing; scaling lives inside the model."""
    with Image.open(path) as image:
        gray = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float32)[..., np.newaxis]


def make_dataset(rows, root, config, training=False):
    import tensorflow as tf

    paths = [str(resolve_image(root, row["path"])) for row in rows]
    labels = [row["label"] for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def decode(path, label):
        def read(value):
            return load_image(value.decode("utf-8"), config.image_size)
        pixels = tf.numpy_function(read, [path], tf.float32)
        pixels.set_shape((config.image_size, config.image_size, 1))
        return pixels, label

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).cache()
    if training:
        dataset = dataset.shuffle(len(rows), seed=config.seed, reshuffle_each_iteration=True)
    return dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
