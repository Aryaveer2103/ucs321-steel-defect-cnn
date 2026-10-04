"""Integrity tests use generated fixtures, never pretend they are NEU results."""
import tempfile
import csv
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from steel_defect.config import CLASSES
from steel_defect.data import assign_splits, prepare_manifest, read_manifest, validate_manifest


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        rng = np.random.default_rng(19)
        for name in CLASSES:
            folder = self.dataset / name
            folder.mkdir(parents=True)
            for index in range(20):
                Image.fromarray(rng.integers(0, 256, (16, 16), dtype=np.uint8)).save(folder / f"{index}.bmp")

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, output="manifest.csv"):
        return prepare_manifest(self.dataset, self.root / output, seed=42)

    def test_stratified_splits_are_deterministic_and_disjoint(self):
        rows = self.prepare()
        self.assertEqual(rows, self.prepare("second.csv"))
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({r["sha256"] for r in rows}), 120)
        for name in CLASSES:
            self.assertEqual([sum(r["class"] == name and r["split"] == s for r in rows)
                              for s in ("train", "val", "test")], [16, 2, 2])

    def test_seed_is_configurable_without_changing_class_balance(self):
        first = self.prepare()
        second = prepare_manifest(self.dataset, self.root / "seed_7.csv", seed=7)
        self.assertNotEqual([r["split"] for r in first], [r["split"] for r in second])
        for name in CLASSES:
            self.assertEqual([sum(r["class"] == name and r["split"] == s for r in second)
                              for s in ("train", "val", "test")], [16, 2, 2])

    def test_1799_images_have_nearest_integer_balanced_80_10_10_counts(self):
        rows = [{"class": name, "label": label, "split": "train"}
                for label, name in enumerate(CLASSES)
                for _ in range(299 if name == "patches" else 300)]
        assign_splits(rows, seed=42)
        for name in CLASSES:
            self.assertEqual([sum(r["class"] == name and r["split"] == s for r in rows)
                              for s in ("train", "val", "test")],
                             [239 if name == "patches" else 240, 30, 30])

    def test_previous_manifest_schema_is_rejected(self):
        rows = self.prepare()
        fields = [key for key in rows[0] if key != "split_method"]
        legacy = self.root / "legacy.csv"
        with legacy.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "obsolete|Invalid manifest"):
            read_manifest(legacy)

    def test_duplicate_copy_across_packaging_boundaries_is_excluded_before_new_split(self):
        copy = self.dataset / "valid" / CLASSES[0] / "copy.bmp"
        copy.parent.mkdir(parents=True)
        Image.open(self.dataset / CLASSES[0] / "0.bmp").save(copy)
        rows = prepare_manifest(self.dataset, self.root / "pooled.csv", deduplicate=True)
        self.assertEqual(len(rows), 120)
        partitions = [{r["sha256"] for r in rows if r["split"] == s} for s in ("train", "val", "test")]
        self.assertFalse(partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])

    def test_summary_records_ignored_packaging_and_new_method(self):
        self.prepare()
        import json
        summary = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(summary["split_method"], "pooled_stratified_v2")
        self.assertEqual(summary["requested_train_fraction"], 0.8)
        self.assertFalse(summary["packaging_boundaries_retained"])

    def test_existing_manifest_is_not_silently_overwritten(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "exists"):
            self.prepare()

    def test_duplicate_pixels_are_rejected_before_splitting(self):
        source = self.dataset / CLASSES[0] / "0.bmp"
        Image.open(source).save(self.dataset / CLASSES[1] / "duplicate.png")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.prepare()

    def test_unreadable_images_fail_with_path(self):
        (self.dataset / CLASSES[0] / "broken.bmp").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "broken.bmp"):
            self.prepare()

    def test_explicit_deduplication_records_exclusions_without_deleting_images(self):
        source = self.dataset / CLASSES[0] / "0.bmp"
        Image.open(source).save(self.dataset / CLASSES[0] / "duplicate.png")
        rows = prepare_manifest(self.dataset, self.root / "dedup.csv", deduplicate=True)
        self.assertEqual(len(rows), 120)
        self.assertTrue((self.dataset / CLASSES[0] / "duplicate.png").exists())
        import json
        summary = json.loads((self.root / "dedup.json").read_text())
        self.assertEqual(len(summary["excluded_duplicates"]), 1)

    def test_deduplication_never_discards_conflicting_class_labels(self):
        Image.open(self.dataset / CLASSES[0] / "0.bmp").save(self.dataset / CLASSES[1] / "duplicate.png")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            prepare_manifest(self.dataset, self.root / "dedup.csv", deduplicate=True)

    def test_changed_image_is_detected(self):
        self.prepare()
        Image.new("L", (16, 16), 0).save(self.dataset / CLASSES[0] / "0.bmp")
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_manifest(read_manifest(self.root / "manifest.csv"), self.dataset)

    def test_parent_path_escape_is_rejected(self):
        rows = self.prepare()
        rows[0]["path"] = "../outside.bmp"
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_manifest(rows, self.dataset)

    def test_packaging_boundaries_are_ignored_for_final_classification_split(self):
        for folder in list(self.dataset.iterdir()):
            name = folder.name
            for image in list(folder.iterdir()):
                split = "test" if int(image.stem) >= 16 else "train"
                target = self.root / "presplit" / split / "images" / name
                target.mkdir(parents=True, exist_ok=True)
                image.rename(target / image.name)
        self.dataset = self.root / "presplit"
        rows = self.prepare()
        self.assertEqual(sum(r["split"] == "test" for r in rows), 12)
        self.assertEqual(sum(r["split"] == "val" for r in rows), 12)
        self.assertEqual(sum(r["split"] == "train" for r in rows), 96)
        self.assertTrue(any(r["path"].startswith("train/") for r in rows if r["split"] == "test"))


if __name__ == "__main__":
    unittest.main()
