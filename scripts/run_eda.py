"""Generate class counts, training image examples and split figures."""
import argparse
from pathlib import Path

from steel_defect.config import DEFAULT_DATA_DIR, DEFAULT_MANIFEST, DEFAULT_RUN_DIR
from steel_defect.data import read_manifest, split_counts, validate_manifest
from steel_defect.utils import cli_guard, save_json
from steel_defect.visualize import plot_eda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    validate_manifest(rows, args.data_dir)
    plot_eda(rows, args.data_dir, args.output_dir / "figures", args.seed)
    save_json(args.output_dir / "metrics" / "dataset_counts.json", split_counts(rows))
    print(f"EDA figures: {args.output_dir / 'figures'} (sample grid uses training images only).")


if __name__ == "__main__":
    cli_guard(main)
