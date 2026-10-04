"""Download the pinned public mirror (optional), inspect images and save splits."""
import argparse
import hashlib
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from steel_defect.config import (DEFAULT_DATA_DIR, DEFAULT_MANIFEST, DEFAULT_SPLIT_SEED,
                                 DEFAULT_VAL_FRACTION, DEFAULT_TEST_FRACTION)
from steel_defect.data import prepare_manifest, read_manifest, split_counts, validate_manifest
from steel_defect.utils import cli_guard, save_json

DOWNLOAD_URL = "https://ndownloader.figshare.com/files/54094775"
ARCHIVE_MD5 = "19af5a1250d8ab0cb0828d31ee2b349d"
ARCHIVE_SIZE = 27316154


def download_dataset(destination):
    destination = Path(destination)
    if destination.exists():
        print(f"Using existing dataset: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "NEU-CLS.zip"
    if not archive.exists():
        print(f"Downloading {ARCHIVE_SIZE / 1e6:.1f} MB from the versioned Figshare mirror...")
        partial = archive.with_suffix(".zip.part")
        try:
            request = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "UCS321-steel-defect/0.1"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            partial.rename(archive)
        except (OSError, urllib.error.URLError) as error:
            raise ValueError(f"Download failed: {error}. Follow data/README.md for manual setup.") from error
    if archive.stat().st_size != ARCHIVE_SIZE or hashlib.md5(archive.read_bytes()).hexdigest() != ARCHIVE_MD5:
        raise ValueError(f"Archive checksum/size mismatch: {archive}. Remove this archive explicitly "
                         "and retry, or use the official manual source.")
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        staging = Path(staging)
        with zipfile.ZipFile(archive) as compressed:
            for entry in compressed.infolist():
                target = (staging / entry.filename).resolve()
                if not target.is_relative_to(staging.resolve()) or entry.file_size > 100_000_000:
                    raise ValueError(f"Unsafe archive entry: {entry.filename}")
            compressed.extractall(staging)
        staging.rename(destination)
        # TemporaryDirectory cleanup tolerates the moved staging path.
    save_json(destination.parent / "download_source.json", {
        "source": "https://doi.org/10.6084/m9.figshare.28903550.v1", "url": DOWNLOAD_URL,
        "md5": ARCHIVE_MD5, "size_bytes": ARCHIVE_SIZE,
        "note": "Third-party NEU image mirror with detection labels and a train/valid split; "
                "not the original official NEU-CLS archive. Classification uses filename labels.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download a checksum-pinned public Figshare mirror.")
    parser.add_argument("--deduplicate", action="store_true", help="Explicitly exclude exact same-class pixel copies; record exclusions.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Actual extracted image root.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--regenerate", action="store_true", help="Invalidate/archive an existing manifest and explicitly create a new split.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    args = parser.parse_args()
    if args.download:
        download_dataset(args.data_dir)
    if args.regenerate and args.manifest.exists():
        previous = [args.manifest, args.manifest.with_suffix(".json")]
        archives = [(path, path.with_name(f"{path.stem}.invalid{path.suffix}")) for path in previous if path.exists()]
        if any(target.exists() for _, target in archives):
            raise FileExistsError("An invalidated-manifest archive already exists. Choose a new --manifest path.")
        for path, target in archives:
            path.rename(target)
            print(f"Invalidated previous manifest file: {target}")
    if args.manifest.exists():
        rows = read_manifest(args.manifest)
        validate_manifest(rows, args.data_dir)
        print(f"Reusing frozen final manifest: {args.manifest}. Split flags do not change an existing manifest.")
    else:
        rows = prepare_manifest(args.data_dir, args.manifest, args.seed, args.val_fraction, args.test_fraction,
                                deduplicate=args.deduplicate)
        print(f"Created {args.manifest}")
    print("Class                 Train   Val  Test")
    counts = split_counts(rows)
    for name in counts["train"]:
        print(f"{name:22} {counts['train'][name]:5} {counts['val'][name]:5} {counts['test'][name]:5}")
    print(f"{'TOTAL':22} {sum(counts['train'].values()):5} {sum(counts['val'].values()):5} {sum(counts['test'].values()):5}")
    print("Final classification split: pooled stratification; source train/valid folder boundaries ignored.")
    print(f"Total images: {len(rows)}. Exact decoded-pixel duplicates: 0.")


if __name__ == "__main__":
    cli_guard(main)
