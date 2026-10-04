# Dataset setup and provenance

Images, archives and split manifests belong here and are excluded from Git.
Do not upload the dataset to the project repository.

## Original source

The dataset is the Northeastern University **NEU Surface Defect Database**, by
Kechen Song and Yunhui Yan. The official page distinguishes NEU-CLS
(classification images) from NEU-DET (images plus localization annotations):

- [Official NEU page](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm)
- [Official NEU-CLS archive on Google Drive](https://drive.google.com/file/d/1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc/view)

The official database description specifies 1,800 grayscale images, six classes,
300 images per class and 200 × 200 pixels. It is a benchmark of cropped steel
surface defects, not full-resolution Tata Steel production photographs.

## Reproducible automatic option

After installing the project, run from the repository root:

```powershell
python scripts/prepare_data.py --download --deduplicate
```

This downloads the public **third-party mirror** uploaded by Weilin Cao:

- [Figshare record, version 1](https://doi.org/10.6084/m9.figshare.28903550.v1)
- File ID: `54094775`, file: `NEU-CLS.zip`
- Direct file URL: `https://ndownloader.figshare.com/files/54094775`
- Size: `27,316,154` bytes
- MD5: `19af5a1250d8ab0cb0828d31ee2b349d`

The mirror record lists CC BY 4.0. This is the uploader's license statement;
it does not establish separate licensing terms for the original NEU archive.
Cite both the original dataset and the mirror when using this route.

**The archive was actually inspected.** Despite its NEU-CLS title, it contains
JPEG images and YOLO-format detection label files, with this layout:

```text
data/raw/NEU-CLS/
├── train/train/images/        # 1,770 JPEGs
├── train/train/labels/        # detection annotations (unused)
└── valid/valid/
    ├── images/               # 30 JPEGs
    └── labels/               # detection annotations (unused)
```

Class labels come from complete filename prefixes such as `pitted_surface_1.jpg`.
The detection annotations are not needed for image classification. This is a
NEU-derived distribution, not a claim of byte identity with the official archive.
There are 300 filenames per class, and all images decode at 200 × 200 pixels.
`patches_101.jpg` and `patches_105.jpg` have identical decoded grayscale pixels.

`--deduplicate` explicitly excludes the latter from the manifest without deleting
either downloaded file. Every exclusion is recorded in `data/processed/manifest.json`.
Without this flag, preparation fails on the duplicate. Conflicting class labels
always fail, even with deduplication. Deduplication takes place across the entire
pool before final splitting, including copies in different packaging folders.
The first lexicographically sorted path represents each same-class pixel hash.
Near-duplicate images and common production batches cannot be ruled out.

## Final classification split

The original Figshare train/valid packaging was inspected but **not retained**.
Its 30-image validation set (only five per class) is unsuitable for reliable
early stopping, best-checkpoint selection and learning-rate decisions. This
packaging is not a classification benchmark split that we are required to preserve.

Start with the **1,799 unique images** after exact-pixel deduplication. Pool them
across source folders, then create a deterministic **stratified 80/10/10**
train/validation/test split with seed **42**. No original folder boundary restricts
the final image assignment. Exact final counts are:

| Class | Train | Validation | Test | Distinct total |
| --- | ---: | ---: | ---: | ---: |
| crazing | 240 | 30 | 30 | 300 |
| inclusion | 240 | 30 | 30 | 300 |
| patches | 239 | 30 | 30 | 299 |
| pitted_surface | 240 | 30 | 30 | 300 |
| rolled-in_scale | 240 | 30 | 30 | 300 |
| scratches | 240 | 30 | 30 | 300 |
| **Total** | **1,439** | **180** | **180** | **1,799** |

Both held-out partitions target `round(0.10 × 1,799) = 180` images, apportioned
by class using stratification. Training gets the remainder. Class balance is
as close as possible: only patches has one fewer training image.
The seed is configurable with `--seed`, but the final split is fixed before
training. The final test set never influences augmentation, early stopping,
checkpoint selection, learning-rate decisions or any tuning. Validation alone
supports model selection. Do not replace this split after viewing test scores.

`manifest.json` records the source packaging counts, ignored-boundary policy,
seed, requested ratios, resulting counts and deduplication exclusions. Each CSV
row includes `split_method=pooled_stratified_v2`. Old manifests without this
version are rejected, including previous smoke manifests. The superseded CSV/JSON
are archived with `.invalid` in their filenames; they are development history.
Regeneration was performed with:

```powershell
python scripts/prepare_data.py --download --deduplicate --regenerate
```

Normal later invocations reuse the final frozen manifest. `--regenerate` is
explicit and refuses to overwrite an existing invalidated archive. Choose a new
manifest path for a deliberately separate protocol; never redraw the final split
based on model scores.

## Manual official option

1. Open the official page and choose **NEU-CLS**, rather than NEU-CLS-64 or NEU-DET.
2. Download the linked `NEU-CLS.rar` from Google Drive (or the official Baidu link).
3. Extract it locally with an archive tool. The project does not require a RAR library.
4. Pass the directory that actually contains the extracted image collection:

```powershell
python scripts/prepare_data.py --data-dir "C:/datasets/NEU-CLS" --manifest data/processed/official.csv --deduplicate
python scripts/run_eda.py --data-dir "C:/datasets/NEU-CLS" --manifest data/processed/official.csv --output-dir outputs/official
python -m steel_defect.train --data-dir "C:/datasets/NEU-CLS" --manifest data/processed/official.csv --run-dir outputs/official
python -m steel_defect.evaluate --run-dir outputs/official
```

No paths are guessed. Missing roots or ambiguous labels produce actionable errors.
Supported image types are BMP, PNG, JPEG and TIFF. Supported layouts include:

```text
dataset/crazing/*.bmp                       # six class folders, aliases accepted
dataset/Cr_1.bmp                           # original abbreviated filename labels
dataset/train/images/crazing_1.jpg          # filename labels inside split folders
dataset/train/crazing/*.jpg                 # class folders inside split folders
dataset/test/crazing/*.jpg
dataset/validation/crazing/*.jpg
```

These layouts affect image discovery and label extraction only. Source split
names do not define final partitions: all distinct images are pooled and
stratified 80% / 10% / 10%, with seed 42 by default. The algorithm first samples
the stratified test target, then the stratified validation target from the
remaining pool. Exact counts depend on rounding and deduplication; the saved
counts are authoritative. Every class must occur in each final partition.

## Dataset citation

K. Song and Y. Yan, “A noise robust method based on completed local binary
patterns for hot-rolled steel strip surface defects,” *Applied Surface Science*,
285, 858–864, 2013.

For the automatic mirror: Cao, Weilin (2025). *NEU-CLS*. Figshare. Dataset.
DOI: `10.6084/m9.figshare.28903550.v1`.
