# First-version design and verification plan

The user's detailed project brief is the specification. Work directly in the
empty `project1-build` branch, with no new worktree, commits or approval pauses.

1. Verify NEU provenance and inspect the real archive. Prefer the pinned public
   Figshare mirror; retain official manual download instructions.
2. Test dataset integrity and split isolation first. Implement readable-image
   checks, content hashes, supported layouts, deterministic stratified splits,
   and an immutable manifest. For final classification, pool unique images and
   ignore source packaging boundaries; stratify 80/10/10 with configurable seed.
3. Implement shared grayscale loading, a modest custom CNN and training-only
   augmentation. Save the best validation checkpoint, configuration, history,
   environment and the exact split manifest in a separate run directory.
4. Implement EDA, held-out evaluation, single-image inference and a matplotlib
   demo. Keep all figures and machine-readable results reproducible.
5. Document exact commands, academic scope and limitations. Verify syntax,
   CLI help, integrity tests, actual EDA and a short end-to-end training smoke
   run if dependencies and data are available. Smoke results are not final results.
6. Request one independent review of the complete implementation, repair
   material findings, inspect artifacts and Git status.

Key interfaces: manifest rows contain relative path, class, integer label,
decoded-pixel SHA-256, split and split-method version. Model accepts 128 x 128 x 1 float32 pixels in
the range 0–255 and embeds rescaling. Evaluation and inference load the same
saved model and class order. EDA sample grids use training images only; test
prediction examples are generated only during final evaluation.

## Implementation record

- Inspected the pinned archive rather than relying on its record title. It uses
  a train/valid detection-style layout, filename class labels and 200 x 200 JPEGs.
- Added an explicit deduplication flag after inspection found one exact pair.
  Source files remain intact; the exclusion is recorded. The initial development
  split retained mirror packaging (1,499 / 30 / 270); that protocol is superseded.
- Implemented all planned commands and figures. A real one-epoch 32 x 32 smoke
  run completed training, evaluation, inference and noninteractive demo export.
- All 11 integrity/pipeline tests passed, including an independent review run.
  CLI help, missing-data errors, module imports, archive HTTP access, saved
  prediction/metric agreement, PNG decoding and figure layouts were checked.
- Independent review reported no material findings. Git remains uncommitted.
  Full 128 x 128 training, the official RAR distribution and an interactive GUI
  window were not verified in the initial pass. Smoke results are not final results.

## Final splitting correction

The user requested a methodological correction before academic training.
The inspected Figshare packaging contains just 30 validation images, five per
class. It is not a required classification benchmark split and is unsuitable
for reliable early stopping, checkpoint selection and learning-rate decisions.

Pool the 1,799 exact-pixel-unique images and stratify 80/10/10 with seed 42.
Final counts are 1,439 training / 180 validation / 180 test. Every class has
30 validation and 30 test images; training has 240 per class except patches (239).
The test partition has no role in fitting or model-selection decisions.

The schema now marks `pooled_stratified_v2` in each manifest row. Legacy CSVs
are rejected and have been renamed to `.invalid.csv`, with the default old
summary archived too. The old smoke run is marked `superseded_development`.
Regenerate the manifest, run the complete tests and syntax/import checks, and
regenerate EDA. Do not train or evaluate a model during this correction.
