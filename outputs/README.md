# Generated outputs

These files are generated locally and ignored by Git. The default academic run
uses this directory. Alternate experiments use `--run-dir outputs/<name>`.

- `figures/`: EDA, accuracy/loss curves, confusion matrices, test prediction grid.
- `metrics/`: frozen manifest, dataset counts/summary, run configuration,
  environment, history, classification reports and test predictions.
- `models/`: `best.keras`, input/class metadata and readable architecture summary.

`outputs/smoke/` contains **superseded non-final development artifacts** from
the former 1,499 / 30 / 270 packaging-based split. Its manifest has been renamed
`manifest.invalid.csv` and its run status is `superseded_development`.
Its model, scores and plots must not be reused for the academic run.

The final manifest pools 1,799 unique images into a stratified 80/10/10 split:
1,439 training / 180 validation / 180 held-out test images. EDA in `figures/`
and `metrics/dataset_counts.json` now describes this final protocol. Full
128 × 128 training exposed a BatchNormalization inference-statistic failure.
The original run is marked `invalidated_validation_failure`; its checkpoint is
preserved for debugging and must not be used for final reporting or evaluation.
`validation_audit/` contains the training/validation audit.
`validation_diagnostic/` contains the unsuccessful eight-epoch momentum=0.9
trial. The original momentum was restored.
`validation_calibrated_diagnostic/` contains the five-epoch diagnostic with
training-only population moments refreshed before validation. Diagnostic runs
are marked `diagnostic_only` and are not eligible for final evaluation.
No held-out test images were opened or evaluated in this investigation.
Start final training from fresh weights in a new directory, for example
`python -m steel_defect.train --run-dir outputs/final_clean`.
Use validation only for model-selection decisions.

Training refuses to overwrite an existing run configuration or model. Choose a
new run directory for a new experiment. EDA and evaluation can regenerate their
figures and metrics. Running evaluation again does not turn the test set into a
validation set: keep all future model selection independent of those test scores.
