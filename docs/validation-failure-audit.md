# Validation failure audit — 2026-10-04

## Scope and final outcome

An earlier development run had unreliable BatchNormalization (BN) inference
statistics. The investigation used training and validation images only. It did
not open or evaluate held-out test images, redraw the frozen split, or change the
CNN architecture. The final correction refreshes BN population moments from
unaugmented training images before each validation pass.

The subsequent completed run is `outputs/final`: 50 epochs, selected checkpoint
at epoch 45. Independent checkpoint reload before test evaluation reproduced
validation accuracy 97.7778% (176/180) and loss 0.0841048211. Final held-out test
accuracy is 97.78%; the definitive results are in the root README and final metrics.
Development diagnostic scores below are historical evidence, not final results.

## Pipeline investigation

- The manifest is class-grouped, but the actual `tf.data` training pipeline caches
  originals, shuffles all 1,439 image/label pairs with seed 42, then batches.
  Epoch order changes and each of the first eight observed batches contained
  all six classes. Class-blocked training was ruled out.
- Class indices, filenames, pixel hashes, dimensions and image/label pairing
  were consistent across all 1,619 training/validation images. Twelve validation
  thumbnails agreed visually with their labels.
- Both splits used the same Pillow grayscale/bilinear loader and float32 pixels
  in [0,255]. The model supplies the only rescaling layer. Augmentation and
  dropout are inactive during inference.
- The Keras fit `shuffle` argument is ignored for Dataset inputs; the redundant
  argument was removed. Dataset shuffling determines training order.
- BN momentum remained 0.99. After 45 first-epoch updates, `0.99**45 = 0.6362`
  of the initial variance remains. The first-layer moving variance was
  0.6362–0.6394 while current unaugmented training population variance was
  0.000118–0.009165. Deeper layers also showed mismatched moments.

See [Keras training APIs](https://keras.io/api/models/model_training_apis/) and
[BN behavior](https://keras.io/api/layers/normalization_layers/batch_normalization/)
for the distinction between training batch statistics and inference moving moments.

## Causal check and rejected alternative

The invalid development checkpoint (epoch 1 of a nine-epoch run) had validation
accuracy 21.67% and loss 1.9418, predicting inclusion for 170 images and
pitted_surface for ten. Holding every learned weight fixed and replacing only
BN moving moments with population moments from training images raised validation
accuracy to 73.89% and reduced loss to 0.7688. Class prediction counts became
[34,45,25,23,28,25]. The original checkpoint was not overwritten.

An eight-epoch momentum=0.9 diagnostic remained unstable: validation accuracy
ranged from 28.89% to 66.11% and ended at 37.22%. That change was reverted.
The final architecture, momentum, batch size, augmentation, dropout, learning-rate
defaults and callback hyperparameters retain the original design.

## Implemented correction

`TrainingBatchNormCalibration` in `src/steel_defect/normalization.py` runs immediately
before fit's validation loop. Its separate training dataset is unshuffled.
Inference-mode probes disable augmentation and dropout. BN layers are calibrated
in order so deeper probes see already-calibrated earlier layers. Moments are
weighted by the actual activation count, including the smaller last batch.
Only moving means and variances change; learned weights and optimizer state do not.

Checkpointing, early stopping and learning-rate scheduling then consume corrected
validation scores. No validation or test pixels compute population moments.
The saved checkpoint contains these statistics, so prediction needs no calibration.

A five-epoch diagnostic produced validation accuracies 73.89%, 77.78%, 80.56%,
77.78% and 82.78%; the reloaded epoch-5 checkpoint reproduced loss 0.4246.
Its image-access guard recorded 1,439 training images, 180 validation images and
zero held-out images. Diagnostic runs were marked `diagnostic_only`, so the
evaluation CLI rejects them. The original invalid run was marked
`invalidated_validation_failure`.

## Evidence and preservation

The diagnosis and correction are preserved in commit `4c89dfa`.
Its historical scripts `scripts/audit_validation.py` and
`scripts/diagnose_training.py` are accessible through Git history. They depended
on local invalid-run directories or launched short training runs, so they were
removed from the completed project's working tree.

Raw diagnostic artifacts remain local under ignored `outputs/validation_audit`,
`outputs/validation_diagnostic` and `outputs/validation_calibrated_diagnostic`.
They are not required for final inference and are not part of the published
final artifacts. This document preserves the relevant measured findings.

Regression tests cover sample pairing, reshuffling, deterministic nontraining
order, train/validation-only content checks, weighted BN population moments,
inactive augmentation, sequential layer calibration, unchanged learned weights
and prediction independence from validation batch composition.
