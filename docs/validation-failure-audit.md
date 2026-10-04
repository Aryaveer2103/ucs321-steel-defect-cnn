# Validation failure audit — 2026-10-04

The proximate cause is stale BatchNormalization (BN) inference statistics.
Class-blocked training batches are ruled out. The final correction refreshes
population moments using only training images before each validation pass.
The CNN architecture, BN momentum, learning rate, batch size, augmentation,
dropout, and callback hyperparameters retain their original defaults.
No held-out test images were opened, inspected, predicted, or evaluated in this
investigation. The manifest and split assignments were not regenerated.

## Pipeline audit

1. **Manifest ordering:** training starts with 64 labels of class 0. Its main
   class runs have lengths 237, 236, 234, 237, 235, and 237, followed by smaller
   source-packaging runs. The on-disk manifest is class-grouped.
2. **Input pipeline:** paths and labels enter one `tf.data.Dataset` together.
   Decoding uses the shared Pillow loader, followed by cache, training-only
   full-buffer shuffle, batching, and prefetch. This is not a NumPy fit input.
3. **Actual shuffling:** the buffer is all 1,439 training samples, with seed 42
   and `reshuffle_each_iteration=True`. Shuffling occurs after caching and before
   batching. Observed label order changes across passes. Validation order does
   not change.
4. **Fit argument:** `shuffle=False` did not disable the dataset's shuffle. The
   installed Keras 3.15.1 adapter and the [Keras model training API](https://keras.io/api/models/model_training_apis/)
   confirm that this argument is ignored for `tf.data.Dataset` inputs. The
   redundant argument was removed. Keras can warn about its ignored default
   `shuffle=True`; the actual sample ordering remains owned by the dataset.
5. **Batch composition:** each of the first eight training batches contains all
   six classes. Counts in class-index order for the first three batches are:

   | Batch | crazing | inclusion | patches | pitted surface | rolled-in scale | scratches |
   |---|---:|---:|---:|---:|---:|---:|
   | 1 | 4 | 4 | 3 | 10 | 6 | 5 |
   | 2 | 6 | 5 | 5 | 5 | 4 | 7 |
   | 3 | 7 | 6 | 7 | 3 | 4 | 5 |

6. **Class mapping:** all 1,619 training/validation rows consistently map indices
   0–5 to crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches.
7. **Preprocessing:** both splits use grayscale conversion, bilinear resizing,
   and float32 tensors. The saved model performs the only `1/255` rescaling.
8. **Tensor checks:** train shape is `(1439, 128, 128, 1)` and validation shape
   is `(180, 128, 128, 1)`. Both have finite float32 pixels in `[0, 255]`.
   Mean raw intensities are 128.2057 and 129.5852, respectively.
9. **Image/label correspondence:** every training/validation filename resolves
   to the manifest class; every decoded image matches its frozen dimensions and
   pixel hash. Twelve validation thumbnails (two per class) were visually
   inspected, and their appearances agree with their labels. This checks
   correspondence; it does not independently re-annotate all images.
10. **BN:** default momentum is 0.99. After the first epoch's 45 updates,
    `0.99**45 = 0.6362` of the initial variance remains. The saved first-layer
    moving variance ranges from 0.6362 to 0.6394, whereas its current unaugmented
    training population variance ranges from 0.000118 to 0.009165. Later layers
    also have mismatched moments. [Keras BN documentation](https://keras.io/api/layers/normalization_layers/batch_normalization/)
    describes the separate batch-statistic training and moving-statistic
    inference behavior.
11. **Augmentation:** calling the actual augmentation block with
    `training=False` is an exact identity on the validation batch. Validation,
    population calibration, and prediction disable augmentation and dropout.
12. **Pooled split paths:** final split membership comes from the manifest, not
    the source `train/valid` folder. No training/validation path or label bug was
    found. Training previously opened held-out pixels during its integrity
    check. It now checks global manifest metadata but opens image content only
    for training/validation. Preparation/final evaluation retain their explicit
    full-content integrity checks. Neither was run against real held-out images
    during this task.

## Causal diagnostic on the original saved checkpoint

The saved best checkpoint is from epoch 1, not epoch 9. On validation it predicts
inclusion for 170 images and pitted_surface for 10; the other classes receive no
predictions. Accuracy is 21.67% and cross-entropy is 1.9418.

Keeping all learned weights unchanged, replacing only the three BN layers'
inference moments with population moments from the 1,439 unaugmented training
images raises validation accuracy to 73.89% and reduces loss to 0.7688.
Prediction counts become `[34, 45, 25, 23, 28, 25]`. Moments are computed in
layer order, so each deeper layer sees already-corrected inference activations.
No validation moments, gradients, augmentation, or dropout contribute. The
original checkpoint file was not overwritten.

Raw evidence: `outputs/validation_audit/audit.json` and
`outputs/validation_audit/validation_examples.png`.

## Rejected momentum-only correction

An eight-epoch fresh run with momentum 0.9 improved early validation but remained
unstable. It did not establish a sufficient fix. That change was reverted.

| Epoch | Train accuracy | Validation accuracy | Train loss | Validation loss |
|---|---:|---:|---:|---:|
| 1 | 0.5789 | 0.3500 | 1.1594 | 1.6271 |
| 2 | 0.7359 | 0.6611 | 0.7813 | 0.9651 |
| 3 | 0.7665 | 0.5889 | 0.6471 | 1.1063 |
| 4 | 0.8151 | 0.3278 | 0.5611 | 3.2914 |
| 5 | 0.8277 | 0.2889 | 0.5255 | 3.3952 |
| 6 | 0.8603 | 0.3778 | 0.4304 | 2.8111 |
| 7 | 0.8819 | 0.5111 | 0.3678 | 1.4892 |
| 8 | 0.8839 | 0.3722 | 0.3286 | 3.0739 |

Artifacts: `outputs/validation_diagnostic/`. No architecture changes were made.

## Final correction and five-epoch diagnostic

`TrainingBatchNormCalibration` refreshes BN moments immediately before fit's
validation loop. It uses a deterministic, unshuffled training dataset and
inference-mode probes, then weights channel moments by the actual number of
pixels, including the smaller final batch. It changes only the moving means and
variances. ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau then receive
validation scores from the corrected inference model. Saved checkpoints contain
the corrected moments; prediction requires no additional calibration.

Only the ordinary training dataset is shuffled. Validation and the existing
held-out dataset constructor remain deterministic and unshuffled. Original BN
momentum 0.99 and all original model/hyperparameter defaults are retained.
The additional training-only forward passes increase epoch time and cache use.

Command:

```powershell
.venv/Scripts/python -u scripts/diagnose_training.py --epochs 5 --run-dir outputs/validation_calibrated_diagnostic
```

| Epoch | Train accuracy | Validation accuracy | Train loss | Validation loss |
|---|---:|---:|---:|---:|
| 1 | 0.5789 | 0.7389 | 1.1594 | 0.7688 |
| 2 | 0.7359 | 0.7778 | 0.7813 | 0.6114 |
| 3 | 0.7665 | 0.8056 | 0.6471 | 0.5119 |
| 4 | 0.8151 | 0.7778 | 0.5611 | 0.4839 |
| 5 | 0.8277 | 0.8278 | 0.5255 | 0.4246 |

Validation now tracks learning. Learning rate stays at 0.001 throughout these
five epochs. The first four training curves exactly match the original failed
run; epoch 5 differs because the original run reduced its learning rate based on
the invalid validation signal. Training accuracy includes augmentation/dropout
and updates throughout the epoch; validation uses final weights without them.
Validation accuracy exceeding training accuracy early is therefore compatible
with this protocol.

Reloading the selected epoch-5 checkpoint reproduces validation accuracy 82.78%
and loss 0.4246. Predicted-class counts are `[35, 41, 25, 27, 29, 23]`: there is
no one-class collapse. The access guard records 1,439 unique training and 180
unique validation images opened, with zero held-out images opened.

Curves, history, run configuration, checkpoint, and access report are under
`outputs/validation_calibrated_diagnostic/`. Both short runs are marked
`diagnostic_only`, which prevents the existing evaluation CLI accepting them.
Training stopped after this diagnostic; no final clean run was started.

## Verification and original-run status

All 20 tests pass. Regression coverage includes sample pairing, full-epoch
coverage, mixed training batches, reproducible seeds, epoch reshuffling,
deterministic nontraining order, and integrity checks without held-out files.
BN calibration tests check inactive augmentation, weighted population moments,
sequential calibration of two BN layers, unchanged learned weights, and
prediction independence from validation batch composition.
Compilation and all project module imports pass. `git diff --check` passes.
Installed TensorFlow/Keras/matplotlib emit dependency warnings; there were no
test failures after the final correction.

The original nine-epoch run is marked `invalidated_validation_failure` in
`outputs/metrics/run_config.json`. Its model selection, learning-rate schedule,
and early stopping were driven by defective validation scores. Its checkpoint
and curves are preserved for debugging, but it is invalid for final reporting
or final evaluation. The original model SHA256 remains
`fec6ad70a62435fc95331b366ac43438d4c20eeb15bc8a4fb302f1eee2689579`.
The frozen manifest SHA256 remains
`a3e658e5cdebcb3b5b9d5ddf6399b626f612367c56dd1bbf5d2f9f1c42690a75`.

When ready, start final training separately from fresh weights:

```powershell
.venv/Scripts/python -m steel_defect.train --run-dir outputs/final_clean
```

This command trains and validates only. Do not run held-out evaluation until
final model selection is complete.
