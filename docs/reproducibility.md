# Final experiment and repository verification

The completed run is `outputs/final`. The final model was selected at epoch 45
from a 50-epoch run using validation loss. Its held-out test accuracy is
97.77777777777777% (176/180). The root README presents rounded values; the
machine-readable final metrics retain their original precision.

## Artifact integrity

| Artifact | SHA-256 |
| --- | --- |
| `outputs/final/models/best.keras` | `24e83202be8281c790e606ff7d29856cc7985d7f07398e81cfe630af38549a5d` |
| `outputs/final/metrics/manifest.csv` | `a3e658e5cdebcb3b5b9d5ddf6399b626f612367c56dd1bbf5d2f9f1c42690a75` |

The manifest contains 1,799 unique paths and pixel hashes, split 1,439/180/180
with seed 42. Each validation/test class has 30 images. `dataset_summary.json`
records the one duplicate excluded from the original 1,800 images. The committed
manifest is used directly by the README reproduction commands; it is not redrawn.

`.gitattributes` disables line-ending conversion for final artifacts so the
manifest checksum remains valid on Windows and other platforms. Final model and
metric values were preserved during repository finalization. Only `data_dir` in
`run_config.json` and `dataset_summary.json` was changed from an absolute local
path to `data/raw/NEU-CLS` for portability. Run commands from the repository root,
or supply `--data-dir` when the dataset is elsewhere.
The history CSV's doubled carriage returns were normalized to standard CRLF
record endings; all 50 epochs and numerical values match the original JSON.

The final run used Python 3.12.8, TensorFlow 2.20.0 and Keras 3.15.1 on CPU.
All six direct dependencies are pinned in `requirements.txt`; `pyproject.toml`
reads that same file. The recorded `environment.json` supplies the full package
inventory from training. Reproducible seeds and deterministic operations do not
guarantee identical new training results across hardware or framework versions.

## Finalization verification — 2026-10-04

- All 20 unit tests passed. Tests use synthetic data and temporary models.
- Compilation and imports of every `steel_defect` module passed.
- All six CLI entry points accepted `--help`; README flags match their parsers.
- The final checkpoint loaded with input 128 × 128 × 1, six outputs and 101,990
  parameters (101,542 trainable plus 448 BN moving statistics).
- Both artifact hashes matched the authoritative values.
- Every final JSON parsed. Saved per-image probabilities reproduced the published
  accuracy, macro/weighted metrics, classification report and confusion matrix.
  The matrix contains 176 correct predictions out of 180. Cross entropy recomputed
  from the saved CSV agreed within floating-point precision.
- The 50-epoch JSON/CSV histories agreed, and minimum validation loss selected
  epoch 45. Independent checkpoint validation before the original test evaluation
  had reproduced 97.7778% accuracy and loss 0.0841048211.
- The documented dataset preparation command reused the frozen manifest and
  verified image integrity. EDA wrote training-only sample figures to a disposable
  ignored verification directory.
- Single-image prediction using the final model returned `crazing` with 99.74%
  confidence for `crazing_10.jpg`. Noninteractive demo export passed. The example
  is a UI check and does not establish an additional evaluation metric.
- All nine committed PNGs decoded and their layouts were visually inspected.
- Installed dependency compatibility and requirement resolution passed. Source
  distribution and wheel builds passed. The documented `uv` setup installed the
  pinned requirements and editable project in a fresh disposable Python 3.12
  environment. Local documentation links resolved.
- Stale pending-run instructions, superseded split references and unsupported
  company claims were removed from the current project documentation.

No final training, tuning, BN recalibration or held-out model evaluation was run
during finalization. The final checkpoint, frozen split and published metrics
remain unchanged. Dataset integrity checks read pixels without model inference;
result consistency checks use the saved prediction CSV.

TensorFlow and plotting dependencies emit some informational/deprecation warnings
in this environment. They did not cause test, inference, plotting or build failures.
The existing environment was created with `uv` and has no `pip` module; dependency
checks used `uv pip`. The README uses the tested `uv` setup as its primary
installation route. The Windows launcher also confirmed Python 3.12.8 when given
access to its installed runtime. Standard venv/pip commands are provided as an
alternative when launcher-managed Python 3.12 is already installed.

## Preserved engineering history

The original pipeline commit is `50d133b`. The BN diagnosis and correction are
in `4c89dfa` and documented in `validation-failure-audit.md`. Old invalid/smoke
outputs remain ignored local history. Diagnostic helper scripts are recoverable
from that commit; they are not required by the final pipeline.

The final repository commit includes the README, documentation cleanup, artifact
allowlist, exact-byte Git attributes, dependency pin and all 24 curated final
files. Raw data, archives, caches, virtual environments and diagnostic outputs
are excluded. The annotated release tag is `v1.0-project1-final`.
