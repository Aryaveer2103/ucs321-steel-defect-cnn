# Final and generated outputs

`final/` is the completed academic run. Its explicitly allowlisted artifacts are
committed: selected Keras checkpoint, architecture and input metadata, frozen
manifest, dataset summary, run configuration, environment, 50-epoch history, final
test metrics and classification reports, confusion matrix, per-image predictions,
EDA figures, training curves, test examples and demo.

The checkpoint was selected at epoch 45. Final held-out accuracy is 97.78%
(176/180). See the root README for complete results and exact commands.

All other generated output directories remain ignored. Earlier invalid runs,
smoke results and validation diagnostics are local historical evidence, not final
submission artifacts. The diagnosis is retained in `docs/validation-failure-audit.md`
and the diagnostic scripts remain accessible in commit `4c89dfa`.

Use a new directory such as `outputs/reproduction` for any independent rerun.
Do not overwrite or regenerate `final/` for submission. Training refuses to
overwrite an existing model or run configuration. Prediction and demo verification
can write disposable results under `outputs/verification`.
