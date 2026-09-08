# v08 complete AI evidence package

This local package preserves all420 checkpoints and their saved results, 997 learning examples, 12 splits, frozen sources, the original training plan, recovery evidence and final independent statistical/engineering reviews. It excludes large finite-element fields and dependency binaries. It is not a public release and adds no license.

Install the separately supplied NumPy/PyTorch versions or a compatible environment; see `environment.json` and `requirements_inference.txt`. From any working directory, invoke these files by their actual path:

```text
python verify_relocation.py
python portable_infer.py
python verify_results.py
```

The default replay is the predeclared first IID GNN seed42, with complete, radius and symmetric-kNN views. Select another included checkpoint using `--protocol`, `--model`, `--seed`; non-graph baselines require `--rules complete`. `--device cuda` requests the recorded cuDNN execution policy and checks bitwise identity; successful reproduction on another hardware/software combination is not assumed. CPU replay retains a strict3e-6 diagnostic threshold and reports deviations without widening it. These are software-replay differences, not errors against experiments.

The original plan remains byte-identical, with its historical SHA and paths. `relocation_map.json` is a separate14-field relative-path map. The runtime ignores the original disk location and checks every bundled source hash. No original plan is rewritten or given a misleading old SHA. The inference and saved-result verifier require neither Abaqus nor PyG and perform no optimizer updates.

On Windows, the builder and portable runtime use the extended path namespace internally so that the preserved nested evidence snapshots can exceed legacy path-length limits. This does not rewrite original source files or their provenance paths. Copy the complete directory using a tool that supports long paths.

`verify_results.py` checks all420 complete predictions,360 sparse views, saved metrics, per-case tables, model summaries and paired point estimates. Add `--bootstrap` to recompute both primary5000-repeat comparisons across all12 protocols. Engineering bootstrap outputs and independent reviews are retained and file-bound; this verifier does not rebootstrap them. All verification reports are created without overwriting prior reports; use `--output` for another destination.

This package emulates the original archived scalar running-minimum/interpolated response. It is not a reconstructed pointwise capacity or physical fire-resistance validation. The absent large FE files are referenced as historical provenance and cannot be regenerated from this small package. N=9–15 count-transfer tests are source-batch-confounded; sparse graph views test representation changes with the same trained weights. New training, universal topology generalization and equivalence across backends are not claimed.

`package_manifest.json` binds every delivered file except itself. `source_provenance.json` and `checkpoint_registry.json` preserve original locations and selection-free complete coverage. Move/copy the entire package to another directory and run the same checks before claiming relocation success. A passing default replay covers only the stated checkpoint and graph views; it is not proof that all420 were re-inferred.


Companion evidence addition, 7 September 2026: the builder also preserves the figure/table sources and saved artifacts, and the inference timing directory, available at build time. The explicit companion inventory lists each included file. Their individual QA records state completion; the package relocation checks do not claim to regenerate every figure or remeasure latency. Recorded historical paths remain provenance, and a separately adapted execution interface may be needed to rerun these companion generators outside the original workspace. The portable learning-data, prediction and result-arithmetic interface is unchanged.
