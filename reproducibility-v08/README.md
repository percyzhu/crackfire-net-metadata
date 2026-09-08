# CrackFireNet v08 reproducibility release

This directory releases the materials needed to reproduce the reported learning-study results without access to the large finite-element archives.

## Contents

- `base/` contains the 997 processed learning examples, the 12 frozen partitions, data preparation and feature code, model and training/evaluation code, 420 controlled-study checkpoints, saved predictions, aggregate reports, and figure/table sources.
- `attention-addendum/` contains the 60 later exploratory set-attention checkpoints, saved predictions, code and reports. It is included because the manuscript's core comparison includes the exploratory attention control.
- `reproduce_core_artifacts.py` verifies all released saved predictions, writes a fresh core-comparison table, and redraws the count-transfer figure in a new output directory.

The finite-element meshes, ODBs, full temperature fields and raw experimental records are deliberately excluded. The released target is the archived numerical thermal-response index described in the manuscript; this package cannot recreate the excluded finite-element calculations.

## Quick start

Create a compatible Python environment with the versions recorded in `base/environment.json`, then run from this directory:

```text
python reproduce_core_artifacts.py
```

The command writes a new timestamped directory under `reproduced_outputs/`. It verifies the 420 controlled-study and 60 attention-extension saved-prediction records, generates `Table3_core_comparison.csv` and `Table3_core_comparison.md`, and exports the count-transfer figure as PDF and PNG. It performs no training and does not overwrite bundled files. Add `--bootstrap` to repeat the two primary 5,000-repeat bootstrap calculations for the 420-run study.

## Released material and study scope

The 997 processed cases and all saved model outputs are provided to support computational reproduction. The complete 420-run matrix is the prespecified controlled study. The attention extension was added afterwards and remains exploratory. The count-transfer test changes source batch together with crack count; it should not be interpreted as a causal count-only experiment.

Use the repository citation information when reusing these materials. The repository's CC BY 4.0 license applies to metadata and documentation; users should contact the corresponding author for any reuse outside the stated computational-reproduction purpose.
