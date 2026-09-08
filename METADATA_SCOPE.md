# Public data, code and reproducibility scope

This repository supports reproduction and audit of the reported learning experiments. The root-level files retain their original metadata role. The complete public reproducibility release is in `reproducibility-v08/`.

Included:

- case and split metadata for the 997-case benchmark;
- experiment plans, run counts and status records;
- aggregate reports and an independent attention-matrix audit;
- hashes and versioned identifiers that bind records to the executed studies.
- complete learning tensors for the 997-case archive;
- twelve frozen train/validation/test partitions;
- data preparation, feature-processing, model, training and evaluation sources;
- 420 controlled-study checkpoints and saved predictions, plus the 60-run attention extension;
- a one-command entry point that verifies saved results, recreates the manuscript's core comparison table and redraws the count-transfer figure.

Excluded:

- finite-element meshes, ODB files and full temperature fields;
- raw experimental records;
- dependency binaries and unpublished manuscript source files.

The excluded finite-element and experimental source archives do not change the released learning results. The reproduction package targets the archived learning task, rather than reconstruction of the full finite-element simulations or a new physical fire-resistance calculation.
