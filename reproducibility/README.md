# EAAI reproducibility materials

Companion to *Local geometry-aware crack-graph learning for section-modulus prediction in fire-exposed timber beams*, Xiuzhi Zheng, Bo Peng and Shaojun Zhu.

## Quick reproduction

Use Python 3.12 and install `requirements.txt`. PyTorch CPU works for all checks; the archived training used PyTorch 2.5.1 + CUDA 12.4 on an RTX 3060 Ti.

```
python reproduce.py
```

This verifies release hashes (if the manifest is present), checks all 15 checkpoint hashes, compares checkpoint forward outputs with saved predictions, reconstructs five cases at five times, re-scores all 90 standard-grid and 420 dense prediction files, recomputes the paired family intervals from 10 frozen resampling-index files, and regenerates Table 4, supplementary weak-set/event tables and core figures. All generated files go to `reproduced/`. No solver or model training is started. CPU evaluation takes several minutes.

Additional commands:

```
python prepare_inputs.py
python reproduce_statistics.py
python reproduce_dense.py
python reconstruct_labels.py --case ms500_0064 --all-times
```

The first rebuilds every input feature from raw geometry/fire records and compares with the exact prepared arrays. Floating-point operation order can produce sub-micro-scale differences. The exact archived prepared arrays are used to reproduce the paper's predictions.

To explicitly retrain a model, choose a fresh output directory:

```
python train.py --model local_graph --seed 2601 --output my_run
```

Supported models: `global_graph`, `global_contrast`, `local_graph`, `local_set`, `local_no_contrast`; seeds 2601, 2602, 2603. Training uses only the 860 train and 27 validation cases, 100 epochs, the published optimizer/loss and validation score. It never evaluates test roles or overwrites archived weights. Hardware/library differences can affect floating-point training trajectories; stored weights and full predictions define the published evaluations.

## Contents

- `data/learning_inputs.npz`: all 1,482 prepared inputs, IDs, roles, families, times and physical dimensions.
- `data/section_eta.npy`: memory-mappable `[1482,361,31,4]` section labels. Member histories are `min(axis=2)`.
- `data/raw_inputs.npz`: physical crack attributes, masks, physical dimensions and every actual 61-point fire table.
- `data/first300/`: all 1,482 compact material masks and first-crossing records, sufficient to reconstruct the published geometric labels without the full fields.
- `data/case_geometry_manifest.jsonl`, `case_evidence_manifest.csv`: cavity geometry, roles and mesh/field hashes; compact paths are relative to this package.
- `data/split_manifest.csv`: the single final paper partition. `provenance/legacy_splits/` retains the 12 original split definitions as historical records, not the current experiment partition.
- `models.py`, `train.py`, `metrics.py`, `prepare_inputs.py`, `reconstruct_labels.py`, `reproduce.py`: portable prediction, training, preprocessing, label reconstruction and evaluation.
- `checkpoints/`: all 15 selected weights and pre-evaluation freeze records.
- `predictions/`: all 90 full standard-grid arrays (5 models x 3 seeds x 6 evaluated roles).
- `training_records/`: histories, selected epochs and timing records for all 15 fits.
- `metrics/`: complete published seed/case scores, bootstrap intervals and protocols.
- `dense_queries/`: 7 selected cases, 4 coordinate grids, all 5 models x 3 seeds; separately supplied FE section labels and raw metrics. Five non-training cases enter Supplementary Figure S2/Table S8, with two training diagnostics separately identified.
- `generation/`: exact production source snapshots, parameter generator, geometry/meshing/thermal solve/audit/export scripts, material tables, frozen manifests, actual input files, environment records and full-field readers. Start with `generation/multisize_reproducibility_addendum_v1/WORKDIR_RESTORE.md` to restore the HPC work-directory layout. Solver execution is a separate explicit action; it is not needed for paper reproduction.
- `paper/`: LaTeX, figures, source values, plotting scripts and the manuscript's smaller forward example.
- `provenance/`: origin hashes and historical analysis records, including numerical qualification results retained without reclassification.

## Numerical target and usage

The model predicts the fixed simulator's geometric remaining-section modulus ratio after irreversible first reaching 300 C. Outputs are four directional I/c ratios on 361 times and 31 standard axial sections, normalized by each member's intact gross rectangle. Cells are 2 mm squares with own inertia and boundary-based extreme fibers. Initial cracks are removed; empty sections have zero ratio. Neither target nor prediction is clipped to one or forced monotone. These ratios are not a mechanically reduced effective section or an independently validated bending resistance.

Learning, model selection and transfer evaluations use the common specified numerical response. Historical solver qualification findings are preserved in provenance and are not labelled as passed. Full original temperature fields remain in the source HPC archive, located through the delivered field index; they are not duplicated here. Dataset membership comprises 982 reference-size cases and 500 variable-size cases. The original 13 rejected geometries and 2 conservative exclusions remain in the generation records; they are not part of the 1,482 learning cases.

## Public access and citation

The complete package, manuscript PDFs and editable LaTeX source are available from the [GitHub release](https://github.com/percyzhu/crackfire-net-metadata/releases/tag/eaai-benchmark). Download `EAAI_reproducibility.zip` for the full learning data; GitHub's automatically generated source archives contain the repository checkout only.

Package names are unversioned for manuscript presentation. Original source identifiers, simulation versions, freeze records and hashes are retained to identify exactly which evidence generated the published results. See `CITATION.cff`, `LICENSE` and `THIRD_PARTY_NOTICES.md` for attribution and rights. No new rights are granted over third-party materials. Contact: zhushaojun@tongji.edu.cn.
