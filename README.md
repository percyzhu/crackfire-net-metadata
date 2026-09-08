# CrackFireNet study data, code and reproducibility materials

This repository accompanies the manuscript *Inductive crack-graph learning for thermal-response screening of fire-exposed timber beams* by Xiuzhi Zheng, Bo Peng and Shaojun Zhu.

It publishes metadata, complete learning tensors, frozen data partitions, source code, checkpoints, saved predictions and reproduction utilities for the completed 420-run controlled study and the subsequent 60-run exploratory set-attention extension. The research task is prediction of a fixed numerical thermal-response index for timber beams with prescribed surface slots under specified fire histories.

## Contents

| File | Content |
|---|---|
| `archive997_manifest.json` | Manifest for the 997-case archival benchmark, including case identifiers, partition bindings, source hashes and response metadata. |
| `original_420_run_plan.json` | Frozen original study plan: seven representations, twelve protocols and five seeds. |
| `original_420_run_report.json` | Aggregate completion and integrity report for the 420 original runs. |
| `attention_60_run_plan.json` | Plan for the later exploratory attention extension. |
| `attention_60_run_report.json` | Aggregate completion and integrity report for the 60 new attention runs and 120 reused comparator runs. |
| `attention_independent_matrix_review.json` | Independent arithmetic and coverage review of the complete attention comparison matrix. |
| `reproducibility-v08/` | Public reproducibility release: 997 learning tensors, 12 splits, generation and preprocessing code, model/training code, 480 checkpoints and saved predictions, and a one-command reproduction entry point. |

The reproducibility release does not include finite-element meshes, Abaqus ODB files, complete temperature fields or raw experimental records. These large source archives are not necessary to reproduce the released learning experiments and remain separately retained by the authors.

## Study scope

The 420-run study trains seven model representations across 12 frozen protocols with seeds 42--46. The primary count-transfer protocol trains on one--eight prescribed cracks and evaluates direct prediction on nine--fifteen cracks. The 60-run attention extension was specified after the original results and is exploratory. Consult the manuscript for target definition, experimental validation context, model details, statistical analysis and interpretation.

## Reuse and citation

Unless a file states otherwise, the metadata and documentation in this repository are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the associated manuscript and this repository version when using the materials. A stable repository release and DOI can be added after journal submission.

## Contact

Shaojun Zhu, corresponding author  
Tongji University, Shanghai, China  
zhushaojun@tongji.edu.cn
