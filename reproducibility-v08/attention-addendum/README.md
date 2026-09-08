# Attention60 and budget-analysis addendum

Place this directory beside the previously sealed `EAAI_reproducibility_v08_complete` package. Together they retain the original420 controlled runs,60 later exploratory attention runs and the original-seven exploratory budget analysis. Only the new60 weights are included here. The base manifest SHA is recorded in `base_dependency.json`; it is verified rather than replaced. Historical plans remain byte-identical, while `relocation_map.json` provides a separate path map. No large FE data or dependency binaries are included, and no license or public release is made.

Using an environment compatible with the base `environment.json`, run from any working directory:

```text
python /path/to/addendum/verify_addendum.py --output /new/report_path.json
python /path/to/addendum/portable_attention_infer.py --output /new/inference_report.json
```

Both commands accept `--base ../another_base_directory`, resolved relative to this addendum. The default is `../EAAI_reproducibility_v08_complete`. Reports are created exclusively and cannot overwrite a prior report. Original scientific files and sealed manifests are never edited.

The first command verifies both package file identities, all60 run metadata/prediction bindings and recomputed saved-prediction metrics. It checks supplied statistical summary identities; it does not repeat every bootstrap, budget calculation or figure generator. Independent source-backed audits and their stated scope accompany the evidence.

The second command is fixed before testing to IID attention seed42, CPU, all160 test geometries and61 response times, with no retraining. It retains the original strict maximum software-replay difference <3e-6. A reported difference beyond this bound is not silently accepted. Passing this command covers only this checkpoint, rather than all60 re-inferences. The set attention model ignores raw edges, so this is not a claim of graph-adjacency transfer.

These artifacts predict the fixed archival response index. The count experiment changes source batch as well as crack count, and budget utility denotes available random-to-oracle selection improvement, not risk reduction or measured time savings. The60-run and budget additions are post-review exploratory analyses, distinct from the original protocol. This package records numerical evidence and is not manuscript acceptance.

The payload also preserves the independently checked 6,435-record three-model inference timing session, its raw records and methods, and the final attention figures (`figures/complete_reviewed_v3`, with `figures/final_validation` QA). Earlier figure versions remain identified in their own directories. The thermal comparison companion includes the supplied observation workbook, the original saved vector comparison, all recovered curves, separate-group errors, plotting sources and numerical/visual audits. This is recovery of the prior experiment/FE comparison, not a new FE solution. Companion recovery and plotting sources retain their historical paths; their inclusion does not extend the portable execution claims below.

`addendum_manifest.json` binds the payload and construction gate. Actual relocation/inference receipts are retained separately so the sealed files do not change. Companion source and output inclusion does not mean that every original absolute-path generator has been ported. The portable commands above define the tested execution scope.
