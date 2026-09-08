# Four engineering examples — prepared, no partial-data figure

The implementation is bound to the reviewed selection plan SHA256 `1850acfce80828fa7db4cd7fc751e9c86b8c47f8366fa3c91cc24135ce5d8c1e`. It imports the existing fire-transfer completeness/source validator by its fixed hash, without modifying or executing that helper's plotting code. Only this directory receives outputs.

## Readiness and final generation

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/engineering_examples/draw_engineering_examples_v08.py --gate-only
```

Exit 2 means a prerequisite is missing. The first gate reads the selection plan and authoritative global completion metadata only. It checks existence of every family's engineering-CI JSON/CSV, independent-results audit v2 and development-review text. Until all ten families have 35 completed audited runs and these documents exist, it opens no family performance file or prediction archive.

After the first gate, each engineering CI and independent review must have a complete status and matching protocol, seed, case-count, plan, summary and prediction-hash bindings. Existing CI/review source mismatches stop execution; they must be resolved by the research audit workflow, not silently accepted here. The imported validator then checks all 350 run replays/bindings and per-case statistics before prediction arrays are opened.

When all prerequisites are satisfied:

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/engineering_examples/draw_engineering_examples_v08.py --render
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/engineering_examples/audit_engineering_examples.py
```

The generator reads the saved predictions of the two primary models for five seeds and ten complete family holdouts (100 small NPZ archives). Each archive must match the predictive replay, engineering CI and independent-review hashes; checkpoint files are hashed without running new inference. Reference histories are matched to the frozen tensor collection. Prescribed-fire functions are extracted from the hash-bound archived source and reproduced exactly against the recorded time-input tensor. No full FE temperature field is loaded.

## Selection and outputs

- Exactly 997 GNN case scores, each a mean of five time-mean absolute errors; fixed quantile ranks 498/946/996, exact-tie lexicographic resolution, then largest remaining mean positive peak.
- Four actual seed42 paired trajectories, separate five-seed min/max envelopes, four prescribed-fire strips, and discrete first-crossing markers at q=0.8 and q=0.6.
- `all_candidate_scores.csv`, `selection_score_components.csv`, `selected_cases.csv`, `curve_data.csv`, `all_seed_curves.csv`, `threshold_events.csv`, recorded fire parameters, source/checkpoint hashes, source snapshots, caption and vector PDF/SVG plus PNG.
- Canvas 183 × 200 mm, minimum 8 pt. No smoothing, inferred ensemble trace, invented crossing, or confidence label for the five observed seeds.

The separate QA script recomputes selection from all retained score components, extrema and seed42 traces from individual seed curves, and threshold events from the 61 values. It also checks final dimensions, fonts, vector export and source snapshot hashes. Real-data visual inspection is still required before final delivery. No complete-data branch or formal graphic is claimed to have run during preparation.

The later complete-branch code/source review is recorded separately in `complete_branch_code_review.md/.json`. It repaired explicit UTF-8 path handling and retention of already validated review hashes during snapshot creation; derived tables now also have provenance hashes. Its real-input checks inspect completed ISO metadata and NPZ headers only, never prediction arrays or selected examples.
