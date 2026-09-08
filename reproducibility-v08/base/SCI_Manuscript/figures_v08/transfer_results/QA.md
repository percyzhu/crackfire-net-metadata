# Transfer figure: final QA passed

The contract, fixed model order, full shared ranges, seven-facet layout, and paired-seed connectivity display were prepared before reading partial or complete LCRO performance values. No placeholder or partial-results plot was created.

## Data and audit snapshot

- The source summary reports `COMPLETE_PROTOCOL_35_RUNS` for `lcro_9_15`; the paired report confirms all 35 expected model/seed combinations are complete.
- Every one of the 35 run audit entries passes prediction/plan/split/checkpoint bindings and the documented replay rule. The CPU 3e-6 diagnostic remains unchanged; any flagged cross-backend discrepancy requires bitwise exact replay on the original GPU backend.
- All 9,240 per-case rows reconcile with the 264 unique split IDs and manifest geometry/count labels. The seven model means and sample SDs reconcile with the summary. All 245 crack-count/seed means reconcile independently with the published stratum CSV.
- All 30 sparse-view rows reuse the corresponding complete-graph checkpoint; each reported MAE change reconciles with the same seed's complete-graph MAE. The plotted connectivity table has 45 rows including 15 zero-reference complete views.
- Original report/integrity and all plotted source tables were copied into this directory. Each copied file was verified against the exact hash recorded at generation; this snapshot is independent of later monitor writes to the live evaluation directory.
- Only this completed protocol is represented. Other protocol status and the full 420-run matrix are not inferred from it.

## Display integrity

- All seven models appear in the fixed order, with shared zero-based y ranges covering every seed value.
- All 245 count/seed points and all 45 connectivity/seed points are retained. Lines in b pair the same seed and trained weights; mean ticks do not represent confidence intervals.
- No CI is invented for seed-level connectivity values, and no time points are treated as independent observations.
- Source-005-only high-count coverage and source/count confounding are explicit in the figure and caption.
- The observed mean-aggregation sensitivity is shown without claiming universal normalization robustness or a causal degree effect.

## Visual and export checks

- Final PDF: 183.000006 × 173.999995 mm, minimum font 8.0 pt, zero raster objects.
- The 300 dpi PNG and independent PDF-derived render were visually inspected: readable text, all shared-axis ranges intact, no clipping, overlap, missing curve, or cut-off seed point.
- Included: PDF/SVG/PNG, source snapshots, plotted-source CSVs, provenance, caption, repeatable generator and export auditor, and an artifact hash manifest.
- No training, FE computation, or manuscript edit was performed by this figure task.
