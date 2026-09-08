# IID figure: final QA passed

The layout, model order, contrast selection, uncertainty method, and complete-distribution display were fixed before reading complete IID results. No partial result or placeholder data was plotted.

## Evidence gate

- Final sources are the immutable `replay_precision_diagnostic/sealed_iid35` summary and CSV, with `seal_manifest.json` status `SEALED_COMPLETE_IID_35_RUNS`.
- The separate expert arithmetic audit checked all 35 prediction archives and 5,600 per-case rows, independently recomputed both bootstrap contrasts, and recorded status `IID_ARITHMETIC_DEVELOPMENT_AUDIT_PASSED_NOT_FULL_MATRIX_OR_SUBMISSION_REVIEW`.
- This figure generator verified both gates, all five sealed file hashes, the plan/split hashes, the 35 model/seed identities, the common 160 test identities, prediction hashes shared by both audits, and agreement of all seven summary means/SDs and both paired point estimates with per-case CSV arithmetic. All IID CPU replay differences satisfy the original 3e-6 criterion.
- A later-protocol replay diagnosis does not enter the figure gate, data, or interpretation. No full-matrix or transfer-completion claim is made.

## Statistical display

- All 35 seed MAEs are plotted, in the frozen seven-model order, with arithmetic mean ticks. No timepoint error bar or ranking-based ordering.
- Both prespecified two-way geometry/seed 95% intervals cross zero; both complete intervals and their means remain visible. No model-advantage statement is added.
- Panel c includes all 160 cases for each of seven models and the complete maximum-error range. Each value is a case MAE averaged over seeds, not an ensemble prediction error or a pooled case-time quantile.
- All displayed values are source dimensionless MAEs multiplied by 100, explicitly labelled percentage points.

## Visual and export checks

- Final PDF size: 183.000006 × 146.000003 mm. Minimum PDF font: 8.0 pt. Raster objects: 0.
- PNG preview and independently rasterized PDF were visually inspected. No text collision, clipped interval, missing curve, or boundary overflow.
- Initial QA detected locator ticks outside displayed limits; only out-of-range locator labels were suppressed, without changing any data limits. The 95% guide label was moved below its line to avoid the upper ECDFs. Text colors were darkened for print legibility.
- PDF, SVG, 300 dpi PNG, copied source tables/audits, plotted-source CSVs, caption, reproducible scripts, and hashes are included in this directory.
