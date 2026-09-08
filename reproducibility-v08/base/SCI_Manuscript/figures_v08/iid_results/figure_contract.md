# IID model comparison — design fixed before complete results

- Purpose: report the complete frozen IID comparison without presuming which model performs best.
- Evidence: exactly seven models × five prespecified seeds from the independently audited `iid997` protocol. No partial model rankings, training results, other protocols, or legacy results enter this figure.
- Panel a: five seed-level, equal-case MAEs per model and their arithmetic mean; frozen plan model order, never sorted by result.
- Panel b: the two prespecified paired MAE contrasts, with the existing 5,000-resample two-way geometry/seed percentile 95% intervals. The zero reference and both interval ends are always visible. No additional contrast or bootstrap calculation.
- Panel c: empirical cumulative distributions over test cases of each case's MAE averaged over the same five seeds. This is a distribution of seed-mean errors, not the error of ensemble-averaged predictions. The complete distribution and a 95% guide are shown.
- All MAEs and contrasts are multiplied by 100 and labelled percentage points; they refer to the dimensionless archived scalar, not experimentally validated capacity.
- Journal/export: 183 × 146 mm, three panels, Arial 8 pt or larger, vector PDF/SVG and 300 dpi PNG. Same seven model colors across panels; no significance stars, bars from pooled time points, or unexplained uncertainty ribbons.
- Data gate: require the complete-protocol status, expected 35 model/seed identities, passed audit bindings, exact test identities, finite complete per-case data, and agreement with summary values. Copy source tables and hashes only after all checks pass.
- Scope: IID accuracy and seed/case variability. This panel does not establish transfer, physical validation, or completion of other protocols.
- Backend: Python/matplotlib, continued from the existing explicitly language-specific manuscript figure workflow.

Final source gate: the immutable complete-IID seal and the independent expert arithmetic audit replace the global evaluation report, because that report also covers later, unfinished protocols. The plot's estimands, model order, contrasts, size, and display range rules did not change.
