# All-family fire-transfer figure — fixed before complete outcomes

## Claim and evidence

The figure will describe how the seven prespecified representations generalize to each completely held-out prescribed-fire family, without presuming a winning model or universal transfer. It is a quantitative grid with a primary MAE heatmap and two subordinate paired-contrast forests.

- Fixed family order, shared with the population figure: ISO 834, ASTM E119, external fire, linear, bilinear, plateau, decay, perturbed ISO, log variant, smoldering. Never order by observed performance.
- Fixed model order: fire only, global statistics, DeepSets, capacity-matched DeepSets, sum GNN, zero-edge-feature GNN, mean-message GNN.
- Panel a: all 70 family/model MAEs, averaged over the five seeds. Append two clearly separated descriptive rows: all 997 cross-held-out cases weighted equally, and all ten fire families weighted equally. These rows combine ten separately fitted holdout experiments; they are not the output of one common trained checkpoint.
- Panels b/c: all ten family-level instances of each prespecified contrast, with the source two-way geometry/seed 95% percentile intervals from 5,000 resamples. Preserve both interval ends, zero and every family, even for unfavorable or inconclusive results. No stars, interval-based filtering, multiplicity claim or invented uncertainty for the aggregate rows.
- Smoldering must remain: 86 of its 97 cases are restored constant responses; the other 11 are retained nonconstant cases. The figure will disclose this composition, and a companion CSV will retain subgroup seed MAEs. Neither subgroup is silently removed.

## Gate and scope

Formal output is permitted only after all ten protocols have 35 independently audited runs, all summaries and tables exist, every protocol is marked COMPLETE in the authoritative evaluation report, and all 350 bindings/replay records pass. Read completeness metadata first; do not open any family performance table before this joint gate passes. Incomplete data must return a waiting status and generate no PDF/SVG/PNG or source-data performance snapshot.

After the joint gate, verify ordered model/seed identities, frozen plan/manifest/splits, complete test populations, per-case means and seed SDs, both paired effects, 5,000-resample provenance, source evaluator hashes, and stable source-file hashes. Retain source snapshots for final provenance. Do not rerun training, change frozen rules, tune on test outcomes, or claim completion of journal review.

## Display contract

- Python/matplotlib only, continuing the existing manuscript workflow and nature-figure skill.
- Final canvas: 183 × 190 mm; Arial 8 pt minimum; editable SVG and vector PDF, plus 300 dpi PNG.
- MAEs and differences are multiplied by 100, labelled percentage points of the dimensionless archived scalar-envelope target.
- The heatmap uses one zero-based sequential color scale for all cells. Forest panels share one symmetric scale covering every complete interval and zero.
- Family sample counts accompany the labels. Existing intervals resample whole geometry clusters and training seeds, never individual time points. The 20 per-family intervals are unadjusted and do not establish a global family-population claim.
- Final visual QA remains required after real data become available; no synthetic plot is used to imply readiness.
