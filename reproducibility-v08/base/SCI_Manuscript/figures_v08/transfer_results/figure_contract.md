# Graph-size and connectivity transfer — pre-result figure contract

- Purpose: measure behavior on previously unseen crack counts and under changed graph connectivity, without presuming transfer success or a GNN advantage.
- Sources: only the completed, independently audited 35-run `lcro_9_15` protocol. No incomplete run result is read to decide the plot.
- Panel a: seven small multiples in frozen model order. Each displays the mean MAE at N = 9–15 over five seeds, with all five seed points. Every facet uses the same zero-based y range covering every seed value. The final grid cell explains the 264-case holdout, counts by N, and source-005-only scope.
- Panel b: three GNN variants in fixed order (sum, zero edge features, mean messages). Complete, radius, and symmetric kNN views are evaluated using the same trained checkpoint. Plot MAE changes from complete for each seed and the mean, with a shared scale and visible zero reference. Pairing is retained within seed; no new model is fit.
- The seed points show training variability, not confidence intervals across cases. No timepoint pooling, bootstrap fabrication, significance stars, model selection, or causal claim about message normalization.
- All errors refer to the archived scalar-envelope target. Values are multiplied by 100 and labelled percentage points.
- Physical graph-size generalization is tested on held-out geometries, but source batch is confounded with high crack count. Connectivity changes are representation stress tests on those same cases, not additional physical FE geometries or mesh transfer.
- Export: Python/matplotlib; 183 × 174 mm; Arial 8 pt; vector PDF/SVG and 300 dpi PNG. Continue the manuscript's existing Python figure workflow.
- Complete-data gate: full model/seed identities, complete protocol summary, passed prediction bindings, computational replay under the documented unchanged CPU diagnostic plus exact original-GPU identity for flagged cases, exact split IDs, all per-case strata, and same-checkpoint topology rows.
- Rendering is authorized only after the root confirms complete-protocol aggregation. Before that, the program and contract may be syntax-checked, but no data-dependent figure is generated.
