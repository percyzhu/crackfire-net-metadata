# Complete-data code and interface review

The figure's field names, group structure and units agree with the frozen `benchmark_inference.py`, declared `timing_plan.json` and independent `audit_inference_timing.py`. None of those files was modified or executed for timing during this review.

The expected raw count is **14,355 = 3 × (160 + 5) × (2 × 7 × 2 + 1)**: three rounds; 160 single-case batches and five batches of 32; two devices, seven models, two interfaces, plus the common CPU preparation stage. This produces 58 summary groups. The figure uses 56 numbers: two devices × seven models × two single-query quantiles, plus two devices × seven models × two batch-throughput interfaces. Every displayed group represents 480 case instances across three repeats: 480 batch-one observations or 15 batch-32 observations.

Single-query median/p95 values are milliseconds for a complete 61-time response, not per-time-step or amortized batch latency. The p95 marker is an observed timing quantile, not a confidence interval. Throughput is total cases divided by total seconds. The common interface includes the shared features for all models; it does not imply optimized baseline deployments. Historical training costs and feature-preparation/per-round/crack-count summaries remain companion data, with no mixed-hardware FE speedup claim.

Two provenance weaknesses were repaired: the generator now retains the independent audit's exact hashes across source reading and snapshot creation, and it snapshots every dependency bound by that audit using workspace-relative paths. Copies and current originals must still match the validated digests. Derived plotted values, companion summaries and exports receive hashes. The CUDA title is now taken from the actual environment record rather than a hard-coded GPU name.

A separate `audit_inference_figure.py` recomputes all 56 plotted values from the retained raw records without importing the generator, checks complete source bindings and units, verifies that all plotted values lie inside the axes, and checks the vector PDF dimensions and minimum font size. Its future successful result remains pending visual review.

**Actual preparation test:** the generator's `--render` command and the independent figure audit both returned exit 2 with missing real measurements. No timing rows were read, no model execution was started, and no `rendered` directory or mock speed plot was created. Both scripts passed syntax parsing. The frozen timing-plan source hash still matches the untouched benchmark code.

**Not yet validated:** the real-data rendering branch, execution of the independent 56-value/PDF audit, and visual inspection at final manuscript size. The code/interface review is not a passed experiment or submission review. Exact hashes and gate evidence are in the adjacent JSON records.
