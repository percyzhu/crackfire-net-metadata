# Actual inference-efficiency figure QA

Standalone actual visual, arithmetic and export checks passed. All 7 models, 2 devices and both interfaces are retained; one query is a full 61-time trajectory. Latency p95 is not a confidence interval. The first rendering safely stopped on an out-of-canvas automatic tick; the display locator was corrected without changing measurements. The figure is 183 × 154 mm, minimum 8 pt, pure vector.

The CPU was an Intel Core i5-13600KF with 4 PyTorch threads; GPU was an RTX 3060 Ti. These are the measured common implementation and resident-input forward paths, not independently optimized deployments or FEM speedup. Manuscript placement and final paper review are separate checks.
