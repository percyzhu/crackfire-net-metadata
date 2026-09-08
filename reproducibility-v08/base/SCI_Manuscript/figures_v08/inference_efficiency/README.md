# Inference efficiency — prepared, actual measurements pending

The figure uses the existing Python workflow and waits for the complete real timing experiment and its independent raw-record audit. It does not start model evaluation or open raw timing records before that audit. No placeholder chart or assumed speed ordering is generated.

From the project root:

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/inference_efficiency/draw_inference_efficiency_v08.py --gate-only
```

After all 420 training runs, final prediction replay, quiet-machine measurement and the independent timing audit have finished:

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/inference_efficiency/draw_inference_efficiency_v08.py --render
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/inference_efficiency/audit_inference_figure.py
```

The four panels separate CPU/CUDA single-query median and p95 latency from batch-32 throughput. Seven models remain in frozen order. Every case is a full 61-time trajectory. The generator independently recomputes all 56 plotted numbers from the 14,355 raw records and compares them with the audited summary. All source hashes, raw records, common preparation, crack-count strata, repeated-round summaries and historical training costs are retained in the rendered snapshot. No measurements are excluded.

Export size is 183 x 154 mm with minimum 8 pt text and vector PDF/SVG. PNG is for preview only. The common interface includes identical feature construction for every model; it is not a collection of individually optimized deployments. Median-to-p95 segments are timing quantiles, not confidence intervals. Training cost and historical FE cost are not plotted as comparable isolated benchmark timings.

Current validation is limited to code review, syntax, current source-schema compatibility and an actual absent-measurement gate. Full-data execution, independent figure arithmetic and PDF/embedded-page visual review remain pending. A future rendered directory is preserved rather than overwritten. The caption is a prepared methods description; it cannot be inserted as an executed result before the real experiment finishes.

The complete-branch source/interface review is documented in `complete_branch_code_review.md/.json`. Rendering now retains the audit's exact source hashes through calculation and snapshot creation, copies every audited dependency with its workspace-relative path, and derives the CUDA hardware label from the measured environment. The separate figure audit recomputes the 56 numbers from the retained raw records and checks source bindings, units, axis inclusion, vector dimensions and minimum font size; an automated pass still requires subsequent visual inspection.
