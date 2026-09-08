# Fire-transfer figure: prepared, awaiting all ten complete evaluations

The design and program are fixed without reading incomplete family performance. They follow the existing Python/matplotlib publication workflow. No all-family plot or synthetic placeholder is generated during preparation.

## Gate-only readiness check

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/fire_transfer/draw_fire_transfer_v08.py --gate-only
```

Exit 2 means at least one family evaluation is missing/incomplete or final replay auditing is not ready. `gate_status.json` lists completeness metadata only. Raw run counts are insufficient: every family must be marked COMPLETE with all 35 expected audited cells in `evaluation/report.json`. The gate never opens any per-family summary or performance CSV.

## Final generation entry point

After all ten families are complete and the independent aggregator has finished:

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/fire_transfer/draw_fire_transfer_v08.py --render
```

The same gate runs before any performance read. Afterward the program validates all 350 bindings/replays, 34,895 per-case rows, model/seed/split identity, summaries and paired effects; preserves source hashes and snapshots; and exports the vector figure and all plotted CSVs. No training or manuscript file is modified.

The complete figure will be 183 × 190 mm, with 8 pt text. Its 84 heatmap cells contain 70 family/model means plus two descriptive aggregate rows; 20 existing paired intervals retain every family. Aggregate averages have no invented confidence interval. A companion CSV preserves the 86-flat/11-nonflat smoldering split by model and seed.

Final generation status deliberately remains `PENDING_VISUAL_QA`: inspect the PDF-derived page at final size, check vector objects/fonts and caption placement, then create the final artifact manifest. A complete statistical source does not imply visual or journal acceptance.

For the PDF-derived preview and font/vector audit:

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/figures_v08/fire_transfer/audit_pdf_export.py
```

The actual preparation-time missing-data check is documented in `QA.md` and `gate_test.json`: two complete families, eight pending, no performance source opened and no formal figure generated.
