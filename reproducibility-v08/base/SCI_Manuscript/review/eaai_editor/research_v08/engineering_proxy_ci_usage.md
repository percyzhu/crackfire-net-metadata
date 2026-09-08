# Complete-protocol engineering-proxy intervals

`engineering_proxy_ci.py` adds the final paired-interval wrapper without modifying the frozen learner, target, plan or partitions. It waits for all7 models×5 seeds in one protocol before reading prediction arrays. A pending protocol returns `WAITING_FOR_COMPLETE_35_RUN_PROTOCOL`, reports its missing-run count and writes no result table.

Run from `D:/Work/Abstract` after a protocol finishes:

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' -X utf8 `
  'SCI_Manuscript/review/eaai_editor/research_v08/engineering_proxy_ci.py' `
  --plan 'SCI_Manuscript/experiments/research_v08/comparison_plan_v08_420.json' `
  --protocol iid997 `
  --output 'SCI_Manuscript/review/eaai_editor/research_v08/iid997_engineering_proxy_ci.json'
```

Replace `iid997` in both arguments for each of the other11 protocols. JSON and adjacent CSV are exclusive-create snapshots; do not overwrite an earlier completed result. The JSON includes all35 seed-level screening/ranking summaries and event/censoring counts. The CSV contains the two prespecified contrasts, graph vs capacity-matched set and mean-message vs sum-message graph.

## Interval definition

- Use exactly the same5 seeds, ordered test geometries and frozen61-point targets across models. The wrapper checks identities against the small frozen tensor snapshot and binds every prediction-file hash.
- The secondary intervals use1,000 fixed bootstrap repetitions with seed20260907. The main trajectory-MAE analysis separately uses5,000 repetitions; these numbers must be disclosed accurately rather than called one shared count.
- Resample geometry indices and seed indices independently. Each geometry retains its entire history, and each model uses the **same** draws. Subtract the two model metrics inside each repeat, then take the2.5/97.5 percentiles of those paired differences. Never subtract separately computed interval endpoints.
- Every effect is defined as **baseline minus candidate**. For MAE, FPR and absolute timing error, a positive value favors the candidate. For correlation, ranking agreement, recall and balanced accuracy, a positive value instead favors the baseline. A signed-bias difference has no universally favorable sign. The CSV records this direction for each row.
- Point effects require all5 paired seeds to define the metric. Missing classes, constant rankings and unsupported conditional timing remainNA. Intervals use common finite repeats only and report valid-replicate counts/fractions; absent denominators never become zeros. Full event tables and both-crossing coverage accompany conditional timing metrics.
- A model can change which cases have both reference and predicted crossings. Conditional-timing contrasts therefore compare the declared model-specific conditional means, not necessarily the same crossing cohort; do not interpret lower conditional timing error without its missed-event and coverage results.

The wrapper does not automatically convert these descriptive effects into a scientific or submission pass. It assesses fixed archival numeric proxies, not load capacity or fire resistance.

## Hand-checks and first invocation

`check_engineering_proxy_ci.py` constructs cases and seeds with correlated errors: individual model-MAE intervals are wide, while the true paired differences are exactly0.05 and0.02 in **every** draw. It verifies that both final-response and time-average intervals collapse to those known differences, which would fail if independent interval endpoints were used. Synthetic examples are labelled as software checks and never substituted for trained outcomes. Existing10 metric checks separately cover censoring, threshold equality, ties, delays and the completion gate.

The first real IID invocation returnedpending with13 runs missing and `predictions_or_scores_read=false`; no actual engineering interval was created. Rerun after the whole protocol completes.
