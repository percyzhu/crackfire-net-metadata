# Secondary engineering-proxy evaluation: v08

Declared 2026-09-07 by the independent reviewer before viewing any v08 training predictions or scores. This is a secondary evaluation recommendation, not a retrospective change to the frozen primary metric (equal-case MAE). The root experiment record should bind the adoption timestamp and whether any run had already completed at that time. No new physical meaning or acceptance threshold is assigned to the archival response.

## Fixed interpretation and quantities

Let a_i(t_k) be the frozen archival scalar-envelope reference for geometry i at the 61 stored evaluation times t_k=60k seconds, k=0,...,60. The predictor returns p_i(t_k). Both are numeric proxy responses, not measured section capacity or fire resistance. Lower values mean a smaller response under this archive's convention; the comparison establishes agreement with that convention only.

Two prespecified trajectory summaries are the final response a_i(3600) and the time-averaged response J_i=(1/3600) Σ_k 30[a_i(t_k)+a_i(t_{k+1})]. The latter is the trapezoidal integral of the piecewise-linear evaluation trajectory; no additional FE state is implied. Compute the same summaries on p without scalar clipping or cumulative-minimum repair. Report equal-case MAE of both summaries and signed bias.

## Threshold-crossing proxy, q=0.8 and q=0.6

For each q, define the event time as the first evaluation-grid time at which the response is ≤q. This is explicitly a **discrete archival-index crossing time**. If there is no crossing, retain right censoring beyond3600s; never replace it by failure at3600s. Report both q values, without selecting the one that favors a model.

- Compare observed event/non-event by3600s: counts, confusion table, recall among reference events, false-positive rate among reference non-events, and balanced accuracy when both classes exist. Undefined denominators produce NA with the class count.
- A reference event missed by the predictor, or a later predicted crossing, means optimistic delay **relative to the archive**. An earlier crossing is conservative only in that same proxy sense; neither is a proven physical safety classification.
- On cases for which both curves cross, report median and mean absolute timing difference and signed timing difference p_time−a_time. This conditional timing score must appear with both-event coverage and false/missed event counts; never discard censoring silently to make timing look better.
- Report the fraction of neural trajectories that recross above q after their first crossing and the existing temporal monotonicity violations. Keep raw first crossing; do not enforce irreversibility only during evaluation.
- With a60s grid, these are grid-event errors. Do not imply subminute accuracy or interpolate physical threshold events from them.

## Ranking and prioritization within a stated test population

For final response and J separately, calculate Spearman correlation with average ranks for ties, plus pairwise order agreement over **only pairs with unequal reference scores**. Count a predictor tie as0.5 agreement, report the number/fraction of excluded reference-tied pairs, and return NA if the reference is constant. These metrics assess ranking for the mixture represented by the test set; they do not isolate geometry if fire histories differ.

Report these rankings for the full test population and within each fire family with at least two distinct reference values. Equal-family summaries average defined family scores and state how many families qualify. Do not present cross-family rank correlation as controlled crack-geometry sensitivity. Whole-family holdouts contain one family, so only their supported within-family ordering is interpreted.

## Pairing, strata and uncertainty

Use exactly the frozen test IDs and5 training seeds for each model/protocol. Preserve pairing when contrasting models. Bootstrap independent geometry IDs, and account separately for seed variation; never resample the61 time points as independent specimens. State95% intervals and counts rather than introducing post-hoc engineering pass/fail tolerances.

Report the original911/restored86 status, source batch, fire family and crack-count strata when nonempty. A stratum with too few outcomes gets counts and NA where appropriate, not a favorable pooled inference. The two thresholds and all summaries above are secondary; they cannot replace the primary graph-vs-capacity-matched-set comparison or be used for model selection.

The engineering claim permitted by successful results is agreement in screening or ordering **under the fixed archived-response workflow**. This specification does not establish true thermal damage, char-front location, load capacity, failure time, or a design criterion.
