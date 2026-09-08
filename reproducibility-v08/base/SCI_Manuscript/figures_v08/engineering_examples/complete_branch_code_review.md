# Complete-data branch review

The review found and repaired two full-data risks in this directory: the independent audit used Windows' default encoding for JSON containing Chinese paths; source snapshot creation could rehash a newer engineering/review file after validating an older version. JSON reads now specify UTF-8, and snapshot creation retains the validated review hashes and verifies both the copied and current originals.

The independent table audit now also checks each selected record's rank, selection score and metadata against the candidate table; exactly-once membership in the ten saved holdout splits; the selected family's two models and five checkpoint paths; seed/time/sample identities; event uniqueness and censoring; and hashes of the derived score and curve files.

Read-only checks used complete ISO 834 metadata, its saved seed42 NPZ **headers only**, checkpoint/prediction digests, and frozen dataset inputs. The NPZ format is case × time = 158 × 61, consistent with the generator. The CI, primary replay and independent review field names and source hashes agree. All 997 prescribed-fire input histories reproduce the frozen 61 × 4 tensors exactly. The ten frozen LOCO test splits contain exactly 997 distinct cases, and the declared order-statistic ranks are 498, 946 and 996.

Code inspection confirms that selection uses each case's GNN mean of five per-seed time-mean absolute errors; the fourth case uses the largest remaining mean positive peak. Curves use each selected family's actual seed42 checkpoints and separate five-seed minima/maxima. The response axis includes all raw reference/prediction/envelope extrema with padding, including values outside [0,1]. Threshold markers use the first stored value at or below the threshold without changing the curves.

The updated real `--render` gate stopped with exit 2 at seven completed families. It read no family-performance, review-performance or prediction arrays and generated no score tables or figure. The separate structural review read no prediction array payload and did not select cases. Exact hashes and checks are retained in `complete_branch_code_review.json` and `gate_test.json`.

**Still pending:** the complete 350-run prediction branch, generated-table/export audit and actual visual QA. This is a code/source-interface review, not a completed numerical-figure or manuscript acceptance review.
