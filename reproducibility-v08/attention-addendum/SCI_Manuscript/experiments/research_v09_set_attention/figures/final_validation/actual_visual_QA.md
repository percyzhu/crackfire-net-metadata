# Actual visual and LaTeX checks of the reviewed extension figures

The final v3 paired-effect figure was inspected both as its exported PNG and as a rendering of the actual PDF. The enlarged LCRO panels show the principal crack-count extrapolation comparison, with explicitly labelled expanded linear axes. The lower panels retain all 12 protocols and their complete confidence intervals on the full linear scale. Panel labels, signs, zero references, numerical annotations and protocol names are readable, without clipping. Negative and inconclusive findings are retained.

The three-model MAE figure was inspected in its visually identical v2 rendering; v3 retains the same MAE drawing. Its 12 protocol rows, five-seed dots, mean and sample standard deviation are visible. Both final figure PDFs passed text and page checks and were rendered for inspection.

Both LaTeX table fragments were compiled together in two successful PDFLaTeX passes. The two resulting pages were actually viewed. All 12 protocol rows, including Smoldering, are present; means, standard deviations, interval signs and units are readable. The compilation reported zero overfull boxes. The spacious standalone test-page layout is a validation wrapper and is not the final manuscript placement.

The separate statistics export contains 36 protocol–model rows, with five-seed mean and sample standard deviation for both MAE and RMSE. Each seed's RMSE is the square root of the mean squared error over all case–time observations; its five-seed summary is not a mean of per-case RMSE values.

Bound generated artifact manifest: `../complete_reviewed_v3/artifact_manifest.json`, SHA-256 `3e7cf34ff0121d9392824407943cdf59db15202dd995679cb3bdc7181f2864d6`.

Compile and export evidence: `validation_manifest.json`, `tables_compile.pdf`, `pdflatex_run1.log`, `pdflatex_run2.log`, `three_model_12_protocol_statistics.csv`, and `three_model_12_protocol_statistics.md` in this directory. The original generation manifest is preserved; this note records the subsequent actual inspection separately.

Decision: standalone figure and table visual checks passed. Placement in the final manuscript remains to be checked. This is not a journal acceptance decision or a final manuscript review.
