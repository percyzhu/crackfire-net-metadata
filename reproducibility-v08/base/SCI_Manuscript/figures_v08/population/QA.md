# v0.8 population figure QA

- Inputs: only the frozen997-case manifest and design_audit composition/limitations. No model predictions, training results, tensor or temperature fields were read.
-997 unique caseIDs and997 exact geometry hashes; total exactly matches the independently aggregated source-batch/fire-family/count design table.
- Heatmap has150 explicitly labelled cells includingzeros, ten row totals, and a correctly bounded0–21 colour scale. No interpolation or hidden categories.
- Split check:620train+113validation=733 forN1–8;264test forN9–15. All latter cases are source005; this confounding is visible in thefigure and stated in thecaption.
- Source totals:330 source004 and667 source005. Restoration arithmetic:911+86=997 overall,11+86=97 smoldering. All86 restored cases are constant-response cases and smoldering.
- Final output183×164mm, all labels8pt,SVGeditabletext,PDFTrueType. Heatmap cells and plots arevector geometry.
- Figure was rendered and visually inspected at full output size: all cells legible, count-boundary line clear, legends and lower annotations separated. An initial lower-axis label overlap and colourbar tick beyond thedata range were corrected before delivery.
- Programmatic canvas check found no text outside thefigure. No statistical intervals or tests are presented.
- The figure is a dataset/design description. It neither establishes trained topology transfer nor claims thermally qualified or experimentally validated archival labels.
