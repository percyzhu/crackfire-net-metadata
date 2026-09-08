# v08 manuscript placement review — passed for Figures 3/4 and Table 4

The reviewed layout is bound to the packaged 25-page main PDF with SHA256 `cbfb7d115d3bb4376a90d71513d20ac3c2bc6fb4fff950b98a1d1dec722b97de` and three-page supplement with SHA256 `7c0c12e1a3b1735cad954bc12a13dc968c4bee2590f09a14911fd11131c98ee7`. The root reports that this final rebuild changes only the homepage progress sentence; the new sentence was verified in the PDF. Original figure-source hashes, 8 pt embedded fonts, caption presence and recorded page boundaries on pages 15–17 are identical to the visually reviewed build. Prior page PNG hashes were not retained, so this is a source/layout continuity check rather than a claim of pixelwise equality. Current PNG hashes are now retained in the JSON record. This is a placement/unit review, not a fresh statistical audit or submission approval.

- **Figure 3, p15:** readable model labels, seed points, both paired intervals and full ECDF range. Caption is complete on the same page, with clear separation from the figure and following paragraph. No clipping or overlap.
- **Figure 4, p17:** all seven count facets and three connectivity facets remain legible; shared axes, source-005 scope and seed legend are readable. The full caption stays on the same page and does not collide with the footer or page number.
- **Embedded fonts:** the actual main PDF retains 8.0008 pt Arial for both figures. Neither graphic has been reduced below the nominal 8 pt size. The copied source figure hashes exactly match the delivered, visually checked vector PDFs.
- **Table 4, p16:** caption, headings and all seven rows fit without collision. Every MAE mean, seed SD and RMSE value matches the complete LCRO summary multiplied by 100 and rounded to three decimals. Percentage-point table/figure units remain distinct from the dimensionless values in the prose. The stated 5.48% is a relative reduction, not a percentage-point difference.
- **Supplement:** the three-page contact sheet and the thermal-benchmark page show no evident clipping or caption collision. No additional FE figure work is needed for this layout check.

No blocking visual or unit-consistency issue was found. Figure 3 is followed cleanly by the crack-count subsection, and Figure 4 occupies a dedicated page with its full caption. No layout expansion is required. This record applies to the current hashes above.

Machine-readable page dimensions, font sizes, source-figure hashes and the expected seven Table 4 rows are in `manuscript_visual_QA.json`. No manuscript file was changed.
