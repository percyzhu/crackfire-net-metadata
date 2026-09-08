# Actual artifact QA

Standalone visual and independent arithmetic checks passed.

Reviewed UTC: 2026-09-07T08:44:40.301762+00:00

PDF SHA256: `698dfcab2151c758cb95df9a5473f2a3c795618252ec3993d8fa0f961dcfd009`

- Both tables and their captions fit on one page and remain readable.
- No overfull boxes, out-of-page text, overlapping labels, or raster objects were detected.
- The 10^-3 effect scale, confidence limits, counts, and NA entries agree with the checked source values.
- Smoldering q=0.6 has zero reference events and NA event coverage; counts are not multiplied by five seeds.

Interpretation boundaries:

- Maximum FN/FP and minimum common-event counts may occur in different seeds.
- Secondary intervals are unadjusted and do not replace the prespecified primary comparisons.
- The 9-TeX-point body equals approximately 8.966 PDF big points; smaller mathematical subscripts are standard typography.

Checker-only corrections:

- The new independent table checker initially referenced c instead of the defined cens variable; this checker-only typo was fixed before the successful audit.
- The one-off export check threshold was corrected for the TeX-point/PDF-big-point conversion; the table font and layout were not changed.

The original generators and frozen source evidence were not modified. Final manuscript placement and manuscript expert review remain separate checks.
