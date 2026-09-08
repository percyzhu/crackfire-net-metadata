# Application workflow v0.8 QA

- Exact three-slot geometry matched uniquely to997-case sampleb004_0006,ISO834. The case was selected by geometry identity rather than model performance. All three panels use this same case.
- The previous geometryCSV, nodeIDs and directed pair correspondence are preserved. Slots1/2/3 and graph nodes1/2/3 retain matchingblue/teal/ochre colours.
- The sourceNPZ file hash matches the prior full-source audit; the currenttime_steps+charring_ratios byte hash matches the frozen997-case manifest. Geometry arrays and beam dimensions were separately checked against the same manifest.
- Onlytime_steps,charring_ratios,crack_params,beam_dims were read for the final source bundle. No temperature field or model result/prediction enters the figure.
- Native571 observations→running minimum→61 linearly interpolatedfloat32 values. Native maximum envelope correction0.01780059 occurs at1287.82605s; the20–23min detail contains this event. Do not confuse this native maximum with the preparation manifest's61-time maximum difference0.01545609.
- Panelc contains noP1polygons, section integration, retained-temperature-domain labels or manufactured prediction. PreviousP1method figure remains unchanged in the earlier directory.
- The seven variants match the frozen420-run plan; the GNN connectivity statement describes an evaluation design, not a claimed achieved accuracy.
- Final183×170mm, all textual labels8pt, editablePDF/SVG, vectorbeam/graph/curves. ActualPNG/PDF reviewed for label overlap and clipping; an initial fire-axis/transfer-note overlap was removed.
- Following root review, module labels readExposureLSTM andScalarDecoder to avoid implying parameter sharing across independently trained variants. The footer now states population size, fire-family count and crack-count range; geometry and quantitative charts are unchanged.
- SourceCSV/JSON, hashes, the beam-helper extraction and generator remain in this folder. Rebuilding rechecks source identity and does not edit any old figure, source data, experiment plan or manuscript.
