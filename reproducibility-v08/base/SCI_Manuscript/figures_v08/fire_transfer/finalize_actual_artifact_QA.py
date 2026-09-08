"""Record the completed agent visual inspection and bind actual figure/table outputs.

This script does not generate or alter scientific results. The visual decisions
below were made by inspecting the rendered final PDFs at original image detail.
"""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
STAMP = datetime.now(timezone.utc).isoformat()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


records = [
    (ROOT / "figures_v08/fire_transfer", "fire_transfer_v08", {
        "size_mm": [183, 190], "minimum_font_PDF_pt": 8.0,
        "checks": [
            "All 84 heatmap cells and seven model labels are legible.",
            "The 20 primary intervals share a symmetric axis and are all included without clipping.",
            "The percentage-point scale and 86/97 smoldering flat-case annotation are explicit.",
            "No overlapping labels, clipped marks, or raster objects were observed.",
            "Wide linear/bilinear intervals compress small effects near zero; the shared scale is retained faithfully."
        ],
        "arithmetic_evidence": "independent_arithmetic_QA.json",
        "limits": ["The descriptive case-weighted and equal-family aggregates have no invented confidence interval.",
                   "All 20 family-specific primary intervals cross zero; this figure does not establish uniform model superiority."]
    }),
    (ROOT / "figures_v08/engineering_examples", "engineering_examples_v08", {
        "size_mm": [183, 200], "minimum_font_PDF_pt": 8.0,
        "checks": [
            "The four response panels and aligned prescribed-fire strips are readable and separated.",
            "All raw seed predictions and five-seed ranges fit inside the displayed axes [-0.04, 1.04].",
            "The real seed-42 traces, five-seed min-max bands (not confidence intervals), and fixed selection scores are distinguished.",
            "No label overlap, clipping, or raster objects were observed; bottom annotations remain inside the page."
        ],
        "arithmetic_evidence": "independent_QA.json",
        "limits": ["The selection is conditioned on the sum-GNN error across the complete 997-case cross-holdout population.",
                   "The archived response index is not a validated structural capacity or safety criterion.",
                   "Three adverse bilinear-fire cases emerged from the fixed selection rule; the examples are not a controlled causal comparison."]
    }),
    (ROOT / "tables_v08/engineering_screening/generated", "preview", {
        "pages": 1, "body_font_TeX_pt": 9,
        "body_font_min_PDF_bp": 8.966400146484375,
        "math_subscript_min_PDF_bp": 5.97760009765625,
        "checks": [
            "Both tables and their captions fit on one page and remain readable.",
            "No overfull boxes, out-of-page text, overlapping labels, or raster objects were detected.",
            "The 10^-3 effect scale, confidence limits, counts, and NA entries agree with the checked source values.",
            "Smoldering q=0.6 has zero reference events and NA event coverage; counts are not multiplied by five seeds."
        ],
        "arithmetic_evidence": "independent_arithmetic_QA.json",
        "limits": ["Maximum FN/FP and minimum common-event counts may occur in different seeds.",
                   "Secondary intervals are unadjusted and do not replace the prespecified primary comparisons.",
                   "The 9-TeX-point body equals approximately 8.966 PDF big points; smaller mathematical subscripts are standard typography."],
        "checker_corrections": [
            "The new independent table checker initially referenced c instead of the defined cens variable; this checker-only typo was fixed before the successful audit.",
            "The one-off export check threshold was corrected for the TeX-point/PDF-big-point conversion; the table font and layout were not changed."
        ]
    })
]

label_revision = '--fire-label-revision' in sys.argv
if label_revision:
    records = records[:1]
    records[0][2]['checks'].append('The revised Zero edge / features column header fits on two lines; the caption explicitly states that graph connectivity is retained.')
    backup = records[0][0] / 'backup_before_zero_edge_label'
    data_files = sorted(records[0][0].glob('*.csv'))
    assert all(digest(p) == digest(backup / p.name) for p in data_files)
    records[0][2]['label_revision'] = {
        'old': 'GNN / zero-edge', 'new': 'Zero edge / features',
        'all_source_CSVs_byte_identical': True,
        'source_CSV_count': len(data_files), 'backup': str(backup.name),
        'scope': 'Column display label and caption clarification only; no frozen training or scores changed.'
    }

for folder, base, details in records:
    pdf = folder / f"{base}.pdf"
    png = folder / "pdf_final_QA.png"
    qa = {
        "status": "PASSED_ACTUAL_STANDALONE_VISUAL_AND_SOURCE_ARITHMETIC_QA",
        "reviewed_utc": STAMP,
        "reviewer": "/root/engineering_validation_v08",
        "inspection": "Agent inspected the rasterized final PDF at original image detail; vector and source arithmetic were separately checked.",
        "pdf_sha256": digest(pdf), "inspected_render_sha256": digest(png),
        "source_generator_modified_in_actual_generation": label_revision,
        "manuscript_placement_QA": "Pending integration; this record is standalone artifact QA, not manuscript acceptance.",
        **details,
    }
    write_json(folder / "visual_QA.json", qa)
    md = ["# Actual artifact QA", "", "Standalone visual and independent arithmetic checks passed.", "",
          f"Reviewed UTC: {STAMP}", "", f"PDF SHA256: `{qa['pdf_sha256']}`", ""]
    md += [f"- {x}" for x in details["checks"]]
    md += ["", "Interpretation boundaries:", ""] + [f"- {x}" for x in details["limits"]]
    if details.get("checker_corrections"):
        md += ["", "Checker-only corrections:", ""] + [f"- {x}" for x in details["checker_corrections"]]
    scope = ('Only the display label and caption were clarified. All source CSVs are byte-identical to the preserved previous export; frozen source evidence was not modified.'
             if label_revision else 'The original generators and frozen source evidence were not modified.')
    md += ["", scope + " Final manuscript placement and manuscript expert review remain separate checks.", ""]
    (folder / "actual_QA.md").write_text("\n".join(md), encoding="utf-8")
    selected = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name != "artifact_manifest.json" and p.suffix.lower() in {".pdf", ".svg", ".png", ".csv", ".json", ".tex", ".py", ".md"}:
            selected.append({"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)})
    external_checks = []
    if base == "preview":
        for name in ("generate_engineering_tables.py", "audit_generated_tables.py"):
            p = folder.parent / name
            external_checks.append({"path": "../" + name, "sha256": digest(p)})
    write_json(folder / "artifact_manifest.json", {
        "status": "ACTUAL_ARTIFACTS_GENERATED_AND_STANDALONE_QA_PASSED",
        "created_utc": STAMP, "files": selected, "external_code": external_checks,
        "source_binding": "The provenance or generation_manifest file binds the complete source snapshots and review evidence; nested snapshots are not duplicated in this top-level artifact list.",
        "frozen_training_modified": False, "timing_executed": False,
        "manuscript_placement_checked": False,
    })
    print(json.dumps({"folder": str(folder), "pdf_sha256": digest(pdf),
                      "manifest_sha256": digest(folder / "artifact_manifest.json")}, ensure_ascii=False))
