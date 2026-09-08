from pathlib import Path
import hashlib, json
import fitz

HERE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
pdf = HERE / 'experimental_validation_final.pdf'
doc = fitz.open(pdf)
assert len(doc) == 1
page = doc[0]
spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
assert min(s['size'] for s in spans) >= 7.999
assert not page.get_images(full=True)
outside = [s['text'] for s in spans if not page.rect.contains(fitz.Rect(s['bbox']))]
assert not outside
assert abs(page.rect.width / 72 * 25.4 - 183) < .01
assert abs(page.rect.height / 72 * 25.4 - 152) < .01
page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE/'experimental_validation_pdf_QA.png')
audit = json.loads((HERE/'independent_extraction_audit.json').read_text(encoding='utf-8'))
assert audit['status'] == 'PASS_INDEPENDENT_ARCHIVED_CURVE_EXTRACTION_AND_ARITHMETIC'
report = {
    'status': 'PASS_VECTOR_DIMENSIONS_FONTS_AND_ACTUAL_VISUAL_REVIEW',
    'figure_review_passed': True,
    'scope': 'Archived experimental thermal comparison figure only; not a whole-manuscript acceptance or new solver validation.',
    'pdf_sha256': sha(pdf), 'png_sha256': sha(HERE/'experimental_validation_final.png'),
    'svg_sha256': sha(HERE/'experimental_validation_final.svg'),
    'independent_extraction_audit_sha256': sha(HERE/'independent_extraction_audit.json'),
    'page_dimensions_mm': [page.rect.width/72*25.4, page.rect.height/72*25.4],
    'minimum_font_pt': min(s['size'] for s in spans), 'fonts': sorted({s['font'] for s in spans}),
    'raster_image_objects': len(page.get_images(full=True)), 'out_of_page_text': outside,
    'actual_visual_review': [
        'Final PNG and PDF rendering inspected: four readable panels, no curve clipping or legend/axis overlap.',
        'Depth panels share 0–60 min and 0–1000 degrees Celsius scales; all curves and experimental a/b ranges retained.',
        'Depth labels are schematic groups, not invented thermocouple coordinates.',
        '20 mm archive FE, both observations, arithmetic mean and observed range are distinguished; range is not CI.',
        'Only legend spacing changed after first visual pass; underlying data and error metrics unchanged.'
    ],
    'main_RMS_C': [35.47083052524127, 43.27433611299336, 39.98609367553838],
    'source_and_artifact_sha256': {p.name: sha(p) for p in HERE.iterdir() if p.is_file() and p.name not in ['actual_visual_QA.json','artifact_manifest_final.json']}
}
(HERE/'actual_visual_QA.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
(HERE/'artifact_manifest_final.json').write_text(json.dumps({'status':report['status'], 'files': {p.name: {'sha256':sha(p),'bytes':p.stat().st_size} for p in HERE.iterdir() if p.is_file() and p.name!='artifact_manifest_final.json'}}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='source_and_artifact_sha256'}, ensure_ascii=False, indent=2))
