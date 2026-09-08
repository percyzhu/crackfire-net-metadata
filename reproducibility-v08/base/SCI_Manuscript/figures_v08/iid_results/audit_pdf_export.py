"""Audit final-size text/vector export and make a PDF-derived visual preview."""
from pathlib import Path
import hashlib
import json
import fitz

HERE = Path(__file__).resolve().parent
pdf = HERE / 'iid_results_v08.pdf'
doc = fitz.open(pdf)
assert len(doc) == 1
page = doc[0]
size_mm = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
font_min = min(s['size'] for s in spans)
assert abs(size_mm[0] - 183) < .001 and abs(size_mm[1] - 146) < .001
assert font_min >= 7.99999
assert len(page.get_images(full=True)) == 0
for text in ['Matched DeepSets', 'zero edge features', 'mean messages', '35-run IID', '95%']:
    assert text in page.get_text()
page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / 'pdf_final_QA.png')
qa = {'status': 'PASSED_VECTOR_AND_FINAL_SIZE_AUDIT', 'size_mm': size_mm,
      'minimum_pdf_font_pt': font_min, 'raster_objects': len(page.get_images(full=True)),
      'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(), 'visual_preview': 'pdf_final_QA.png'}
(HERE / 'pdf_export_QA.json').write_text(json.dumps(qa, indent=2) + '\n', encoding='utf-8')
print(json.dumps(qa, indent=2))
