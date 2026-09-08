"""Run after real all-family generation, before visual review and publication."""
from pathlib import Path
import hashlib
import json
import fitz

HERE = Path(__file__).resolve().parent
pdf = HERE / 'fire_transfer_v08.pdf'
if not pdf.exists():
    raise SystemExit('NO_COMPLETE_FIGURE: finish all-family evaluation and generation first.')
doc = fitz.open(pdf)
assert len(doc) == 1
page = doc[0]
size = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
assert abs(size[0] - 183) < .001 and abs(size[1] - 190) < .001
spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
font_min = min(s['size'] for s in spans)
assert font_min >= 7.99999 and not page.get_images(full=True)
text = page.get_text()
for expected in ['ISO 834', 'ASTM E119', 'Smoldering', 'Case-weighted', 'Equal-family', '350 complete audited runs', 'unadjusted']:
    assert expected in text
page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / 'pdf_final_QA.png')
result = {'status': 'PASSED_EXPORT_PENDING_VISUAL_REVIEW', 'size_mm': size,
          'minimum_pdf_font_pt': font_min, 'raster_objects': 0,
          'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
          'preview': 'pdf_final_QA.png', 'requires_human_or_agent_visual_review': True}
(HERE / 'pdf_export_QA.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
