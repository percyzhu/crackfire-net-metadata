"""Record the actual inspected PDF and hash-bind the final budget figure."""
from pathlib import Path
import datetime
import hashlib
import json

HERE=Path(__file__).resolve().parent
OUT=HERE/'rendered'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

audit=read(OUT/'independent_figure_QA.json')
assert audit['plotted_cells']==216 and audit['minimum_pdf_font_pt']==8
assert audit['pdf_sha256']==sha(OUT/'priority_budget_v08.pdf')
visual={'status':'PASSED_ACTUAL_COMPLETE_BUDGET_MATRIX_VISUAL_AND_INDEPENDENT_ARITHMETIC_QA',
        'reviewer':'/root/engineering_validation_v08','reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'pdf_sha256':sha(OUT/'priority_budget_v08.pdf'),'inspected_preview_sha256':sha(OUT/'pdf_final_QA.png'),
        'inspection':'Actual final PDF rendering inspected at original image detail.',
        'dimensions_mm':[183,150],'minimum_pdf_font_pt':8,'raster_objects':0,
        'checks':['All three fixed models, twelve protocols and six endpoint/budget columns are visible.',
                  'All216 numeric cells are legible; redundant .0 labels were removed after the initial visual check to improve cell spacing.',
                  'Panel titles, endpoint groups, nominal budget labels, test counts and shared colorbar do not overlap or clip.',
                  'Raw ranges include tiny floating-point excursions beyond0 and100; no source value is clipped.',
                  'No undefined cells occur; the NA handling rule remains stated.',
                  'The displayed percentage measures available oracle improvement, not risk reduction or time savings.',
                  'The caption maps manuscript eta_b to the unchanged source CSV metric L.'],
        'independent_checks':'2520seed rows,504all-model summaries×5metrics,72reference budgets and216shown cells reconciled without importing the generator.',
        'all7_source_tables_retained':True,'original420_or_sealed_package_modified':False,
        'new_model_or_timing_executions':0,'manuscript_placement_QA':'Pending root integration',
        'scientific_manuscript_or_submission_pass':False}
write(OUT/'visual_QA.json',visual)
(OUT/'actual_QA.md').write_text('# Actual budget-priority figure QA\n\nPassed independent arithmetic, vector export and actual standalone visual review. The three fixed panels retain all12 protocols, both endpoints and all3 budgets (216 cells). Font size is at least8pt at183×150mm. All7 models remain in companion seed/summary tables containing U, regret, gain and oracle-improvement fraction. The separate checker reconciled2520seed rows,504summaries over5metrics,72references and all216plotted numbers without importing the generator or reading prediction arrays.\n\nThe actual raw percentage range is approximately −1.288×10^-12 to100.00000000000537, reflecting only machine-rounding excursions; the common scale includes these raw values. No undefined ratios occur. Cells are rounded for display only. The first rendering was preserved in backup_initial_decimal_labels; redundant trailing .0 was then omitted to improve adjacent-cell spacing. No statistical value or selected model/protocol changed.\n\nThe PDF rendering was inspected at original detail: labels, numbers, endpoint/budget groups, common colorbar and footnotes are legible, with no overlap or clipping. Manuscript eta_b corresponds to unchanged CSV L. Exact-tie expected selection and post-review exploratory status are explicit. The figure does not claim geometric uncertainty intervals, risk reduction, wall-time savings or one common checkpoint across protocol holdouts. Final manuscript placement is a separate root check.\n',encoding='utf-8')
files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'}
sources={p.name:sha(p) for p in [HERE/'draw_priority_budget_v08.py',HERE/'audit_priority_budget_figure.py',HERE/'figure_contract.md',HERE/'caption_en.tex',Path(__file__)]}
write(OUT/'artifact_manifest.json',{'status':'ACTUAL_COMPLETE_BUDGET_FIGURE_AND_COMPANION_TABLES_STANDALONE_QA_PASSED',
                                'files':files,'local_source_sha256':sources,
                                'nested_sources':'provenance.json binds all retained source snapshot files.',
                                'original420_and_sealed_package_modified':False,'submission_pass':False})
print(json.dumps({'PDF':sha(OUT/'priority_budget_v08.pdf'),'SVG':sha(OUT/'priority_budget_v08.svg'),
                  'manifest':sha(OUT/'artifact_manifest.json'),'local_source_sha256':sources},indent=2))
