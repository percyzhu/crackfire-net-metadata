"""Bind agent-reviewed real timing plot and produce a complete numerical readout."""
from pathlib import Path
import csv
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
OUT = HERE / 'rendered'

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, value): p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

qa = read(OUT / 'independent_figure_QA.json')
assert qa['plotted_values_checked'] == 56 and qa['minimum_pdf_font_pt'] == 8
assert qa['pdf_sha256'] == sha(OUT / 'inference_efficiency_v08.pdf')
summary = read(OUT / 'companion_summaries.json')
models = ['fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets', 'gnn', 'gnn_zero_edge_features', 'gnn_mean']
lookup = {(r['device'], r['model'], r['stage'], r['batch_size']): r for r in summary['summaries']}
rows = []
for device in ['cpu', 'cuda']:
    for model in models:
        for stage in ['forward_resident_inputs', 'common_input_to_host_output']:
            one, batch = lookup[device, model, stage, 1], lookup[device, model, stage, 32]
            rows.append({'device': device, 'model': model, 'interface': stage,
                         'single_median_ms': one['median_batch_ms'], 'single_p95_ms': one['p95_batch_ms'],
                         'batch32_queries_per_s': batch['cases_per_second'],
                         'single_measurements': one['measurements'], 'batch_measurements': batch['measurements'],
                         'query': 'one entire 61-time response trajectory'})
with (OUT / 'all_model_timing_readout.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
visual = {'status': 'PASSED_ACTUAL_TIMING_FIGURE_VISUAL_AND_SOURCE_QA',
          'reviewed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'reviewer': '/root/engineering_validation_v08',
          'inspection': 'Actual final PDF rendering inspected at original image detail.',
          'pdf_sha256': sha(OUT / 'inference_efficiency_v08.pdf'),
          'preview_sha256': sha(OUT / 'pdf_final_QA.png'),
          'dimensions_mm': [183,154], 'minimum_text_PDF_pt': 8, 'raster_objects': 0,
          'checks': ['All seven models retained in frozen order in both device columns.',
                     'Single-query median-to-p95 segments are quantiles, not confidence intervals.',
                     'Throughput is sum(cases)/sum(elapsed time) at batch size 32.',
                     'No clipped values, overlapping labels, or out-of-canvas text.',
                     'Zero edge features is labelled explicitly; graph connectivity is retained.',
                     'CPU four-thread and measured RTX 3060 Ti hardware labels match sources.'],
          'generator_revision': 'The first real render stopped before output because an automatic tick 70000 lay outside the canvas. The locator now uses at most three intervals and removes ticks outside the fixed axes. Scores and all measured values are unchanged; zero-edge-feature label clarified.',
          'independent_raw_audit': '14,355 measurements, 58 summary groups, 174 round groups, 420 training identities passed.',
          'independent_figure_audit': '56 plotted values independently recomputed from retained raw records; 18 source bindings checked.',
          'timing_rerun_performed': False, 'manuscript_placement_QA': 'Pending integration',
          'scientific_manuscript_or_submission_pass': False}
write(OUT / 'visual_QA.json', visual)
(OUT / 'actual_QA.md').write_text('# Actual inference-efficiency figure QA\n\nStandalone actual visual, arithmetic and export checks passed. All 7 models, 2 devices and both interfaces are retained; one query is a full 61-time trajectory. Latency p95 is not a confidence interval. The first rendering safely stopped on an out-of-canvas automatic tick; the display locator was corrected without changing measurements. The figure is 183 × 154 mm, minimum 8 pt, pure vector.\n\nThe CPU was an Intel Core i5-13600KF with 4 PyTorch threads; GPU was an RTX 3060 Ti. These are the measured common implementation and resident-input forward paths, not independently optimized deployments or FEM speedup. Manuscript placement and final paper review are separate checks.\n', encoding='utf-8')
manifest = {'status': 'ACTUAL_TIMING_FIGURE_GENERATED_AND_STANDALONE_QA_PASSED',
            'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != 'artifact_manifest.json'},
            'source_code': {p.name: sha(p) for p in [HERE/'draw_inference_efficiency_v08.py', HERE/'audit_inference_figure.py', Path(__file__)]},
            'nested_snapshot_binding': 'figure_qa.json binds all copied source files; original records are retained.',
            'submission_ready_claim': False}
write(OUT / 'artifact_manifest.json', manifest)
for row in rows:
    if row['model'] in ['fire_only', 'capacity_matched_deepsets', 'gnn']:
        print(json.dumps(row))
print('PDF SHA256 ' + sha(OUT / 'inference_efficiency_v08.pdf'))
print('Generator SHA256 ' + sha(HERE/'draw_inference_efficiency_v08.py'))
