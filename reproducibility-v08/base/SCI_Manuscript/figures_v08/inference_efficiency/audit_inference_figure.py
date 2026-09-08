"""Independent figure-number/export audit; never runs a timed operation."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

import numpy as np
import fitz

HERE = Path(__file__).resolve().parent
OUT = HERE / 'rendered'
MODELS = ['fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets',
          'gnn', 'gnn_zero_edge_features', 'gnn_mean']


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_rows(path):
    with Path(path).open(encoding='utf-8', newline='') as stream: return list(csv.DictReader(stream))


def main():
    required = ['figure_qa.json', 'plotted_values.csv', 'inference_efficiency_v08.pdf']
    if not all((OUT / name).is_file() for name in required):
        print('WAITING: no complete real figure; zero raw timing records read.')
        return 2
    qa = read(OUT / 'figure_qa.json')
    assert qa['status'] == 'RENDERED_PENDING_INDEPENDENT_AND_VISUAL_REVIEW'
    sources = qa['source_snapshot']
    assert len({x['original_path'] for x in sources}) == len(sources)
    for item in sources:
        assert sha(OUT / item['snapshot_path']) == qa['source_sha256'][item['original_path']] == item['sha256']
    for name, digest in qa['derived_artifact_sha256'].items(): assert sha(OUT / name) == digest
    def source(name):
        matches = [x for x in sources if Path(x['original_path']).name == name]
        assert len(matches) == 1, name
        return OUT / matches[0]['snapshot_path']
    audit = read(source('inference_timing_independent_audit.json'))
    assert audit['status'] == 'REAL_TIMING_RAW_RECORDS_COVERAGE_AND_ARITHMETIC_AUDIT_PASSED'
    assert audit['raw_measurements_checked'] == 14355 and audit['training_runs'] == 420
    for path, digest in audit['source_sha256'].items(): assert qa['source_sha256'][path] == digest
    timing = read(source('timing_plan.json')); environment = read(source('environment.json'))
    assert timing['rounds'] == 3 and len(timing['sample_ids']) == len(set(timing['sample_ids'])) == 160
    assert timing['times_per_case'] == 61 and timing['seed'] == 42
    assert environment['GPU'] == qa['measured_GPU'] and environment['threads'] == qa['CPU_threads'] == 4
    raw = csv_rows(source('raw_timings.csv')); plotted = csv_rows(OUT / 'plotted_values.csv')
    assert len(raw) == 14355 and len(plotted) == 56
    expected = {(d, m, 'common_input_to_host_output', 1, metric) for d in ['cpu', 'cuda'] for m in MODELS
                for metric in ['median_batch_ms', 'p95_batch_ms']}
    expected.update((d, m, stage, 32, 'cases_per_second') for d in ['cpu', 'cuda'] for m in MODELS
                    for stage in ['forward_resident_inputs', 'common_input_to_host_output'])
    keys = [(x['device'], x['model'], x['stage'], int(x['batch_size']), x['metric']) for x in plotted]
    assert len(set(keys)) == 56 and set(keys) == expected
    for item, key in zip(plotted, keys):
        device, model, stage, size, metric = key
        matched = [r for r in raw if (r['device'], r['model'], r['stage'], int(r['batch_size'])) == key[:4]]
        assert len(matched) == int(item['measurements']) == (480 if size == 1 else 15)
        assert sum(int(r['cases']) for r in matched) == int(item['cases_total']) == 480
        values = np.asarray([float(r['elapsed_s']) for r in matched])
        assert np.isfinite(values).all() and (values > 0).all()
        if metric == 'median_batch_ms': expected_value = float(np.median(values) * 1000)
        elif metric == 'p95_batch_ms': expected_value = float(np.percentile(values, 95) * 1000)
        else: expected_value = 480 / math.fsum(float(r['elapsed_s']) for r in matched)
        assert math.isclose(float(item['value']), expected_value, rel_tol=1e-11, abs_tol=1e-12)
        assert item['unit'] == ('ms per 61-time query' if size == 1 else '61-time queries per second')
        limits = qa['axes_limits'][device]['single_latency_xlim' if size == 1 else 'batch_throughput_xlim']
        assert limits[0] == 0 < expected_value < limits[1]
    doc = fitz.open(OUT / 'inference_efficiency_v08.pdf'); assert len(doc) == 1
    page = doc[0]; dimensions = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
    assert np.allclose(dimensions, [183, 154], atol=.001, rtol=0)
    spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
    minimum = min(s['size'] for s in spans); assert minimum >= 7.99999 and not page.get_images(full=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(OUT / 'pdf_final_QA.png')
    result = {'status': 'PASSED_INDEPENDENT_FIGURE_ARITHMETIC_AND_EXPORT_PENDING_VISUAL_QA',
              'plotted_values_checked': 56, 'raw_records_preserved': 14355, 'timed_models_executed': 0,
              'single_query_unit': 'ms per full 61-time trajectory', 'p95_is_confidence_interval': False,
              'throughput_definition': 'sum(cases) / sum(elapsed_seconds)', 'all_plotted_values_inside_axes': True,
              'snapshot_sources_checked': len(sources), 'dimensions_mm': dimensions, 'minimum_pdf_font_pt': minimum,
              'raster_objects': 0, 'pdf_sha256': sha(OUT / 'inference_efficiency_v08.pdf'),
              'visual_QA_passed': False, 'remaining': 'Inspect actual PDF at final size and the embedded manuscript page.'}
    (OUT / 'independent_figure_QA.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2)); return 0


if __name__ == '__main__': sys.exit(main())
