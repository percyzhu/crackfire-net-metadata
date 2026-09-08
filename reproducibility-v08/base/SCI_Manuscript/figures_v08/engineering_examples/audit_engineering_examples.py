"""Independent table-level selection/envelope/event checks and vector export QA.

Uses retained score components and individual seed curves, not the generator's
selection or plotting functions. Never substitutes unavailable data.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import fitz

HERE = Path(__file__).resolve().parent
files = ['all_candidate_scores.csv', 'selection_score_components.csv', 'selected_cases.csv',
         'curve_data.csv', 'all_seed_curves.csv', 'threshold_events.csv', 'engineering_examples_v08.pdf', 'provenance.json',
         'figure_qa.json', 'selected_fire_parameters.json']
if not all((HERE / name).exists() for name in files):
    raise SystemExit('NO_COMPLETE_EXAMPLES: finish all ten reviewed family protocols and generate actual examples first.')

def load(name): return pd.read_csv(HERE / name, float_precision='round_trip')
def close(a, b): assert np.allclose(a, b, atol=2e-13, rtol=2e-12, equal_nan=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

scores, pieces, selected = load(files[0]), load(files[1]), load(files[2])
curves, all_seed, events = load(files[3]), load(files[4]), load(files[5])
assert len(scores) == scores.sample_id.nunique() == 997 and len(pieces) == 4985
assert scores.geometry_id.nunique() == 997
assert np.isfinite(scores[['mean_case_MAE', 'mean_positive_peak']]).all().all()
assert scores[['mean_case_MAE', 'mean_positive_peak']].ge(0).all().all()
assert set(scores.sample_id) == set(pieces.sample_id)
assert not pieces.duplicated(['sample_id', 'seed']).any()
assert set(pieces.seed) == {42, 43, 44, 45, 46}
for row in scores.itertuples():
    source = pieces[pieces.sample_id == row.sample_id].sort_values('seed')
    assert len(source) == 5
    assert set(source.protocol) == {row.protocol}
    close(source.case_MAE.to_numpy().mean(), row.mean_case_MAE)
    close(source.positive_peak.to_numpy().mean(), row.mean_positive_peak)
ordered = sorted(scores.to_dict('records'), key=lambda r: (r['mean_case_MAE'], r['sample_id']))
chosen = []
for rank in [498, 946, 996]:
    value = ordered[rank]['mean_case_MAE']
    candidates = sorted(r['sample_id'] for r in ordered if r['mean_case_MAE'] == value and r['sample_id'] not in chosen)
    assert candidates; chosen.append(candidates[0])
remaining = [r for r in ordered if r['sample_id'] not in chosen]
peak = max(r['mean_positive_peak'] for r in remaining)
chosen.append(min(r['sample_id'] for r in remaining if r['mean_positive_peak'] == peak))
assert selected.sort_values('example').sample_id.tolist() == chosen and len(set(chosen)) == 4
assert sorted(selected.example) == [1, 2, 3, 4]
for number, row in enumerate(selected.sort_values('example').itertuples(), 1):
    original = scores[scores.sample_id == row.sample_id].iloc[0]
    for key in ('protocol', 'geometry_id', 'fire_family', 'source_batch', 'crack_count', 'flat_response', 'restored'):
        assert getattr(row, key) == original[key]
    expected_kind = 'mean_case_MAE' if number < 4 else 'mean_positive_peak'
    assert row.selection_kind == expected_kind
    close(row.selection_score, original[expected_kind])
    if number < 4:
        assert row.rank_zero_based == [498, 946, 996][number - 1] and not row.no_remaining_positive_error
    else:
        assert np.isnan(row.rank_zero_based) and bool(row.no_remaining_positive_error) == (peak == 0)
assert len(curves) == 244 and len(all_seed) == 2440 and len(events) == 24
assert not all_seed.duplicated(['example', 'model', 'seed', 'time_s']).any()
assert not events.duplicated(['example', 'series', 'threshold']).any()
for item in selected.itertuples():
    sample = curves[curves.example == item.example].sort_values('time_s')
    assert len(sample) == 61 and set(sample.sample_id) == {item.sample_id}
    assert np.array_equal(sample.time_s, np.arange(61) * 60)
    for model in ['capacity_matched_deepsets', 'gnn']:
        block = all_seed[(all_seed.example == item.example) & (all_seed.model == model)]
        assert len(block) == 305 and set(block.seed) == {42, 43, 44, 45, 46}
        assert set(block.sample_id) == {item.sample_id}
        for _, trace in block.groupby('seed'): assert np.array_equal(trace.sort_values('time_s').time_s, sample.time_s)
        matrix = np.vstack([block[block.seed == s].sort_values('time_s').prediction for s in [42, 43, 44, 45, 46]])
        close(matrix[0], sample[model + '_seed42']); close(matrix.min(axis=0), sample[model + '_min']); close(matrix.max(axis=0), sample[model + '_max'])
        error = matrix[0] - sample.reference.to_numpy()
        close(abs(error).mean(), getattr(item, model + '_seed42_MAE'))
        close(np.maximum(error, 0).max(), getattr(item, model + '_seed42_positive_peak'))
    for model in ['reference', 'capacity_matched_deepsets', 'gnn']:
        values = sample.reference.to_numpy() if model == 'reference' else sample[model + '_seed42'].to_numpy()
        for q in [.8, .6]:
            record = events[(events.example == item.example) & (events.series == model) & (events.threshold == q)].iloc[0]
            assert record.sample_id == item.sample_id
            first = next((k for k, value in enumerate(values) if value <= q), None)
            assert bool(record.event) == (first is not None)
            if first is None:
                assert np.isnan(record.first_grid_time_s) and np.isnan(record.response_at_event)
                assert record.censor_label == '>3600 s' and not record.recross
            else:
                close(record.first_grid_time_s, first * 60); close(record.response_at_event, values[first])
                assert bool(record.recross) == any(value > q for value in values[first + 1:])
provenance = json.loads((HERE / 'provenance.json').read_text(encoding='utf-8'))
assert provenance['selection_plan_sha256'] == '1850acfce80828fa7db4cd7fc751e9c86b8c47f8366fa3c91cc24135ce5d8c1e'
for source in provenance['source_snapshot']: assert sha(HERE / source['snapshot_path']) == source['sha256']
for name, digest in provenance['derived_artifact_sha256'].items(): assert sha(HERE / name) == digest
manifest_source = next(x for x in provenance['source_snapshot'] if Path(x['original_path']).name == 'manifest.json')
manifest = json.loads((HERE / manifest_source['snapshot_path']).read_text(encoding='utf-8'))
assert set(scores.sample_id) == {c['sample_id'] for c in manifest['cases']}
for protocol, group in scores.groupby('protocol'):
    source = next(x for x in provenance['source_snapshot'] if Path(x['original_path']).name == protocol + '.json')
    split = json.loads((HERE / source['snapshot_path']).read_text(encoding='utf-8'))
    assert set(group.sample_id) == set(split['test']) and len(group) == len(split['test'])
prediction_sources = provenance['prediction_and_checkpoint_sources']
assert len(prediction_sources) == 100
assert len({(x['protocol'], x['model'], x['seed']) for x in prediction_sources}) == 100
for item in selected.itertuples():
    sources = [x for x in prediction_sources if x['protocol'] == item.protocol]
    assert {(x['model'], x['seed']) for x in sources} == {
        (m, s) for m in ['capacity_matched_deepsets', 'gnn'] for s in [42, 43, 44, 45, 46]}
    for source in sources:
        expected_parent = Path(source['protocol']) / source['model'] / ('seed_' + str(source['seed']))
        for name in ('prediction_path', 'checkpoint_path'):
            assert tuple(Path(source[name]).parent.parts[-3:]) == expected_parent.parts
figure_qa = json.loads((HERE / 'figure_qa.json').read_text(encoding='utf-8'))
response_columns = curves.filter(regex='reference|_seed42|_min$|_max$')
raw_min, raw_max = float(response_columns.min().min()), float(response_columns.max().max())
close([raw_min, raw_max], figure_qa['raw_response_extrema'])
lo, hi = figure_qa['response_axis_limits']
assert lo < raw_min <= raw_max < hi and not figure_qa['response_clipped_to_unit_interval']
doc = fitz.open(HERE / 'engineering_examples_v08.pdf'); assert len(doc) == 1
page = doc[0]; size = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
assert abs(size[0] - 183) < .001 and abs(size[1] - 200) < .001
spans = [s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
font_min = min(s['size'] for s in spans); assert font_min >= 7.99999 and not page.get_images(full=True)
page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / 'pdf_final_QA.png')
result = {'status': 'PASSED_INDEPENDENT_NUMERICAL_AND_VECTOR_CHECKS_PENDING_VISUAL_REVIEW',
    'selection_population': 997, 'selection_score_components': 4985, 'ranks': [498, 946, 996],
    'selected_cases': chosen, 'curve_rows': 244, 'individual_seed_curve_rows': 2440, 'event_records': 24,
    'seed42_is_actual_trace': True, 'pointwise_extrema_recomputed': True,
    'events_recomputed_without_interpolation': True, 'right_censoring_preserved': True,
    'all_raw_response_extrema_inside_axes': True, 'response_axis_limits': [lo, hi],
    'size_mm': size, 'minimum_pdf_font_pt': font_min, 'raster_objects': 0,
    'pdf_sha256': sha(HERE / 'engineering_examples_v08.pdf'), 'full_temperature_payload_read': False,
    'remaining': 'Inspect pdf_final_QA.png at final size; inspect caption in manuscript; create final artifact manifest.'}
(HERE / 'independent_QA.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
