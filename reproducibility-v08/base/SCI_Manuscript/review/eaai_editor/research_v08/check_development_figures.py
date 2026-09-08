"""Independent development figure checks; reads no learned predictions/results."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
import collections
import csv
import hashlib
import json
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SCI = HERE.parents[2]
ROOT = SCI.parent
METHOD = SCI / 'figures_v08/method'
POPULATION = SCI / 'figures_v08/population'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest_path = SCI / 'experiments/research_v08/archive997/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    cases = manifest['cases']
    case = next(c for c in cases if c['sample_id'] == 'b004_0006')
    provenance = json.loads((METHOD / 'source_provenance.json').read_text(encoding='utf-8'))
    assert provenance['sample_id'] == case['sample_id']
    assert provenance['geometry_id'] == case['geometry_id']
    assert case['fire'] == {'type': 'iso834', 'params': {}}
    source_path = ROOT / case['source_npz']
    with np.load(source_path, allow_pickle=False) as source:
        times = source['time_steps']
        native = source['charring_ratios']
        crack_params = source['crack_params']
        beam_dims = source['beam_dims']
    assert len(times) == len(native) == 571
    assert np.array_equal(beam_dims.ravel(), [1.5, .14, .2])
    slots = rows(METHOD / 'geometry_source.csv')
    nodes = rows(METHOD / 'graph_nodes.csv')
    edges = rows(METHOD / 'graph_edges.csv')
    expected_params = [[float(c[k]) for k in ('face','z','h','w','l','d')] for c in case['cracks']]
    csv_params = [[float(c[k]) for k in ('face','z_center_m','local_h_m','width_m','length_m','depth_m')] for c in slots]
    assert np.array_equal(crack_params, expected_params)
    assert np.array_equal(crack_params, csv_params)
    centers = []
    for c, slot, node in zip(case['cracks'], slots, nodes):
        f, z, d, h = int(c['face']), c['z'], c['d'], c['h']
        q = -h if f in (0, 3) else h
        point = [q,.1-d/2,z] if f == 0 else [q,-.1+d/2,z] if f == 1 else [.07-d/2,q,z] if f == 2 else [-.07+d/2,q,z]
        assert np.allclose(point, [float(node[k]) for k in ('x_m','y_m','z_m')], atol=1e-15, rtol=0)
        xyz = ['x', 'y', 'z']
        assert np.allclose(point, [(float(slot[k+'_min_m'])+float(slot[k+'_max_m']))/2 for k in xyz], atol=1e-15, rtol=0)
        centers.append(np.asarray(point))
    assert {(int(e['source']), int(e['target'])) for e in edges} == {(i,j) for i in (1,2,3) for j in (1,2,3) if i != j}
    for edge in edges:
        i, j = int(edge['source'])-1, int(edge['target'])-1
        assert abs(float(edge['distance_m']) - np.linalg.norm(centers[i]-centers[j])) < 1e-14
        assert abs(float(edge['delta_z_m']) - abs(centers[i][2]-centers[j][2])) < 1e-14
    curve = rows(METHOD / 'native_scalar_and_envelope.csv')
    curve_array = np.asarray([[float(r[k]) for k in ('time_s','native_scalar','running_minimum')] for r in curve])
    envelope = np.minimum.accumulate(native)
    assert np.allclose(curve_array, np.column_stack((times,native,envelope)), atol=1e-12, rtol=0)
    target = rows(METHOD / 'target_61.csv')
    grid = np.arange(61)*60.
    target_csv = np.asarray([float(r['target_float32']) for r in target], np.float32)
    expected = np.interp(grid, times, envelope).astype(np.float32)
    assert np.array_equal([float(r['time_s']) for r in target], grid)
    assert np.array_equal(target_csv, expected)
    tensors_path = SCI / 'experiments/research_v08/archive997/tensors.pt'
    assert digest(tensors_path) == manifest['tensor_sha256']
    tensors = torch.load(tensors_path, map_location='cpu', weights_only=True)
    assert np.array_equal(tensors['targets'][case['sample_id']].numpy(), expected)
    gas = rows(METHOD / 'prescribed_iso834.csv')
    assert np.allclose([float(r['prescribed_gas_C']) for r in gas], 20+345*np.log10(8*grid/60+1), atol=1e-12, rtol=0)

    # Counts are reconstructed directly from the frozen manifest, independently of plotting aggregation.
    family_count = collections.Counter((c['fire_family'],c['num_cracks']) for c in cases)
    batch_count = collections.Counter((c['source_batch'],c['num_cracks']) for c in cases)
    for r in rows(POPULATION / 'fire_family_by_crack_count.csv'):
        assert int(r['cases']) == family_count[(r['fire_family'],int(r['crack_count']))]
    assert len(rows(POPULATION / 'fire_family_by_crack_count.csv')) == 150
    for r in rows(POPULATION / 'source_batch_by_crack_count.csv'):
        assert int(r['cases']) == batch_count[(r['source_batch'],int(r['crack_count']))]
    assert len(rows(POPULATION / 'source_batch_by_crack_count.csv')) == 30
    identifiers = rows(POPULATION / 'population_case_identifiers.csv')
    byid = {c['sample_id']:c for c in cases}
    assert len(identifiers) == len({r['sample_id'] for r in identifiers}) == 997
    assert set(byid) == {r['sample_id'] for r in identifiers}
    for r in identifiers:
        c = byid[r['sample_id']]
        for key in ('geometry_id','source_batch','fire_family'):
            assert r[key] == c[key]
        assert int(r['crack_count']) == c['num_cracks']
        assert (r['legacy911'] == 'True') == bool(c['legacy_final_id'])
        assert (r['flat_response'] == 'True') == c['flat_response']
    restored = [c for c in cases if c['legacy_final_id'] is None]
    high_count = [c for c in cases if c['num_cracks'] >= 9]
    assert len(restored) == 86 and all(c['flat_response'] and c['fire_family'] == 'smoldering' for c in restored)
    assert len(high_count) == 264 and all(c['source_batch'] == 'batch_005' for c in high_count)
    composition = rows(POPULATION / 'retained_and_restored_composition.csv')
    assert [(int(r['legacy_retained']),int(r['restored']),int(r['total'])) for r in composition] == [(911,86,997),(11,86,97)]
    related = SCI / 'latex_v08/sections/01_related_work.tex'
    engineering = SCI / 'latex_v08/sections/02_engineering_data.tex'
    related_text, engineering_text = related.read_text(encoding='utf-8'), engineering.read_text(encoding='utf-8')
    assert 'The present archival scalar envelope retains a scalar history convention rather than that material-point state.' in related_text
    assert 'thesis supplies the model comparison and reports temperature RMSEs of 36, 43 and 40' in engineering_text or ('temperature RMSE' in engineering_text and 'thesis-reported validation' in engineering_text)
    result = {
      'created_at_utc': datetime.now(timezone.utc).isoformat(),
      'review_type': 'DEVELOPMENT_QA_ONLY_NO_SUBMISSION_DECISION',
      'checks_passed': True, 'prediction_results_read': False,
      'read_scope': 'Frozen manifest/tensors; source case time_steps, charring_ratios, crack_params, beam_dims; figure small CSVs/provenance; current methods text. No full temperature array, solver, learned prediction or performance summary.',
      'method_case': case['sample_id'], 'geometry_rows':3, 'directed_edges':6,
      'native_points':571, 'reference_points':61,
      'native_envelope_max_difference':float(np.max(native-envelope)),
      'reference_vs_frozen_target_max_difference':float(np.max(abs(target_csv-tensors['targets'][case['sample_id']].numpy()))),
      'population_count':997, 'legacy_count':911, 'restored_count':86,
      'N9_15_count':264, 'N9_15_all_batch005':True,
      'development_prose_items': {'D08-P01':'RESOLVED', 'D08-P02':'RESOLVED', 'D08-P03':'Implementation additions recorded separately; actual complete-protocol statistical results not reviewed here.'},
      'source_sha256': {str(p.relative_to(SCI)).replace('\\','/'):digest(p) for p in (manifest_path, METHOD/'draw_method_v08.py',METHOD/'method_v08.pdf',METHOD/'caption_en.tex',POPULATION/'population_v08.pdf',POPULATION/'caption_en.tex',related,engineering)}
    }
    output = HERE / 'development_figure_checks.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
