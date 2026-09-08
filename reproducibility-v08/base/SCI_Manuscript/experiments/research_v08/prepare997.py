"""Prepare all 997 nonempty production scalar responses, including 86 flats."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import time
import numpy as np
import torch
import research_runner as rr


def create(output):
    output = rr.safe_output(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refuse to overwrite the snapshot')
    output.mkdir(parents=True, exist_ok=True)
    archive = rr.WORKSPACE / 'Abaqus_Archive_20260906/abaqus'
    mapping_path = rr.SCI / 'audit/temperature_coverage/npz_legacy_target_mapping.csv'
    records = list(csv.DictReader(mapping_path.open(encoding='utf-8-sig')))
    m4p = archive / 'data/batch_004/params/batch_manifest.json'
    m5p = archive / 'data/batch_005/params/batch_manifest.json'
    m4 = json.loads(m4p.read_text(encoding='utf-8'))
    m5 = json.loads(m5p.read_text(encoding='utf-8'))
    m5byid = {int(c['id']): c for c in m5['samples']}
    ns = rr.legacy_functions(rr.WORKSPACE)
    times = np.linspace(0., 3600., 61)
    cases, exclusions, graphs, targets, legacy_diffs = [], [], {}, {}, []
    source_hashes = {str(p.relative_to(rr.WORKSPACE)).replace('\\', '/'): rr.digest(p) for p in (mapping_path, m4p, m5p)}
    for item in records:
        path = archive / item['archive_npz']
        source_batch = path.parent.name
        source_id = int(path.stem.rsplit('_', 1)[1])
        sid = ('b004_' if source_batch == 'batch_004' else 'b005_') + f'{source_id:04d}'
        with np.load(path, allow_pickle=False) as z:
            t, raw_y = z['time_steps'], z['charring_ratios']
            raw_cracks, dims = z['crack_params'], z['beam_dims']
        if not len(t) and not len(raw_y):
            exclusions.append({'sample_id': sid, 'reason': 'EMPTY_RESPONSE', 'source': str(path),
                               'response_filter_not_based_on_constant_value': True})
            continue
        if source_batch == 'batch_004':
            cracks = m4['crack_params'][source_id]
            fire = m4['fire_curves'][source_id]
        else:
            cp = archive / f'data/batch_005/params/cracks_{source_id:04d}.json'
            fp = archive / f'data/batch_005/params/configs/fire_curve_{source_id:04d}.json'
            cracks = json.loads(cp.read_text(encoding='utf-8'))
            mr = m5byid[source_id]
            fire = {'type': mr['curve_type'], 'params': mr['curve_params']}
            fire_file = json.loads(fp.read_text(encoding='utf-8'))
            if {'type': fire_file['fire_curve']['type'], 'params': fire_file['fire_curve']['params']} != fire or fire_file['fire']['duration'] != 3600:
                raise ValueError('Manifest/fire config disagreement: ' + sid)
            if len(cracks) != mr['num_cracks']:
                raise ValueError('Manifest crack count mismatch')
            source_hashes[str(cp.relative_to(rr.WORKSPACE)).replace('\\', '/')] = rr.digest(cp)
            source_hashes[str(fp.relative_to(rr.WORKSPACE)).replace('\\', '/')] = rr.digest(fp)
        expected = np.asarray([[float(c[k]) for k in ('face', 'z', 'h', 'w', 'l', 'd')] for c in cracks])
        if not np.array_equal(raw_cracks, expected): raise ValueError('Geometry mismatch ' + sid)
        if not np.array_equal(np.asarray(dims).ravel(), [1.5, .14, .2]): raise ValueError('Beam mismatch')
        if t.ndim != 1 or raw_y.shape != t.shape or len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(raw_y).all():
            raise ValueError('Invalid scalar array ' + sid)
        if t[0] != 0 or t[-1] != 3600 or np.any(np.diff(t) <= 0): raise ValueError('Incomplete times ' + sid)
        if raw_y.min() < 0 or raw_y.max() > 1 + 1e-8: raise ValueError('Invalid label range')
        # Exactly one explicit transformation from recovered scalar member to target.
        already_cumulative = not bool(np.any(np.diff(raw_y) > 0))
        envelope = raw_y.copy() if already_cumulative else np.minimum.accumulate(raw_y)
        y64 = np.interp(times, t, envelope)
        case = {'sample_id': sid, 'beam': dict(rr.pf.DEFAULT_BEAM), 'cracks': cracks,
            'fire': fire, 'num_cracks': len(cracks), 'fire_family': fire['type'],
            'source_batch': source_batch, 'source_id': source_id,
            'legacy_final_id': Path(item['legacy_final_npz']).stem if item['legacy_final_npz'] else None,
            'source_npz': str(path.relative_to(rr.WORKSPACE)).replace('\\', '/'),
            'source_npz_sha256_prior_full_audit': item['file_sha256'],
            'source_npz_bytes': path.stat().st_size,
            'source_scalar_arrays_sha256_current': hashlib.sha256(t.tobytes() + raw_y.tobytes()).hexdigest(),
            'native_time_count': len(t), 'native_history_already_nonincreasing': already_cumulative,
            'target_operation': 'native identity -> linear interpolation' if already_cumulative else 'native cumulative minimum -> linear interpolation',
            'flat_response': bool(np.ptp(raw_y) == 0), 'all_ones_response': bool(np.all(raw_y == 1)),
            'legacy_envelope_change_max': float(np.max(np.abs(np.interp(times, t, raw_y) - y64)))}
        case['geometry_id'] = rr.geometry_id(case)
        if case['legacy_final_id']:
            with np.load(rr.WORKSPACE / 'Abaqus' / item['legacy_final_npz'], allow_pickle=False) as old:
                oldy = np.interp(times, old['time_steps'], np.minimum.accumulate(old['charring_ratios']))
            diff = float(np.max(np.abs(oldy - y64)))
            if diff != 0: raise ValueError('Archive target not equal to retained legacy target')
            legacy_diffs.append(diff)
        tf = ns['make_time_features'](61, 3600., fire['type'], fire['params']).squeeze(0)
        e, a = rr.pf.build_crack_edges(cracks)
        graphs[sid] = (rr.pf.encode_crack_node_features(cracks), e, a,
                       rr.pf.encode_global_features(cracks), tf)
        if not all(torch.isfinite(v).all() for v in graphs[sid]): raise ValueError('Nonfinite features')
        targets[sid] = torch.tensor(y64, dtype=torch.float32)
        cases.append(case)
    if len(cases) != 997 or len(exclusions) != 3 or len(legacy_diffs) != 911:
        raise ValueError('Unexpected population or mappings')
    groups = collections.defaultdict(list)
    for c in cases: groups[c['geometry_id']].append(c['sample_id'])
    if len(groups) != len(cases):
        raise ValueError('Repeated geometries require grouped allocation before freezing')
    torch.save({'graphs': graphs, 'targets': targets, 'time_s': torch.tensor(times)}, output / 'tensors.pt')
    manifest = {'status': 'PREPARED_ARCHIVAL_TARGET_NOT_NUMERICALLY_QUALIFIED', 'purpose': rr.PURPOSE,
        'target_version': rr.TARGET_VERSION, 'feature_version': rr.FEATURE_VERSION,
        'target_interpretation': 'Historical scalar envelope only, not reconstructed pointwise irreversible timber capacity.',
        'target_formula': 'native monotone identity or native cumulative minimum; linear interpolation at 0:60:3600; float32',
        'temporal_interpretation': 'Archived prescribed-fire function/parameters, not per-case realized amplitude evidence.',
        'case_admission': 'All available complete finite in-range nonempty scalar responses; no constant-target filtering.',
        'tensor_path': 'tensors.pt', 'tensor_sha256': rr.digest(output / 'tensors.pt'),
        'feature_tensor_sha256': rr.tensor_collection_digest(graphs),
        'target_tensor_sha256': rr.tensor_collection_digest(targets), 'cases': cases,
        'exclusions': exclusions, 'source_sha256': rr.sources(), 'metadata_source_sha256': source_hashes,
        'full_temperature_payload_read_this_run': False,
        'current_arrays_read': ['time_steps', 'charring_ratios', 'crack_params', 'beam_dims'],
        'legacy911_target_exact_reconciliation_count': len(legacy_diffs),
        'legacy911_max_target_difference': max(legacy_diffs),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    rr.write_json(output / 'manifest.json', manifest)
    families = sorted({c['fire_family'] for c in cases})
    rng = np.random.default_rng(42)

    def write_split(name, train, validation, test, explanation):
        split = {'status': 'DRAFT_PRETRAINING', 'purpose': rr.PURPOSE, 'protocol': name,
            'dataset_sha256': rr.digest(output / 'manifest.json'), 'split_seed': 42,
            'train': train, 'validation': validation, 'test': test, 'rule': explanation}
        rr.check_split(split, manifest)
        selected = set(train)
        fitted = rr.fit_radius_on_training_geometries([dict(c, role='train') for c in cases if c['sample_id'] in selected])
        split['topology_recipe'] = {'radius': fitted['normalized_radius'], 'radius_fit_quantile': .5,
            'radius_fit_only_training': True, 'symmetric_knn_k': 3,
            'radius_fit_geometry_count': fitted['unique_training_geometries']}
        rr.write_json(output / 'splits' / (name + '.json'), split)

    roles = [[], [], []]
    for family in families:
        ids = sorted(c['sample_id'] for c in cases if c['fire_family'] == family)
        ids = list(rng.permutation(ids))
        ntrain, nval = int(.70 * len(ids)), int(.15 * len(ids))
        for dst, seq in zip(roles, [ids[:ntrain], ids[ntrain:ntrain + nval], ids[ntrain + nval:]]): dst.extend(seq)
    write_split('iid997', *roles, 'Per-fire-family deterministic 70/15/remaining split; exact geometry disjoint.')
    rng = np.random.default_rng(42)
    tr, va, te = [], [], []
    for family in families:
        development = sorted(c['sample_id'] for c in cases if c['fire_family'] == family and c['num_cracks'] <= 8)
        ids = list(rng.permutation(development)); n = int(.85 * len(ids))
        tr += ids[:n]; va += ids[n:]
        te += sorted(c['sample_id'] for c in cases if c['fire_family'] == family and c['num_cracks'] >= 9)
    write_split('lcro_9_15', tr, va, te, 'Train/validation only N1–8; test N9–15; per-family 85/15 development split.')
    for heldout in families:
        rng = np.random.default_rng(42)
        tr, va, te = [], [], []
        for family in families:
            ids = sorted(c['sample_id'] for c in cases if c['fire_family'] == family)
            if family == heldout: te.extend(ids); continue
            ids = list(rng.permutation(ids)); n = int(.85 * len(ids)); tr += ids[:n]; va += ids[n:]
        write_split('loco_' + heldout, tr, va, te, 'Entire named fire family held out; remaining families stratified85/15.')
    summary = {'status': manifest['status'], 'cases': len(cases), 'source_exports': len(records),
        'excluded_empty': exclusions, 'retained_historical_cases': len(legacy_diffs),
        'restored_flat_cases': sum(c['all_ones_response'] and not c['legacy_final_id'] for c in cases),
        'unique_geometries': len(groups), 'duplicate_geometry_groups': [],
        'families': dict(collections.Counter(c['fire_family'] for c in cases)),
        'counts': dict(sorted(collections.Counter(c['num_cracks'] for c in cases).items())),
        'feature_shapes': {'nodes': '[N,11]', 'edges': '[N(N-1),5]', 'globals': '[1,3]', 'time': '[61,4]', 'target': '[61]'},
        'tensor_bytes': (output / 'tensors.pt').stat().st_size, 'training_started': False,
        'legacy911_target_exact_reconciliation_count': 911, 'legacy911_max_target_difference': 0,
        'manifest_sha256': rr.digest(output / 'manifest.json'),
        'feature_tensor_sha256': manifest['feature_tensor_sha256'],
        'target_tensor_sha256': manifest['target_tensor_sha256'], 'splits': {}}
    for p in sorted((output / 'splits').glob('*.json')):
        s = json.loads(p.read_text(encoding='utf-8'))
        summary['splits'][p.stem] = {r: len(s[r]) for r in ('train', 'validation', 'test')}
    rr.write_json(output / 'readiness.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=str(rr.HERE / 'archive997'))
    create(p.parse_args().output)
