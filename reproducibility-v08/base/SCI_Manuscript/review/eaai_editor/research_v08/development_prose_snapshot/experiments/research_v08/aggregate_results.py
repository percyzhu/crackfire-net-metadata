"""Independent v08 complete-protocol statistics and run-integrity checks.

Never modifies the frozen learner/plan, and never publishes partial-seed model
rankings. Archived targets remain explicitly different from physical capacity.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import sys
import time
import numpy as np
import torch
import research_runner as rr

sys.path.insert(0, str(rr.EXPERIMENTS / 'evaluation'))
import evaluation_stats as es

METRICS = ['equal_case_MAE', 'equal_case_RMSE', 'mean_case_RMSE', 'R2_pooled',
           'maximum_positive_error', 'maximum_absolute_error', 'positive_error_q95', 'positive_error_q99']


def csv_write(path, rows):
    if not rows: return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def scalar_mean_sd(values):
    if any(v is None for v in values): return {'mean': None, 'sample_sd': None}
    return {'mean': float(np.mean(values)), 'sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def audit_cell(path, plan_path, plan, manifest, data, protocol, mode, seed, split, replay=False):
    meta = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8'))
    result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
    def require(test, message):
        if not test: raise ValueError(f'{path}: {message}')
    require(meta['purpose'] == result['purpose'] == rr.PURPOSE, 'wrong purpose')
    for key, val in [('mode', mode), ('seed', seed), ('protocol', protocol)]:
        require(meta[key] == result[key] == val, 'model/seed/protocol mismatch')
    require(meta['plan_sha256'] == rr.digest(plan_path), 'plan hash mismatch')
    require(meta['dataset_sha256'] == plan['manifest_sha256'] == rr.digest(plan['manifest_path']), 'dataset hash mismatch')
    require(meta['split_sha256'] == plan['protocols'][protocol]['split_sha256'], 'split hash mismatch')
    require(meta['source_sha256'] == plan['source_sha256'] == rr.sources(), 'frozen code hash mismatch')
    require(meta['target_version'] == manifest['target_version'] == rr.TARGET_VERSION, 'wrong target version')
    require(meta['feature_version'] == manifest['feature_version'] == rr.FEATURE_VERSION, 'wrong feature version')
    require(meta['feature_tensor_sha256'] == manifest['feature_tensor_sha256'], 'feature tensor hash mismatch')
    require(meta['target_tensor_sha256'] == manifest['target_tensor_sha256'], 'target tensor hash mismatch')
    require(meta['configuration'] == plan['training_configuration'], 'training budget changed')
    require(result['checkpoint_sha256'] == rr.digest(path / 'best.pt'), 'checkpoint hash mismatch')
    checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
    for key in ('plan_sha256', 'dataset_sha256', 'split_sha256'):
        require(checkpoint[key] == meta[key], 'checkpoint binding mismatch')
    model = rr.make_model(mode)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    params = sum(v.numel() for v in model.parameters() if v.requires_grad)
    require(params == meta['parameters'], 'parameter count mismatch')
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()), 'nonfinite weights')
    log = list(csv.DictReader((path / 'learning_curve.csv').open(encoding='utf-8')))
    require(bool(log), 'empty training log')
    require([int(row['epoch']) for row in log] == list(range(1, len(log) + 1)), 'nonsequential epochs')
    values = np.asarray([[float(row[k]) for k in ('train_mse', 'validation_mse', 'learning_rate', 'elapsed_s')] for row in log])
    require(np.isfinite(values).all(), 'nonfinite training log')
    require(np.all(values[:, :2] >= 0) and np.all(np.diff(values[:, 3]) >= 0), 'invalid loss/time log')
    best = int(np.argmin(values[:, 1])) + 1
    require(best == result['best_epoch'] == checkpoint['epoch'], 'best epoch not earliest validation minimum')
    require(values[best - 1, 1] == result['best_validation_mse'] == checkpoint['best_validation_mse'], 'best validation value differs')
    require(len(log) == result['epochs_run'] <= plan['training_configuration']['epochs_max'], 'epoch budget exceeded')
    expected_truth = np.stack([data['targets'][sid].numpy() for sid in split['test']])
    with np.load(path / 'test_predictions.npz', allow_pickle=False) as z:
        bindings = es.aligned_predictions(z['sample_ids'], z['time_s'], z['truth'], z['predictions'],
            split['test'], data['time_s'].numpy(), expected_truth)
        prediction = z['predictions'].copy()
    metric = es.metrics(expected_truth, prediction)
    metric_keys = {'equal_case_MAE': 'equal_case_MAE', 'equal_case_RMSE': 'RMSE',
        'R2_pooled': 'pooled_R2', 'maximum_positive_error': 'maximum_overprediction',
        'maximum_absolute_error': 'maximum_absolute_error'}
    for computed, saved in metric_keys.items():
        a, b = metric[computed], result[saved]
        require((a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, abs_tol=1e-13, rel_tol=1e-13)),
                'saved metric differs from independent recomputation ' + computed)
    cpu_error = None
    if replay:
        cpu_pred = rr.predict(model, data['graphs'], split['test'], 32, 'cpu')
        cpu_error = float(np.max(np.abs(cpu_pred - prediction)))
        require(cpu_error < 3e-6, 'CPU prediction replay differs from saved GPU output')
    topology = {}
    if mode in ('gnn', 'gnn_mean', 'gnn_zero_edge_features'):
        for rule in ('radius', 'symmetric_knn'):
            declared = result['topology_results'][rule]
            require(declared['same_trained_checkpoint'] and not declared['refitting'] and
                    declared['checkpoint_sha256'] == result['checkpoint_sha256'], 'topology refit/binding mismatch')
            with np.load(path / ('topology_' + rule + '.npz'), allow_pickle=False) as z:
                es.aligned_predictions(z['sample_ids'], z['time_s'], z['truth'], z['predictions'],
                    split['test'], data['time_s'].numpy(), expected_truth)
                sparse = z['predictions'].copy()
            sm = es.metrics(expected_truth, sparse)
            change = float(np.abs(sparse - prediction).mean())
            require(math.isclose(change, declared['mean_abs_prediction_change'], abs_tol=1e-13, rel_tol=1e-13), 'topology prediction change mismatch')
            for computed, saved in metric_keys.items():
                a, b = sm[computed], declared[saved]
                require((a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, abs_tol=1e-13, rel_tol=1e-13)),
                        'topology saved metric differs')
            topology_replay_error = None
            if replay:
                by_id = {c['sample_id']: c for c in manifest['cases']}
                recipe = split['topology_recipe']
                sparse_graphs = {sid: rr.graph_view(by_id[sid], rule, recipe['radius'],
                    recipe['symmetric_knn_k'], data['graphs'][sid][-1]) for sid in split['test']}
                require(sum(g[1].shape[1] for g in sparse_graphs.values()) == declared['total_edges'], 'topology edge count mismatch')
                sparse_replay = rr.predict(model, sparse_graphs, split['test'], 32, 'cpu')
                topology_replay_error = float(np.max(np.abs(sparse_replay - sparse)))
                require(topology_replay_error < 3e-6, 'same-checkpoint topology replay mismatch')
            topology[rule] = {'predictions': sparse, 'metrics': sm,
                'mean_abs_prediction_change': change, 'total_edges': declared['total_edges'],
                'checkpoint_sha256': declared['checkpoint_sha256'],
                'cpu_replay_max_abs': topology_replay_error}
    audit = {'protocol': protocol, 'model': mode, 'seed': seed, 'status': 'PASSED_BINDINGS',
        'plan_dataset_feature_target_split_checkpoint_hashes': True, 'ordered_ids_truth_times_exact': True,
        'best_epoch_is_earliest_validation_minimum': True, 'parameters': params,
        'predictions_sha256': rr.digest(path / 'test_predictions.npz'),
        'checkpoint_sha256': result['checkpoint_sha256'], **bindings,
        'cpu_replay_max_abs_difference': cpu_error,
        'topology_cpu_replay_max_abs': {k: v['cpu_replay_max_abs'] for k, v in topology.items()},
        'topology_files_bound_to_same_checkpoint': len(topology),
        'epoch_count': len(log), 'training_wall_s': result['training_wall_s']}
    return {'audit': audit, 'metrics': metric, 'prediction': prediction, 'truth': expected_truth,
            'topology': topology, 'training_wall_s': result['training_wall_s']}


def describe_protocol(name, cells, plan, manifest, split, output):
    seeds, models = plan['seeds'], plan['models']
    by_id = {c['sample_id']: c for c in manifest['cases']}
    cases = [by_id[sid] for sid in split['test']]
    geometry_ids = [c['geometry_id'] for c in cases]
    model_summary, contrast, strata, per_case, per_time, topology_rows = [], {}, [], [], [], []
    metric_arrays = {}
    for mode in models:
        current = [cells[(mode, s)] for s in seeds]
        y = current[0]['truth']
        preds = np.stack([c['prediction'] for c in current])
        errors = preds.astype(float) - y[None].astype(float)
        case_mae = np.abs(errors).mean(axis=2)
        metric_arrays[mode] = case_mae
        summary = {'model': mode, 'seed_count': len(seeds), 'seeds': seeds,
            'test_case_count': len(cases), 'parameters': current[0]['audit']['parameters'],
            'metrics': {k: scalar_mean_sd([c['metrics'][k] for c in current]) for k in METRICS},
            'training_wall_s': scalar_mean_sd([c['training_wall_s'] for c in current]),
            'monotonicity_violation_fraction': scalar_mean_sd([(np.diff(c['prediction'], axis=1) > 1e-6).mean() for c in current])}
        family_mae = []
        for i, seed in enumerate(seeds):
            groups = {}
            for kind, getter in [('fire_family', lambda c: c['fire_family']),
                    ('crack_count', lambda c: str(c['num_cracks'])),
                    ('source_batch', lambda c: c['source_batch']),
                    ('retained_or_restored', lambda c: 'retained911' if c['legacy_final_id'] else 'restored86'),
                    ('flat_response', lambda c: 'flat' if c['flat_response'] else 'nonflat')]:
                assignments = [getter(c) for c in cases]
                groups[kind] = {}
                for group in sorted(set(assignments)):
                    mask = np.asarray([v == group for v in assignments])
                    sm = es.metrics(y[mask], preds[i, mask])
                    groups[kind][group] = sm
                    strata.append({'protocol': name, 'model': mode, 'seed': seed, 'stratum': kind,
                        'group': group, 'case_count': int(mask.sum()),
                        'geometry_count': len({geometry_ids[k] for k in np.flatnonzero(mask)}),
                        **{k: sm[k] for k in METRICS}})
            family_mae.append(float(np.mean([v['equal_case_MAE'] for v in groups['fire_family'].values()])))
            for j, c in enumerate(cases):
                per_case.append({'protocol': name, 'model': mode, 'seed': seed, 'sample_id': c['sample_id'],
                    'geometry_id': c['geometry_id'], 'fire_family': c['fire_family'], 'crack_count': c['num_cracks'],
                    'source_batch': c['source_batch'], 'legacy_final_id': c['legacy_final_id'], 'flat_response': c['flat_response'],
                    'MAE': float(case_mae[i, j]), 'RMSE': float(np.sqrt(np.square(errors[i, j]).mean())),
                    'maximum_overprediction': float(np.maximum(errors[i, j], 0).max())})
            for j in range(y.shape[1]):
                per_time.append({'protocol': name, 'model': mode, 'seed': seed, 'time_s': j * 60,
                    'case_MAE': float(np.abs(errors[i, :, j]).mean()),
                    'case_RMSE': float(np.sqrt(np.square(errors[i, :, j]).mean())),
                    'maximum_overprediction': float(np.maximum(errors[i, :, j], 0).max())})
            for rule, sparse in current[i]['topology'].items():
                topology_rows.append({'protocol': name, 'model': mode, 'seed': seed, 'rule': rule,
                    'same_trained_checkpoint': True, 'refitting': False,
                    'checkpoint_sha256': sparse['checkpoint_sha256'], 'directed_edge_total': sparse['total_edges'],
                    'mean_abs_prediction_change': sparse['mean_abs_prediction_change'],
                    'MAE_change_vs_complete': sparse['metrics']['equal_case_MAE'] - current[i]['metrics']['equal_case_MAE'],
                    **{k: sparse['metrics'][k] for k in METRICS}})
        summary['equal_family_macro_MAE'] = scalar_mean_sd(family_mae)
        model_summary.append(summary)
    for baseline, graph, tag in [('capacity_matched_deepsets', 'gnn', 'graph_vs_capacity_matched'),
                                  ('gnn', 'gnn_mean', 'mean_vs_sum')]:
        interval = es.paired_bootstrap(metric_arrays[baseline], metric_arrays[graph], geometry_ids,
            seeds, repeats=5000, random_seed=20260907)
        interval['contrast'] = f'{baseline} MAE minus {graph} MAE; positive favors {graph}'
        interval.pop('G04_or_G05_pass', None)
        interval['purpose'] = rr.PURPOSE
        contrast[tag] = interval
    summary = {'status': 'COMPLETE_PROTOCOL_35_RUNS', 'protocol': name, 'purpose': rr.PURPOSE,
        'target_interpretation': manifest['target_interpretation'], 'seeds': seeds, 'models': models,
        'independent_geometry_clusters': len(set(geometry_ids)), 'test_case_count': len(cases),
        'model_summaries': model_summary, 'paired_contrasts': contrast,
        'interpretation': 'Complete prespecified protocol only; not full420 matrix completion, physical qualification, or peer-review acceptance.',
        'topology_interpretation': 'Same-checkpoint representation-connectivity stress; not independent physical FE topology transfer.'}
    output.mkdir(parents=True, exist_ok=True)
    rr.write_json(output / 'summary.json', summary)
    csv_write(output / 'strata_by_seed.csv', strata)
    csv_write(output / 'per_case_metrics.csv', per_case)
    csv_write(output / 'time_metrics.csv', per_time)
    csv_write(output / 'topology_by_seed.csv', topology_rows)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=str(rr.HERE / 'comparison_plan_v08_420.json'))
    parser.add_argument('--output', default=str(rr.HERE / 'evaluation'))
    parser.add_argument('--replay', action='store_true', help='CPU replay all completed predictions as well as bindings')
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    output = rr.safe_output(args.output); output.mkdir(parents=True, exist_ok=True)
    try:
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['purpose'] != rr.PURPOSE or plan['status'] != 'FROZEN_BEFORE_TRAINING': raise ValueError('Wrong plan')
        if plan['source_sha256'] != rr.sources(): raise ValueError('Frozen code changed')
        if rr.digest(plan['manifest_path']) != plan['manifest_sha256']: raise ValueError('Manifest changed')
        manifest, data = rr.load_dataset(plan['manifest_path'])
        rr.configure(42, 4)
        allowed_paths = {str((Path(plan['runs_root']) / name / mode / ('seed_' + str(seed)) / 'result.json').resolve())
            for name in plan['protocols'] for mode in plan['models'] for seed in plan['seeds']}
        unexpected = [str(p) for p in Path(plan['runs_root']).glob('*/*/*/result.json') if str(p.resolve()) not in allowed_paths]
        if unexpected: raise ValueError('Unregistered or duplicate run paths: ' + str(unexpected))
        protocols, audits = {}, []
        for name, spec in plan['protocols'].items():
            split_path = Path(spec['split_path'])
            if rr.digest(split_path) != spec['split_sha256']: raise ValueError('Split changed')
            split = json.loads(split_path.read_text(encoding='utf-8'))
            if split['dataset_sha256'] != plan['manifest_sha256']: raise ValueError('Wrong split dataset binding')
            rr.check_split(split, manifest)
            train_ids = set(split['train'])
            radius_check = rr.fit_radius_on_training_geometries([dict(c, role='train') for c in manifest['cases'] if c['sample_id'] in train_ids])
            if radius_check['normalized_radius'] != split['topology_recipe']['radius']:
                raise ValueError('Frozen radius differs from training-only fit')
            found, missing = {}, []
            for mode in plan['models']:
                for seed in plan['seeds']:
                    path = Path(plan['runs_root']) / name / mode / ('seed_' + str(seed))
                    if not (path / 'result.json').exists():
                        missing.append({'model': mode, 'seed': seed}); continue
                    cell = audit_cell(path, plan_path, plan, manifest, data, name, mode, seed, split, args.replay)
                    found[(mode, seed)] = cell
                    audits.append(cell['audit'])
            protocols[name] = {'status': 'PENDING' if missing else 'COMPLETE',
                'audited_run_count': len(found), 'expected_run_count': len(plan['models']) * len(plan['seeds']),
                'missing_cells': missing, 'scientific_summary_published': not bool(missing)}
            if not missing:
                describe_protocol(name, found, plan, manifest, split, output / name)
        report = {'status': 'COMPLETE_420_RUN_MATRIX' if all(v['status'] == 'COMPLETE' for v in protocols.values()) else 'PENDING_MATRIX',
            'purpose': rr.PURPOSE, 'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'plan_sha256': rr.digest(plan_path), 'protocols': protocols, 'audited_completed_runs': len(audits),
            'pending_protocols_have_no_model_performance_summaries': True,
            'audit_source_sha256': {p.name: rr.digest(p) for p in (Path(__file__), Path(es.__file__))},
            'full_temperature_payload_read_this_check': False, 'cpu_prediction_replay_enabled': args.replay,
            'no_training_or_model_changes': True, 'peer_review_pass': False}
        rr.write_json(output / 'run_integrity.json', {'audits': audits})
        rr.write_json(output / 'report.json', report)
        print(json.dumps({'status': report['status'], 'audited_completed_runs': len(audits),
            'complete_protocols': [k for k, v in protocols.items() if v['status'] == 'COMPLETE']}, indent=2))
    except BaseException as exc:
        rr.write_json(output / 'report.json', {'status': 'ERROR_INCOMPLETE_CHECK', 'error': str(exc),
            'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'peer_review_pass': False})
        raise


if __name__ == '__main__': main()
