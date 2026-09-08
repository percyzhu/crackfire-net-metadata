"""Reproducible common-interface timings; never run beside research training."""
from pathlib import Path
import argparse
import collections
import csv
import json
import math
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
import torch
import research_runner as rr

HERE = Path(__file__).resolve().parent
OUT = HERE / 'inference_benchmark_v1'
TRAIN_PLAN = HERE / 'comparison_plan_v08_420.json'
PLAN_PATH = OUT / 'timing_plan.json'
FINAL_REPORT = HERE / 'evaluation/report.json'
FINAL_INTEGRITY = HERE / 'evaluation/run_integrity.json'
AUDIT_SOURCES = {
    'aggregate_results.py': HERE / 'aggregate_results.py',
    'evaluation_stats.py': rr.EXPERIMENTS / 'evaluation/evaluation_stats.py',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def powershell(script):
    result = subprocess.run(['powershell', '-NoProfile', '-Command', script],
                            capture_output=True, text=True, timeout=30, check=True)
    return result.stdout.strip()


def load_sources():
    plan = read(TRAIN_PLAN)
    if rr.sources() != plan['source_sha256']:
        raise ValueError('Frozen training sources changed')
    if rr.digest(plan['manifest_path']) != plan['manifest_sha256']:
        raise ValueError('Manifest changed')
    manifest, data = rr.load_dataset(plan['manifest_path'])
    split_path = plan['protocols']['iid997']['split_path']
    if rr.digest(split_path) != plan['protocols']['iid997']['split_sha256']:
        raise ValueError('IID split changed')
    return plan, manifest, data, read(split_path)


def validate_selection(timing, plan, split):
    """Reassert declared selection against source identities, not just paths."""
    if timing['protocol'] != 'iid997' or timing['seed'] != 42 or timing['sample_ids'] != split['test']:
        raise ValueError('Timing sample/seed/protocol selection differs from frozen IID rule')
    if [r['model'] for r in timing['checkpoints']] != plan['models'] or timing['times_per_case'] != 61:
        raise ValueError('Timing model inventory or response grid changed')
    for record in timing['checkpoints']:
        folder = Path(plan['runs_root']) / 'iid997' / record['model'] / 'seed_42'
        if Path(record['checkpoint']).resolve() != (folder / 'best.pt').resolve():
            raise ValueError('Selected checkpoint path violates first-seed rule')
        meta, result = read(folder / 'run_metadata.json'), read(folder / 'result.json')
        for key, value in [('protocol', 'iid997'), ('mode', record['model']), ('seed', 42)]:
            if meta[key] != value or result[key] != value:
                raise ValueError('Selected checkpoint model/seed/protocol identity mismatch')
        if meta['plan_sha256'] != rr.digest(TRAIN_PLAN) or meta['dataset_sha256'] != plan['manifest_sha256']:
            raise ValueError('Selected checkpoint plan/dataset mismatch')
        if record['checkpoint_sha256'] != result['checkpoint_sha256'] or record['parameters'] != meta['parameters']:
            raise ValueError('Selected checkpoint declaration differs from run provenance')


def prepare():
    if PLAN_PATH.exists():
        raise ValueError('Do not overwrite declared timing selection')
    plan, manifest, _, split = load_sources()
    records = []
    for mode in plan['models']:
        folder = Path(plan['runs_root']) / 'iid997' / mode / 'seed_42'
        result, meta = read(folder / 'result.json'), read(folder / 'run_metadata.json')
        if meta['plan_sha256'] != rr.digest(TRAIN_PLAN):
            raise ValueError('Checkpoint training-plan mismatch')
        if result['checkpoint_sha256'] != rr.digest(folder / 'best.pt'):
            raise ValueError('Checkpoint hash mismatch')
        records.append({'model': mode, 'checkpoint': str(folder / 'best.pt'),
                        'checkpoint_sha256': result['checkpoint_sha256'],
                        'parameters': meta['parameters']})
    timing = {
        'status': 'DECLARED_BEFORE_TIMING_AFTER_ACCURACY_RESULTS',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'train_plan_sha256': rr.digest(TRAIN_PLAN),
        'benchmark_source_sha256': rr.digest(__file__),
        'protocol': 'iid997', 'seed': 42, 'checkpoints': records,
        'selection_rule': 'First training seed and every IID test case in frozen split order; no speed/accuracy selection.',
        'sample_ids': split['test'], 'times_per_case': 61,
        'devices': ['cpu', 'cuda'], 'CPU_threads': 4,
        'rounds': 3, 'batch_sizes': [1, 32], 'warmup_batches_per_shape': 10,
        'warmup_full_population_passes': 1,
        'model_order': 'Rotate frozen model order by round; alternate device order by round.',
        'stages': ['forward_resident_inputs', 'common_input_to_host_output'],
        'separate_common_CPU_preparation_stage': 'feature_preparation_only: all five tensors rebuilt before collation, once per round and batch size; not repeated for each model/device.',
        'feature_preparation': 'Reconstruct all five common input tensors from in-memory recorded geometry and fire parameters.',
        'common_interface_caveat': 'Every model pays for the same common features, including unused edge/node attributes; not individually optimized deployment pipelines.',
        'excluded_costs': ['Python import', 'model loading', 'disk I/O', 'JSON parsing', 'network transfer', 'FE generation'],
        'synchronization': 'CUDA synchronize before and after each timed operation; host output conversion included only in common_input_to_host_output.',
        'statistics': 'Raw batch durations; mean, p50, p95 by model/device/stage/batch size; throughput total cases / total seconds; per-case amortized latency distinguished from serial query latency.',
        'variance_scope': 'Three repeated rounds on fixed hardware, one checkpoint seed; no accuracy confidence interpretation.',
        'training_cost': 'All 420 recorded host wall times summarized separately; includes training/validation/checkpoint I/O and best-checkpoint restore API, excludes initial setup and final test/topology inference. Frozen timing has no explicit CUDA synchronization after final restore. As-executed costs may include concurrent analysis load and are not isolated architecture speed benchmarks. Failed-attempt/replay overhead and process setup are not included in the sum of successful runs.',
        'execution_gate': '420 unique complete units; queue status COMPLETE_ARCHIVAL_AI_MATRIX; queue lock absent; final COMPLETE_420_RUN_MATRIX replay audit with current evaluator source hashes; all420 actual result/status/metadata/checkpoint/prediction identities bound before measurement; no detected project Python training/audit/figure compute.',
        'execution_gate_rechecks': 'Full artifact hashes verified once before timing. At each round/model and completion, recheck known project-compute processes, queue state, and final report/integrity/source hashes; do not hash all420 weights within latency collection.',
        'comparison_scope': 'Absolute inference and recorded training costs; historical FE timings separate; no mixed-hardware FEM/GNN speedup.',
        'feature_version': manifest['feature_version'],
        'precision_and_unit': 'Float32 model inputs/outputs (int64 graph indices); one case is its entire61-point trajectory, not one time step.',
    }
    validate_selection(timing, plan, split)
    rr.write_json(PLAN_PATH, timing)
    print(json.dumps({'status': timing['status'], 'test_cases': len(split['test']),
                      'checkpoints': len(records), 'timing_plan_sha256': rr.digest(PLAN_PATH)}))


def is_project_compute(process, current_pid):
    if int(process['ProcessId']) == current_pid:
        return False
    command = (process.get('CommandLine') or '').lower().replace('\\', '/')
    # Include figure generation and validation as well as model/audit scripts.
    # Short relative invocations need explicit known basenames. Anonymous stdin
    # Python jobs cannot be attributed to a workspace from this OS inventory.
    if 'sci_manuscript/' in command or '1_代码/' in command or str(rr.WORKSPACE).lower().replace('\\', '/') + '/' in command:
        return True
    known = ('research_runner', 'io_recovery_patch1', 'run_queue', 'aggregate_results',
             'audit_completed_protocol', 'audit_same_weight_topology', 'audit_iid_results',
             'audit_prepared_archive', 'engineering_proxy_ci', 'engineering_proxy_metrics',
             'check_engineering_proxy_ci', 'check_engineering_proxy_metrics',
             'check_development_figures', 'benchmark_inference', 'build_v08', 'render_review')
    return any(re.search(r'(?<![\w])' + re.escape(name) + r'\.py\b', command) for name in known)


def final_report_ready(plan):
    if not FINAL_REPORT.exists() or not FINAL_INTEGRITY.exists():
        return False
    report = read(FINAL_REPORT)
    if (report.get('status') != 'COMPLETE_420_RUN_MATRIX' or
        report.get('cpu_prediction_replay_enabled') is not True or
        report.get('purpose') != rr.PURPOSE or
        report.get('plan_sha256') != rr.digest(TRAIN_PLAN) or
        report.get('audited_completed_runs') != 420 or
        set(report.get('protocols', {})) != set(plan['protocols'])):
        return False
    if report.get('audit_source_sha256') != {name: rr.digest(path) for name, path in AUDIT_SOURCES.items()}:
        return False
    return all(value.get('status') == 'COMPLETE' and value.get('audited_run_count') == 35 and
               value.get('expected_run_count') == 35 and not value.get('missing_cells') and
               value.get('scientific_summary_published') is True for value in report['protocols'].values())


def readiness(plan, bound_evidence=None):
    queue = read(Path(plan['runs_root']) / 'queue_status.json')
    expected = {(p, m, s) for p in plan['protocols'] for m in plan['models'] for s in plan['seeds']}
    completed = [(x['protocol'], x['mode'], x['seed']) for x in queue['completed']]
    reasons = []
    if queue['status'] != 'COMPLETE_ARCHIVAL_AI_MATRIX': reasons.append('queue_incomplete')
    if len(completed) != len(expected) or set(completed) != expected: reasons.append('matrix_not_420_unique_units')
    if queue.get('active') is not None: reasons.append('active_training_cell')
    if queue['plan_sha256'] != rr.digest(TRAIN_PLAN): reasons.append('queue_plan_mismatch')
    if (Path(plan['runs_root']) / 'queue.lock').exists(): reasons.append('queue_lock_present')
    procs = powershell("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress")
    inventory = json.loads(procs) if procs else []
    if isinstance(inventory, dict): inventory = [inventory]
    matching = [p for p in inventory if is_project_compute(p, os.getpid())]
    matching_ids = {p['ProcessId'] for p in matching}
    logged_inventory = [p if p['ProcessId'] in matching_ids else
                        {'ProcessId': p['ProcessId'], 'Name': p['Name']} for p in inventory]
    if matching: reasons.append('concurrent_research_compute')
    if not final_report_ready(plan): reasons.append('final_420_replay_audit_missing_or_changed')
    if bound_evidence:
        for key, path in [('report_sha256', FINAL_REPORT), ('integrity_sha256', FINAL_INTEGRITY)]:
            if not path.exists() or rr.digest(path) != bound_evidence[key]:
                reasons.append('bound_final_audit_changed_' + key)
    return {'status': 'READY' if not reasons else 'DEFERRED_NO_MEASUREMENTS',
            'reasons': reasons, 'completed_units': len(completed),
            'matching_processes': matching, 'python_process_inventory': logged_inventory,
            'process_scope': 'Known project paths and compute-script names; anonymous stdin jobs and unrelated OS activity are not proven absent.',
            'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}


def regenerate(case, make_time):
    cracks, beam, fire = case['cracks'], case['beam'], case['fire']
    edges, attributes = rr.pf.build_crack_edges(cracks, beam)
    return (rr.pf.encode_crack_node_features(cracks, beam), edges, attributes,
            rr.pf.encode_global_features(cracks, beam),
            make_time(61, 3600., fire['type'], fire['params']).squeeze(0))


def verify_features(manifest, data, ids):
    cases = {c['sample_id']: c for c in manifest['cases']}
    make_time = rr.legacy_functions(rr.WORKSPACE)['make_time_features']
    for sid in ids:
        rebuilt = regenerate(cases[sid], make_time)
        if not all(torch.equal(a, b) for a, b in zip(rebuilt, data['graphs'][sid])):
            raise ValueError('Common input reconstruction mismatch: ' + sid)
    return cases, make_time


def sync(device):
    if device == 'cuda': torch.cuda.synchronize()


def replay_identity_passed(replay):
    if not replay or replay.get('cpu_diagnostic_tolerance') != 3e-6:
        return False
    if replay.get('cpu_within_original_tolerance') is True:
        return (replay.get('status') == 'CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE' and
                0 <= replay.get('cpu_max_abs_difference', math.inf) < 3e-6)
    return (replay.get('status') == 'GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED' and
            replay.get('same_backend_gpu_bitwise_equal') is True and
            replay.get('gpu_max_abs_difference') == 0 and
            replay.get('cpu_max_abs_difference', 0) >= 3e-6)


def validate_final_matrix(plan, manifest, data):
    """One premeasurement scan binds the final replay audit to actual files.

    No checkpoint is run here and no latency is measured. Repeated collection
    gates compare these small report hashes rather than rescan420 weight files.
    """
    if not final_report_ready(plan):
        raise ValueError('A final complete420 independent replay audit is required')
    report_sha, integrity_sha = rr.digest(FINAL_REPORT), rr.digest(FINAL_INTEGRITY)
    audits = read(FINAL_INTEGRITY)['audits']
    expected = {(p, m, s) for p in plan['protocols'] for m in plan['models'] for s in plan['seeds']}
    keys = [(a['protocol'], a['model'], a['seed']) for a in audits]
    if len(expected) != 420 or len(keys) != 420 or len(set(keys)) != 420 or set(keys) != expected:
        raise ValueError('Final audit is not exactly420 distinct frozen identities')
    by_key = dict(zip(keys, audits))
    plan_sha = rr.digest(TRAIN_PLAN)
    amendment_path = HERE / 'recovery_io_patch1/amendment.json'
    amendment = read(amendment_path)
    amendment_sha = rr.digest(amendment_path)
    wrapper_sha = rr.digest(HERE / 'io_recovery_patch1.py')
    if amendment['original_plan_sha256'] != plan_sha or amendment['wrapper_sha256'] != wrapper_sha:
        raise ValueError('I/O recovery lineage changed')
    rows, artifact_rows = [], []
    for protocol, spec in plan['protocols'].items():
        if rr.digest(spec['split_path']) != spec['split_sha256']:
            raise ValueError('Frozen split changed before timing')
        split = read(spec['split_path']); rr.check_split(split, manifest)
        if split['dataset_sha256'] != plan['manifest_sha256']:
            raise ValueError('Split dataset binding changed')
        expected_truth = np.stack([data['targets'][sid].numpy() for sid in split['test']])
        for mode in plan['models']:
            for seed in plan['seeds']:
                folder = Path(plan['runs_root']) / protocol / mode / ('seed_' + str(seed))
                audit = by_key[(protocol, mode, seed)]
                meta, result, status = (read(folder / name) for name in ('run_metadata.json', 'result.json', 'status.json'))
                if (status['status'] != 'COMPLETE_ARCHIVAL_RESEARCH' or audit['status'] != 'PASSED_BINDINGS' or
                    meta['purpose'] != rr.PURPOSE or result['purpose'] != rr.PURPOSE):
                    raise ValueError('Incomplete or mismatched training-cost source')
                for name, value in [('protocol', protocol), ('mode', mode), ('seed', seed)]:
                    if meta[name] != value or result[name] != value:
                        raise ValueError('Actual result/metadata identity differs from final audit')
                required = {'plan_sha256': plan_sha, 'dataset_sha256': plan['manifest_sha256'],
                            'split_sha256': spec['split_sha256'], 'source_sha256': plan['source_sha256'],
                            'target_version': manifest['target_version'], 'feature_version': manifest['feature_version'],
                            'feature_tensor_sha256': manifest['feature_tensor_sha256'],
                            'target_tensor_sha256': manifest['target_tensor_sha256'],
                            'configuration': plan['training_configuration'], 'parameters': audit['parameters']}
                if any(meta.get(name) != value for name, value in required.items()):
                    raise ValueError('Actual training metadata differs from frozen provenance')
                if not all(audit.get(name) is True for name in ('plan_dataset_feature_target_split_checkpoint_hashes',
                        'ordered_ids_truth_times_exact', 'best_epoch_is_earliest_validation_minimum')):
                    raise ValueError('Final audit did not validate required identities')
                if not replay_identity_passed(audit.get('computational_replay')):
                    raise ValueError('Missing or failed final checkpoint replay identity')
                rules = ('radius', 'symmetric_knn') if mode in ('gnn', 'gnn_mean', 'gnn_zero_edge_features') else ()
                if (set(audit['topology_computational_replay']) != set(rules) or
                    audit['topology_files_bound_to_same_checkpoint'] != len(rules) or
                    not all(replay_identity_passed(x) for x in audit['topology_computational_replay'].values())):
                    raise ValueError('Final audit lacks declared same-weight topology replays')
                names = ['run_metadata.json', 'result.json', 'status.json', 'learning_curve.csv',
                         'best.pt', 'test_predictions.npz'] + ['topology_' + rule + '.npz' for rule in rules]
                hashes = {name: rr.digest(folder / name) for name in names}
                if (hashes['best.pt'] != audit['checkpoint_sha256'] or
                    hashes['best.pt'] != result['checkpoint_sha256'] or
                    hashes['test_predictions.npz'] != audit['predictions_sha256']):
                    raise ValueError('Actual checkpoint/predictions differ from final audited files')
                checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
                if any(checkpoint[name] != meta[name] for name in ('plan_sha256', 'dataset_sha256', 'split_sha256')):
                    raise ValueError('Checkpoint internal provenance changed')
                chain = audit['runtime_amendment_chain']
                if not chain or chain['amendment_sha256'] != amendment_sha:
                    raise ValueError('Final audit runtime amendment lineage differs')
                patch = meta.get('runtime_io_patch')
                if patch:
                    if (patch['sha256'] != wrapper_sha or chain['wrapper_sha256'] != wrapper_sha or
                        patch['amendment_sha256'] != amendment_sha or
                        Path(patch['path']).resolve() != (HERE / 'io_recovery_patch1.py').resolve() or
                        Path(patch['amendment_path']).resolve() != amendment_path.resolve()):
                        raise ValueError('Actual recovered-run wrapper/amendment differs')
                else:
                    key = str((folder / 'run_metadata.json').relative_to(HERE)).replace('\\', '/')
                    if amendment['preserved_prior_completed_artifacts'].get(key) != hashes['run_metadata.json']:
                        raise ValueError('Unbound run before/after recovery')
                elapsed = result['training_wall_s']
                if (not np.isfinite(elapsed) or elapsed <= 0 or elapsed != audit['training_wall_s'] or
                    elapsed != status['training_wall_s'] or result['epochs_run'] != audit['epoch_count'] or
                    result['epochs_run'] != status['epochs_run'] or result['best_epoch'] != status['best_epoch'] or
                    result['best_epoch'] != checkpoint['epoch']):
                    raise ValueError('Actual training cost/epoch differs from final audit/status/checkpoint')
                log = list(csv.DictReader((folder / 'learning_curve.csv').open(encoding='utf-8')))
                vals = np.asarray([float(x['validation_mse']) for x in log])
                if (len(log) != result['epochs_run'] or not np.isfinite(vals).all() or
                    [int(x['epoch']) for x in log] != list(range(1, len(log) + 1)) or
                    int(np.argmin(vals)) + 1 != result['best_epoch'] or
                    vals[result['best_epoch'] - 1] != result['best_validation_mse'] or
                    result['best_validation_mse'] != checkpoint['best_validation_mse']):
                    raise ValueError('Actual validation-selection provenance differs')
                for name in ['test_predictions.npz'] + ['topology_' + rule + '.npz' for rule in rules]:
                    with np.load(folder / name, allow_pickle=False) as saved:
                        if (saved['sample_ids'].tolist() != split['test'] or
                            not np.array_equal(saved['time_s'], data['time_s'].numpy()) or
                            not np.array_equal(saved['truth'], expected_truth)):
                            raise ValueError('Actual prediction IDs/time/truth differ from frozen arrays')
                        recomputed = rr.metrics(expected_truth, saved['predictions'])
                    recorded = result if name == 'test_predictions.npz' else result['topology_results'][name[9:-4]]
                    for metric, value in recomputed.items():
                        other = recorded[metric]
                        if value is None:
                            valid = other is None
                        elif isinstance(value, list):
                            valid = np.allclose(value, other, rtol=1e-13, atol=1e-13)
                        else:
                            valid = other is not None and math.isclose(value, other, rel_tol=1e-13, abs_tol=1e-13)
                        if not valid: raise ValueError('Actual result no longer matches saved predictions')
                rows.append({'protocol': protocol, 'model': mode, 'seed': seed, 'parameters': meta['parameters'],
                             'training_wall_s': elapsed, 'epochs_run': result['epochs_run'], 'best_epoch': result['best_epoch'],
                             'result_sha256': hashes['result.json']})
                artifact_rows.append({'protocol': protocol, 'model': mode, 'seed': seed,
                                      'folder': str(folder), 'files_sha256': hashes})
    if rr.digest(FINAL_REPORT) != report_sha or rr.digest(FINAL_INTEGRITY) != integrity_sha:
        raise ValueError('Final audit changed during source binding')
    return {'status': 'ALL420_FINAL_REPLAY_AND_ACTUAL_ARTIFACTS_BOUND_BEFORE_TIMING',
            'plan_sha256': plan_sha, 'report_sha256': report_sha, 'integrity_sha256': integrity_sha,
            'audit_source_sha256': {name: rr.digest(path) for name, path in AUDIT_SOURCES.items()},
            'runtime_wrapper_sha256': wrapper_sha, 'runtime_amendment_sha256': amendment_sha,
            'actual_runs_checked': len(rows), 'artifacts': artifact_rows,
            'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}, rows


def training_costs(plan, rows, target):
    """Write already-validated costs before the first timed operation."""
    with (target / 'training_costs_420.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summaries = []
    for protocol in plan['protocols']:
        for mode in plan['models']:
            selected = [r for r in rows if r['protocol'] == protocol and r['model'] == mode]
            values = np.asarray([r['training_wall_s'] for r in selected])
            summaries.append({'protocol': protocol, 'model': mode, 'seeds': len(values),
                              'mean_wall_s': float(values.mean()), 'SD_wall_s': float(values.std(ddof=1)),
                              'min_wall_s': float(values.min()), 'max_wall_s': float(values.max()),
                              'mean_epochs_run': float(np.mean([r['epochs_run'] for r in selected]))})
    return {'completed_runs': len(rows), 'sum_successful_training_wall_s': sum(r['training_wall_s'] for r in rows),
            'summaries': summaries, 'CSV_sha256': rr.digest(target / 'training_costs_420.csv'),
            'not_total_project_elapsed_or_isolated_training_speed': True}


def run():
    plan, manifest, data, split = load_sources()
    timing = read(PLAN_PATH)
    if timing['train_plan_sha256'] != rr.digest(TRAIN_PLAN) or timing['benchmark_source_sha256'] != rr.digest(__file__):
        raise ValueError('Timing declaration changed; document amendment before measurement')
    ready = readiness(plan)
    rr.write_json(OUT / 'readiness_latest.json', ready)
    if ready['status'] != 'READY':
        print(json.dumps(ready)); return
    validate_selection(timing, plan, split)
    bound_evidence, cost_rows = validate_final_matrix(plan, manifest, data)
    ready = readiness(plan, bound_evidence)
    rr.write_json(OUT / 'readiness_latest.json', ready)
    if ready['status'] != 'READY':
        print(json.dumps(ready)); return
    target = OUT / 'measured'
    if target.exists(): raise ValueError('Preserve the earlier measurement attempt')
    target.mkdir()
    rr.write_json(target / 'premeasurement_artifact_binding.json', bound_evidence)
    costs = training_costs(plan, cost_rows, target)
    cases, make_time = verify_features(manifest, data, timing['sample_ids'])
    rr.configure(42, timing['CPU_threads'])
    if not torch.cuda.is_available(): raise ValueError('Declared CPU/CUDA comparison requires CUDA')
    rows = []
    environment = {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'numpy': np.__version__, 'platform': platform.platform(), 'pid': os.getpid(),
                   'CPU': powershell('Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress'),
                   'GPU': torch.cuda.get_device_name(0), 'threads': torch.get_num_threads(),
                   'cudnn_enabled': torch.backends.cudnn.enabled,
                   'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                   'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                   'timing_plan_sha256': rr.digest(PLAN_PATH), 'readiness': ready,
                   'precision': 'float32 inputs/outputs; int64 graph indices',
                   'latency_unit': 'one entire61-point trajectory per case',
                   'artifact_binding_sha256': rr.digest(target / 'premeasurement_artifact_binding.json')}
    rr.write_json(target / 'environment.json', environment)
    try:
        with (target / 'raw_timings.csv').open('x', encoding='utf-8', newline='') as stream, torch.inference_mode():
            fields = ['round', 'device', 'model', 'stage', 'batch_size', 'cases', 'sample_ids', 'num_cracks', 'elapsed_s']
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for round_index in range(timing['rounds']):
                if readiness(plan, bound_evidence)['status'] != 'READY': raise ValueError('Concurrent compute or changed final audit detected')
                for size in timing['batch_sizes']:
                    prep_groups = [timing['sample_ids'][i:i+size] for i in range(0, len(timing['sample_ids']), size)]
                    for _ in range(timing['warmup_full_population_passes']):
                        for group in prep_groups:
                            prepared = [regenerate(cases[sid], make_time) for sid in group]
                    for j in range(timing['warmup_batches_per_shape']):
                        prepared = [regenerate(cases[sid], make_time) for sid in prep_groups[j % len(prep_groups)]]
                    for group in prep_groups:
                        start = time.perf_counter()
                        prepared = [regenerate(cases[sid], make_time) for sid in group]
                        elapsed = time.perf_counter() - start
                        row = dict(round=round_index, device='cpu', model='common_features', stage='feature_preparation_only',
                                   batch_size=size, cases=len(group), sample_ids='|'.join(group),
                                   num_cracks='|'.join(str(cases[s]['num_cracks']) for s in group), elapsed_s=elapsed)
                        writer.writerow(row); rows.append(row)
                    stream.flush()
                order = timing['checkpoints'][round_index:] + timing['checkpoints'][:round_index]
                devices = timing['devices'] if round_index % 2 == 0 else list(reversed(timing['devices']))
                for device in devices:
                    for record in order:
                        if readiness(plan, bound_evidence)['status'] != 'READY': raise ValueError('Concurrent compute or changed final audit detected')
                        if rr.digest(record['checkpoint']) != record['checkpoint_sha256']: raise ValueError('Weights changed')
                        model = rr.make_model(record['model']).to(device).eval()
                        checkpoint = torch.load(record['checkpoint'], map_location=device, weights_only=True)
                        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                        if sum(p.numel() for p in model.parameters()) != record['parameters']: raise ValueError('Parameter mismatch')
                        for size in timing['batch_sizes']:
                            groups = [timing['sample_ids'][i:i+size] for i in range(0, len(timing['sample_ids']), size)]
                            resident = [rr.collate([data['graphs'][sid] for sid in group], device) for group in groups]
                            def operation(index, stage):
                                batch = resident[index] if stage == 'forward_resident_inputs' else rr.collate(
                                    [regenerate(cases[sid], make_time) for sid in groups[index]], device)
                                prediction = model(*batch)
                                return prediction if stage == 'forward_resident_inputs' else prediction.cpu().numpy()
                            for stage in timing['stages']:
                                for _ in range(timing['warmup_full_population_passes']):
                                    for j in range(len(groups)): operation(j, stage)
                                for j in range(timing['warmup_batches_per_shape']): operation(j % len(groups), stage)
                                sync(device)
                                for index, group in enumerate(groups):
                                    sync(device); start = time.perf_counter()
                                    prediction = operation(index, stage)
                                    sync(device); elapsed = time.perf_counter() - start
                                    if tuple(prediction.shape) != (len(group), 61, 1): raise ValueError('Output shape changed')
                                    finite = bool(torch.isfinite(prediction).all()) if isinstance(prediction, torch.Tensor) else bool(np.isfinite(prediction).all())
                                    if not finite: raise ValueError('Nonfinite benchmark output')
                                    row = dict(round=round_index, device=device, model=record['model'], stage=stage,
                                               batch_size=size, cases=len(group), sample_ids='|'.join(group),
                                               num_cracks='|'.join(str(cases[s]['num_cracks']) for s in group), elapsed_s=elapsed)
                                    writer.writerow(row); rows.append(row)
                                stream.flush()
                        del model, checkpoint, resident, prediction
                        sync(device)
        grouped = collections.defaultdict(list)
        for row in rows: grouped[(row['device'], row['model'], row['stage'], row['batch_size'])].append(row)
        summaries = []
        for key, group in grouped.items():
            values = np.asarray([r['elapsed_s'] for r in group])
            ncases = sum(r['cases'] for r in group)
            summaries.append(dict(zip(['device', 'model', 'stage', 'batch_size'], key),
                measurements=len(group), cases_total=ncases, mean_batch_ms=float(values.mean()*1000),
                median_batch_ms=float(np.median(values)*1000), p95_batch_ms=float(np.quantile(values,.95)*1000),
                amortized_ms_per_case=float(values.sum()*1000/ncases), cases_per_second=float(ncases/values.sum())))
        by_count = collections.defaultdict(list)
        for row in rows:
            if row['batch_size'] == 1:
                by_count[(row['device'], row['model'], row['stage'], int(row['num_cracks']))].append(row)
        count_summaries = []
        for key, group in by_count.items():
            values = np.asarray([r['elapsed_s'] for r in group])
            count_summaries.append(dict(zip(['device', 'model', 'stage', 'num_cracks'], key),
                measurements=len(group), unique_cases=len({r['sample_ids'] for r in group}),
                mean_single_query_ms=float(values.mean()*1000), median_single_query_ms=float(np.median(values)*1000),
                p95_single_query_ms=float(np.quantile(values,.95)*1000)))
        final_ready = readiness(plan, bound_evidence)
        rr.write_json(target / 'readiness_at_completion.json', final_ready)
        if final_ready['status'] != 'READY': raise ValueError('Final quiet-machine/source gate failed')
        round_groups = collections.defaultdict(list)
        for row in rows:
            round_groups[(row['round'], row['device'], row['model'], row['stage'], row['batch_size'])].append(row)
        round_summaries = []
        for key, group in round_groups.items():
            elapsed = sum(r['elapsed_s'] for r in group)
            count = sum(r['cases'] for r in group)
            round_summaries.append(dict(zip(['round', 'device', 'model', 'stage', 'batch_size'], key),
                cases_total=count, measurements=len(group), mean_batch_ms=elapsed * 1000 / len(group),
                amortized_ms_per_case=elapsed * 1000 / count, cases_per_second=count / elapsed))
        rr.write_json(target / 'summary.json', {'status': 'COMPLETE_INFERENCE_TIMING',
            'summaries': summaries, 'rows': len(rows), 'timing_plan_sha256': rr.digest(PLAN_PATH),
            'single_query_by_crack_count': count_summaries, 'recorded_training_cost': costs,
            'per_round_summaries': round_summaries,
            'artifact_binding_sha256': rr.digest(target / 'premeasurement_artifact_binding.json'),
            'raw_csv_sha256': rr.digest(target / 'raw_timings.csv'), 'environment_sha256': rr.digest(target / 'environment.json')})
        print(json.dumps({'status': 'COMPLETE_INFERENCE_TIMING', 'raw_measurements': len(rows)}))
    except BaseException as exc:
        rr.write_json(target / 'failure.json', {'status': 'FAILED_RETAIN_PARTIAL_DO_NOT_REPORT', 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'check-ready', 'validate-features', 'run'])
    action = parser.parse_args().action
    if action == 'prepare': prepare()
    elif action == 'run': run()
    else:
        plan, manifest, data, split = load_sources()
        if action == 'check-ready': report = readiness(plan)
        else:
            verify_features(manifest, data, split['test'])
            report = {'status': 'EXACT_INPUT_RECONSTRUCTION', 'cases': len(split['test']), 'tensors_per_case': 5,
                      'latencies_measured': False, 'source_sha256': rr.digest(__file__)}
        rr.write_json(OUT / (action + '.json'), report)
        print(json.dumps(report))


if __name__ == '__main__': main()
