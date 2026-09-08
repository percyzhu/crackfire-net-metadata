"""Seal existing complete420 artifacts without recomputing scientific metrics.

Requires the finished queue, a complete replay audit, and all twelve saved
engineering CI outputs. Weights/predictions remain at their original paths;
their actual hashes are rebound in the snapshot, not duplicated here.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
SCI = HERE.parents[1]
ROOT = SCI.parent
PLAN = HERE / 'comparison_plan_v08_420.json'
PLAN_SHA = 'fd9088427cbc9234191b71e619626b188a98c43c9af248c9dd3e435afdb8f068'
WRAPPER_SHA = '3e863d1c6f9e71cada06dd6ef9274e9c35343f7802faf4b70ad35320f9636566'
AMENDMENT_SHA = 'b861837f1f30ef1df0803118ef3cdee9c616721e1106674ad1c5b9318a5d7611'
EVAL = HERE / 'evaluation'
REVIEW = SCI / 'review/eaai_editor/research_v08'
SNAPSHOTS = HERE / 'completed_protocol_snapshots'
READ_SHA = {}


def read(p):
    p = Path(p).resolve(); payload = p.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
    if p in READ_SHA: assert READ_SHA[p] == digest, 'Source changed within sealing invocation'
    READ_SHA[p] = digest
    return json.loads(payload.decode('utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ready():
    assert sha(PLAN) == PLAN_SHA
    plan = read(PLAN)
    queue = read(Path(plan['runs_root']) / 'queue_status.json')
    expected = {(p, m, s) for p in plan['protocols'] for m in plan['models'] for s in plan['seeds']}
    actual = {(v['protocol'], v['mode'], v['seed']) for v in queue['completed']}
    assert len(expected) == 420 and len(plan['protocols']) == 12
    assert len(plan['models']) == 7 and plan['seeds'] == [42, 43, 44, 45, 46]
    assert len(actual) == len(queue['completed']) and actual <= expected
    report = read(EVAL / 'report.json')
    ci_paths = [REVIEW / (p + '_engineering_proxy_ci' + suffix)
                for p in plan['protocols'] for suffix in ('.json', '.csv')]
    queue_done = (queue['status'] == 'COMPLETE_ARCHIVAL_AI_MATRIX' and
                  queue.get('active') is None and actual == expected)
    audit_done = (report['status'] == 'COMPLETE_420_RUN_MATRIX' and
                  report['audited_completed_runs'] == 420 and report['cpu_prediction_replay_enabled'])
    complete = queue_done and audit_done and all(p.is_file() for p in ci_paths)
    status = {'status': 'READY_FOR_FINAL420_SOURCE_BINDING' if complete else 'WAITING_FINAL420_QUEUE_REPLAY_AND_CI',
              'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'queue_status': queue['status'], 'queue_completed_unique': len(actual),
              'required_unique_identities': 420, 'models': 7, 'seeds': 5, 'protocols': 12,
              'complete_replay_available': audit_done,
              'missing_CI_file_count': sum(not p.is_file() for p in ci_paths),
              'prediction_arrays_read': 0, 'metric_payloads_read': 0, 'snapshot_created': False}
    return (plan, queue, report, expected, ci_paths) if complete else None, status


def seal(data):
    plan, queue, report, expected, ci_paths = data
    assert queue['plan_sha256'] == report['plan_sha256'] == PLAN_SHA
    assert sha(HERE / 'io_recovery_patch1.py') == queue['runtime_io_patch_sha256'] == WRAPPER_SHA
    assert sha(HERE / 'recovery_io_patch1/amendment.json') == queue['runtime_amendment_sha256'] == AMENDMENT_SHA
    assert (HERE / 'recovery_io_patch1/worker_stderr.log').stat().st_size == 0
    assert len(plan['source_sha256']) == 12
    assert {p: sha(ROOT / p) for p in plan['source_sha256']} == plan['source_sha256']
    assert sha(plan['manifest_path']) == plan['manifest_sha256']
    assert all(v['status'] == 'COMPLETE' and v['audited_run_count'] == 35 and
               v['expected_run_count'] == 35 and v['missing_cells'] == [] and
               v['scientific_summary_published'] for v in report['protocols'].values())
    assert set(report['protocols']) == set(plan['protocols'])
    audits = read(EVAL / 'run_integrity.json')['audits']
    indexed = {(a['protocol'], a['model'], a['seed']): a for a in audits}
    assert len(audits) == len(indexed) == 420 and set(indexed) == expected
    paths = [PLAN, EVAL / 'report.json', EVAL / 'run_integrity.json',
             Path(plan['runs_root']) / 'queue_status.json', Path(plan['manifest_path']),
             HERE / 'io_recovery_patch1.py', HERE / 'recovery_io_patch1/amendment.json',
             SNAPSHOTS / 'heartbeat_final420_runtime_20260907.json',
             SNAPSHOTS / 'aggregate_final420_20260907.log',
             SNAPSHOTS / 'final_matrix_statistics_20260907.json',
             SNAPSHOTS / 'loco_smoldering_35/snapshot_manifest.json',
             Path(__file__).resolve()]
    paths += [ROOT / p for p in plan['source_sha256']]
    paths += [HERE / 'aggregate_results.py', SCI / 'experiments/evaluation/evaluation_stats.py']
    for source in paths[-2:]: assert sha(source) == report['audit_source_sha256'][source.name]
    paths += ci_paths
    run_bindings, replay_count, flagged_count = [], 0, 0
    for identity in sorted(expected):
        protocol, model, seed = identity
        a = indexed[identity]
        assert a['status'] == 'PASSED_BINDINGS' and a['ordered_ids_truth_times_exact']
        assert a['plan_dataset_feature_target_split_checkpoint_hashes']
        folder = Path(plan['runs_root']) / protocol / model / ('seed_' + str(seed))
        binding = {'protocol': protocol, 'model': model, 'seed': seed, 'files': {}}
        for name in ('result.json', 'run_metadata.json', 'status.json', 'learning_curve.csv',
                     'best.pt', 'test_predictions.npz'):
            p = folder / name; digest = sha(p)
            binding['files'][name] = {'path': str(p), 'sha256': digest, 'bytes': p.stat().st_size}
            if name not in ('best.pt', 'test_predictions.npz'): paths.append(p)
        assert binding['files']['best.pt']['sha256'] == a['checkpoint_sha256']
        assert binding['files']['test_predictions.npz']['sha256'] == a['predictions_sha256']
        result = read(folder / 'result.json')
        assert set(result['topology_results']) == set(a['topology_computational_replay'])
        for rule, declared in result['topology_results'].items():
            assert declared['checkpoint_sha256'] == a['checkpoint_sha256']
            p = folder / ('topology_' + rule + '.npz')
            binding['files'][p.name] = {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}
        checks = [a['computational_replay']] + list(a['topology_computational_replay'].values())
        for c in checks:
            replay_count += 1
            assert c['cpu_diagnostic_tolerance'] == 3e-6
            if c['cpu_within_original_tolerance']:
                assert c['status'] == 'CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE' and c['cpu_max_abs_difference'] < 3e-6
            else:
                flagged_count += 1
                assert c['status'] == 'GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED'
                assert c['cpu_max_abs_difference'] >= 3e-6 and c['same_backend_gpu_bitwise_equal'] is True
                assert c['gpu_max_abs_difference'] == 0
        run_bindings.append(binding)
    assert replay_count == 780
    for protocol, spec in plan['protocols'].items():
        split_path = Path(spec['split_path']); assert sha(split_path) == spec['split_sha256']; paths.append(split_path)
        paths += [EVAL / protocol / name for name in ('summary.json', 'per_case_metrics.csv',
                  'strata_by_seed.csv', 'time_metrics.csv', 'topology_by_seed.csv')]
        ci = read(REVIEW / (protocol + '_engineering_proxy_ci.json'))
        assert ci['status'] == 'COMPLETE_35_RUN_PAIRED_ENGINEERING_PROXY_CI'
        assert ci['protocol'] == protocol and ci['completed_runs_checked'] == 35 and ci['seeds'] == plan['seeds']
        assert ci['plan_sha256'] == PLAN_SHA and ci['manifest_sha256'] == plan['manifest_sha256']
        assert len(ci['source_prediction_sha256']) == 35
        for p, digest in ci['source_prediction_sha256'].items(): assert sha(p) == digest
        for p, digest in ci['code_sha256'].items(): assert sha(p) == digest; paths.append(Path(p))
        p = REVIEW / 'engineering_proxy_evaluation_prespecification.md'
        assert sha(p) == ci['prespecification_sha256']; paths.append(p)
    # Fix all bytes before creating the directory; preserve workspace-relative paths.
    unique_paths = sorted(set(p.resolve() for p in paths), key=str)
    payloads = [(p, p.read_bytes()) for p in unique_paths]
    source_hashes = {str(p): hashlib.sha256(payload).hexdigest() for p, payload in payloads}
    assert all(sha(p) == digest for p, digest in READ_SHA.items())
    assert all(sha(p) == digest for p, digest in source_hashes.items())
    target = (SNAPSHOTS / 'final_matrix_420').resolve(); target.relative_to(SNAPSHOTS.resolve())
    target.mkdir()  # exclusive; a partial failure must never be silently overwritten
    files = []
    for source, payload in payloads:
        relative = source.relative_to(ROOT.resolve()); destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream: stream.write(payload)
        assert sha(destination) == sha(source) == source_hashes[str(source)]
        files.append({'relative_path': relative.as_posix(), 'source': str(source),
                      'sha256': source_hashes[str(source)], 'bytes': len(payload)})
    manifest = {'status': 'COMPLETE420_COMPUTATION_SNAPSHOT_PENDING_FINAL_FAMILY_INDEPENDENT_REVIEW',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'aggregate_generated_utc': report['generated_utc'], 'plan_sha256': PLAN_SHA,
                'unique_run_identities': 420, 'protocols': list(plan['protocols']),
                'models': plan['models'], 'seeds': plan['seeds'], 'complete_and_sparse_replay_calls': replay_count,
                'original_CPU_tolerance': 3e-6, 'CPU_backend_flags_requiring_original_GPU_exact': flagged_count,
                'copied_files': files, 'run_artifact_bindings': run_bindings,
                'large_weights_and_predictions_copied': False, 'journal_or_manuscript_pass': False,
                'does_not_include_formal_inference_timings': True}
    with (target / 'snapshot_manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False); stream.write('\n')
    return {'status': manifest['status'], 'snapshot': str(target), 'unique_run_identities': 420,
            'files_copied': len(files), 'manifest_sha256': sha(target / 'snapshot_manifest.json')}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seal', action='store_true'); args = parser.parse_args()
    data, status = ready()
    if data is None: print(json.dumps(status, indent=2)); return 2
    print(json.dumps(seal(data) if args.seal else status, indent=2)); return 0


if __name__ == '__main__': raise SystemExit(main())
