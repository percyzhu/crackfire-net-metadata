"""Independent raw-record arithmetic and coverage audit after real timings finish."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import math

import numpy as np

HERE = Path(__file__).resolve().parent
SCI = HERE.parents[2]
EXP = SCI / 'experiments/research_v08'
BENCH = EXP / 'inference_benchmark_v1'
MEASURED = BENCH / 'measured'
OUTPUT = HERE / 'inference_timing_independent_audit.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert np.isfinite(a) and np.isfinite(b)
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12), (a, b)


def record_index(records, names):
    result = {tuple(r[n] for n in names): r for r in records}
    assert len(result) == len(records), 'Duplicate summary keys'
    return result


def replay_confirmed(record):
    assert record and record['cpu_diagnostic_tolerance'] == 3e-6
    if record['cpu_within_original_tolerance']:
        assert record['status'] == 'CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE'
        assert 0 <= record['cpu_max_abs_difference'] < 3e-6
    else:
        assert record['status'] == 'GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED'
        assert record['cpu_max_abs_difference'] >= 3e-6
        assert record['same_backend_gpu_bitwise_equal'] is True and record['gpu_max_abs_difference'] == 0


def verify_artifact_binding(train, binding):
    """Check the real420 source files, not just an asserted binding status."""
    plan_sha = sha(EXP / 'comparison_plan_v08_420.json')
    assert binding['plan_sha256'] == plan_sha
    final = read(EXP / 'evaluation/report.json')
    assert final['status'] == 'COMPLETE_420_RUN_MATRIX' and final['cpu_prediction_replay_enabled'] is True
    assert final['purpose'] == 'ARCHIVAL_SURROGATE_RESEARCH_V08' and final['plan_sha256'] == plan_sha
    assert final['audited_completed_runs'] == 420 and set(final['protocols']) == set(train['protocols'])
    for value in final['protocols'].values():
        assert value['status'] == 'COMPLETE' and value['audited_run_count'] == value['expected_run_count'] == 35
        assert value['missing_cells'] == [] and value['scientific_summary_published'] is True
    source_paths = {'aggregate_results.py': EXP / 'aggregate_results.py',
                    'evaluation_stats.py': SCI / 'experiments/evaluation/evaluation_stats.py'}
    assert binding['audit_source_sha256'] == final['audit_source_sha256'] == {k: sha(v) for k,v in source_paths.items()}
    for relative, digest in train['source_sha256'].items():
        assert sha(SCI.parent / relative) == digest, 'Frozen training source changed'
    amendment_path = EXP / 'recovery_io_patch1/amendment.json'
    amendment = read(amendment_path)
    assert amendment['original_plan_sha256'] == plan_sha
    assert binding['runtime_wrapper_sha256'] == amendment['wrapper_sha256'] == sha(EXP / 'io_recovery_patch1.py')
    assert binding['runtime_amendment_sha256'] == sha(amendment_path)
    expected = {(p,m,s) for p in train['protocols'] for m in train['models'] for s in train['seeds']}
    assert len(expected) == 420
    recorded = record_index(binding['artifacts'], ('protocol', 'model', 'seed'))
    audits = record_index(read(EXP / 'evaluation/run_integrity.json')['audits'], ('protocol', 'model', 'seed'))
    assert set(recorded) == set(audits) == expected
    manifest = read(train['manifest_path'])
    assert sha(Path(train['manifest_path']).parent / manifest['tensor_path']) == manifest['tensor_sha256']
    for spec in train['protocols'].values():
        assert sha(spec['split_path']) == spec['split_sha256']
        assert read(spec['split_path'])['dataset_sha256'] == train['manifest_sha256']
    actual = {}
    for key, record in recorded.items():
        protocol, model, seed = key
        folder = Path(train['runs_root']) / protocol / model / ('seed_' + str(seed))
        assert Path(record['folder']).resolve() == folder.resolve()
        rules = ('radius', 'symmetric_knn') if model in ('gnn', 'gnn_mean', 'gnn_zero_edge_features') else ()
        names = {'run_metadata.json', 'result.json', 'status.json', 'learning_curve.csv', 'best.pt', 'test_predictions.npz'}
        names.update('topology_' + rule + '.npz' for rule in rules)
        hashes = record['files_sha256']; assert set(hashes) == names
        for name, digest in hashes.items():
            assert sha(folder / name) == digest, 'Actual artifact changed after premeasurement binding'
        audit = audits[key]
        assert audit['status'] == 'PASSED_BINDINGS'
        assert all(audit[k] is True for k in ('plan_dataset_feature_target_split_checkpoint_hashes',
                                            'ordered_ids_truth_times_exact', 'best_epoch_is_earliest_validation_minimum'))
        assert audit['checkpoint_sha256'] == hashes['best.pt'] and audit['predictions_sha256'] == hashes['test_predictions.npz']
        replay_confirmed(audit['computational_replay'])
        assert set(audit['topology_computational_replay']) == set(rules)
        assert audit['topology_files_bound_to_same_checkpoint'] == len(rules)
        for value in audit['topology_computational_replay'].values(): replay_confirmed(value)
        meta, source, status = (read(folder / name) for name in ('run_metadata.json','result.json','status.json'))
        for name, value in (('protocol', protocol), ('mode', model), ('seed', seed)):
            assert meta[name] == source[name] == value
        assert meta['purpose'] == source['purpose'] == 'ARCHIVAL_SURROGATE_RESEARCH_V08'
        assert meta['plan_sha256'] == plan_sha and meta['dataset_sha256'] == train['manifest_sha256']
        assert meta['split_sha256'] == train['protocols'][protocol]['split_sha256']
        assert meta['source_sha256'] == train['source_sha256'] and meta['configuration'] == train['training_configuration']
        for name in ('target_version', 'feature_version', 'feature_tensor_sha256', 'target_tensor_sha256'):
            assert meta[name] == manifest[name]
        assert source['checkpoint_sha256'] == hashes['best.pt'] and meta['parameters'] == audit['parameters']
        assert status['status'] == 'COMPLETE_ARCHIVAL_RESEARCH'
        assert source['epochs_run'] == status['epochs_run'] == audit['epoch_count']
        assert source['best_epoch'] == status['best_epoch']
        assert np.isfinite(source['training_wall_s']) and source['training_wall_s'] > 0
        assert source['training_wall_s'] == status['training_wall_s'] == audit['training_wall_s']
        assert audit['runtime_amendment_chain']['amendment_sha256'] == binding['runtime_amendment_sha256']
        patch = meta.get('runtime_io_patch')
        if patch:
            assert patch['sha256'] == binding['runtime_wrapper_sha256']
            assert patch['amendment_sha256'] == binding['runtime_amendment_sha256']
            assert Path(patch['path']).resolve() == (EXP / 'io_recovery_patch1.py').resolve()
            assert Path(patch['amendment_path']).resolve() == amendment_path.resolve()
        else:
            relative = str((folder / 'run_metadata.json').relative_to(EXP)).replace('\\','/')
            assert amendment['preserved_prior_completed_artifacts'][relative] == hashes['run_metadata.json']
        actual[key] = {'result': source, 'metadata': meta, 'hashes': hashes}
    return actual


def audit():
    result_path = MEASURED / 'summary.json'
    result = read(result_path) if result_path.exists() and not (MEASURED / 'failure.json').exists() else None
    if result is None or result.get('status') != 'COMPLETE_INFERENCE_TIMING':
        ready = {'status': 'WAITING_FOR_COMPLETE_REAL_TIMINGS', 'latency_records_read': 0,
                 'has_failure_record': (MEASURED / 'failure.json').exists(),
                 'summary_status': result.get('status') if result else None,
                 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        (HERE / 'inference_timing_audit_readiness.json').write_text(json.dumps(ready, indent=2), encoding='utf-8')
        print(json.dumps(ready)); return
    if OUTPUT.exists(): raise ValueError('Preserve an earlier independent timing review')
    assert result['status'] == 'COMPLETE_INFERENCE_TIMING'
    timing_path = BENCH / 'timing_plan.json'
    timing, train = read(timing_path), read(EXP / 'comparison_plan_v08_420.json')
    assert sha(timing_path) == result['timing_plan_sha256']
    assert timing['train_plan_sha256'] == sha(EXP / 'comparison_plan_v08_420.json')
    assert timing['benchmark_source_sha256'] == sha(EXP / 'benchmark_inference.py')
    assert timing['seed'] == 42 and timing['rounds'] == 3 and timing['batch_sizes'] == [1, 32]
    assert timing['protocol'] == 'iid997' and timing['times_per_case'] == 61 and timing['CPU_threads'] == 4
    assert timing['warmup_batches_per_shape'] == 10 and timing['warmup_full_population_passes'] == 1
    assert timing['stages'] == ['forward_resident_inputs', 'common_input_to_host_output']
    split = read(train['protocols']['iid997']['split_path'])
    manifest = read(train['manifest_path'])
    assert sha(train['manifest_path']) == train['manifest_sha256']
    assert sha(train['protocols']['iid997']['split_path']) == train['protocols']['iid997']['split_sha256']
    ids = timing['sample_ids']; assert ids == split['test'] and len(ids) == len(set(ids)) == 160
    counts = {c['sample_id']: c['num_cracks'] for c in manifest['cases']}
    modes = [r['model'] for r in timing['checkpoints']]; assert modes == train['models']
    assert timing['devices'] == ['cpu', 'cuda']
    for record in timing['checkpoints']:
        expected_checkpoint = Path(train['runs_root']) / 'iid997' / record['model'] / 'seed_42/best.pt'
        assert Path(record['checkpoint']).resolve() == expected_checkpoint.resolve()
        assert sha(record['checkpoint']) == record['checkpoint_sha256']
    environment = read(MEASURED / 'environment.json')
    binding = read(MEASURED / 'premeasurement_artifact_binding.json')
    assert binding['status'] == 'ALL420_FINAL_REPLAY_AND_ACTUAL_ARTIFACTS_BOUND_BEFORE_TIMING'
    assert binding['actual_runs_checked'] == 420
    assert binding['report_sha256'] == sha(EXP / 'evaluation/report.json')
    assert binding['integrity_sha256'] == sha(EXP / 'evaluation/run_integrity.json')
    for ready in (environment['readiness'], read(MEASURED / 'readiness_at_completion.json')):
        assert ready['status'] == 'READY' and ready['completed_units'] == 420
        assert ready['reasons'] == [] and ready['matching_processes'] == []
    assert environment['threads'] == 4 and environment['timing_plan_sha256'] == sha(timing_path)
    assert sha(MEASURED / 'premeasurement_artifact_binding.json') == result['artifact_binding_sha256'] == environment['artifact_binding_sha256']
    assert sha(MEASURED / 'environment.json') == result['environment_sha256']
    assert sha(MEASURED / 'raw_timings.csv') == result['raw_csv_sha256']
    actual_sources = verify_artifact_binding(train, binding)
    for record in timing['checkpoints']:
        actual = actual_sources[('iid997', record['model'], 42)]
        assert record['parameters'] == actual['metadata']['parameters']
        assert record['checkpoint_sha256'] == actual['hashes']['best.pt']
    with (MEASURED / 'raw_timings.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    expected = set()
    combinations = [('cpu', 'common_features', 'feature_preparation_only')]
    combinations += [(d, m, stage) for d in ('cpu', 'cuda') for m in modes
                     for stage in ('forward_resident_inputs', 'common_input_to_host_output')]
    for repeat in range(3):
        for device, mode, stage in combinations:
            for batch in (1, 32):
                for start in range(0, len(ids), batch):
                    expected.add((repeat, device, mode, stage, batch, tuple(ids[start:start+batch])))
    observed = set(); groups = collections.defaultdict(list); rounds = collections.defaultdict(list); strata = collections.defaultdict(list)
    for row in rows:
        samples = tuple(row['sample_ids'].split('|'))
        key = (int(row['round']), row['device'], row['model'], row['stage'], int(row['batch_size']), samples)
        assert key in expected and key not in observed, 'Missing/duplicate/unexpected timing group'
        observed.add(key)
        assert int(row['cases']) == len(samples)
        assert row['num_cracks'].split('|') == [str(counts[s]) for s in samples]
        seconds = float(row['elapsed_s']); assert np.isfinite(seconds) and seconds > 0
        value = (seconds, len(samples), samples)
        groups[key[1:5]].append(value); rounds[key[:5]].append(value)
        if key[4] == 1: strata[key[1:4] + (counts[samples[0]],)].append(value)
    assert observed == expected and len(rows) == result['rows'] == 14355
    main = record_index(result['summaries'], ('device', 'model', 'stage', 'batch_size'))
    assert set(main) == set(groups) and len(groups) == 58
    for key, values in groups.items():
        durations = np.asarray([v[0] for v in values]); cases = sum(v[1] for v in values); stated = main[key]
        assert stated['measurements'] == len(values) and stated['cases_total'] == cases
        for name, calculated in [('mean_batch_ms', durations.mean()*1000), ('median_batch_ms', np.median(durations)*1000),
                                 ('p95_batch_ms', np.percentile(durations, 95)*1000),
                                 ('amortized_ms_per_case', durations.sum()*1000/cases), ('cases_per_second', cases/durations.sum())]:
            close(stated[name], calculated)
    by_round = record_index(result['per_round_summaries'], ('round', 'device', 'model', 'stage', 'batch_size'))
    assert set(by_round) == set(rounds) and len(rounds) == 174
    for key, values in rounds.items():
        elapsed = math.fsum(v[0] for v in values); cases = sum(v[1] for v in values); stated = by_round[key]
        assert stated['cases_total'] == cases == 160 and stated['measurements'] == len(values)
        close(stated['mean_batch_ms'], elapsed*1000/len(values)); close(stated['amortized_ms_per_case'], elapsed*1000/cases)
        close(stated['cases_per_second'], cases/elapsed)
    by_count = record_index(result['single_query_by_crack_count'], ('device', 'model', 'stage', 'num_cracks'))
    assert set(by_count) == set(strata)
    for key, values in strata.items():
        durations = np.asarray([v[0] for v in values]); stated = by_count[key]
        assert stated['measurements'] == len(values) and stated['unique_cases'] == len({v[2] for v in values})
        close(stated['mean_single_query_ms'], durations.mean()*1000)
        close(stated['median_single_query_ms'], np.median(durations)*1000)
        close(stated['p95_single_query_ms'], np.percentile(durations,95)*1000)
    costs_path = MEASURED / 'training_costs_420.csv'
    assert sha(costs_path) == result['recorded_training_cost']['CSV_sha256']
    with costs_path.open(encoding='utf-8', newline='') as stream: cost_rows = list(csv.DictReader(stream))
    cost_keys = {(r['protocol'], r['model'], int(r['seed'])) for r in cost_rows}
    assert len(cost_keys) == len(cost_rows) == 420
    assert cost_keys == {(p,m,s) for p in train['protocols'] for m in modes for s in train['seeds']}
    cost_groups = collections.defaultdict(list)
    for row in cost_rows:
        actual = actual_sources[(row['protocol'], row['model'], int(row['seed']))]
        source = actual['result']
        assert actual['hashes']['result.json'] == row['result_sha256']
        close(source['training_wall_s'], float(row['training_wall_s']))
        assert source['epochs_run'] == int(row['epochs_run']) and source['best_epoch'] == int(row['best_epoch'])
        assert actual['metadata']['parameters'] == int(row['parameters'])
        cost_groups[(row['protocol'],row['model'])].append(row)
    cost_summary = result['recorded_training_cost']
    assert cost_summary['completed_runs'] == 420 and cost_summary['not_total_project_elapsed_or_isolated_training_speed'] is True
    close(cost_summary['sum_successful_training_wall_s'], math.fsum(float(r['training_wall_s']) for r in cost_rows))
    recorded = record_index(cost_summary['summaries'], ('protocol', 'model')); assert set(recorded) == set(cost_groups)
    for key, group in cost_groups.items():
        v = np.asarray([float(r['training_wall_s']) for r in group]); stated = recorded[key]; assert stated['seeds'] == len(v) == 5
        for name, val in [('mean_wall_s',np.mean(v)), ('SD_wall_s',np.std(v,ddof=1)), ('min_wall_s',min(v)), ('max_wall_s',max(v)),
                          ('mean_epochs_run',np.mean([int(r['epochs_run']) for r in group]))]: close(stated[name],val)
    report = {'status': 'REAL_TIMING_RAW_RECORDS_COVERAGE_AND_ARITHMETIC_AUDIT_PASSED',
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'raw_measurements_checked':len(rows),
              'actual_training_artifact_identities_checked':len(actual_sources),
              'summary_groups':len(groups),'round_groups':len(rounds),'single_query_N_groups':len(strata),'training_runs':420,
              'whole_61_time_curve_per_query':True,'repeated_rounds_are_not_independent_accuracy_samples':True,
              'scientific_manuscript_or_submission_pass':False,
              'scope':'Independent source/coverage/statistical/unit audit binding the prior full420 computational replay; does not rerun models, independently measure another machine, or prove an idle OS.',
              'source_sha256':{str(p):sha(p) for p in (Path(__file__), timing_path, result_path,
                  MEASURED/'raw_timings.csv', costs_path, MEASURED/'environment.json',
                  MEASURED/'premeasurement_artifact_binding.json', MEASURED/'readiness_at_completion.json',
                  EXP/'comparison_plan_v08_420.json', EXP/'benchmark_inference.py',
                  EXP/'evaluation/report.json', EXP/'evaluation/run_integrity.json', Path(train['manifest_path']))}}
    OUTPUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','raw_measurements_checked','summary_groups','round_groups','training_runs')}))


if __name__ == '__main__': audit()
