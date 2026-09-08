"""Independent all60-only arithmetic/source audit. No model import or inference.

The checker deliberately does not call the aggregate's statistical functions.
Only metadata is read until the complete60 queue and final aggregate exist.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import collections
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
PLAN_SHA = '8d21e13af116dc97556b37060daaf86afed5f7ba645a5bdae18e3d4ffb3b05a9'
AGGREGATE_SHA = 'aa06a3280fed95be571f1b39677c84c20d193804897d4f4d55ec7a99053a2e7a'
MODELS = ['capacity_matched_deepsets', 'gnn', 'set_attention']
SEEDS = [42, 43, 44, 45, 46]
COUNTERS = dict(performance_JSON_reads=0, prediction_array_loads=0, tensor_array_loads=0, model_inference=0)


def read(p, performance=False):
    if performance:
        COUNTERS['performance_JSON_reads'] += 1
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def close(a, b):
    if a is None or b is None or isinstance(a, str) and a == '' or isinstance(b, str) and b == '':
        assert a in (None, '') and b in (None, ''), (a, b)
    else:
        assert np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=2e-12, atol=2e-13), (a, b)


def match(expected, actual):
    for key, value in expected.items():
        if isinstance(value, dict):
            match(value, actual[key])
        elif isinstance(value, (bool, str)):
            assert actual[key] == value, (key, value, actual[key])
        else:
            close(value, actual[key])


def csv_rows(p):
    with Path(p).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def gate(plan_path, aggregate):
    assert sha(plan_path) == PLAN_SHA
    p = read(plan_path)
    assert p['seeds'] == SEEDS and p['new_run_count'] == 60 and len(p['protocols']) == 12
    assert p['new_model'] == 'set_attention'
    assert p['epistemic_status'] == 'AFTER_ORIGINAL_TEST_RESULTS_REVIEW_EXPLORATORY'
    for path, h in p['extension_source_sha256'].items():
        assert sha(path) == h
    for rel, h in p['original_source_sha256'].items():
        assert sha(WORK/rel) == h
    assert sha(p['manifest_path']) == p['manifest_sha256']
    assert sha(p['original_completed_audit_path']) == p['original_completed_audit_sha256']
    assert sha(p['original_completed_report_path']) == p['original_completed_report_sha256']
    for spec in p['protocols'].values():
        assert sha(spec['split_path']) == spec['split_sha256']
    expected = {(pr, 'set_attention', seed) for pr in p['protocols'] for seed in SEEDS}
    root = Path(p['runs_root'])
    allowed_results = {(root/pr/model/f'seed_{seed}'/'result.json').resolve() for pr, model, seed in expected}
    unexpected_results = [str(path) for path in root.glob('*/*/*/result.json') if path.resolve() not in allowed_results]
    assert not unexpected_results, ('Unregistered extension result paths', unexpected_results)
    assert not (aggregate/'failure.json').exists(), 'Aggregate failure evidence cannot be bypassed by existing report files'
    completed = []
    for pr, model, seed in sorted(expected):
        folder = root/pr/model/f'seed_{seed}'
        names = ['status.json', 'run_metadata.json', 'result.json', 'best.pt', 'test_predictions.npz', 'learning_curve.csv']
        if not all((folder/n).is_file() for n in names):
            continue
        meta = read(folder/'run_metadata.json')
        assert (meta['protocol'], meta['mode'], meta['seed']) == (pr, model, seed)
        assert meta['plan_sha256'] == PLAN_SHA
        if read(folder/'status.json')['status'] == 'COMPLETE_EXPLORATORY_EXTENSION':
            completed.append((pr, model, seed))
    queue = read(root/'queue_status.json') if (root/'queue_status.json').exists() else None
    if queue and queue['status'] == 'ERROR_STOPPED_EXTENSION':
        raise AssertionError('Failed queue cannot be interpreted as incomplete healthy work')
    state = 'WAITING_FOR_ALL60_COMPLETE'
    if len(completed) == 60 and queue and queue['status'] == 'COMPLETE_60_EXPLORATORY_EXTENSION':
        identities = [(r['protocol'], r['model'], r['seed']) for r in queue['completed']]
        assert len(identities) == len(set(identities)) == 60 and set(identities) == expected
        assert queue['active'] is None and queue['plan_sha256'] == PLAN_SHA
        required = [aggregate/'report.json', aggregate/'run_integrity.json']
        required += [aggregate/pr/name for pr in p['protocols'] for name in
                     ['summary.json', 'engineering_ci.json', 'engineering_ci.csv', 'per_case_metrics.csv', 'time_metrics.csv', 'strata_by_seed.csv']]
        state = 'READY_FOR_INDEPENDENT_FULL_MATRIX_REVIEW' if all(x.is_file() for x in required) else 'WAITING_FOR_COMPLETE_AGGREGATE_ARTIFACTS'
    return p, dict(status=state, completed_metadata_runs=len(completed), required_new_runs=60,
                   aggregate_directory=str(aggregate), partial_performance_read=False, counters=dict(COUNTERS))


def scalar_metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    e = p-a
    positive = np.maximum(e, 0)
    maes, maxima = np.abs(e).mean(1), positive.max(1)
    denominator = np.square(a-a.mean()).sum()
    out = dict(equal_case_MAE=float(maes.mean()), equal_case_RMSE=float(np.sqrt(np.square(e).mean())),
               mean_case_RMSE=float(np.sqrt(np.square(e).mean(1)).mean()),
               R2_pooled=float(1-np.square(e).sum()/denominator) if denominator else None,
               maximum_positive_error=float(positive.max()), maximum_absolute_error=float(np.abs(e).max()),
               case_time_positive_error_q95=float(np.quantile(positive, .95)),
               case_time_positive_error_q99=float(np.quantile(positive, .99)))
    for q in (50, 90, 95, 99):
        out[f'per_case_MAE_q{q}'] = float(np.quantile(maes, q/100))
        out[f'per_case_max_positive_error_q{q}'] = float(np.quantile(maxima, q/100))
    return out


def ranks(a, p):
    ra, rp = rankdata(a), rankdata(p)
    rho = None if np.ptp(ra) == 0 or np.ptp(rp) == 0 else float(np.corrcoef(ra, rp)[0, 1])
    i, j = np.triu_indices(len(a), 1)
    da, dp = a[i]-a[j], p[i]-p[j]
    keep = da != 0
    score = np.where(dp[keep] == 0, .5, (np.sign(da[keep]) == np.sign(dp[keep])).astype(float))
    return dict(spearman=rho, pairwise_order_agreement=float(score.mean()) if len(score) else None,
                eligible_pairs=int(keep.sum()), excluded_reference_tied_pairs=int((~keep).sum()),
                excluded_reference_tie_fraction=float((~keep).mean()) if len(keep) else None)


def crossings(a, p, t, q):
    # Explicit first-hit scan is independent of the generator's argmax vectorization.
    at = np.asarray([next((float(t[k]) for k, v in enumerate(row) if v <= q), np.inf) for row in a])
    pt = np.asarray([next((float(t[k]) for k, v in enumerate(row) if v <= q), np.inf) for row in p])
    ae, pe = np.isfinite(at), np.isfinite(pt)
    tp, fn, fp, tn = [int(v.sum()) for v in (ae & pe, ae & ~pe, ~ae & pe, ~ae & ~pe)]
    delay = pt[ae & pe]-at[ae & pe]
    recross = sum(bool(np.any(row[t > first] > q)) for row, first in zip(p, pt) if np.isfinite(first))
    recall, fpr = tp/(tp+fn) if tp+fn else None, fp/(fp+tn) if fp+tn else None
    return dict(threshold=q, reference_events=int(ae.sum()), reference_censored=int((~ae).sum()),
                predicted_events=int(pe.sum()), predicted_censored=int((~pe).sum()),
                true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn,
                recall=recall, false_positive_rate=fpr,
                balanced_accuracy=(recall+1-fpr)/2 if recall is not None and fpr is not None else None,
                both_event_count=tp, both_event_population_fraction=tp/len(a), both_event_reference_event_fraction=recall,
                conditional_timing_MAE_s=float(np.abs(delay).mean()) if len(delay) else None,
                conditional_timing_median_abs_s=float(np.median(np.abs(delay))) if len(delay) else None,
                conditional_timing_bias_s=float(delay.mean()) if len(delay) else None,
                optimistic_late_or_missed_count=fn+int((delay > 0).sum()), recross_above_threshold_count=recross,
                recross_fraction_all_cases=recross/len(a), recross_fraction_predicted_event_cases=recross/int(pe.sum()) if pe.any() else None)


def engineering_points(a, p, t, families):
    a, p = np.asarray(a, float), np.asarray(p, float)
    weights = np.diff(t)/(2*(t[-1]-t[0]))
    ja = np.sum((a[:, :-1]+a[:, 1:])*weights, axis=1)
    jp = np.sum((p[:, :-1]+p[:, 1:])*weights, axis=1)
    end = {'final': (a[:, -1], p[:, -1]), 'time_average': (ja, jp)}
    per_family = {}
    for fam in sorted(set(families)):
        mask = families == fam
        per_family[fam] = dict(cases=int(mask.sum()), **{k: ranks(x[mask], y[mask]) for k, (x, y) in end.items()})
    equal = {}
    for endpoint in end:
        equal[endpoint] = {}
        for metric in ('spearman', 'pairwise_order_agreement'):
            eligible = [r for r in per_family.values() if r[endpoint]['eligible_pairs'] > 0]
            vals = [r[endpoint][metric] for r in eligible if r[endpoint][metric] is not None]
            full = bool(eligible) and len(vals) == len(eligible)
            equal[endpoint][metric] = dict(mean_over_defined_families=float(np.mean(vals)) if full else None,
                defined_families=len(vals), total_families=len(per_family), reference_eligible_families=len(eligible),
                full_reference_family_support=full, missing_model_scores_not_silently_averaged=True)
    out = dict(cases=len(a), grid_points=len(t), horizon_s=float(t[-1]),
        summaries={k: dict(MAE=float(np.abs(y-x).mean()), bias=float((y-x).mean())) for k, (x, y) in end.items()},
        crossings={str(q): crossings(a, p, t, q) for q in (.8, .6)},
        rankings={k: ranks(x, y) for k, (x, y) in end.items()},
        positive_step_fraction=float((np.diff(p, axis=1) > 1e-6).mean()), within_family_rankings=per_family,
        equal_family_ranking_summaries=equal)
    flat = {f'{k}_{m}': v[m] for k, v in out['summaries'].items() for m in ('MAE', 'bias')}
    for q, row in out['crossings'].items():
        for m in ('recall', 'false_positive_rate', 'balanced_accuracy', 'conditional_timing_MAE_s', 'conditional_timing_median_abs_s', 'conditional_timing_bias_s'):
            flat[f'q{q}_{m}'] = row[m]
    for k, row in out['rankings'].items():
        for m in ('spearman', 'pairwise_order_agreement'):
            flat[f'{k}_{m}'] = row[m]
            flat[f'{k}_equal_family_{m}'] = equal[k][m]['mean_over_defined_families']
    return out, flat


def bootstrap_primary(difference, geometry, repetitions=5000):
    ns, nc = difference.shape
    assert len(set(geometry)) == nc
    order = np.argsort(geometry)
    ordered = difference[:, order]
    rng = np.random.default_rng(20260907)
    vals = {k: [] for k in ('geometry_only', 'seed_only', 'two_way')}
    for _ in range(repetitions):
        sw = rng.multinomial(ns, np.ones(ns)/ns)
        gw = rng.multinomial(nc, np.ones(nc)/nc)
        vals['geometry_only'].append(float(np.sum(ordered*gw)/(ns*nc)))
        vals['seed_only'].append(float(np.sum(difference.mean(1)*sw)/ns))
        vals['two_way'].append(float(np.sum(ordered*sw[:, None]*gw[None, :])/(ns*nc)))
    return {k: dict(percentile_95=np.quantile(v, [.025, .975]).tolist(), bootstrap_sd=float(np.std(v, ddof=1))) for k, v in vals.items()}


def replay_check(r):
    maximum = r['cpu_max_abs_difference']
    assert np.isfinite(maximum) and maximum >= 0 and r['cpu_diagnostic_tolerance'] == 3e-6
    if maximum < 3e-6:
        assert r['cpu_within_original_tolerance'] is True and r['status'] == 'CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE'
    else:
        assert r['cpu_within_original_tolerance'] is False
        assert r['same_backend_gpu_bitwise_equal'] is True and r['gpu_max_abs_difference'] == 0
        assert r['status'] == 'GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED'


def analytic_checks():
    a = np.array([[1., .7, .5], [1., 1., 1.]])
    p = np.array([[1., .9, .5], [1., .7, .9]])
    out = crossings(a, p, np.array([0., 60., 120.]), .8)
    assert out['conditional_timing_MAE_s'] == 60 and out['false_positive'] == 1
    assert out['recross_above_threshold_count'] == 1 and out['balanced_accuracy'] == .5
    zero = crossings(a, p, np.array([0., 60., 120.]), .4)
    assert zero['recall'] is None and zero['conditional_timing_MAE_s'] is None and zero['false_positive_rate'] == 0
    assert ranks(np.ones(3), np.arange(3.))['spearman'] is None
    assert scalar_metrics(np.ones((2, 3)), np.ones((2, 3)))['R2_pooled'] is None
    constant = bootstrap_primary(np.full((5, 3), .2), ['c', 'a', 'b'], repetitions=100)
    for v in constant.values():
        close(v['percentile_95'], [.2, .2])
    seed_only_d = np.repeat(np.arange(5.)[:, None], 3, axis=1)
    close(bootstrap_primary(seed_only_d, ['c', 'a', 'b'], 100)['geometry_only']['percentile_95'], [2., 2.])
    case_only_d = np.repeat(np.arange(3.)[None, :], 5, axis=0)
    close(bootstrap_primary(case_only_d, ['c', 'a', 'b'], 100)['seed_only']['percentile_95'], [1., 1.])
    return dict(status='ANALYTIC_SIGN_CROSSED_RESAMPLING_CENSORING_TIE_TESTS_PASSED', actual_extension_performance_used=False)


def audit(plan, aggregate, output):
    assert not output.exists()
    final = read(aggregate/'report.json', performance=True)
    assert final['status'] == 'COMPLETE_EXPLORATORY_60_NEW_PLUS120_REUSED_MATRIX'
    assert final['plan_sha256'] == PLAN_SHA and final['new_runs_audited'] == 60 and final['reused_comparator_runs'] == 120
    assert final['protocols'] == list(plan['protocols']) and final['models'] == MODELS and final['seeds'] == SEEDS
    assert final['epistemic_status'] == plan['epistemic_status'] and final['cpu_prediction_replay_enabled'] is True
    assert final['analysis_source_sha256'][str(HERE/'aggregate_extension.py')] == AGGREGATE_SHA
    for path, h in final['analysis_source_sha256'].items():
        assert sha(path) == h
    for rel, h in final['outputs_sha256'].items():
        assert sha(aggregate/rel) == h
    auth_path = HERE/'execution_authorization_root.json'
    auth = read(auth_path)
    assert sha(auth_path) == final['execution_authorization_sha256'] and auth['plan_sha256'] == PLAN_SHA
    assert sha(auth['independent_software_review_path']) == auth['independent_software_review_sha256']
    software = read(auth['independent_software_review_path'])
    assert software['software_review_passed'] is True and software['extension_plan_sha256'] == PLAN_SHA
    assert software['optimizer_updates'] == 0 and software['scientific_target_accuracy_read'] is False
    # Bind the prior real software/gradient/mask checks; do not rerun model inference.
    smoke = read(HERE/'software_smoke_cpu_final.json')
    assert sha(HERE/'software_smoke_cpu_final.json') in software['source_sha256'].values()
    assert smoke['padding_invariance_max_abs'] == 0 and smoke['finite_forward_backward'] is True
    integrity = read(aggregate/'run_integrity.json')
    new = {(r['protocol'], r['model'], r['seed']): r for r in integrity['new_run_audits']}
    assert len(new) == len(integrity['new_run_audits']) == 60
    old = {}
    for row in integrity['reused_original_comparators']:
        ref = row['reference']
        key = (ref['protocol'], ref['model'], ref['seed'])
        assert key not in old
        old[key] = row
    assert len(old) == 120
    planned_refs = {(r['protocol'], r['model'], r['seed']): r for r in plan['reused_comparator_artifacts']}
    original_audits = {(r['protocol'], r['model'], r['seed']): r for r in read(plan['original_completed_audit_path'])['audits']}
    replay_counts = collections.Counter()
    paths = {}
    for protocol in plan['protocols']:
        for model in MODELS:
            for seed in SEEDS:
                key = (protocol, model, seed)
                if model == 'set_attention':
                    row = new[key]
                    assert row['status'] == 'PASSED_EXTENSION_BINDINGS_AND_REPLAY' and row['parameters'] == 194273
                    assert row['plan_sha256'] == PLAN_SHA
                    folder = Path(plan['runs_root'])/protocol/model/f'seed_{seed}'
                    for name, h in row['source_artifact_sha256'].items():
                        assert sha(folder/name) == h
                    meta = read(folder/'run_metadata.json')
                    assert meta['extension_source_sha256'] == plan['extension_source_sha256']
                    assert meta['backend_configuration'] == plan['backend_configuration']
                    assert meta['configuration'] == plan['training_configuration']
                    assert meta['dataset_sha256'] == plan['manifest_sha256']
                    assert meta['split_sha256'] == plan['protocols'][protocol]['split_sha256']
                    log = csv_rows(folder/'learning_curve.csv')
                    assert [int(v['epoch']) for v in log] == list(range(1, len(log)+1))
                    val = np.asarray([float(v['validation_mse']) for v in log])
                    assert np.isfinite(val).all() and np.all(val >= 0)
                    best = int(np.argmin(val))+1
                    result = read(folder/'result.json', performance=True)
                    assert best == row['best_epoch'] == result['best_epoch']
                    assert len(log) == row['epoch_count'] == result['epochs_run']
                    assert result['best_validation_mse'] == val[best-1]
                    config = plan['training_configuration']
                    assert len(log) <= config['epochs_max']
                    assert len(log) == config['epochs_max'] or len(log)-best == config['patience']
                    rates = [float(v['learning_rate']) for v in log]
                    expected_rates = [config['learning_rate']*(1+math.cos(math.pi*i/config['epochs_max']))/2 for i in range(len(log))]
                    assert np.allclose(rates, expected_rates, rtol=1e-12, atol=1e-15)
                    path = folder/'test_predictions.npz'
                    r = row['computational_replay']
                else:
                    entry = old[key]
                    assert entry['reference'] == planned_refs[key] and entry['audit'] == original_audits[key]
                    for item in entry['reference']['artifacts'].values():
                        assert sha(item['path']) == item['sha256']
                    path = Path(entry['reference']['artifacts']['test_predictions.npz']['path'])
                    r = entry['audit']['computational_replay']
                replay_check(r)
                replay_counts[r['status']] += 1
                paths[key] = path
    import torch
    manifest = read(plan['manifest_path'])
    tensor_path = Path(plan['manifest_path']).parent/manifest['tensor_path']
    assert sha(tensor_path) == manifest['tensor_sha256']
    COUNTERS['tensor_array_loads'] += 1
    data = torch.load(tensor_path, map_location='cpu', weights_only=True)
    for x, edge, attrs, glob, time in data['graphs'].values():
        n = len(x)
        assert 1 <= n <= 15 and x.shape == (n, 11) and glob.shape == (1, 3) and time.shape == (61, 4)
        valid = np.arange(15) < n
        assert int(valid.sum()) == n and valid.any()
    cases = {c['sample_id']: c for c in manifest['cases']}
    times = data['time_s'].numpy()
    assert np.array_equal(times, np.arange(61)*60)
    reports = []
    total_case_rows = total_strata_rows = 0
    for protocol, spec in plan['protocols'].items():
        split = read(spec['split_path'])
        ids = split['test']; nc = len(ids)
        geometry = [cases[s]['geometry_id'] for s in ids]
        assert nc == spec['counts']['test'] and len(set(geometry)) == nc
        truth = np.stack([data['targets'][s].numpy() for s in ids]).astype(float)
        family = np.asarray([cases[s]['fire_family'] for s in ids])
        summary = read(aggregate/protocol/'summary.json', performance=True)
        eng = read(aggregate/protocol/'engineering_ci.json', performance=True)
        assert summary['status'] == 'COMPLETE_EXPLORATORY_PROTOCOL_5_NEW_PLUS10_REUSED_RUNS'
        assert summary['models'] == MODELS and summary['seeds'] == SEEDS and summary['test_case_count'] == nc
        assert eng['status'] == 'COMPLETE_EXPLORATORY_THREE_MODEL_ENGINEERING_CI' and eng['test_case_count'] == nc
        per_case = csv_rows(aggregate/protocol/'per_case_metrics.csv')
        case_index = {(r['model'], int(r['seed']), r['sample_id']): r for r in per_case}
        assert len(per_case) == len(case_index) == 15*nc
        time_rows = csv_rows(aggregate/protocol/'time_metrics.csv')
        time_index = {(r['model'], int(r['seed']), float(r['time_s'])): r for r in time_rows}
        assert len(time_rows) == len(time_index) == 15*61
        point_index = {(r['model'], r['seed']): r for r in eng['per_seed_event_counts_and_metrics']}
        assert len(point_index) == 15
        predictions, mae, endpoints, metrics_by_cell, eng_point_values = {}, {}, {}, {}, {}
        for model in MODELS:
            collected, mvalues, eng_values = [], [], []
            for seed in SEEDS:
                path = paths[(protocol, model, seed)]
                COUNTERS['prediction_array_loads'] += 1
                with np.load(path, allow_pickle=False) as z:
                    assert z['sample_ids'].tolist() == ids and np.array_equal(z['time_s'], times) and np.array_equal(z['truth'], truth)
                    assert z['predictions'].dtype == np.float32
                    p = z['predictions'].astype(float)
                assert p.shape == (nc, 61) and np.isfinite(p).all()
                collected.append(p)
                metrics = scalar_metrics(truth, p); mvalues.append(metrics); metrics_by_cell[model, seed] = metrics
                e = p-truth
                for j, sid in enumerate(ids):
                    rr = case_index[model, seed, sid]
                    close(rr['MAE'], np.abs(e[j]).mean()); close(rr['RMSE'], np.sqrt(np.square(e[j]).mean()))
                    close(rr['maximum_overprediction'], np.maximum(e[j], 0).max())
                    assert rr['geometry_id'] == geometry[j] and rr['source_batch'] == cases[sid]['source_batch']
                for j, time in enumerate(times):
                    rr = time_index[model, seed, float(time)]
                    close(rr['case_MAE'], np.abs(e[:, j]).mean()); close(rr['case_RMSE'], np.sqrt(np.square(e[:, j]).mean()))
                    close(rr['maximum_overprediction'], np.maximum(e[:, j], 0).max())
                points, flat = engineering_points(truth, p, times, family)
                match(points, point_index[model, seed]); eng_values.append(flat)
            predictions[model] = np.stack(collected)
            eng_point_values[model] = eng_values
            mae[model] = np.abs(predictions[model]-truth).mean(2)
            endpoints[model] = dict(final_MAE=np.abs(predictions[model][:, :, -1]-truth[:, -1]),
                time_average_MAE=np.abs(np.trapezoid(predictions[model], times, axis=2)/3600-np.trapezoid(truth, times, axis=1)/3600))
            stated = next(r for r in summary['model_summaries'] if r['model'] == model)
            for metric in mvalues[0]:
                vals = [r[metric] for r in mvalues]
                close(stated['metrics'][metric]['mean'], float(np.mean(vals)) if None not in vals else None)
                close(stated['metrics'][metric]['sample_sd'], float(np.std(vals, ddof=1)) if None not in vals else None)
            for metric in eng_values[0]:
                vals = [r[metric] for r in eng_values if r[metric] is not None]
                s = eng['model_seed_summaries'][model][metric]
                assert s['defined_seeds'] == len(vals)
                close(s['mean_over_defined_seeds'], float(np.mean(vals)) if vals else None)
                close(s['seed_sd'], float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
        strata = csv_rows(aggregate/protocol/'strata_by_seed.csv')
        expected_strata = 0
        assignments = {'fire_family': [cases[s]['fire_family'] for s in ids],
                       'crack_count': [str(cases[s]['num_cracks']) for s in ids],
                       'source_batch': [cases[s]['source_batch'] for s in ids],
                       'retained_or_restored': ['retained911' if cases[s]['legacy_final_id'] else 'restored86' for s in ids],
                       'flat_response': ['flat' if cases[s]['flat_response'] else 'nonflat' for s in ids]}
        expected_strata = 15*sum(len(set(v)) for v in assignments.values())
        assert len(strata) == expected_strata
        seen = set()
        for rr in strata:
            key = (rr['model'], int(rr['seed']), rr['stratum'], rr['group'])
            assert key not in seen; seen.add(key)
            mask = np.asarray(assignments[rr['stratum']]) == rr['group']
            assert int(rr['case_count']) == int(mask.sum()) > 0
            metrics = scalar_metrics(truth[mask], predictions[rr['model']][SEEDS.index(int(rr['seed'])), mask])
            for k, v in metrics.items():
                close(rr[k], v)
        effects = {}
        for tag, baseline in [('attention_vs_matched_set', 'capacity_matched_deepsets'), ('attention_vs_sum_GNN', 'gnn')]:
            difference = mae[baseline]-mae['set_attention']
            ci = summary['paired_contrasts'][tag]
            assert ci['contrast'] == baseline+' MAE minus set_attention MAE; positive favors set_attention'
            assert ci['seeds'] == SEEDS and ci['independent_geometry_clusters'] == nc
            assert eng['paired_contrasts'][tag]['baseline'] == baseline
            assert eng['paired_contrasts'][tag]['candidate'] == 'set_attention'
            close(ci['point_estimate_equal_case_seed_mean'], difference.mean())
            close(ci['paired_effect_by_seed'], difference.mean(1))
            assert ci['bootstrap_repeats'] == 5000 and ci['bootstrap_random_seed'] == 20260907
            independent = bootstrap_primary(difference, geometry)
            for method, values in independent.items():
                match(values, ci['intervals'][method])
            for metric in ('final_MAE', 'time_average_MAE'):
                delta = endpoints[baseline][metric]-endpoints['set_attention'][metric]
                rng = np.random.default_rng(20260907); reps = []
                for _ in range(1000):
                    ix = rng.integers(0, nc, nc); ss = rng.integers(0, 5, 5)
                    reps.append(float(delta[ss[:, None], ix[None, :]].mean()))
                stated = eng['paired_contrasts'][tag]['metrics'][metric]
                close(stated['point_baseline_minus_candidate'], delta.mean())
                close(stated['percentile_95_paired_difference'], np.quantile(reps, [.025, .975]))
                assert stated['common_valid_bootstrap_replicates'] == 1000
            effects[tag] = dict(point=float(difference.mean()), paired_seed_effects=difference.mean(1).tolist(),
                                two_way95=independent['two_way']['percentile_95'], positive_favors='set_attention')
        ci_rows = csv_rows(aggregate/protocol/'engineering_ci.csv')
        assert len(ci_rows) == len({(r['contrast'], r['metric']) for r in ci_rows}) == 48
        for rr in ci_rows:
            contrast = eng['paired_contrasts'][rr['contrast']]
            ss = contrast['metrics'][rr['metric']]
            av = [v[rr['metric']] for v in eng_point_values[contrast['baseline']]]
            bv = [v[rr['metric']] for v in eng_point_values[contrast['candidate']]]
            supported = None not in av and None not in bv
            close(ss['point_baseline_minus_candidate'], float(np.mean(av)-np.mean(bv)) if supported else None)
            assert ss['baseline_defined_seeds'] == sum(v is not None for v in av)
            assert ss['candidate_defined_seeds'] == sum(v is not None for v in bv)
            assert ss['required_seed_count'] == 5 and 0 <= ss['common_valid_bootstrap_replicates'] <= 1000
            assert ss['bootstrap_repetitions'] == 1000
            close(ss['valid_bootstrap_fraction'], ss['common_valid_bootstrap_replicates']/1000)
            estimable = supported and ss['common_valid_bootstrap_replicates'] > 0
            if estimable:
                interval = ss['percentile_95_paired_difference']
                assert ss['status'] == 'ESTIMATED' and isinstance(interval, list) and len(interval) == 2
                assert np.isfinite(interval).all() and interval[0] <= interval[1]
            else:
                assert ss['percentile_95_paired_difference'] is None
                assert ss['status'] == 'NOT_ESTIMABLE_ON_ALL_PAIRED_SEEDS'
            close(rr['effect'], ss['point_baseline_minus_candidate'])
            close([float(rr['ci95_low']), float(rr['ci95_high'])] if rr['ci95_low'] else None, ss['percentile_95_paired_difference'])
            assert int(rr['valid_bootstrap']) == ss['common_valid_bootstrap_replicates']
            assert rr['status'] == ss['status'] and int(rr['paired_seed_count']) == 5 and int(rr['bootstrap_repetitions']) == 1000
        reports.append(dict(protocol=protocol, independent_geometries=nc, per_case_rows=len(per_case), strata_rows=len(strata), effects=effects,
                            engineering_event_counts_and_metrics=eng['per_seed_event_counts_and_metrics']))
        total_case_rows += len(per_case); total_strata_rows += len(strata)
        print(json.dumps({'independently_reconciled_protocol': protocol, 'new_runs': 5, 'reused_runs': 10}), flush=True)
    result = dict(status='PASS_INDEPENDENT_NUMERICAL_REVIEW_60_NEW_PLUS120_REUSED', numerical_review_passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), extension_plan_sha256=PLAN_SHA,
        aggregate_report_sha256=sha(aggregate/'report.json'), aggregate_run_integrity_sha256=sha(aggregate/'run_integrity.json'),
        completed_new_runs=60, reused_comparator_runs=120, protocols=list(plan['protocols']), models=MODELS, seeds=SEEDS,
        epistemic_status=plan['epistemic_status'], checker_sha256=sha(__file__), aggregate_directory=str(aggregate),
        whole_matrix_review=True, prediction_files_independently_recomputed=180, per_case_rows=total_case_rows,
        strata_rows=total_strata_rows, independent_primary_bootstrap=dict(contrasts=24, methods=3, repetitions=5000),
        independent_final_J_secondary_CI=dict(contrasts=48, repetitions=1000),
        remaining_secondary_CI_scope='All point metrics and all48CSV rows per protocol reconcile; remaining ranking/timing/classification bootstrap algorithms reviewed, not all interval resamples independently rerun.',
        computational_replay_records_checked=dict(replay_counts), reviewer_repeated_GPU_replay=False,
        mask_input_scope='All997 feature shapes/counts within1..15 with nonempty valid-key mask; prior independently bound mask/permutation/backward software checks. No new model inference in this reviewer.',
        actual_model_input_cases_checked=len(data['graphs']), original420_untouched=True, original7_budget_untouched=True,
        protocol_reviews=reports, counters=dict(COUNTERS), analytic_checks=analytic_checks(), submission_pass=False)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status', 'completed_new_runs', 'reused_comparator_runs', 'per_case_rows')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregate-directory', default=str(HERE/'evaluation_complete60'))
    parser.add_argument('--output', default=str(HERE/'independent_full_matrix_review.json'))
    parser.add_argument('--check-ready', action='store_true')
    parser.add_argument('--gate-evidence')
    parser.add_argument('--analytic-tests', action='store_true')
    args = parser.parse_args()
    if args.analytic_tests:
        print(json.dumps(analytic_checks(), indent=2)); return 0
    aggregate, output = Path(args.aggregate_directory).resolve(), Path(args.output).resolve()
    assert aggregate.is_relative_to(HERE) and output.is_relative_to(HERE)
    plan, state = gate(HERE/'plan_set_attention_60_v1.json', aggregate)
    if args.gate_evidence:
        evidence = Path(args.gate_evidence).resolve()
        assert evidence.is_relative_to(HERE)
        with evidence.open('x', encoding='utf-8') as f:
            json.dump(dict(state, created_utc=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__)), f, indent=2)
    if args.check_ready or state['status'] != 'READY_FOR_INDEPENDENT_FULL_MATRIX_REVIEW':
        print(json.dumps(state, indent=2)); return 0 if state['status'].startswith('READY') else 2
    audit(plan, aggregate, output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
