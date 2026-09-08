"""Independent scalar arithmetic: no import of the budget generator or model code.

Uses Python sorted blocks / fsum / statistics, rather than NumPy partition and
dot product. Checks every frozen prediction endpoint, exported case score,
selection probability, utility, seed summary, and fixed paired difference.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import collections
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results'
BASE = HERE.parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(name):
    with (OUT / name).open(encoding='utf-8', newline='') as f:
        yield from csv.DictReader(f)


def close(a, b):
    if a is None or b is None or a == '' or b == '':
        assert a in (None, '') and b in (None, ''), (a, b)
    else:
        assert math.isclose(float(a), float(b), rel_tol=2e-12, abs_tol=2e-13), (a, b)


def key(r):
    return (r['protocol'], r['model'], int(r['seed']), r['endpoint'])


def scalar_J(y):
    # Uniform 60s grid: integral /3600 = weighted sum /60.
    return math.fsum([.5 * float(y[0]), *map(float, y[1:-1]), .5 * float(y[-1])]) / 60


def metrics(group, k, raw=False):
    score_key = 'raw_prediction_score' if raw else 'ranking_score'
    ordered = sorted(group.values(), key=lambda r: r[score_key])
    cutoff = ordered[k - 1][score_key]
    low = [r for r in ordered if r[score_key] < cutoff]
    tie = [r for r in ordered if r[score_key] == cutoff]
    remaining = k - len(low)
    u = (math.fsum(r['reference_score'] for r in low) +
         remaining * statistics.fmean(r['reference_score'] for r in tie)) / k
    ys = [r['reference_score'] for r in ordered]
    oracle = statistics.fmean(sorted(ys)[:k])
    random = statistics.fmean(ys)
    regret = u - oracle
    assert regret >= -1e-12
    gain = random - u
    denom = random - oracle
    return dict(U=u, regret_raw=regret, regret_display=max(0., regret),
                improvement_over_random=gain, oracle_improvement_fraction=gain/denom if denom > 1e-12 else None,
                cutoff_score=cutoff, strictly_selected_count=len(low), cutoff_tied_count=len(tie),
                selected_from_cutoff_tie=remaining, U_oracle=oracle, U_random=random,
                available_oracle_improvement=denom,
                random_policy_SD=math.sqrt((1-k/len(ys))*statistics.variance(ys)/k))


def main():
    report_path = OUT / 'analysis_report.json'
    report = read(report_path)
    assert report['protocols'] == 12 and report['models'] == 7 and report['seeds'] == 5
    for name, digest in report['output_sha256'].items():
        assert sha(OUT/name) == digest
    for path, digest in report['source_hashes'].items():
        assert sha(path) == digest
    plan = read(BASE/'comparison_plan_v08_420.json')
    groups = collections.defaultdict(dict)
    for r in records('case_endpoint_scores.csv'):
        for field in ('raw_prediction_score', 'ranking_score', 'reference_score', 'normalization_change'):
            r[field] = float(r[field])
        r['input_group_size'] = int(r['input_group_size'])
        assert r['sample_id'] not in groups[key(r)]
        groups[key(r)][r['sample_id']] = r
    assert len(groups) == 840 and sum(map(len, groups.values())) == 99470
    prediction_count = 0
    raw_endpoint_max_error = 0.
    for protocol in plan['protocols']:
        ids = read(plan['protocols'][protocol]['split_path'])['test']
        for model in plan['models']:
            for seed in plan['seeds']:
                path = Path(plan['runs_root'])/protocol/model/f'seed_{seed}'/'test_predictions.npz'
                assert sha(path) == report['prediction_sha256'][str(path)]
                with np.load(path, allow_pickle=False) as z:
                    assert z['sample_ids'].tolist() == ids
                    assert np.array_equal(z['time_s'], np.arange(61)*60)
                    for endpoint in ('terminal', 'J'):
                        gg = groups[(protocol, model, seed, endpoint)]
                        assert set(gg) == set(ids)
                        equivalent = collections.defaultdict(list)
                        for j, sid in enumerate(ids):
                            pred = float(z['predictions'][j, -1]) if endpoint == 'terminal' else scalar_J(z['predictions'][j])
                            truth = float(z['truth'][j, -1]) if endpoint == 'terminal' else scalar_J(z['truth'][j])
                            close(pred, gg[sid]['raw_prediction_score'])
                            close(truth, gg[sid]['reference_score'])
                            raw_endpoint_max_error = max(raw_endpoint_max_error, abs(pred-gg[sid]['raw_prediction_score']))
                            equivalent[gg[sid]['input_group']].append(gg[sid])
                        for members in equivalent.values():
                            assert all(r['input_group_size'] == len(members) for r in members)
                            if model in ('fire_only', 'global_stats'):
                                score = statistics.fmean(r['raw_prediction_score'] for r in members)
                            else:
                                assert len(members) == 1
                                score = members[0]['raw_prediction_score']
                            for r in members:
                                close(r['ranking_score'], score)
                                close(r['normalization_change'], r['ranking_score']-r['raw_prediction_score'])
                prediction_count += 1
    main_rows = list(records('model_seed_utility.csv'))
    main_lookup = {}
    expected = {}
    budget_denominators = {'10%': 10, '25%': 4, '50%': 2}
    for r in main_rows:
        kk = key(r)
        n, k = int(r['N']), int(r['k'])
        assert n == len(groups[kk]) and k == n//budget_denominators[r['budget']]
        close(r['actual_budget_fraction'], k/n)
        calc = metrics(groups[kk], k)
        for field in calc:
            if field in r:
                close(r[field], calc[field])
        expected[(*kk, r['budget'])] = calc
        main_lookup[(*kk, r['budget'])] = r
    assert len(expected) == len(main_rows) == 2520
    weight_sums = collections.defaultdict(list)
    weight_count = 0
    for r in records('case_selection_probabilities.csv'):
        kk = key(r)
        calc = expected[(*kk, r['budget'])]
        score = groups[kk][r['sample_id']]['ranking_score']
        if score < calc['cutoff_score']:
            p = 1.
        elif score > calc['cutoff_score']:
            p = 0.
        else:
            p = calc['selected_from_cutoff_tie']/calc['cutoff_tied_count']
        close(r['inclusion_probability'], p)
        weight_sums[(*kk, r['budget'])].append(float(r['inclusion_probability']))
        weight_count += 1
    assert weight_count == 298410
    for kk, values in weight_sums.items():
        close(math.fsum(values), int(main_lookup[kk]['k']))
    raw_count = 0
    for r in records('raw_uncanonicalized_baseline_diagnostics.csv'):
        assert r['model'] in ('fire_only', 'global_stats')
        calc = metrics(groups[key(r)], int(r['k']), raw=True)
        for field in calc:
            if field in r:
                close(r[field], calc[field])
        raw_count += 1
    assert raw_count == 720
    for r in records('reference_budgets.csv'):
        kk = (r['protocol'], 'fire_only', 42, r['endpoint'], r['budget'])
        calc = expected[kk]
        for field in ('U_oracle', 'U_random', 'available_oracle_improvement', 'random_policy_SD'):
            close(r[field], calc[field])
        ys = [v['reference_score'] for v in groups[kk[:4]].values()]
        assert (r['constant_reference'] == 'True') == (len(set(ys)) == 1)
    metrics_names = ['U', 'regret_raw', 'regret_display', 'improvement_over_random', 'oracle_improvement_fraction']
    for r in records('model_five_seed_summary.csv'):
        rr = [main_lookup[(r['protocol'], r['model'], seed, r['endpoint'], r['budget'])] for seed in plan['seeds']]
        assert int(r['independent_seed_count']) == 5
        for m in metrics_names:
            vals = [float(a[m]) for a in rr if a[m] != '']
            assert int(r[m+'_defined_seeds']) == len(vals)
            statistics_values = dict(mean=statistics.fmean(vals), sample_sd=statistics.stdev(vals), min=min(vals), max=max(vals)) if vals else dict.fromkeys(['mean', 'sample_sd', 'min', 'max'])
            for stat, value in statistics_values.items():
                close(r[m+'_'+stat], value)
    paired_groups = collections.defaultdict(list)
    for r in records('paired_seed_contrasts.csv'):
        baseline = main_lookup[(r['protocol'], r['baseline'], int(r['seed']), r['endpoint'], r['budget'])]
        candidate = main_lookup[(r['protocol'], r['candidate'], int(r['seed']), r['endpoint'], r['budget'])]
        diff = float(baseline['U']) - float(candidate['U'])
        close(r['U_baseline_minus_candidate'], diff)
        close(diff, float(baseline['regret_raw'])-float(candidate['regret_raw']))
        paired_groups[(r['protocol'], r['contrast'], r['endpoint'], r['budget'])].append(diff)
    assert len(paired_groups) == 144
    for r in records('paired_five_seed_summary.csv'):
        vv = paired_groups[(r['protocol'], r['contrast'], r['endpoint'], r['budget'])]
        assert len(vv) == 5
        for stat, value in dict(mean=statistics.fmean(vv), sample_sd=statistics.stdev(vv), min=min(vv), max=max(vv)).items():
            close(r['U_baseline_minus_candidate_'+stat], value)
    single_input = [r for r in main_rows if len({v['input_group'] for v in groups[key(r)].values()}) == 1]
    max_single_input_gain = max(abs(float(r['improvement_over_random'])) for r in single_input)
    assert max_single_input_gain <= 1e-12
    audit = dict(status='COMPLETE_ALL_SEVEN_MODEL_BUDGET_ARITHMETIC_INDEPENDENTLY_RECONCILED_NOT_SUBMISSION_REVIEW',
                 created_utc=datetime.now(timezone.utc).isoformat(), generator_functions_imported=False,
                 independent_algorithm='Python sorted cutoff blocks / fsum / statistics, exact independent trapezoid weighted sum; no partition/dot or generator calls',
                 original_prediction_files_checked=prediction_count, endpoint_case_rows_checked=99470,
                 raw_endpoint_max_float64_recalculation_difference=raw_endpoint_max_error,
                 selection_probability_rows_checked=weight_count, model_seed_utility_rows_checked=2520,
                 reference_budget_rows_checked=72, seed_summary_rows_checked=504,
                 raw_baseline_diagnostic_rows_checked=raw_count, paired_seed_rows_checked=720, paired_summary_rows_checked=144,
                 canonical_single_effective_input_budget_rows=len(single_input), max_abs_single_input_improvement_float_residual=max_single_input_gain,
                 meaningful_negative_improvement_rows=sum(float(r['improvement_over_random']) < -1e-12 for r in main_rows),
                 undefined_oracle_fraction_rows=sum(r['oracle_improvement_fraction'] == '' for r in main_rows),
                 tiny_negative_regret_rows=sum(-1e-12 <= float(r['regret_raw']) < 0 for r in main_rows),
                 numerical_interpretation='Constant-input policies equal random in exact arithmetic. Tiny recorded roundoff is preserved and is not evidence of prioritization information. Larger negative regret would fail.',
                 analysis_report_sha256=sha(report_path), audit_code_sha256=sha(__file__),
                 no_new_training_inference_or_FE=True, no_cross_protocol_population_or_utility_pooling=True,
                 submission_pass=False)
    with (OUT/'independent_arithmetic_audit.json').open('x', encoding='utf-8') as f:
        json.dump(audit, f, indent=2, allow_nan=False)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
