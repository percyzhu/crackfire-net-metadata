"""Verify the complete saved420 predictions and statistics without model execution."""
import argparse
import collections
import csv
import math
import sys
import numpy as np
import portable_lib as p


def same(a, b):
    if a is None or b is None: assert a is None and b is None
    else: assert np.allclose(a, b, rtol=1e-12, atol=1e-13, equal_nan=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--output', default=str(p.BASE / 'verification/saved_results.json'))
    args = parser.parse_args()
    package, rr, plan, manifest, tensors, splits, registry = p.initialize()
    assert package['checkpoints'] == len(registry) == 420, 'Full result verification requires the complete package'
    expected = {(pr, m, s) for pr in plan['protocols'] for m in plan['models'] for s in plan['seeds']}
    assert {(r['protocol'], r['model'], r['seed']) for r in registry} == expected
    import aggregate_results as ar
    counts = collections.Counter(); cases = {}; metric = {}; by_id = {c['sample_id']: c for c in manifest['cases']}
    for record in registry:
        pr, mode, seed = record['protocol'], record['model'], record['seed']; split = splits[pr]
        folder, meta, result, _ = p.check_run(rr, plan, manifest, record)
        for rule in record['graph_rules']:
            y, prediction = p.saved_array(folder, rule, split, tensors)
            original = result if rule == 'complete' else result['topology_results'][rule]
            for key, val in rr.metrics(y, prediction).items(): same(val, original[key])
            counts[rule] += 1
            if rule == 'complete':
                error = prediction.astype(float) - y.astype(float)
                cases[(pr, mode, seed)] = np.abs(error).mean(axis=1)
                metric[(pr, mode, seed)] = ar.scalar_metrics(y, prediction)
                with (p.RESEARCH / 'evaluation' / pr / 'per_case_metrics.csv').open(encoding='utf-8', newline='') as stream:
                    rows = [x for x in csv.DictReader(stream) if x['model'] == mode and int(x['seed']) == seed]
                rows = {x['sample_id']: x for x in rows}; assert set(rows) == set(split['test'])
                for i, sid in enumerate(split['test']):
                    same(cases[(pr, mode, seed)][i], float(rows[sid]['MAE']))
                    same(np.sqrt(np.square(error[i]).mean()), float(rows[sid]['RMSE']))
                    same(np.maximum(error[i], 0).max(), float(rows[sid]['maximum_overprediction']))
    assert dict(counts) == {'complete': 420, 'radius': 180, 'symmetric_knn': 180}
    for protocol in plan['protocols']:
        summary = p.read(p.RESEARCH / 'evaluation' / protocol / 'summary.json')
        assert summary['status'] == 'COMPLETE_PROTOCOL_35_RUNS'
        for model in summary['model_summaries']:
            for name, recorded in model['metrics'].items():
                values = [metric[(protocol, model['model'], s)][name] for s in plan['seeds']]
                same(recorded['mean'], None if any(v is None for v in values) else float(np.mean(values)))
                same(recorded['sample_sd'], None if any(v is None for v in values) else float(np.std(values, ddof=1)))
        for tag, first, second in [('graph_vs_capacity_matched', 'capacity_matched_deepsets', 'gnn'), ('mean_vs_sum', 'gnn', 'gnn_mean')]:
            a = np.stack([cases[(protocol, first, s)] for s in plan['seeds']]); b = np.stack([cases[(protocol, second, s)] for s in plan['seeds']])
            recorded = summary['paired_contrasts'][tag]
            same((a - b).mean(axis=1), recorded['paired_effect_by_seed'])
            same(float((a - b).mean()), recorded['point_estimate_equal_case_seed_mean'])
            if args.bootstrap:
                answer = ar.es.paired_bootstrap(a, b, [by_id[s]['geometry_id'] for s in splits[protocol]['test']], plan['seeds'], repeats=5000, random_seed=20260907)
                for name in ('geometry_only', 'seed_only', 'two_way'):
                    same(answer['intervals'][name]['percentile_95'], recorded['intervals'][name]['percentile_95'])
    report = {'status': 'PASSED_COMPLETE_SAVED_RESULT_ARITHMETIC', 'complete_checkpoints': 420, 'graph_view_counts': dict(counts),
              'cases_in_dataset': 997, 'protocols': 12, 'case_ID_truth_time_bindings_checked': True,
              'primary_bootstrap_recomputed': args.bootstrap, 'bootstrap_repetitions_if_requested': 5000,
              'model_executions': 0, 'optimizer_updates': 0, 'new_training_validated': False,
              'engineering_CI_scope': 'Preserved independent engineering review outputs are file-hash bound; not rebootstrapped by this entry.'}
    p.write_report(args.output, report); print(report['status']); return 0


if __name__ == '__main__': sys.exit(main())
