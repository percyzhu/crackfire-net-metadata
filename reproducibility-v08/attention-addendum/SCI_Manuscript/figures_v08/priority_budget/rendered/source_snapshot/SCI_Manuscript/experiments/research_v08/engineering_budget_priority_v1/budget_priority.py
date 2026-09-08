"""Read-only, post-review exploratory budget prioritization of frozen predictions.

No model import, model construction, forward call, training, or FE calculation.
Torch is used only to deserialize the frozen small feature/target tensors.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import collections
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SCI = BASE.parents[1]
REVIEW = SCI / 'review/eaai_editor/research_v08'
MD_SHA = '77a8c53efe548b3cfde9620e81b36c089558e9a0e52cc7c86dddb152d12ba24d'
JSON_SHA = 'efeaf945c10c7f7563b460f34c82bf7658cb11ff01df1467f1e2c71685d4257d'
BUDGETS = [('10%', 1, 10), ('25%', 1, 4), ('50%', 1, 2)]
METRICS = ['U', 'regret_raw', 'regret_display', 'improvement_over_random', 'oracle_improvement_fraction']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(p, rows):
    with Path(p).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def arr_key(a):
    a = np.ascontiguousarray(a)
    return (a.dtype.str, a.shape, a.tobytes())


def fingerprint(parts):
    h = hashlib.sha256()
    for a in parts:
        dtype, shape, raw = arr_key(a)
        header = json.dumps([dtype, shape, len(raw)]).encode()
        h.update(len(header).to_bytes(8, 'little'))
        h.update(header)
        h.update(raw)
    return h.hexdigest()


def canonical_nodes(x):
    # Node order cannot distinguish a mathematical set/complete graph input.
    x = x.copy()
    x[x == 0] = 0  # signed zero has the same exact numerical value
    order = sorted(range(len(x)), key=lambda j: tuple(x[j]))
    return np.ascontiguousarray(x[order])


def load_fixed():
    md = REVIEW / 'engineering_budget_priority_protocol.md'
    jp = md.with_suffix('.json')
    assert sha(md) == MD_SHA and sha(jp) == JSON_SHA
    protocol = read(jp)
    plan_path = BASE / 'comparison_plan_v08_420.json'
    assert sha(plan_path) == protocol['training_plan_sha256']
    plan = read(plan_path)
    assert plan['models'] == protocol['models'] and plan['seeds'] == protocol['seeds']
    assert list(plan['protocols']) == protocol['protocols']
    manifest_path = Path(plan['manifest_path'])
    assert sha(manifest_path) == plan['manifest_sha256']
    manifest = read(manifest_path)
    tensor_path = manifest_path.parent / manifest['tensor_path']
    assert sha(tensor_path) == manifest['tensor_sha256']
    tensors = torch.load(tensor_path, map_location='cpu', weights_only=True)
    assert np.array_equal(tensors['time_s'].numpy(), np.arange(61) * 60)
    ids = {}
    for name, spec in plan['protocols'].items():
        assert sha(spec['split_path']) == spec['split_sha256']
        ids[name] = read(spec['split_path'])['test']
        assert len(ids[name]) == spec['counts']['test']
    return protocol, plan, manifest, tensors, ids


def groups_for(ids, keys):
    groups = collections.defaultdict(list)
    for sid in ids:
        groups[keys[sid]].append(sid)
    return dict(groups)


def input_gate():
    protocol, plan, manifest, tensors, splits = load_fixed()
    keys = {kind: {} for kind in ('fire_only', 'global_stats', 'node_models')}
    cases = {c['sample_id']: c for c in manifest['cases']}
    for sid, graph in tensors['graphs'].items():
        x, edge, attr, g, t = [a.numpy() for a in graph]
        n = len(x)
        assert x.shape == (n, 11) and g.shape == (1, 3) and t.shape == (61, 4)
        pairs = [tuple(p) for p in edge.T.tolist()]
        expected = set(itertools.permutations(range(n), 2))
        assert len(pairs) == len(expected) and set(pairs) == expected
        assert attr.shape == (n * (n - 1), 5)
        cx = canonical_nodes(x)
        assert np.array_equal(cx, canonical_nodes(x[::-1]))
        keys['fire_only'][sid] = fingerprint([t])
        keys['global_stats'][sid] = fingerprint([t, g])
        keys['node_models'][sid] = fingerprint([cx, g, t])
    records = []
    node_duplicates = []
    for name, ids in splits.items():
        assert len({cases[s]['geometry_id'] for s in ids}) == len(ids)
        for kind in keys:
            groups = groups_for(ids, keys[kind])
            duplicates = {key: members for key, members in groups.items() if len(members) > 1}
            records.append(dict(protocol=name, input_kind=kind, cases=len(ids), unique_input_groups=len(groups),
                                duplicate_groups=duplicates, largest_group=max(map(len, groups.values()))))
            if kind == 'node_models' and duplicates:
                node_duplicates.append(dict(protocol=name, groups=duplicates))
    result = dict(status='INPUT_GATE_PASSED_NO_EQUIVALENT_DISTINCT_NODE_INPUT_CASES' if not node_duplicates else 'STOP_NODE_INPUT_EQUIVALENCE_NEEDS_UNIFORM_POLICY',
                  created_utc=now(), prediction_arrays_read=0, protocols=12, unique_archival_cases=997,
                  node_models=plan['models'][2:], canonical_node_order='lexicographic numerical rows; signed zero normalized; multiplicity preserved',
                  node_equivalence_test='canonical node-feature multiset + exact global + time tensors. Absence of duplicates proves absence of identical complete attributed graph inputs; no claim about latent/model-output equivalence.',
                  complete_graphs_verified=997, reversed_node_order_invariance_verified=997,
                  source_tensor_sha256=manifest['tensor_sha256'], approved_protocol_sha256=JSON_SHA,
                  code_sha256=sha(__file__), groups=records, node_duplicates=node_duplicates, input_fingerprints=keys)
    write_json(HERE / 'input_equivalence_gate.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'prediction_arrays_read', 'complete_graphs_verified')}))


def weights(score, k):
    assert 0 < k <= len(score) and np.isfinite(score).all()
    tau = np.partition(score, k - 1)[k - 1]
    below, tie = score < tau, score == tau
    h, e = int(below.sum()), int(tie.sum())
    r = k - h
    w = below.astype(float) + tie * (r / e)
    assert np.isclose(w.sum(), k, atol=1e-12, rtol=0) and np.all((w >= 0) & (w <= 1))
    return w, dict(cutoff_score=float(tau), strictly_selected_count=h, cutoff_tied_count=e, selected_from_cutoff_tie=r)


def reference(y, k):
    oracle = float(np.sort(y)[:k].mean())
    random = float(y.mean())
    return dict(U_oracle=oracle, U_random=random, available_oracle_improvement=random - oracle,
                random_policy_SD=float(np.sqrt((1 - k / len(y)) * np.var(y, ddof=1) / k)))


def utility(y, w, k, ref):
    u = float(np.dot(w, y) / k)
    regret = u - ref['U_oracle']
    assert regret >= -1e-12, regret
    gain = ref['U_random'] - u
    available = ref['available_oracle_improvement']
    return dict(U=u, regret_raw=regret, regret_display=max(0., regret), improvement_over_random=gain,
                oracle_improvement_fraction=gain / available if available > 1e-12 else None)


def analytic_tests():
    # Exact boundary ties: one fixed case plus either of two cases equally likely.
    y = np.array([.1, .5, .9, 1.]); s = np.array([0., 1., 1., 2.])
    w, tie = weights(s, 2)
    assert np.array_equal(w, [1., .5, .5, 0.]) and tie['selected_from_cutoff_tie'] == 1
    result = utility(y, w, 2, reference(y, 2))
    assert np.isclose(result['U'], .4) and np.isclose(result['regret_raw'], .1)
    assert np.isclose(result['improvement_over_random'], .225)
    # Enumerate real selection-policy possibilities to verify the expected mean.
    alternatives = [np.mean(y[[0, j]]) for j in (1, 2)]
    assert np.isclose(result['U'], np.mean(alternatives))
    means = [np.mean(y[list(ix)]) for ix in itertools.combinations(range(4), 2)]
    ref = reference(y, 2)
    assert np.isclose(np.mean(means), ref['U_random'])
    assert np.isclose(np.std(means, ddof=0), ref['random_policy_SD'])
    cw, _ = weights(np.ones(4), 2)
    cr = utility(y, cw, 2, ref)
    assert cr['improvement_over_random'] == 0 and cr['oracle_improvement_fraction'] == 0
    for k in (1, 2, 4):
        const = np.ones(4); ww, _ = weights(s, k)
        ur = utility(const, ww, k, reference(const, k))
        assert ur['U'] == 1 and ur['regret_raw'] == ur['improvement_over_random'] == 0
        assert ur['oracle_improvement_fraction'] is None
    poor, _ = weights(-y, 2)
    pr = utility(y, poor, 2, ref)
    assert pr['improvement_over_random'] < 0 and pr['oracle_improvement_fraction'] < 0
    optimal, _ = weights(y, 2)
    assert utility(y, optimal, 2, ref)['regret_raw'] == 0
    return dict(status='ANALYTIC_IDENTITY_CHECKS_PASSED', cutoff_tie_expected_U=.4,
                cutoff_tie_oracle_regret=.1, cutoff_tie_random_improvement=.225,
                random_without_replacement_all_subsets_checked=6, constant_predictions_equal_random=True,
                constant_truth_zero_regret_gain_and_NA_fraction=True, negative_gain_retained=True)


def normalize(score, ids, kind, fingerprints):
    out = score.copy()
    index = {sid: j for j, sid in enumerate(ids)}
    for members in groups_for(ids, fingerprints[kind]).values():
        ix = [index[s] for s in members]
        out[ix] = float(np.mean(score[ix]))
    return out


def summary_rows(rows, fields, metrics):
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[tuple(r[k] for k in fields)].append(r)
    output = []
    for key, group in grouped.items():
        assert sorted(r['seed'] for r in group) == [42, 43, 44, 45, 46]
        row = dict(zip(fields, key))
        row['independent_seed_count'] = 5
        for metric in metrics:
            vals = [r[metric] for r in group if r[metric] is not None]
            row[metric + '_defined_seeds'] = len(vals)
            for stat, fn in [('mean', np.mean), ('sample_sd', lambda a: np.std(a, ddof=1)), ('min', np.min), ('max', np.max)]:
                row[metric + '_' + stat] = float(fn(vals)) if vals else None
        output.append(row)
    return output


def run():
    assert not (HERE / 'results').exists(), 'Preserve previous execution; no overwrite'
    protocol, plan, manifest, tensors, splits = load_fixed()
    gate = read(HERE / 'input_equivalence_gate.json')
    assert gate['status'] == 'INPUT_GATE_PASSED_NO_EQUIVALENT_DISTINCT_NODE_INPUT_CASES'
    assert gate['source_tensor_sha256'] == manifest['tensor_sha256'] and gate['code_sha256'] == sha(__file__)
    adoption = read(HERE / 'adoption_record.json')
    assert adoption['approved_protocol_md_sha256'] == MD_SHA and adoption['approved_protocol_json_sha256'] == JSON_SHA
    timing_path = BASE / 'inference_benchmark_v1/measured/summary.json'
    assert read(timing_path)['status'] == 'COMPLETE_INFERENCE_TIMING'
    final_review_path = REVIEW / 'final_matrix_420_independent_binding_review.json'
    final = read(final_review_path)
    assert final['unique_completed_runs'] == 420 and final['independent_primary_prediction_audits'] == 420
    for p, h in final['protocol_review_source_sha256'].items():
        assert sha(p) == h
    snap_path = BASE / 'completed_protocol_snapshots/final_matrix_420/snapshot_manifest.json'
    assert sha(snap_path) == protocol['final_matrix_manifest_sha256']
    integrity_path = BASE / 'evaluation/run_integrity.json'
    assert sha(integrity_path) == final['final_integrity_sha256']
    integrity = {(r['protocol'], r['model'], r['seed']): r for r in read(integrity_path)['audits']}
    assert len(integrity) == 420
    tests = analytic_tests()
    out = HERE / 'results'
    out.mkdir()
    cases = {c['sample_id']: c for c in manifest['cases']}
    refs, rows, raw_rows, score_rows, weight_rows, prediction_hashes = [], [], [], [], [], {}
    fingerprints = gate['input_fingerprints']
    start = now()
    for name, ids in splits.items():
        truth = np.stack([tensors['targets'][sid].numpy() for sid in ids]).astype(float)
        times = tensors['time_s'].numpy()
        endpoints = {'terminal': truth[:, -1], 'J': np.trapezoid(truth, times, axis=1) / 3600}
        n = len(ids)
        ref_lookup = {}
        for endpoint, y in endpoints.items():
            for b, numerator, denominator in BUDGETS:
                k = n * numerator // denominator
                ref = reference(y, k)
                ref_lookup[endpoint, b] = ref
                refs.append(dict(protocol=name, endpoint=endpoint, budget=b, N=n, k=k, actual_budget_fraction=k/n,
                                 constant_reference=bool(np.ptp(y) == 0), **ref))
        for mode in plan['models']:
            kind = mode if mode in ('fire_only', 'global_stats') else 'node_models'
            sizes = {sid: len(members) for members in groups_for(ids, fingerprints[kind]).values() for sid in members}
            for seed in plan['seeds']:
                folder = Path(plan['runs_root']) / name / mode / f'seed_{seed}'
                meta = read(folder / 'run_metadata.json')
                assert read(folder / 'status.json')['status'] == 'COMPLETE_ARCHIVAL_RESEARCH'
                assert (meta['protocol'], meta['mode'], meta['seed']) == (name, mode, seed)
                assert meta['plan_sha256'] == protocol['training_plan_sha256']
                bound = integrity[name, mode, seed]
                assert bound['status'] == 'PASSED_BINDINGS'
                ppath = folder / 'test_predictions.npz'
                phash = sha(ppath)
                assert phash == bound['predictions_sha256']
                assert meta['dataset_sha256'] == plan['manifest_sha256']
                assert meta['split_sha256'] == plan['protocols'][name]['split_sha256']
                assert meta['target_version'] == plan['target_version'] and meta['feature_version'] == plan['feature_version']
                checkpoint = folder / 'best.pt'
                assert sha(checkpoint) == bound['checkpoint_sha256']
                prediction_hashes[str(ppath)] = phash
                with np.load(ppath, allow_pickle=False) as z:
                    assert z['sample_ids'].tolist() == ids
                    assert np.array_equal(z['truth'], truth) and np.array_equal(z['time_s'], times)
                    p = z['predictions'].astype(float)
                assert p.shape == (n, 61) and np.isfinite(p).all()
                pred_endpoints = {'terminal': p[:, -1], 'J': np.trapezoid(p, times, axis=1) / 3600}
                for endpoint, raw_score in pred_endpoints.items():
                    y = endpoints[endpoint]
                    score = normalize(raw_score, ids, kind, fingerprints) if kind != 'node_models' else raw_score.copy()
                    prefix = dict(protocol=name, model=mode, seed=seed, endpoint=endpoint)
                    for j, sid in enumerate(ids):
                        score_rows.append(dict(**prefix, sample_id=sid, geometry_id=cases[sid]['geometry_id'],
                                               raw_prediction_score=float(raw_score[j]), ranking_score=float(score[j]),
                                               reference_score=float(y[j]), input_group=fingerprints[kind][sid],
                                               input_group_size=sizes[sid], normalization_change=float(score[j]-raw_score[j])))
                    for b, numerator, denominator in BUDGETS:
                        k = n * numerator // denominator
                        ref = ref_lookup[endpoint, b]
                        w, tie = weights(score, k)
                        values = utility(y, w, k, ref)
                        if len(set(score)) == 1:
                            assert abs(values['improvement_over_random']) <= 1e-12
                        rows.append(dict(**prefix, budget=b, N=n, k=k, actual_budget_fraction=k/n, **values, **tie))
                        for j, sid in enumerate(ids):
                            weight_rows.append(dict(**prefix, budget=b, sample_id=sid, inclusion_probability=float(w[j])))
                        if kind != 'node_models':
                            raw_w, raw_tie = weights(raw_score, k)
                            raw_rows.append(dict(**prefix, budget=b, N=n, k=k, diagnostic_only=True,
                                                 **utility(y, raw_w, k, ref), **raw_tie))
        print(json.dumps({'protocol_complete': name, 'N': n, 'models': 7, 'seeds': 5}), flush=True)
    summary = summary_rows(rows, ['protocol', 'model', 'endpoint', 'budget', 'N', 'k', 'actual_budget_fraction'], METRICS)
    lookup = {(r['protocol'], r['model'], r['seed'], r['endpoint'], r['budget']): r for r in rows}
    paired = []
    for name, ids in splits.items():
        for tag, baseline, candidate in [('graph_vs_capacity_matched', 'capacity_matched_deepsets', 'gnn'),
                                         ('mean_vs_sum', 'gnn', 'gnn_mean')]:
            for endpoint in ('terminal', 'J'):
                for b, numerator, denominator in BUDGETS:
                    for seed in plan['seeds']:
                        a, c = [lookup[name, m, seed, endpoint, b] for m in (baseline, candidate)]
                        diff = a['U'] - c['U']
                        assert np.isclose(diff, a['regret_raw'] - c['regret_raw'], atol=1e-12, rtol=0)
                        paired.append(dict(protocol=name, contrast=tag, baseline=baseline, candidate=candidate, endpoint=endpoint,
                                           budget=b, N=len(ids), k=len(ids)*numerator//denominator, seed=seed,
                                           U_baseline_minus_candidate=diff))
    paired_summary = summary_rows(paired, ['protocol', 'contrast', 'baseline', 'candidate', 'endpoint', 'budget', 'N', 'k'], ['U_baseline_minus_candidate'])
    expected = [(refs, 72), (rows, 2520), (summary, 504), (paired, 720), (paired_summary, 144), (raw_rows, 720)]
    for data, count in expected:
        assert len(data) == count
    artifacts = {'reference_budgets.csv': refs, 'model_seed_utility.csv': rows, 'model_five_seed_summary.csv': summary,
                 'paired_seed_contrasts.csv': paired, 'paired_five_seed_summary.csv': paired_summary,
                 'raw_uncanonicalized_baseline_diagnostics.csv': raw_rows, 'case_endpoint_scores.csv': score_rows,
                 'case_selection_probabilities.csv': weight_rows}
    for filename, data in artifacts.items():
        write_csv(out / filename, data)
    write_json(out / 'analytic_identity_checks.json', tests)
    report = dict(status='COMPLETE_FIXED_SEVEN_MODEL_POST_REVIEW_EXPLORATORY_BUDGET_ANALYSIS_PENDING_INDEPENDENT_ARITHMETIC_REVIEW',
                  started_utc=start, completed_utc=now(), protocols=12, models=7, seeds=5,
                  unique_original_geometries=997, overlapping_protocol_test_occurrences=1421,
                  rows={name: len(data) for name, data in artifacts.items()},
                  no_new_training_inference_FE=True, original_primary_analysis_unchanged=True,
                  budget_outputs_known_before_protocol=False, original420_outcomes_known_before_protocol=True,
                  approval_record_sha256=sha(HERE/'adoption_record.json'), approved_protocol_md_sha256=MD_SHA,
                  approved_protocol_json_sha256=JSON_SHA, code_sha256=sha(__file__),
                  input_gate_sha256=sha(HERE/'input_equivalence_gate.json'),
                  source_hashes={str(p): sha(p) for p in (timing_path, final_review_path, snap_path, integrity_path,
                                 Path(plan['manifest_path']), Path(plan['manifest_path']).parent/manifest['tensor_path'])},
                  prediction_sha256=prediction_hashes, all420_current_checkpoint_sha256_rechecked=True,
                  output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
                  no_cross_protocol_pooled_budget_utility=True, no_new_confirmatory_CI_or_significance_tests=True,
                  submission_pass=False)
    write_json(out / 'analysis_report.json', report)
    print(json.dumps({k: report[k] for k in ('status', 'rows')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-gate', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--analytic-tests', action='store_true')
    args = parser.parse_args()
    assert sum((args.input_gate, args.run, args.analytic_tests)) == 1
    if args.input_gate:
        input_gate()
    elif args.run:
        run()
    else:
        print(json.dumps(analytic_tests(), indent=2))
