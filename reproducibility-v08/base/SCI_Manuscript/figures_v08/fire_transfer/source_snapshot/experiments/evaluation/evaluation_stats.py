"""Pure NumPy evaluation primitives; no training or scientific-data fallback."""
import hashlib
import json
import numpy as np

MODELS = ('fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets', 'gnn', 'gnn_zero_edge_features')
FORMAL_SEEDS = (42, 43, 44, 45, 46)


def array_hash(value):
    a = np.ascontiguousarray(value)
    if a.dtype.hasobject:
        raise ValueError('Object arrays are forbidden.')
    header = json.dumps(dict(dtype=a.dtype.str, shape=a.shape), sort_keys=True).encode()
    return hashlib.sha256(header + b'\0' + a.tobytes()).hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def aligned_predictions(ids, times, truth, prediction, expected_ids, expected_times, expected_truth):
    ids = [str(x) for x in ids]
    if len(ids) != len(set(ids)) or ids != list(expected_ids):
        raise ValueError('Prediction sample IDs must exactly match the frozen ordered test IDs; no silent realignment.')
    t, y, p = np.asarray(times), np.asarray(truth), np.asarray(prediction)
    if not np.array_equal(t, expected_times) or array_hash(t) != array_hash(expected_times):
        raise ValueError('Prediction times differ from canonical times, including dtype.')
    if not np.array_equal(y, expected_truth) or array_hash(y) != array_hash(expected_truth):
        raise ValueError('Prediction truth differs from frozen labels after the documented training float32 cast.')
    if y.ndim != 2 or y.shape != p.shape or y.shape != (len(ids), len(t)):
        raise ValueError('Require one complete case-by-time prediction matrix.')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite prediction/truth/time.')
    if not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError('Time must be nonempty and strictly increasing.')
    return dict(ids_sha256=object_hash(ids), time_sha256=array_hash(t), truth_sha256=array_hash(y))


def complete_cells(cells, seeds=FORMAL_SEEDS):
    keys = [(r['model'], int(r['seed'])) for r in cells]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate model/seed run; selecting the best repeat is prohibited.')
    expected = {(model, seed) for model in MODELS for seed in seeds}
    extra = set(keys) - expected
    if extra:
        raise ValueError('Unexpected model/seed cells: ' + repr(sorted(extra)))
    missing = sorted(expected - set(keys))
    return dict(status='COMPLETE' if not missing else 'PENDING', expected_cells=len(expected),
                present_cells=len(keys), missing=[dict(model=m, seed=s) for m, s in missing])


def case_errors(truth, prediction):
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 2 or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Require finite, aligned case-by-time arrays.')
    with np.errstate(over='raise', invalid='raise'):
        e = p-y
        sq = e*e
        mae=np.abs(e).mean(1);mse=sq.mean(1)
    if not np.isfinite(e).all() or not np.isfinite(sq).all():
        raise ValueError('Error arithmetic overflow.')
    if not np.isfinite(mae).all() or not np.isfinite(mse).all():raise ValueError('Error reduction overflow.')
    return e,mae,mse


def metrics(truth, prediction):
    y = np.asarray(truth, dtype=np.float64)
    e, mae, mse = case_errors(y, prediction)
    with np.errstate(over='raise',invalid='raise'):
        sst = float(np.sum((y-y.mean())**2))
        sse = float(np.sum(e*e))
    if not np.isfinite(sst) or not np.isfinite(sse):raise ValueError('Metric reduction overflow.')
    r2=None if sst==0 else 1.-sse/sst
    if r2 is not None and not np.isfinite(r2):raise ValueError('R2 overflow.')
    positive = np.maximum(e, 0.)
    return dict(equal_case_MAE=float(mae.mean()), equal_case_RMSE=float(np.sqrt(mse.mean())),
                mean_case_RMSE=float(np.sqrt(mse).mean()),
                R2_pooled=r2,
                R2_denominator_SST=sst, squared_error_sum=sse,
                maximum_positive_error=float(positive.max()), maximum_absolute_error=float(abs(e).max()),
                positive_error_q95=float(np.quantile(positive, .95)), positive_error_q99=float(np.quantile(positive, .99)),
                case_count=len(y), time_count=y.shape[1])


def metric_group_tables(truth, predictions, case_records, seeds):
    """predictions: S x C x T. Geometry/fire/count groups precede outcomes."""
    p = np.asarray(predictions)
    if p.shape != (len(seeds), *np.asarray(truth).shape):
        raise ValueError('Seed/case/time dimensions do not align.')
    definitions = {'all': ['all'] * len(case_records),
                   'geometry_id': [r['geometry_id'] for r in case_records],
                   'fire_family': [r['fire_family'] for r in case_records],
                   'crack_count': [str(len(r['cracks'])) for r in case_records]}
    rows = []
    for stratum, assignments in definitions.items():
        for group in sorted(set(assignments)):
            mask = np.array([a == group for a in assignments])
            for si, seed in enumerate(seeds):
                row = metrics(np.asarray(truth)[mask], p[si, mask])
                rows.append(dict(stratum=stratum, group=group, seed=int(seed),
                    geometry_count=len({r['geometry_id'] for r, keep in zip(case_records, mask) if keep}), **row))
    summaries = []
    for stratum, assignments in definitions.items():
        for group in sorted(set(assignments)):
            items = [r for r in rows if r['stratum'] == stratum and r['group'] == group]
            values = {}
            for key in ['equal_case_MAE','equal_case_RMSE','mean_case_RMSE','R2_pooled','maximum_positive_error']:
                a = [r[key] for r in items]
                values[key] = dict(mean_over_seeds=None if any(v is None for v in a) else float(np.mean(a)),
                                  sample_sd_over_seeds=None if len(a)<2 or any(v is None for v in a) else float(np.std(a, ddof=1)))
            summaries.append(dict(stratum=stratum, group=group, seeds=list(seeds), seed_count=len(seeds),
                                  case_count=items[0]['case_count'], geometry_count=items[0]['geometry_count'], metrics=values))
    macro = {}
    for kind in ['fire_family','geometry_id']:
        # Each family/geometry receives equal weight here, unlike primary equal-case MAE.
        macro[kind] = [float(np.mean([r['equal_case_MAE'] for r in rows if r['stratum']==kind and r['seed']==seed])) for seed in seeds]
    return dict(per_seed_strata=rows, seed_summaries=summaries, equal_group_macro_MAE_by_seed=macro)


def weighted_cluster_effect(difference, groups, seed_counts, cluster_counts):
    """Ratio estimator retaining all cases in each selected cluster, including repeats."""
    d = np.asarray(difference, dtype=float)
    unique, index = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    if d.ndim != 2 or d.shape[1] != len(index):
        raise ValueError('Expected seed-by-case paired error differences.')
    totals = np.zeros((d.shape[0], len(unique)))
    for case, gi in enumerate(index):
        totals[:, gi] += d[:, case]
    sizes = np.bincount(index)
    denominator = np.sum(seed_counts) * np.dot(cluster_counts, sizes)
    if denominator <= 0:
        raise ValueError('Empty resampled set.')
    return float(np.asarray(seed_counts) @ totals @ np.asarray(cluster_counts) / denominator)


def paired_bootstrap(baseline_mae, graph_mae, geometry_ids, seeds, repeats=10000, random_seed=20260907):
    a, b = np.asarray(baseline_mae, dtype=float), np.asarray(graph_mae, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(seeds) or a.shape[1] != len(geometry_ids):
        raise ValueError('Paired seed/case layout mismatch.')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite paired errors.')
    if repeats < 100:
        raise ValueError('Use at least 100 resamples for software interval checks.')
    d=a-b; ns,nc=d.shape
    unique, inverse=np.unique(np.asarray(geometry_ids,dtype=str),return_inverse=True); ng=len(unique)
    totals=np.zeros((ns,ng))
    for j,g in enumerate(inverse): totals[:,g]+=d[:,j]
    sizes=np.bincount(inverse)
    rng=np.random.default_rng(random_seed)
    distributions={k:[] for k in ['geometry_only','seed_only','two_way']}
    for _ in range(repeats):
        sw=rng.multinomial(ns,np.full(ns,1/ns)); gw=rng.multinomial(ng,np.full(ng,1/ng))
        denominator=float(gw@sizes)
        distributions['geometry_only'].append(float(totals.sum(0)@gw/(ns*denominator)))
        distributions['seed_only'].append(float(sw@d.mean(1)/ns))
        distributions['two_way'].append(float(sw@totals@gw/(ns*denominator)))
    intervals={}
    for method,values in distributions.items():
        available=(ns>=2 if method=='seed_only' else ng>=2 if method=='geometry_only' else ns>=2 and ng>=2)
        intervals[method]=dict(status='ESTIMATED' if available else 'NOT_ESTIMABLE_SINGLE_SEED_OR_CLUSTER',
             percentile_95=None if not available else np.quantile(values,[.025,.975]).tolist(),
             bootstrap_sd=None if not available else float(np.std(values,ddof=1)))
    return dict(contrast='capacity_matched_deepsets MAE minus gnn MAE; positive favors GNN',
        point_estimate_equal_case_seed_mean=float(d.mean()), paired_effect_by_seed=d.mean(1).tolist(),
        paired_effect_seed_sample_sd=float(np.std(d.mean(1),ddof=1)) if ns>1 else None,
        seeds=list(seeds), seed_count=ns, independent_geometry_clusters=ng, test_cases=nc,
        cluster_sizes=dict(zip(unique.tolist(),sizes.tolist())), bootstrap_repeats=repeats,
        bootstrap_random_seed=random_seed, intervals=intervals,
        estimand='Mean across training seeds of equal-case time-mean absolute-error difference. Geometry resampling retains each selected cluster whole; unequal cluster sizes use sum of errors / sampled case count.',
        limitations='Five training seeds give limited information about the seed distribution. Crossed product-resampling is an approximate diagnostic, not an exact finite-sample confidence procedure or a model-uncertainty interval. No time points are resampled.',
        G04_or_G05_pass=False)
