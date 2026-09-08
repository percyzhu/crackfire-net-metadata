"""Secondary archival-proxy metrics; never physical capacity/fire resistance.

No training-result file is read until the complete 7-model x 5-seed protocol
passes the completion gate. Importing this module reads no experiment files.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
THRESHOLDS = (.8, .6)
VERSION = 'archival-engineering-proxy-metrics-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_inputs(truth, prediction, times):
    a, p, t = np.asarray(truth, float), np.asarray(prediction, float), np.asarray(times, float)
    if a.ndim != 2 or a.shape != p.shape or t.ndim != 1 or a.shape[1] != len(t):
        raise ValueError('Expected matching case-by-time arrays and one common time vector')
    if not len(a) or len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0):
        raise ValueError('Empty cases or invalid strictly increasing time vector')
    if not np.isfinite(a).all() or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Nonfinite response; no clipping or imputation is permitted')
    return a, p, t


def average_rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind='stable')
    ranked = np.empty(len(x), float)
    boundaries = np.r_[0, 1+np.flatnonzero(np.diff(x[order]) != 0), len(x)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        ranked[order[start:end]] = (start+1+end)/2
    return ranked


def rank_metrics(reference, prediction):
    a, p = np.asarray(reference, float), np.asarray(prediction, float)
    if a.ndim != 1 or p.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError('Invalid ranking inputs')
    total = len(a)*(len(a)-1)//2
    ra, rp = average_rank(a), average_rank(p)
    ra, rp = ra-ra.mean(), rp-rp.mean()
    denominator = float(np.linalg.norm(ra)*np.linalg.norm(rp))
    rho = None if denominator == 0 else float(np.dot(ra, rp)/denominator)
    # Pair comparisons are independent of array order. Reference ties excluded;
    # predictor ties get exactly half credit. No randomized tie breaking.
    ix, jx = np.triu_indices(len(a), 1)
    da, dp = a[jx]-a[ix], p[jx]-p[ix]
    eligible = da != 0
    score = ((da[eligible]*dp[eligible] > 0).astype(float) + .5*(dp[eligible] == 0))
    return {'spearman': rho, 'pairwise_order_agreement': float(score.mean()) if len(score) else None,
            'eligible_pairs': int(eligible.sum()), 'excluded_reference_tied_pairs': int((~eligible).sum()),
            'excluded_reference_tie_fraction': float((~eligible).sum()/total) if total else None}


def event_arrays(values, times, threshold):
    crossing = np.asarray(values) <= threshold
    event = crossing.any(axis=1)
    first_index = crossing.argmax(axis=1)
    event_time = np.where(event, np.asarray(times)[first_index], np.nan)
    positions = np.arange(crossing.shape[1])[None, :]
    recross = event & (((positions > first_index[:, None]) & ~crossing).any(axis=1))
    return event, event_time, recross


def crossing_metrics(truth, prediction, times, threshold):
    ae, at, _ = event_arrays(truth, times, threshold)
    pe, pt, recross = event_arrays(prediction, times, threshold)
    tp, fn, fp, tn = [int(v.sum()) for v in (ae&pe, ae&~pe, ~ae&pe, ~ae&~pe)]
    both = ae & pe
    delay = pt[both]-at[both]
    recall = tp/(tp+fn) if tp+fn else None
    fpr = fp/(fp+tn) if fp+tn else None
    return {'threshold': threshold, 'reference_events': int(ae.sum()), 'reference_censored': int((~ae).sum()),
            'predicted_events': int(pe.sum()), 'predicted_censored': int((~pe).sum()),
            'true_positive': tp, 'false_negative': fn, 'false_positive': fp, 'true_negative': tn,
            'recall': recall, 'false_positive_rate': fpr,
            'balanced_accuracy': .5*(recall+1-fpr) if recall is not None and fpr is not None else None,
            'both_event_count': int(both.sum()), 'both_event_population_fraction': float(both.mean()),
            'both_event_reference_event_fraction': tp/(tp+fn) if tp+fn else None,
            'conditional_timing_MAE_s': float(abs(delay).mean()) if len(delay) else None,
            'conditional_timing_median_abs_s': float(np.median(abs(delay))) if len(delay) else None,
            'conditional_timing_bias_s': float(delay.mean()) if len(delay) else None,
            'optimistic_late_or_missed_count': int(fn + np.sum(delay > 0)),
            'recross_above_threshold_count': int(recross.sum()),
            'recross_fraction_all_cases': float(recross.mean()),
            'recross_fraction_predicted_event_cases': float(recross.sum()/pe.sum()) if pe.any() else None}


def evaluate_population(truth, prediction, times, families=None):
    a, p, t = checked_inputs(truth, prediction, times)
    integral_weights = np.diff(t)/(2*(t[-1]-t[0]))
    ja = np.sum((a[:, :-1]+a[:, 1:])*integral_weights, axis=1)
    jp = np.sum((p[:, :-1]+p[:, 1:])*integral_weights, axis=1)
    errors = {'final': p[:, -1]-a[:, -1], 'time_average': jp-ja}
    out = {'cases': len(a), 'grid_points': len(t), 'horizon_s': float(t[-1]),
           'summaries': {name: {'MAE': float(abs(e).mean()), 'bias': float(e.mean())} for name, e in errors.items()},
           'crossings': {str(q): crossing_metrics(a, p, t, q) for q in THRESHOLDS},
           'rankings': {'final': rank_metrics(a[:, -1], p[:, -1]), 'time_average': rank_metrics(ja, jp)},
           'positive_step_fraction': float((np.diff(p, axis=1)>1e-6).mean())}
    if families is not None:
        f = np.asarray(families)
        if f.shape != (len(a),): raise ValueError('Family identities must match cases')
        per_family = {}
        for name in sorted(set(f)):
            ix = f == name
            per_family[str(name)] = {'cases': int(ix.sum()), 'final': rank_metrics(a[ix, -1], p[ix, -1]),
                                     'time_average': rank_metrics(ja[ix], jp[ix])}
        out['within_family_rankings'] = per_family
        aggregate = {}
        for target in ('final','time_average'):
            aggregate[target] = {}
            for metric in ('spearman','pairwise_order_agreement'):
                eligible = [r for r in per_family.values() if r[target]['eligible_pairs'] > 0]
                values = [r[target][metric] for r in eligible if r[target][metric] is not None]
                full = bool(eligible) and len(values) == len(eligible)
                aggregate[target][metric] = {
                    'mean_over_defined_families': float(np.mean(values)) if full else None,
                    'defined_families': len(values), 'total_families': len(per_family),
                    'reference_eligible_families':len(eligible),
                    'full_reference_family_support':full,
                    'missing_model_scores_not_silently_averaged':True}
        out['equal_family_ranking_summaries'] = aggregate
    return out


def comparative_summary(truth, predictions_by_seed, times, families, repetitions=1000, random_seed=20260907):
    """Seed mean and paired hierarchical geometry/seed bootstrap for one model.

    Root can difference bootstrap replicate arrays from models generated using
    the same IDs/order/random seed. Each replicate uses the same sampled cases
    in every sampled seed, preserving the design's crossed dependence.
    """
    preds = np.asarray(predictions_by_seed, float)
    if preds.ndim != 3 or preds.shape[1:] != np.asarray(truth).shape:
        raise ValueError('Expected seed-by-case-by-time predictions')
    family_array = np.asarray(families)
    if family_array.shape != (len(truth),):
        raise ValueError('Family identities must match the paired cases')
    def values(a,p,t,f):
        v = evaluate_population(a,p,t,f)
        result = {f'{k}_{m}':v['summaries'][k][m] for k in ('final','time_average') for m in ('MAE','bias')}
        for q,c in v['crossings'].items():
            for k in ('recall','false_positive_rate','balanced_accuracy','conditional_timing_MAE_s',
                      'conditional_timing_median_abs_s','conditional_timing_bias_s'):
                result[f'q{q}_{k}'] = c[k]
        for target,r in v['rankings'].items():
            for metric in ('spearman','pairwise_order_agreement'):
                result[f'{target}_{metric}'] = r[metric]
                result[f'{target}_equal_family_{metric}'] = v['equal_family_ranking_summaries'][target][metric]['mean_over_defined_families']
        return result
    seed_values = [values(truth,p,times,family_array) for p in preds]
    keys = list(seed_values[0])
    rng = np.random.default_rng(random_seed)
    boot = {key: np.full(repetitions,np.nan) for key in keys}
    for b in range(repetitions):
        ix = rng.integers(0,len(truth),len(truth))
        seeds = rng.integers(0,len(preds),len(preds))
        vv = [values(np.asarray(truth)[ix],preds[s,ix],times,family_array[ix]) for s in seeds]
        for key in keys:
            valid = [v[key] for v in vv if v[key] is not None]
            # Missing denominators are not silently treated as zeros. A full
            # seed-mean replicate is retained only if all drawn seeds define it.
            if len(valid)==len(seeds): boot[key][b]=float(np.mean(valid))
    result = {}
    for key in keys:
        vv = [v[key] for v in seed_values if v[key] is not None]
        valid = boot[key][np.isfinite(boot[key])]
        result[key] = {'mean_over_defined_seeds':float(np.mean(vv)) if vv else None,
                      'seed_sd':float(np.std(vv,ddof=1)) if len(vv)>1 else None,
                      'defined_seeds':len(vv), 'bootstrap_valid_replicates':len(valid),
                      'bootstrap_repetitions':repetitions,
                      'paired_hierarchical_95_interval':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None}
    return result, boot


def complete_protocol(plan_path, protocol):
    """Check every required status before reading any predictions or scores."""
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('status')!='FROZEN_BEFORE_TRAINING' or plan.get('purpose')!='ARCHIVAL_SURROGATE_RESEARCH_V08':
        raise ValueError('Expected a frozen archival-study plan')
    if len(plan['models'])!=7 or len(set(plan['models']))!=7 or plan['seeds']!=[42,43,44,45,46]:
        raise ValueError('Comparative tables require all seven models and five declared seeds')
    if protocol not in plan['protocols']: raise ValueError('Protocol not in frozen plan')
    required=[];pending=[]
    for model in plan['models']:
        for seed in plan['seeds']:
            p=Path(plan['runs_root'])/protocol/model/('seed_'+str(seed))
            status=p/'status.json'
            if not status.exists() or json.loads(status.read_text(encoding='utf-8')).get('status')!='COMPLETE_ARCHIVAL_RESEARCH':
                pending.append(str(p))
            elif not (p/'test_predictions.npz').exists(): pending.append(str(p))
            required.append((model,seed,p))
    if pending:
        return {'status':'WAITING_FOR_COMPLETE_35_RUN_PROTOCOL','protocol':protocol,'pending_runs':len(pending),
                'predictions_or_scores_read':False},None
    return plan,required


def evaluate_complete_protocol(plan_path,protocol,output):
    plan,runs=complete_protocol(plan_path,protocol)
    if runs is None:return plan
    output=Path(output).resolve()
    if not output.is_relative_to(HERE):raise ValueError('Reviewer outputs must remain under this review directory')
    if output.exists():raise ValueError('Evaluation output is exclusive-create')
    split_path=Path(plan['protocols'][protocol]['split_path'])
    if sha(split_path)!=plan['protocols'][protocol]['split_sha256']:raise ValueError('Split SHA mismatch')
    split=json.loads(split_path.read_text(encoding='utf-8'));ids=split['test']
    manifest_path=Path(plan['manifest_path'])
    if sha(manifest_path)!=plan['manifest_sha256']:raise ValueError('Manifest SHA mismatch')
    cases={c['sample_id']:c for c in json.loads(manifest_path.read_text(encoding='utf-8'))['cases']}
    family=[cases[i]['fire_family'] for i in ids]
    records=[];reference=None;times=None
    for model,seed,folder in runs:
        metadata=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        if metadata['plan_sha256']!=sha(plan_path):raise ValueError('Run binds another plan')
        with np.load(folder/'test_predictions.npz',allow_pickle=False) as f:
            if f['sample_ids'].tolist()!=ids:raise ValueError('Test order/identity mismatch')
            a,p,t=f['truth'],f['predictions'],f['time_s']
        if not np.array_equal(t,np.arange(61)*60):raise ValueError('Expected frozen 61-point evaluation grid')
        if reference is None:reference=a.copy();times=t.copy()
        if not np.array_equal(reference,a) or not np.array_equal(times,t):raise ValueError('Unequal references across runs')
        v=evaluate_population(a,p,t,family)
        v.update(model=model,seed=seed,source_prediction_sha256=sha(folder/'test_predictions.npz'))
        v['strata']={}
        strata={'source_batch':sorted({cases[i]['source_batch'] for i in ids}),
                'fire_family':sorted(set(family)),
                'num_cracks':sorted({cases[i]['num_cracks'] for i in ids}),
                'original_subset':['retained911','restored86']}
        for key,levels in strata.items():
            for level in levels:
                ix=np.asarray([(cases[i][key]==level) if key!='original_subset' else
                    (bool(cases[i]['legacy_final_id'])==(level=='retained911')) for i in ids])
                if ix.any():v['strata'][key+':'+str(level)]=evaluate_population(a[ix],p[ix],t)
        records.append(v)
    result={'status':'COMPLETE_PROTOCOL_SECONDARY_PROXY_EVALUATION','version':VERSION,
            'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':protocol,'runs':len(runs),
            'scope':'Fixed archival-response proxies, not physical capacity or fire resistance',
            'plan_sha256':sha(plan_path),'manifest_sha256':sha(manifest_path),
            'evaluation_source_sha256':sha(Path(__file__)),'thresholds':list(THRESHOLDS),
            'raw_predictions_not_clipped_or_monotonized':True,'records':records,
            'uncertainty_note':'Per-seed metrics here. comparative_summary supplies paired-bootstrap replicates; do not infer a confidence interval from means alone.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return {'status':result['status'],'runs':len(runs),'path':str(output)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(evaluate_complete_protocol(args.plan,args.protocol,args.output),indent=2))
