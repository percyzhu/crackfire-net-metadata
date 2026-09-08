"""Complete-protocol paired CIs for the prespecified archival proxies.

Requires all 35 model/seed runs before reading prediction arrays. Intervals use
common geometry and seed resamples, never differences of interval endpoints.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,json,time
import numpy as np
import engineering_proxy_metrics as ep

REPETITIONS=1000
RANDOM_SEED=20260907
CONTRASTS=(('graph_vs_capacity_matched','capacity_matched_deepsets','gnn'),
           ('mean_vs_sum','gnn','gnn_mean'))


def preference(metric):
    if metric.endswith('bias') or metric.endswith('bias_s'):
        return 'Neither sign of a signed-bias change is automatically better'
    if any(metric.endswith(s) for s in ('spearman','pairwise_order_agreement','recall','balanced_accuracy')):
        return 'Positive baseline-minus-candidate effect favors baseline for this higher-is-better metric'
    return 'Positive baseline-minus-candidate effect favors candidate for this lower-is-better metric'


def paired_intervals(truth,predictions,times,families,seeds,repetitions=REPETITIONS,random_seed=RANDOM_SEED):
    required={m for _,a,b in CONTRASTS for m in (a,b)}
    if set(predictions)!=required:
        raise ValueError('Expected precisely matched-set, sum-graph and mean-graph predictions')
    summaries={};replicates={}
    for model in sorted(required):
        if np.asarray(predictions[model]).shape[0]!=len(seeds):raise ValueError('Missing seed')
        summaries[model],replicates[model]=ep.comparative_summary(
            truth,predictions[model],times,families,repetitions=repetitions,random_seed=random_seed)
    contrasts={}
    for tag,baseline,candidate in CONTRASTS:
        metrics={}
        for key in summaries[baseline]:
            a,b=summaries[baseline][key],summaries[candidate][key]
            ba,bb=replicates[baseline][key],replicates[candidate][key]
            valid=np.isfinite(ba)&np.isfinite(bb)
            difference=ba[valid]-bb[valid]
            supported=a['defined_seeds']==b['defined_seeds']==len(seeds)
            point=a['mean_over_defined_seeds']-b['mean_over_defined_seeds'] if supported else None
            metrics[key]={
                'point_baseline_minus_candidate':point,
                'percentile_95_paired_difference':np.quantile(difference,[.025,.975]).tolist() if supported and len(difference) else None,
                'baseline_defined_seeds':a['defined_seeds'],'candidate_defined_seeds':b['defined_seeds'],
                'required_seed_count':len(seeds),'point_uses_all_paired_seeds':supported,
                'common_valid_bootstrap_replicates':int(valid.sum()),'bootstrap_repetitions':repetitions,
                'valid_bootstrap_fraction':float(valid.mean()),
                'status':'ESTIMATED' if supported and len(difference) else 'NOT_ESTIMABLE_ON_ALL_PAIRED_SEEDS',
                'effect_direction':preference(key),
                'NA_policy':'Undefined class/ranking denominators are not zeros; require all five point-estimate seeds and common defined bootstrap replicates.'}
        contrasts[tag]={'baseline':baseline,'candidate':candidate,'difference_definition':'baseline metric minus candidate metric',
                        'metrics':metrics}
    return summaries,contrasts


def run(plan_path,protocol,output):
    plan,runs=ep.complete_protocol(plan_path,protocol)
    if runs is None:return plan
    output=Path(output).resolve()
    if not output.is_relative_to(ep.HERE):raise ValueError('Output must stay in the v08 reviewer directory')
    csv_path=output.with_suffix('.csv')
    if output.exists() or csv_path.exists():raise ValueError('JSON/CSV outputs are exclusive-create')
    start=time.perf_counter()
    plan_path=Path(plan_path);manifest_path=Path(plan['manifest_path'])
    if ep.sha(manifest_path)!=plan['manifest_sha256']:raise ValueError('Manifest SHA mismatch')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    split_path=Path(plan['protocols'][protocol]['split_path'])
    if ep.sha(split_path)!=plan['protocols'][protocol]['split_sha256']:raise ValueError('Split SHA mismatch')
    split=json.loads(split_path.read_text(encoding='utf-8'));ids=split['test']
    byid={c['sample_id']:c for c in manifest['cases']}
    geometry=[byid[s]['geometry_id'] for s in ids]
    if len(set(geometry))!=len(ids):raise ValueError('This fixed997 wrapper expects one independent geometry per case')
    families=np.asarray([byid[s]['fire_family'] for s in ids])
    # Small frozen learning tensors only; no temperature fields or FE meshes.
    import torch
    torch.set_num_threads(1)
    tensor_path=manifest_path.parent/manifest['tensor_path']
    if ep.sha(tensor_path)!=manifest['tensor_sha256']:raise ValueError('Frozen tensor SHA mismatch')
    data=torch.load(tensor_path,map_location='cpu',weights_only=True)
    expected=np.stack([data['targets'][s].numpy() for s in ids]);times=data['time_s'].numpy()
    if not np.array_equal(times,np.arange(61)*60):raise ValueError('Unexpected target grid')
    records=[];predictions={m:[] for m in ('capacity_matched_deepsets','gnn','gnn_mean')};hashes={}
    for model,seed,folder in runs:
        meta=json.loads((folder/'run_metadata.json').read_text(encoding='utf-8'))
        if meta['plan_sha256']!=ep.sha(plan_path) or meta['dataset_sha256']!=plan['manifest_sha256']:
            raise ValueError('Run plan/dataset binding mismatch')
        if meta['mode']!=model or meta['seed']!=seed or meta['protocol']!=protocol:
            raise ValueError('Model/seed/protocol identity mismatch')
        path=folder/'test_predictions.npz'
        with np.load(path,allow_pickle=False) as z:
            if z['sample_ids'].tolist()!=ids or not np.array_equal(z['time_s'],times) or not np.array_equal(z['truth'],expected):
                raise ValueError('IDs/time/reference are not the frozen paired arrays')
            pred=z['predictions'].copy()
        ep.checked_inputs(expected,pred,times)
        hashes[str(path)]=ep.sha(path)
        record=ep.evaluate_population(expected,pred,times,families)
        record.update(model=model,seed=seed)
        records.append(record)
        if model in predictions:predictions[model].append(pred)
    predictions={m:np.stack(v) for m,v in predictions.items()}
    summaries,contrasts=paired_intervals(expected,predictions,times,families,plan['seeds'])
    result={'status':'COMPLETE_35_RUN_PAIRED_ENGINEERING_PROXY_CI','scope':'Archival numerical-response proxies; no physical capacity or fire-resistance validation',
        'created_utc':datetime.now(timezone.utc).isoformat(),'protocol':protocol,'seeds':plan['seeds'],
        'completed_runs_checked':len(runs),'test_case_count':len(ids),'unique_geometry_groups':len(set(geometry)),
        'thresholds':list(ep.THRESHOLDS),'secondary_bootstrap_repetitions':REPETITIONS,
        'primary_MAE_bootstrap_repetitions_separately_implemented':5000,'bootstrap_random_seed':RANDOM_SEED,
        'bootstrap_design':'Same independently resampled geometry indices and seed indices for every contrasted model; geometry/time histories remain intact.',
        'subtraction':'Subtract the two model metrics within each common bootstrap replicate; never subtract interval endpoints.',
        'model_seed_summaries':summaries,'paired_contrasts':contrasts,'per_seed_event_counts_and_metrics':records,
        'identities':{'sample_ids':ids,'geometry_ids':geometry,'families':families.tolist()},
        'source_prediction_sha256':hashes,'plan_sha256':ep.sha(plan_path),'manifest_sha256':ep.sha(manifest_path),
        'code_sha256':{str(p):ep.sha(p) for p in [Path(__file__),Path(ep.__file__)]},
        'prespecification_sha256':ep.sha(ep.HERE/'engineering_proxy_evaluation_prespecification.md'),
        'elapsed_s':time.perf_counter()-start,'journal_or_submission_pass':False,
        'limitations':['Secondary measures were specified after training began but before the specifying reviewers inspected performance.',
          'Five seeds provide limited initialization-distribution information; these crossed-bootstrap intervals are approximate.',
          'Conditional timing intervals exclude nonjoint events but their counts and coverage are reported for every seed.',
          'Different models can have different joint-event subsets; conditional-timing contrasts are differences of those declared conditional metrics, not a claim of improvement on one common crossing cohort.',
          'Undefined ranking/class denominators remainNA; common-valid bootstrap counts accompany intervals.',
          'Mixed-fire ranking is descriptive; within-family results still need interpretation of varying fire parameters.',
          'No full temperature fields are read or physically qualified by this evaluator.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    rows=[]
    for tag,c in contrasts.items():
        for metric,v in c['metrics'].items():
            interval=v['percentile_95_paired_difference']
            rows.append(dict(protocol=protocol,contrast=tag,metric=metric,
              baseline=c['baseline'],candidate=c['candidate'],effect=v['point_baseline_minus_candidate'],
              ci95_low=interval[0] if interval else None,ci95_high=interval[1] if interval else None,
              status=v['status'],paired_seed_count=len(plan['seeds']),
              valid_bootstrap=v['common_valid_bootstrap_replicates'],bootstrap_repetitions=REPETITIONS,
              effect_direction=v['effect_direction']))
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return {'status':result['status'],'runs':len(runs),'protocol':protocol,'contrasts':len(contrasts),
            'metric_rows':len(rows),'elapsed_s':result['elapsed_s'],'output':str(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    print(json.dumps(run(args.plan,args.protocol,args.output),indent=2))
