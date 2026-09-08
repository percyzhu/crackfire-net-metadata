"""Independent evaluator for the post-results set-attention extension.

Reads no prediction/result performance payload until all60 registered runs are
complete. Original120 comparator artifacts are reused only by frozen hashes and
the original completed independent audit; the original420 evidence is untouched.
"""
from pathlib import Path
import argparse
import csv
import datetime
import json
import math
import sys
import time
import numpy as np
import torch
import run_extension as run
from attention_model import SetAttentionSurrogate, MODEL_NAME, MODEL_VERSION, ARCHITECTURE, parameter_count

sys.path.insert(0,str(run.V08))
import aggregate_results as ag
sys.path.insert(0,str(run.rr.SCI/'review/eaai_editor/research_v08'))
import engineering_proxy_metrics as ep
import engineering_proxy_ci as eci

MODELS=('capacity_matched_deepsets','gnn',MODEL_NAME)
CONTRASTS=(('attention_vs_matched_set','capacity_matched_deepsets',MODEL_NAME),
           ('attention_vs_sum_GNN','gnn',MODEL_NAME))


def require(value,message):
    if not value:raise ValueError(message)


def exclusive_json(path,value):
    path=run.safe_output(path)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def readiness(plan_path):
    """Identity/status/existence only. No result JSON values or NPZ are read."""
    plan=run.verify_plan(plan_path);root=Path(plan['runs_root'])
    expected={(protocol,MODEL_NAME,seed) for protocol in plan['protocols'] for seed in plan['seeds']}
    allowed={str((root/p/m/('seed_'+str(s))/'result.json').resolve()) for p,m,s in expected}
    unexpected=[str(p) for p in root.glob('*/*/*/result.json') if str(p.resolve()) not in allowed]
    require(not unexpected,'Unregistered result paths: '+repr(unexpected))
    present=[];missing=[]
    for protocol,mode,seed in sorted(expected):
        folder=root/protocol/mode/('seed_'+str(seed))
        files=('result.json','run_metadata.json','status.json','best.pt','test_predictions.npz','learning_curve.csv')
        if not all((folder/f).is_file() for f in files):missing.append((protocol,mode,seed));continue
        status=run.read(folder/'status.json');meta=run.read(folder/'run_metadata.json')
        require((meta['protocol'],meta['mode'],meta['seed'])==(protocol,mode,seed),'Metadata identity mismatch')
        require(meta['plan_sha256']==run.rr.digest(plan_path),'Metadata plan hash mismatch')
        if status['status']!='COMPLETE_EXPLORATORY_EXTENSION':missing.append((protocol,mode,seed));continue
        present.append((protocol,mode,seed))
    queue_path=root/'queue_status.json';queue=run.read(queue_path) if queue_path.is_file() else None
    if queue is not None and queue['status']=='ERROR_STOPPED_EXTENSION':
        raise ValueError('Extension queue failed: '+str(queue.get('error')))
    complete=not missing and queue is not None and queue['status']=='COMPLETE_60_EXPLORATORY_EXTENSION'
    if complete:
        keys=[(v['protocol'],v['model'],v['seed']) for v in queue['completed']]
        require(len(keys)==len(set(keys))==60 and set(keys)==expected,'Queue identity matrix differs')
        require(queue['active'] is None and queue['planned_runs']==60,'Complete queue has an active or wrong matrix')
        require(queue['plan_sha256']==run.rr.digest(plan_path),'Queue plan hash differs')
    return plan,{'status':'READY_ALL60_METADATA_COMPLETE' if complete else 'WAITING_FOR_ALL60_COMPLETE',
        'expected_runs':60,'completed_metadata_runs':len(present),'missing_cells':missing,
        'result_performance_payloads_read':0,'prediction_arrays_read':0,
        'queue_status':queue['status'] if queue else None,'plan_sha256':run.rr.digest(plan_path)}


def load_prediction(path,ids,time_s,truth):
    with np.load(path,allow_pickle=False) as z:
        binding=ag.es.aligned_predictions(z['sample_ids'],z['time_s'],z['truth'],z['predictions'],ids,time_s,truth)
        prediction=z['predictions'].copy()
    require(prediction.dtype==np.float32,'Saved prediction dtype must be float32')
    return prediction,binding


def replay_valid(record):
    require(record is not None,'Original comparator lacks a computational replay')
    error=record['cpu_max_abs_difference']
    if error<3e-6:
        require(record['cpu_within_original_tolerance'] is True and record['status']=='CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE','Inconsistent original CPU replay')
    else:
        require(record['cpu_within_original_tolerance'] is False and record['same_backend_gpu_bitwise_equal'] is True,
                'Original flagged comparator lacks exact GPU identity')
        require(record['gpu_max_abs_difference']==0 and record['status']=='GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED',
                'Original GPU identity differs')


def audit_new(folder,plan_path,plan,manifest,data,protocol,seed,split,authorization_path):
    meta=run.read(folder/'run_metadata.json');result=run.read(folder/'result.json');status=run.read(folder/'status.json')
    for key,value in [('purpose',run.PURPOSE),('epistemic_status',plan['epistemic_status']),('mode',MODEL_NAME),('protocol',protocol),('seed',seed)]:
        require(meta[key]==result[key]==value,'New run identity/purpose differs: '+key)
    require(meta['model_version']==MODEL_VERSION and meta['architecture']==ARCHITECTURE,'New model recipe differs')
    for key,value in [('plan_sha256',run.rr.digest(plan_path)),('original_plan_sha256',run.ORIGINAL_PLAN_SHA),
                      ('dataset_sha256',plan['manifest_sha256']),('split_sha256',plan['protocols'][protocol]['split_sha256']),
                      ('target_version',run.rr.TARGET_VERSION),('feature_version',run.rr.FEATURE_VERSION),
                      ('feature_tensor_sha256',manifest['feature_tensor_sha256']),('target_tensor_sha256',manifest['target_tensor_sha256'])]:
        require(meta[key]==value,'New run provenance differs: '+key)
    require(meta['original_source_sha256']==plan['original_source_sha256'],'Original source chain differs')
    require(meta['extension_source_sha256']==plan['extension_source_sha256'],'Extension source chain differs')
    require(meta['configuration']==plan['training_configuration'] and meta['backend_configuration']==plan['backend_configuration'],'Budget/backend differs')
    require(meta['execution_authorization_sha256']==run.rr.digest(authorization_path),'Execution authorization differs')
    require(result['topology_results']=={},'Attention edge-independent control has unexpected topology experiments')
    checkpoint_path=folder/'best.pt';checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    require(run.rr.digest(checkpoint_path)==result['checkpoint_sha256'],'New checkpoint hash differs')
    for key in ('plan_sha256','dataset_sha256','split_sha256'):require(checkpoint[key]==meta[key],'Checkpoint provenance differs: '+key)
    model=SetAttentionSurrogate();model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    require(parameter_count(model)==meta['parameters']==plan['parameters']==194273,'Parameter count differs')
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'Nonfinite checkpoint')
    with (folder/'learning_curve.csv').open(encoding='utf-8') as f:log=list(csv.DictReader(f))
    require(bool(log),'Missing epochs')
    require([int(v['epoch']) for v in log]==list(range(1,len(log)+1)),'Nonsequential epochs')
    values=np.asarray([[float(row[k]) for k in ('train_mse','validation_mse','learning_rate','elapsed_s')] for row in log])
    require(np.isfinite(values).all() and np.all(values[:,:2]>=0) and np.all(np.diff(values[:,3])>=0),'Invalid training log')
    best=int(np.argmin(values[:,1]))+1
    require(best==result['best_epoch']==checkpoint['epoch']==status['best_epoch'],'Checkpoint is not earliest validation minimum')
    require(float(values[best-1,1])==result['best_validation_mse']==checkpoint['best_validation_mse'],'Best validation value differs')
    config=plan['training_configuration'];epochs=len(log)
    require(epochs==result['epochs_run']==status['epochs_run']<=config['epochs_max'],'Epoch budget differs')
    require(epochs==config['epochs_max'] or epochs-best==config['patience'],'Stopping differs from frozen patience')
    expected_lr=np.asarray([config['learning_rate']*(1+math.cos(math.pi*(i-1)/config['epochs_max']))/2 for i in range(1,epochs+1)])
    require(np.allclose(values[:,2],expected_lr,rtol=1e-12,atol=1e-15),'Cosine scheduler differs')
    require(math.isfinite(result['training_wall_s']) and result['training_wall_s']>=values[-1,3] and
            result['training_wall_s']==status['training_wall_s'],'Training wall time differs')
    truth=np.stack([data['targets'][sid].numpy() for sid in split['test']])
    prediction,binding=load_prediction(folder/'test_predictions.npz',split['test'],data['time_s'].numpy(),truth)
    metrics=ag.scalar_metrics(truth,prediction)
    for computed,saved in [('equal_case_MAE','equal_case_MAE'),('equal_case_RMSE','RMSE'),('R2_pooled','pooled_R2'),
                           ('maximum_positive_error','maximum_overprediction'),('maximum_absolute_error','maximum_absolute_error')]:
        a,b=metrics[computed],result[saved]
        require((a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-13,abs_tol=1e-13)),
                'Independent metric differs: '+computed)
    replay=ag.computational_replay(model,data['graphs'],split['test'],prediction)
    replay_valid(replay)
    audit={'status':'PASSED_EXTENSION_BINDINGS_AND_REPLAY','protocol':protocol,'model':MODEL_NAME,'seed':seed,
        'parameters':194273,'best_epoch':best,'epoch_count':epochs,'training_wall_s':result['training_wall_s'],
        'earliest_validation_minimum':True,'plan_sha256':run.rr.digest(plan_path),'computational_replay':replay,**binding,
        'source_artifact_sha256':{name:run.rr.digest(folder/name) for name in
            ('run_metadata.json','result.json','status.json','best.pt','test_predictions.npz','learning_curve.csv')}}
    return {'audit':audit,'metrics':metrics,'truth':truth,'prediction':prediction,'training_wall_s':result['training_wall_s']}


def bind_original(plan):
    report=run.read(plan['original_completed_report_path'])
    require(run.rr.digest(plan['original_completed_report_path'])==plan['original_completed_report_sha256'],'Original report SHA differs')
    require(report['status']=='COMPLETE_420_RUN_MATRIX' and report['audited_completed_runs']==420 and report['cpu_prediction_replay_enabled'],'Original audit incomplete')
    audits=run.read(plan['original_completed_audit_path'])['audits']
    index={(a['protocol'],a['model'],a['seed']):a for a in audits}
    require(len(audits)==len(index)==420,'Original audit identities not420')
    references={}
    for ref in plan['reused_comparator_artifacts']:
        key=(ref['protocol'],ref['model'],ref['seed']);require(key not in references,'Duplicate reused comparator')
        a=index[key];require(a['status']=='PASSED_BINDINGS','Unpassed original comparator')
        replay_valid(a['computational_replay'])
        for name,item in ref['artifacts'].items():require(run.rr.digest(item['path'])==item['sha256'],'Original artifact changed: '+item['path'])
        require(ref['artifacts']['best.pt']['sha256']==a['checkpoint_sha256'] and
                ref['artifacts']['test_predictions.npz']['sha256']==a['predictions_sha256'],'Original artifact differs from final audit')
        references[key]={'reference':ref,'audit':a}
    expected={(p,m,s) for p in plan['protocols'] for m in MODELS[:2] for s in plan['seeds']}
    require(len(references)==120 and set(references)==expected,'Reused comparator matrix differs')
    return references


def original_cell(reference,data,split):
    ref=reference['reference'];a=reference['audit'];files=ref['artifacts']
    truth=np.stack([data['targets'][sid].numpy() for sid in split['test']])
    prediction,binding=load_prediction(files['test_predictions.npz']['path'],split['test'],data['time_s'].numpy(),truth)
    meta=run.read(files['run_metadata.json']['path']);result=run.read(files['result.json']['path'])
    for key,value in [('protocol',ref['protocol']),('mode',ref['model']),('seed',ref['seed'])]:require(meta[key]==result[key]==value,'Original reference identity differs')
    return {'audit':{'status':'REUSED_FROZEN_ORIGINAL_AUDIT','parameters':a['parameters'],'original_audit':a,**binding},
        'metrics':ag.scalar_metrics(truth,prediction),'truth':truth,'prediction':prediction,'training_wall_s':result['training_wall_s']}


def engineering_summary(truth,predictions,times,families,seeds):
    summaries={};replicates={};records=[]
    for mode in MODELS:
        summaries[mode],replicates[mode]=ep.comparative_summary(truth,predictions[mode],times,families,repetitions=1000,random_seed=20260907)
        for i,seed in enumerate(seeds):
            row=ep.evaluate_population(truth,predictions[mode][i],times,families);row.update(model=mode,seed=seed);records.append(row)
    contrasts={}
    for tag,baseline,candidate in CONTRASTS:
        metrics={}
        for key in summaries[baseline]:
            a,b=summaries[baseline][key],summaries[candidate][key]
            ba,bb=replicates[baseline][key],replicates[candidate][key];valid=np.isfinite(ba)&np.isfinite(bb)
            differences=ba[valid]-bb[valid];supported=a['defined_seeds']==b['defined_seeds']==len(seeds)
            metrics[key]={'point_baseline_minus_candidate':a['mean_over_defined_seeds']-b['mean_over_defined_seeds'] if supported else None,
                'percentile_95_paired_difference':np.quantile(differences,[.025,.975]).tolist() if supported and len(differences) else None,
                'baseline_defined_seeds':a['defined_seeds'],'candidate_defined_seeds':b['defined_seeds'],'required_seed_count':len(seeds),
                'common_valid_bootstrap_replicates':int(valid.sum()),'bootstrap_repetitions':1000,'valid_bootstrap_fraction':float(valid.mean()),
                'status':'ESTIMATED' if supported and len(differences) else 'NOT_ESTIMABLE_ON_ALL_PAIRED_SEEDS',
                'effect_direction':eci.preference(key)}
        contrasts[tag]={'baseline':baseline,'candidate':candidate,'difference_definition':'baseline metric minus candidate metric','metrics':metrics}
    return {'status':'COMPLETE_EXPLORATORY_THREE_MODEL_ENGINEERING_CI','model_seed_summaries':summaries,'paired_contrasts':contrasts,
        'per_seed_event_counts_and_metrics':records,'seeds':seeds,'secondary_bootstrap_repetitions':1000,'bootstrap_random_seed':20260907,
        'point_aggregation':'Mean of five seed-level metrics, not five times the independent geometry count.',
        'NA_policy':'Undefined event/class/ranking denominators remain null; point contrasts require all five seeds; report common valid bootstrap counts.',
        'conditional_event_timing_scope':'Each model has its own joint-reference/predicted-event subset. Differences compare those conditional metrics, not one shared crossing cohort.',
        'scope':'Archived response index only, not physical fire resistance or capacity.'}


def describe(protocol,cells,plan,manifest,data,split,output):
    by_id={c['sample_id']:c for c in manifest['cases']};cases=[by_id[s] for s in split['test']]
    geometry=[c['geometry_id'] for c in cases];families=np.asarray([c['fire_family'] for c in cases])
    require(len(set(geometry))==len(cases),'Engineering resampler requires one case per unique geometry')
    seeds=plan['seeds'];predictions={m:np.stack([cells[(m,s)]['prediction'] for s in seeds]) for m in MODELS}
    truth=cells[(MODEL_NAME,seeds[0])]['truth'];model_summaries=[];case_errors={};per_case=[];per_time=[];strata=[]
    for mode in MODELS:
        e=predictions[mode].astype(float)-truth.astype(float)[None];mae=np.abs(e).mean(2);case_errors[mode]=mae
        model_summaries.append({'model':mode,'seed_count':5,'test_case_count':len(cases),
            'parameters':cells[(mode,seeds[0])]['audit']['parameters'],
            'metrics':{key:ag.scalar_mean_sd([cells[(mode,s)]['metrics'][key] for s in seeds]) for key in ag.METRICS},
            'training_wall_s':ag.scalar_mean_sd([cells[(mode,s)]['training_wall_s'] for s in seeds])})
        for i,seed in enumerate(seeds):
            for j,c in enumerate(cases):
                per_case.append({'protocol':protocol,'model':mode,'seed':seed,'sample_id':c['sample_id'],'geometry_id':c['geometry_id'],
                    'fire_family':c['fire_family'],'crack_count':c['num_cracks'],'source_batch':c['source_batch'],
                    'legacy_final_id':c['legacy_final_id'],'flat_response':c['flat_response'],'MAE':float(mae[i,j]),
                    'RMSE':float(np.sqrt(np.square(e[i,j]).mean())),'maximum_overprediction':float(np.maximum(e[i,j],0).max())})
            for j,t in enumerate(data['time_s'].numpy()):
                per_time.append({'protocol':protocol,'model':mode,'seed':seed,'time_s':float(t),
                    'case_MAE':float(np.abs(e[i,:,j]).mean()),'case_RMSE':float(np.sqrt(np.square(e[i,:,j]).mean())),
                    'maximum_overprediction':float(np.maximum(e[i,:,j],0).max())})
            for kind,getter in [('fire_family',lambda c:c['fire_family']),('crack_count',lambda c:str(c['num_cracks'])),
                 ('source_batch',lambda c:c['source_batch']),('retained_or_restored',lambda c:'retained911' if c['legacy_final_id'] else 'restored86'),
                 ('flat_response',lambda c:'flat' if c['flat_response'] else 'nonflat')]:
                assignments=[getter(c) for c in cases]
                for group in sorted(set(assignments)):
                    mask=np.asarray([v==group for v in assignments]);metrics=ag.scalar_metrics(truth[mask],predictions[mode][i,mask])
                    strata.append({'protocol':protocol,'model':mode,'seed':seed,'stratum':kind,'group':group,'case_count':int(mask.sum()),
                                   **{key:metrics[key] for key in ag.METRICS}})
    paired={}
    for tag,baseline,candidate in CONTRASTS:
        ci=ag.es.paired_bootstrap(case_errors[baseline],case_errors[candidate],geometry,seeds,repeats=5000,random_seed=20260907)
        ci.pop('G04_or_G05_pass',None);ci.update(contrast=baseline+' MAE minus '+candidate+' MAE; positive favors '+candidate,
            purpose=run.PURPOSE,epistemic_status=plan['epistemic_status']);paired[tag]=ci
    summary={'status':'COMPLETE_EXPLORATORY_PROTOCOL_5_NEW_PLUS10_REUSED_RUNS','protocol':protocol,'purpose':run.PURPOSE,
        'epistemic_status':plan['epistemic_status'],'models':list(MODELS),'seeds':seeds,'test_case_count':len(cases),
        'independent_geometry_clusters':len(set(geometry)),'model_summaries':model_summaries,'paired_contrasts':paired,
        'no_mechanism_causality_claim':True,'all_original420_retained':True,
        'quantile_scope':'per_case_MAE_qXX uses time-mean absolute error per case; per_case_max_positive_error_qXX uses each case maximum over time, then case quantile; case_time_positive_error quantiles use all entries including zeros.'}
    engineering=engineering_summary(truth,predictions,data['time_s'].numpy(),families,seeds)
    engineering.update(protocol=protocol,test_case_count=len(cases),unique_geometry_groups=len(set(geometry)),epistemic_status=plan['epistemic_status'])
    output.mkdir();exclusive_json(output/'summary.json',summary);exclusive_json(output/'engineering_ci.json',engineering)
    ag.csv_write(output/'per_case_metrics.csv',per_case);ag.csv_write(output/'time_metrics.csv',per_time);ag.csv_write(output/'strata_by_seed.csv',strata)
    rows=[]
    for tag,c in engineering['paired_contrasts'].items():
        for metric,value in c['metrics'].items():
            ci=value['percentile_95_paired_difference']
            rows.append({'protocol':protocol,'contrast':tag,'metric':metric,'baseline':c['baseline'],'candidate':c['candidate'],
                'effect':value['point_baseline_minus_candidate'],'ci95_low':ci[0] if ci else None,'ci95_high':ci[1] if ci else None,
                'status':value['status'],'paired_seed_count':5,'valid_bootstrap':value['common_valid_bootstrap_replicates'],'bootstrap_repetitions':1000})
    ag.csv_write(output/'engineering_ci.csv',rows)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default=str(run.HERE/'plan_set_attention_60_v1.json'))
    parser.add_argument('--authorization');parser.add_argument('--output');parser.add_argument('--replay',action='store_true')
    parser.add_argument('--check-ready',action='store_true');args=parser.parse_args()
    plan,gate=readiness(args.plan)
    if args.check_ready or gate['status']!='READY_ALL60_METADATA_COMPLETE':print(json.dumps(gate,indent=2));return 0 if gate['status'].startswith('READY') else 2
    require(args.replay,'Full extension audit always requires checkpoint prediction replay')
    require(args.authorization,'Original authorization artifact must be supplied for final provenance checks')
    _,authorization=run.execution_gate(args.plan,args.authorization);require(authorization['status']=='AUTHORIZED_EXTENSION_EXECUTION','Missing execution lineage')
    require(args.output,'Provide a new explicit output snapshot directory')
    output=run.safe_output(args.output);require(not output.exists(),'Never overwrite an extension audit snapshot');output.mkdir(parents=True)
    started=time.perf_counter()
    try:
        manifest,data=run.rr.load_dataset(plan['manifest_path']);references=bind_original(plan)
        run.rr.configure(42,4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        cells={};audits=[];splits={}
        # Every new checkpoint is independently cleared before any summaries are written.
        for protocol,spec in plan['protocols'].items():
            split=run.read(spec['split_path']);run.rr.check_split(split,manifest);splits[protocol]=split
            for seed in plan['seeds']:
                folder=Path(plan['runs_root'])/protocol/MODEL_NAME/('seed_'+str(seed))
                cell=audit_new(folder,args.plan,plan,manifest,data,protocol,seed,split,args.authorization)
                cells[(protocol,MODEL_NAME,seed)]=cell;audits.append(cell['audit'])
                for mode in MODELS[:2]:cells[(protocol,mode,seed)]=original_cell(references[(protocol,mode,seed)],data,split)
        require(len(audits)==60 and len(cells)==180,'Final comparison matrix incomplete')
        exclusive_json(output/'run_integrity.json',{'new_run_audits':audits,'reused_original_comparators':[v for v in references.values()]})
        for protocol in plan['protocols']:
            current={(m,s):cells[(protocol,m,s)] for m in MODELS for s in plan['seeds']}
            describe(protocol,current,plan,manifest,data,splits[protocol],output/protocol)
        sources=[Path(__file__),Path(ag.__file__),Path(ag.es.__file__),Path(ep.__file__),Path(eci.__file__)]
        report={'status':'COMPLETE_EXPLORATORY_60_NEW_PLUS120_REUSED_MATRIX','purpose':run.PURPOSE,
            'epistemic_status':plan['epistemic_status'],'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'plan_sha256':run.rr.digest(args.plan),'original_plan_sha256':run.ORIGINAL_PLAN_SHA,
            'original_completed_audit_sha256':plan['original_completed_audit_sha256'],'new_runs_audited':60,'reused_comparator_runs':120,
            'protocols':list(plan['protocols']),'models':list(MODELS),'seeds':plan['seeds'],'cpu_prediction_replay_enabled':True,
            'computational_replay_policy':'Original strict CPU diagnostic maximum <3e-6; flags retain CPU differences and require bitwise exact original GPU backend replay. No tolerance widening.',
            'analysis_source_sha256':{str(p):run.rr.digest(p) for p in sources},'execution_authorization_sha256':run.rr.digest(args.authorization),
            'outputs_sha256':{str(p.relative_to(output)):run.rr.digest(p) for p in output.rglob('*') if p.is_file()},
            'performance_gate':gate,'all_original420_retained':True,'mechanism_causality_claim':False,'peer_review_pass':False,
            'elapsed_s':time.perf_counter()-started}
        exclusive_json(output/'report.json',report)
        print(json.dumps({'status':report['status'],'new_runs':60,'reused_runs':120,'output':str(output),'report_sha256':run.rr.digest(output/'report.json')},indent=2))
    except BaseException as exc:
        exclusive_json(output/'failure.json',{'status':'FAILED_EXTENSION_AUDIT_NO_COMPLETE_CLAIM','error':repr(exc),'peer_review_pass':False})
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
