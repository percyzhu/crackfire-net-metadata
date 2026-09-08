"""Separate three-model timing batch; no modification of original measurements.

Reuses audited pure helpers, not the original hard-coded seven-model driver.
No collection before complete60 replay, independent review, quiet gate and a
separate root authorization. Preparation never reads new accuracy results.
"""
from pathlib import Path
import argparse
import collections
import csv
import datetime
import json
import math
import os
import platform
import re
import sys
import time

HERE=Path(__file__).resolve().parent
EXT=HERE.parent
V08=EXT.parent/'research_v08'
sys.path.insert(0,str(EXT));sys.path.insert(0,str(V08))
import run_extension as ext
import benchmark_inference as bm
import numpy as np
import torch
from attention_model import SetAttentionSurrogate

PLAN=HERE/'timing_plan_v1.json'
EXT_PLAN=EXT/'plan_set_attention_60_v1.json'
EXT_PLAN_SHA='8d21e13af116dc97556b37060daaf86afed5f7ba645a5bdae18e3d4ffb3b05a9'
ORIGINAL_BENCHMARK_SHA='aad7de1e3e332a119adf5ea97f2813bf47dd8d3fe4d136a4ba60cb096f96bcf9'
AGGREGATOR_SHA='aa06a3280fed95be571f1b39677c84c20d193804897d4f4d55ec7a99053a2e7a'
REVIEW_STATUS='PASS_INDEPENDENT_NUMERICAL_REVIEW_60_NEW_PLUS120_REUSED'
MODELS=['capacity_matched_deepsets','gnn','set_attention']
STAGES=['forward_resident_inputs','common_input_to_host_output']


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return bm.read(path)


def sha(path):return ext.rr.digest(path)


def safe(path):
    path=Path(path).resolve();require(path.is_relative_to(HERE.resolve()),'Timing writes must stay in the new timing directory');return path


def write(path,value):
    path=safe(path)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def prepare():
    require(not PLAN.exists(),'Preserve an existing timing declaration; no overwrite')
    extension=ext.verify_plan(EXT_PLAN);require(sha(EXT_PLAN)==EXT_PLAN_SHA,'Wrong extension plan')
    original=read(extension['original_plan_path'])
    require(sha(V08/'benchmark_inference.py')==ORIGINAL_BENCHMARK_SHA,'Original benchmark helper changed')
    split_path=extension['protocols']['iid997']['split_path'];split=read(split_path)
    manifest=read(extension['manifest_path']);cases={c['sample_id']:c for c in manifest['cases']}
    require(len(split['test'])==160 and len(set(split['test']))==160,'Expected all160 IID test geometries')
    records=[]
    for model in MODELS:
        root=Path(extension['runs_root']) if model=='set_attention' else Path(original['runs_root'])
        folder=root/'iid997'/model/'seed_42';meta=read(folder/'run_metadata.json');status=read(folder/'status.json')
        require((meta['protocol'],meta['mode'],meta['seed'])==('iid997',model,42),'Wrong selected checkpoint identity')
        expected_plan=EXT_PLAN_SHA if model=='set_attention' else extension['original_plan_sha256']
        require(meta['plan_sha256']==expected_plan and meta['dataset_sha256']==extension['manifest_sha256'],'Selected source binding differs')
        require(status['status']==('COMPLETE_EXPLORATORY_EXTENSION' if model=='set_attention' else 'COMPLETE_ARCHIVAL_RESEARCH'),'Selected checkpoint unfinished')
        records.append({'model':model,'checkpoint':str(folder/'best.pt'),'checkpoint_sha256':sha(folder/'best.pt'),
            'run_metadata_path':str(folder/'run_metadata.json'),'run_metadata_sha256':sha(folder/'run_metadata.json'),
            'training_plan_sha256':expected_plan,'parameters':meta['parameters']})
    old_timing=V08/'inference_benchmark_v1/measured/summary.json'
    old_review=ext.rr.SCI/'review/eaai_editor/research_v08/inference_timing_independent_audit.json'
    require(read(old_timing)['status']=='COMPLETE_INFERENCE_TIMING' and read(old_timing)['rows']==14355,'Original measurement missing')
    require(read(old_review)['status']=='REAL_TIMING_RAW_RECORDS_COVERAGE_AND_ARITHMETIC_AUDIT_PASSED','Original timing review missing')
    plan={'status':'DECLARED_BEFORE_EXTENSION_ACCURACY_REVIEW_AND_SEPARATE_TIMING',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'extension_plan_sha256':EXT_PLAN_SHA,
        'original_training_plan_sha256':extension['original_plan_sha256'],'benchmark_source_sha256':sha(__file__),
        'original_benchmark_helper_path':str(V08/'benchmark_inference.py'),'original_benchmark_helper_sha256':ORIGINAL_BENCHMARK_SHA,
        'reused_helpers':['regenerate(case,make_time)','verify_features(manifest,data,ids)','sync(device)',
            'powershell(script)','is_project_compute(process,current_pid)','replay_identity_passed(record)'],
        'not_reused':'Original run/prepare/validate_selection have a fixed seven-model inventory and original output paths; this driver is independent.',
        'extension_source_sha256':extension['extension_source_sha256'],'original_source_sha256':extension['original_source_sha256'],
        'manifest_path':extension['manifest_path'],'manifest_sha256':extension['manifest_sha256'],
        'IID_split_path':split_path,'IID_split_sha256':sha(split_path),'feature_version':extension['feature_version'],
        'target_version':extension['target_version'],'protocol':'iid997','seed':42,'models':MODELS,'checkpoints':records,
        'sample_ids':split['test'],'times_per_case':61,'observed_crack_counts':sorted({cases[s]['num_cracks'] for s in split['test']}),
        'selection_rule':'First original seed42; every160 IID test geometry in frozen split order; all three fixed models. No choice based on accuracy, latency or confidence interval.',
        'new_attention_accuracy_inspected_by_preparer':False,'original420_results_already_known':True,
        'devices':['cpu','cuda'],'CPU_threads':4,'rounds':3,'batch_sizes':[1,32],
        'model_orders_by_round':[MODELS[i:]+MODELS[:i] for i in range(3)],
        'device_orders_by_round':[['cpu','cuda'],['cuda','cpu'],['cpu','cuda']],
        'warmup_full_population_passes':1,'warmup_batches_per_shape':10,'stages':STAGES,
        'feature_preparation_stage':'feature_preparation_only: regenerate all five CPU feature tensors once per round and batch-size condition, shared across models/devices.',
        'resident_interface':'Precollated device-resident common inputs to device-resident whole61-point response; forward call only, synchronized around each CUDA duration.',
        'common_interface':'In-memory geometry/fire objects -> reconstruct five tensors -> collate/device transfer -> forward -> CPU NumPy whole61-point response. Includes unused raw edge/node features equally for all three models.',
        'model_source_policy':'Frozen attention implementation, including padding and Python packing, is measured as executed; no speed-driven source changes.',
        'precision':{'float_dtype':'float32','index_dtype':'int64','cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':True,'deterministic_algorithms':True},
        'statistics':'Keep every raw batch duration in seconds. Report mean/p50/p95 batch milliseconds, total cases divided by summed seconds, and summed milliseconds divided by cases. Serial-query N strata for batch1; all26 overall and78 per-round groups.',
        'expected_records':{'model_timings':5940,'common_feature_preparation':495,'total':6435,'overall_groups':26,'round_groups':78},
        'warmup_and_sync':'Each round/device/model/shape/interface: full160-case pass then10 batches before collection; CUDA synchronize before timer and after operation. Shape/finiteness checks and logging outside timer.',
        'excluded_costs':['Imports','checkpoint loading','disk and JSON I/O','network transfer','FE generation'],
        'historical_training_cost_scope':'The separately audited60-new/120-reused aggregate records training/validation/I/O wall time. This inference batch does not measure or reclassify training cost.',
        'variance_scope':'Three repeated rounds and one fixed checkpoint seed on the observed host. No accuracy-CI or multi-hardware claim.',
        'same_new_batch_anchors':['capacity_matched_deepsets','gnn'],
        'old_measurement_policy':'Original seven-model14355 records remain immutable; no pooling, replacement or selection between old/new sessions. Three-model comparisons use only this new common session; discrepancies between sessions remain visible.',
        'old_timing_summary_path':str(old_timing),'old_timing_summary_sha256':sha(old_timing),
        'old_timing_review_path':str(old_review),'old_timing_review_sha256':sha(old_review),
        'required_final_aggregate_source_sha256':AGGREGATOR_SHA,
        'execution_gate':'All60 unique extension cells/queue complete; real complete replay aggregate and bound independent numerical-result review; all60 and120 actual source artifacts verified; no detected project Python compute; separately bound root timing authorization.',
        'gate_rechecks':'Full model/data/artifact scan once before timing. Each round/model and completion: only queue, source/final-evidence hashes and detected project-compute checks, plus selected checkpoint SHA. No repeated full weight scan in timing collection.',
        'execution_authorization_status':'AUTHORIZED_SEPARATE_ATTENTION_TIMING_AFTER_COMPLETE60_REPLAY_AND_QUIET',
        'output_directory':str(HERE/'measured'),'run_count_new_training':0,'latencies_measured_during_preparation':0}
    write(PLAN,plan);validate_plan()
    print(json.dumps({'status':plan['status'],'plan_sha256':sha(PLAN),'checkpoints':3,'cases':160,'expected_raw_records':6435,'latencies_measured':0},indent=2))


def validate_plan():
    extension=ext.verify_plan(EXT_PLAN);require(sha(EXT_PLAN)==EXT_PLAN_SHA,'Extension plan changed')
    timing=read(PLAN)
    require(timing['status']=='DECLARED_BEFORE_EXTENSION_ACCURACY_REVIEW_AND_SEPARATE_TIMING','Wrong timing declaration')
    require(timing['benchmark_source_sha256']==sha(__file__) and timing['original_benchmark_helper_sha256']==sha(V08/'benchmark_inference.py')==ORIGINAL_BENCHMARK_SHA,'Timing source changed')
    require(timing['extension_source_sha256']==extension['extension_source_sha256'] and timing['original_source_sha256']==extension['original_source_sha256'],'Training source chain differs')
    require(timing['extension_plan_sha256']==EXT_PLAN_SHA and timing['manifest_sha256']==sha(timing['manifest_path'])==extension['manifest_sha256'],'Dataset/plan differs')
    require(timing['IID_split_sha256']==sha(timing['IID_split_path'])==extension['protocols']['iid997']['split_sha256'],'IID split differs')
    split=read(timing['IID_split_path'])
    require(timing['protocol']=='iid997' and timing['seed']==42 and timing['models']==MODELS and timing['sample_ids']==split['test'] and len(split['test'])==160,'Timing selection differs')
    require(timing['rounds']==3 and timing['devices']==['cpu','cuda'] and timing['CPU_threads']==4 and timing['batch_sizes']==[1,32] and timing['stages']==STAGES,'Timing recipe differs')
    require(timing['model_orders_by_round']==[MODELS[i:]+MODELS[:i] for i in range(3)] and timing['device_orders_by_round']==[['cpu','cuda'],['cuda','cpu'],['cpu','cuda']],'Round order differs')
    require(timing['warmup_full_population_passes']==1 and timing['warmup_batches_per_shape']==10 and timing['times_per_case']==61,'Warmup/output scope differs')
    for record in timing['checkpoints']:
        require(sha(record['run_metadata_path'])==record['run_metadata_sha256'],'Selected checkpoint metadata changed')
    require([r['model'] for r in timing['checkpoints']]==MODELS,'Missing selected model')
    require(sha(timing['old_timing_summary_path'])==timing['old_timing_summary_sha256'] and sha(timing['old_timing_review_path'])==timing['old_timing_review_sha256'],'Historical timing source changed')
    return timing,extension


def quiet_inventory():
    result=bm.powershell("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress")
    processes=json.loads(result) if result else []
    if isinstance(processes,dict):processes=[processes]
    script_names={p.stem.lower() for p in EXT.rglob('*.py')}
    def matches(process):
        if int(process['ProcessId'])==os.getpid():return False
        if bm.is_project_compute(process,os.getpid()):return True
        command=(process.get('CommandLine') or '').lower()
        return any(re.search(r'(?<![\w])'+re.escape(name)+r'\.py\b',command) for name in script_names)
    matching=[p for p in processes if matches(p)];matching_ids={p['ProcessId'] for p in matching}
    logged=[p if p['ProcessId'] in matching_ids else {'ProcessId':p['ProcessId'],'Name':p['Name']} for p in processes]
    return {'matching_project_processes':matching,'python_process_inventory':logged,
            'scope':'Known project paths and actual script basenames. Anonymous jobs and unrelated OS work are not proven absent.'}


def readiness(audit_dir=None,review_path=None,authorization_path=None,bound=None):
    timing,extension=validate_plan();reasons=[]
    root=Path(extension['runs_root']);queue=read(root/'queue_status.json')
    expected={(p,'set_attention',s) for p in extension['protocols'] for s in extension['seeds']}
    keys=[(v['protocol'],v['model'],v['seed']) for v in queue['completed']]
    if queue['status']=='ERROR_STOPPED_EXTENSION':raise ValueError('Training failed; do not measure or retry')
    if queue['status']!='COMPLETE_60_EXPLORATORY_EXTENSION' or queue['active'] is not None:reasons.append('extension_queue_incomplete')
    if len(keys)!=60 or len(set(keys))!=60 or set(keys)!=expected:reasons.append('extension_matrix_not60_unique')
    require(queue['plan_sha256']==EXT_PLAN_SHA,'Queue source differs')
    if not bm.final_report_ready(read(extension['original_plan_path'])):reasons.append('original420_final_replay_missing_or_changed')
    evidence={};report=None;review=None
    # No score/result/checkpoint payload reads: final report and review are metadata.
    if audit_dir and (Path(audit_dir)/'report.json').is_file() and not reasons:
        audit_dir=Path(audit_dir).resolve();require(audit_dir.is_relative_to(EXT),'Final aggregate outside extension')
        report=read(audit_dir/'report.json')
        require(not (audit_dir/'failure.json').exists(),'Final aggregate failed')
        require(report['status']=='COMPLETE_EXPLORATORY_60_NEW_PLUS120_REUSED_MATRIX' and report['new_runs_audited']==60 and report['reused_comparator_runs']==120 and report['cpu_prediction_replay_enabled'] is True,'Final replay aggregate incomplete')
        require(report['plan_sha256']==EXT_PLAN_SHA and report['protocols']==list(extension['protocols']) and report['models']==MODELS and report['seeds']==extension['seeds'],'Final aggregate identities differ')
        require(report['analysis_source_sha256'][str(EXT/'aggregate_extension.py')]==sha(EXT/'aggregate_extension.py')==AGGREGATOR_SHA,'Final aggregate source changed')
        evidence={'aggregate_report_sha256':sha(audit_dir/'report.json'),'aggregate_run_integrity_sha256':sha(audit_dir/'run_integrity.json')}
    else:reasons.append('complete60_replay_aggregate_not_available')
    if report is not None and review_path and Path(review_path).is_file():
        review=read(review_path)
        require(review['status']==REVIEW_STATUS and review['numerical_review_passed'] is True,'Independent result review not passed')
        require(review['extension_plan_sha256']==EXT_PLAN_SHA and all(review[k]==v for k,v in evidence.items()),'Independent review is not bound to current aggregate')
        require(review['completed_new_runs']==60 and review['reused_comparator_runs']==120 and review['protocols']==list(extension['protocols']) and review['models']==MODELS and review['seeds']==extension['seeds'],'Independent result review scope differs')
        evidence['independent_review_sha256']=sha(review_path)
    else:reasons.append('independent60_result_review_not_available')
    inventory=quiet_inventory()
    if inventory['matching_project_processes']:reasons.append('concurrent_project_compute')
    if bound:
        require(all(evidence.get(k)==v for k,v in bound['evidence_hashes'].items()),'Bound final evidence changed during timing')
        require(sha(PLAN)==bound['timing_plan_sha256'],'Bound timing plan changed')
    authorization_ok=False
    if authorization_path and Path(authorization_path).is_file() and report is not None and review is not None:
        authorization=read(authorization_path)
        require(authorization['status']==timing['execution_authorization_status'],'Wrong authorization scope')
        require(authorization['timing_plan_sha256']==sha(PLAN) and authorization['extension_plan_sha256']==EXT_PLAN_SHA,'Authorization is not bound to this timing/learning plan')
        require(all(authorization[k]==v for k,v in evidence.items()),'Timing authorization final-evidence mismatch')
        authorization_ok=True
    if not authorization_ok:reasons.append('separate_root_timing_authorization_required')
    return {'status':'READY_AUTHORIZED_SEPARATE_TIMING' if not reasons else 'DEFERRED_NO_MEASUREMENTS',
        'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reasons':reasons,
        'completed_extension_metadata_runs':len(keys),'timing_plan_sha256':sha(PLAN),'evidence_hashes':evidence,
        'latency_records_read':0,'latencies_measured':0,**inventory}


def validate_artifacts(timing,extension,audit_dir,review_path):
    """One full scan after execution gate; no inference or timers."""
    audit_dir=Path(audit_dir);report=read(audit_dir/'report.json')
    for path,digest in report['analysis_source_sha256'].items():require(sha(path)==digest,'Analysis source changed')
    for relative,digest in report['outputs_sha256'].items():
        file=(audit_dir/relative).resolve();require(file.is_relative_to(audit_dir.resolve()) and sha(file)==digest,'Audit output changed')
    integrity=read(audit_dir/'run_integrity.json');audits=integrity['new_run_audits'];root=Path(extension['runs_root'])
    expected={(p,'set_attention',s) for p in extension['protocols'] for s in extension['seeds']}
    require(len(audits)==60 and {(a['protocol'],a['model'],a['seed']) for a in audits}==expected,'Final replay identities not60')
    for audit in audits:
        require(audit['status']=='PASSED_EXTENSION_BINDINGS_AND_REPLAY' and audit['plan_sha256']==EXT_PLAN_SHA,'Unpassed final run')
        require(bm.replay_identity_passed(audit['computational_replay']),'Incomplete/failed strictCPU or GPUexact replay')
        folder=root/audit['protocol']/audit['model']/('seed_'+str(audit['seed']))
        for name,digest in audit['source_artifact_sha256'].items():require(sha(folder/name)==digest,'Actual extension source differs from audit')
    require(len(integrity['reused_original_comparators'])==120,'Reused comparator count differs')
    for ref in extension['reused_comparator_artifacts']:
        for item in ref['artifacts'].values():require(sha(item['path'])==item['sha256'],'Original reused artifact changed')
    bound_checkpoints=[]
    for record in timing['checkpoints']:
        require(sha(record['checkpoint'])==record['checkpoint_sha256'],'Prespecified checkpoint changed')
        meta=read(record['run_metadata_path']);require(sha(record['run_metadata_path'])==record['run_metadata_sha256'],'Selected metadata changed')
        require(meta['parameters']==record['parameters'] and meta['plan_sha256']==record['training_plan_sha256'],'Selected checkpoint metadata differs')
        checkpoint=torch.load(record['checkpoint'],map_location='cpu',weights_only=True)
        for key in ('plan_sha256','dataset_sha256','split_sha256'):require(checkpoint[key]==meta[key],'Selected checkpoint internal provenance differs')
        bound_checkpoints.append(record)
    return {'status':'ALL60_NEW_AND120_REUSED_ARTIFACTS_BOUND_BEFORE_SEPARATE_TIMING','timing_plan_sha256':sha(PLAN),
        'evidence_hashes':{'aggregate_report_sha256':sha(audit_dir/'report.json'),'aggregate_run_integrity_sha256':sha(audit_dir/'run_integrity.json'),
                           'independent_review_sha256':sha(review_path)},'selected_checkpoints':bound_checkpoints,
        'new_artifact_runs_checked':60,'reused_artifact_runs_checked':120,'no_latency_measurements_in_binding':True}


def summarize(rows):
    grouped=collections.defaultdict(list);by_count=collections.defaultdict(list);per_round=collections.defaultdict(list)
    for row in rows:
        grouped[(row['device'],row['model'],row['stage'],row['batch_size'])].append(row)
        per_round[(row['round'],row['device'],row['model'],row['stage'],row['batch_size'])].append(row)
        if row['batch_size']==1:by_count[(row['device'],row['model'],row['stage'],int(row['num_cracks']))].append(row)
    summaries=[];round_summaries=[];counts=[]
    for key,group in grouped.items():
        values=np.asarray([r['elapsed_s'] for r in group]);n=sum(r['cases'] for r in group)
        require(np.isfinite(values).all() and np.all(values>0),'Invalid measured duration')
        summaries.append(dict(zip(['device','model','stage','batch_size'],key),measurements=len(group),cases_total=n,
            mean_batch_ms=float(values.mean()*1000),median_batch_ms=float(np.median(values)*1000),p95_batch_ms=float(np.quantile(values,.95)*1000),
            amortized_ms_per_case=float(values.sum()*1000/n),cases_per_second=float(n/values.sum())))
    for key,group in per_round.items():
        elapsed=sum(r['elapsed_s'] for r in group);n=sum(r['cases'] for r in group)
        round_summaries.append(dict(zip(['round','device','model','stage','batch_size'],key),cases_total=n,measurements=len(group),
            mean_batch_ms=elapsed*1000/len(group),amortized_ms_per_case=elapsed*1000/n,cases_per_second=n/elapsed))
    for key,group in by_count.items():
        values=np.asarray([r['elapsed_s'] for r in group])
        counts.append(dict(zip(['device','model','stage','num_cracks'],key),measurements=len(group),unique_cases=len({r['sample_ids'] for r in group}),
            mean_single_query_ms=float(values.mean()*1000),median_single_query_ms=float(np.median(values)*1000),p95_single_query_ms=float(np.quantile(values,.95)*1000)))
    require(len(rows)==6435 and len(summaries)==26 and len(round_summaries)==78,'Timing record/group count differs')
    return summaries,round_summaries,counts


def collect(args):
    ready=readiness(args.audit_dir,args.review,args.authorization)
    if ready['status']!='READY_AUTHORIZED_SEPARATE_TIMING':print(json.dumps(ready,indent=2));return 2
    timing,extension=validate_plan();target=safe(timing['output_directory']);require(not target.exists(),'Preserve an existing measurement attempt')
    bound=validate_artifacts(timing,extension,args.audit_dir,args.review)
    original,manifest,data,split=bm.load_sources()
    require(split['test']==timing['sample_ids'],'Common input IDs changed')
    cases,make_time=bm.verify_features(manifest,data,timing['sample_ids'])
    ready=readiness(args.audit_dir,args.review,args.authorization,bound)
    if ready['status']!='READY_AUTHORIZED_SEPARATE_TIMING':print(json.dumps(ready,indent=2));return 2
    target.mkdir();write(target/'premeasurement_artifact_binding.json',bound)
    rows=[];gates=[ready]
    try:
        ext.rr.configure(42,4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        require(torch.cuda.is_available(),'Both declared CPU and CUDA conditions are required')
        environment={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__,'platform':platform.platform(),
            'pid':os.getpid(),'CPU':bm.powershell('Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress'),
            'GPU':torch.cuda.get_device_name(0),'CPU_threads':torch.get_num_threads(),'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'timing_plan_sha256':sha(PLAN),'authorization_sha256':sha(args.authorization),'premeasurement_artifact_binding_sha256':sha(target/'premeasurement_artifact_binding.json'),
            'old7_measurements_pooled':False,'stage_units':'one entire61-point response trajectory per case, float32 outputs; seconds per measured batch'}
        write(target/'environment.json',environment)
        records={r['model']:r for r in timing['checkpoints']}
        def recheck():
            state=readiness(args.audit_dir,args.review,args.authorization,bound);gates.append(state)
            require(state['status']=='READY_AUTHORIZED_SEPARATE_TIMING','Concurrent compute or final source change during timing')
        with (target/'raw_timings.csv').open('x',encoding='utf-8',newline='') as stream,torch.inference_mode():
            fields=['round','device','model','stage','batch_size','cases','sample_ids','num_cracks','elapsed_s']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            def save_row(round_index,device,model,stage,size,group,elapsed):
                row=dict(round=round_index,device=device,model=model,stage=stage,batch_size=size,cases=len(group),
                    sample_ids='|'.join(group),num_cracks='|'.join(str(cases[s]['num_cracks']) for s in group),elapsed_s=elapsed)
                writer.writerow(row);rows.append(row)
            for round_index in range(3):
                recheck()
                for size in timing['batch_sizes']:
                    groups=[timing['sample_ids'][i:i+size] for i in range(0,160,size)]
                    for group in groups:prepared=[bm.regenerate(cases[s],make_time) for s in group]
                    for j in range(10):prepared=[bm.regenerate(cases[s],make_time) for s in groups[j%len(groups)]]
                    for group in groups:
                        start=time.perf_counter();prepared=[bm.regenerate(cases[s],make_time) for s in group];elapsed=time.perf_counter()-start
                        save_row(round_index,'cpu','common_features','feature_preparation_only',size,group,elapsed)
                    stream.flush()
                for device in timing['device_orders_by_round'][round_index]:
                    for mode in timing['model_orders_by_round'][round_index]:
                        recheck();record=records[mode];require(sha(record['checkpoint'])==record['checkpoint_sha256'],'Selected weights changed')
                        model=(SetAttentionSurrogate() if mode=='set_attention' else ext.rr.make_model(mode)).to(device).eval()
                        checkpoint=torch.load(record['checkpoint'],map_location=device,weights_only=True);model.load_state_dict(checkpoint['model_state_dict'],strict=True)
                        require(sum(p.numel() for p in model.parameters())==record['parameters'],'Model parameter count differs')
                        for size in timing['batch_sizes']:
                            groups=[timing['sample_ids'][i:i+size] for i in range(0,160,size)]
                            resident=[ext.rr.collate([data['graphs'][s] for s in group],device) for group in groups]
                            def operation(index,stage):
                                batch=resident[index] if stage=='forward_resident_inputs' else ext.rr.collate([bm.regenerate(cases[s],make_time) for s in groups[index]],device)
                                prediction=model(*batch)
                                return prediction if stage=='forward_resident_inputs' else prediction.cpu().numpy()
                            for stage in STAGES:
                                for j in range(len(groups)):operation(j,stage)
                                for j in range(10):operation(j%len(groups),stage)
                                bm.sync(device)
                                for index,group in enumerate(groups):
                                    bm.sync(device);start=time.perf_counter();prediction=operation(index,stage);bm.sync(device);elapsed=time.perf_counter()-start
                                    require(tuple(prediction.shape)==(len(group),61,1),'Timing output shape differs')
                                    finite=bool(torch.isfinite(prediction).all()) if isinstance(prediction,torch.Tensor) else bool(np.isfinite(prediction).all())
                                    require(finite,'Nonfinite timing output');save_row(round_index,device,mode,stage,size,group,elapsed)
                                stream.flush()
                        del model,checkpoint,resident,prediction
                        bm.sync(device)
        recheck();summaries,rounds,counts=summarize(rows);write(target/'readiness_checks.json',gates)
        report={'status':'COMPLETE_SEPARATE_THREE_MODEL_ATTENTION_TIMING','rows':len(rows),'models':MODELS,'summaries':summaries,
            'per_round_summaries':rounds,'single_query_by_crack_count':counts,'timing_plan_sha256':sha(PLAN),
            'artifact_binding_sha256':sha(target/'premeasurement_artifact_binding.json'),'raw_csv_sha256':sha(target/'raw_timings.csv'),
            'environment_sha256':sha(target/'environment.json'),'authorization_sha256':sha(args.authorization),
            'original7_14355_records_pooled_or_replaced':False,'comparison_scope':'Three fixed models measured in this new common session only.',
            'independent_timing_arithmetic_review_still_required':True,'submission_pass':False}
        write(target/'summary.json',report);print(json.dumps({'status':report['status'],'raw_records':len(rows),'summary_sha256':sha(target/'summary.json')},indent=2))
    except BaseException as error:
        write(target/'failure.json',{'status':'FAILED_SEPARATE_TIMING_RETAIN_PARTIAL_DO_NOT_REPORT','error':repr(error)});raise
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','check-ready','run']);parser.add_argument('--audit-dir')
    parser.add_argument('--review',default=str(EXT/'independent_full_matrix_review.json'));parser.add_argument('--authorization')
    args=parser.parse_args(argv)
    if args.action=='prepare':prepare();return 0
    if args.action=='run':return collect(args)
    report=readiness(args.audit_dir,args.review,args.authorization);print(json.dumps(report,indent=2));return 0 if report['status']=='READY_AUTHORIZED_SEPARATE_TIMING' else 2


if __name__=='__main__':raise SystemExit(main())
