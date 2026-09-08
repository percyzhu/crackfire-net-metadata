"""Isolated 60-run exploratory extension; original420 sources are read-only.

Training is impossible without a separately supplied execution authorization,
an independent software review and the completed original7 timing audit.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
V08 = HERE.parent / 'research_v08'
sys.path.insert(0, str(V08))
import research_runner as rr
import numpy as np
import torch
from attention_model import SetAttentionSurrogate, MODEL_NAME, MODEL_VERSION, ARCHITECTURE, parameter_count

PURPOSE = 'ARCHIVAL_INTERACTION_SET_EXPLORATORY_EXTENSION_V09'
ORIGINAL_PLAN_SHA = 'fd9088427cbc9234191b71e619626b188a98c43c9af248c9dd3e435afdb8f068'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def safe_output(p):
    p = Path(p).resolve()
    if not p.is_relative_to(HERE.resolve()): raise ValueError('Extension writes must remain inside research_v09_set_attention')
    return p


def write_json(p, value):
    p = safe_output(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    start, delay = time.monotonic(), .05
    while True:
        try: temporary.replace(p); return
        except PermissionError:
            if time.monotonic() - start >= 10: raise
            time.sleep(delay); delay = min(.8, delay * 2)


def new_best(validation_mse, previous):
    if not math.isfinite(validation_mse): raise FloatingPointError('Nonfinite validation MSE')
    return validation_mse < previous


def verify_plan(plan_path):
    p = read(plan_path)
    if p['purpose'] != PURPOSE or p['epistemic_status'] != 'AFTER_ORIGINAL_TEST_RESULTS_REVIEW_EXPLORATORY':
        raise ValueError('Wrong extension purpose or falsely pre-registered status')
    if p['status'] != 'FROZEN_BEFORE_EXTENSION_TRAINING_AWAITING_EXECUTION_AUTHORIZATION':
        raise ValueError('Extension configuration is not frozen')
    original = read(p['original_plan_path'])
    assert rr.digest(p['original_plan_path']) == p['original_plan_sha256'] == ORIGINAL_PLAN_SHA
    assert original['source_sha256'] == rr.sources()
    assert p['original_source_sha256'] == original['source_sha256']
    assert p['training_configuration'] == original['training_configuration']
    assert p['protocols'] == original['protocols'] and p['seeds'] == original['seeds']
    assert p['manifest_path'] == original['manifest_path'] and p['manifest_sha256'] == original['manifest_sha256']
    assert rr.digest(p['manifest_path']) == p['manifest_sha256']
    assert p['architecture'] == ARCHITECTURE and p['model_version'] == MODEL_VERSION
    assert p['new_model'] == MODEL_NAME and p['new_run_count'] == 60
    assert len(p['protocols']) == 12 and p['seeds'] == [42,43,44,45,46]
    for path, digest in p['extension_source_sha256'].items(): assert rr.digest(path) == digest
    assert rr.digest(p['original_completed_audit_path']) == p['original_completed_audit_sha256']
    for spec in p['protocols'].values(): assert rr.digest(spec['split_path']) == spec['split_sha256']
    safe_output(p['runs_root'])
    return p


def execution_gate(plan_path, authorization_path):
    plan = verify_plan(plan_path)
    if authorization_path is None or not Path(authorization_path).is_file():
        return plan, {'status':'BLOCKED_PENDING_EXPLICIT_EXECUTION_AUTHORIZATION','optimizer_steps':0}
    auth = read(authorization_path)
    assert auth['status'] == 'AUTHORIZED_AFTER_ORIGINAL_TIMING_AND_INDEPENDENT_SOFTWARE_REVIEW'
    assert auth['plan_sha256'] == rr.digest(plan_path)
    reviewer = read(auth['independent_software_review_path'])
    assert rr.digest(auth['independent_software_review_path']) == auth['independent_software_review_sha256']
    assert reviewer['extension_plan_sha256'] == rr.digest(plan_path) and reviewer['software_review_passed'] is True
    timing_path = V08 / 'inference_benchmark_v1/measured/summary.json'
    timing_audit_path = rr.SCI / 'review/eaai_editor/research_v08/inference_timing_independent_audit.json'
    assert read(timing_path)['status'] == 'COMPLETE_INFERENCE_TIMING'
    assert not (timing_path.parent / 'failure.json').exists()
    assert read(timing_audit_path)['status'] == 'REAL_TIMING_RAW_RECORDS_COVERAGE_AND_ARITHMETIC_AUDIT_PASSED'
    assert rr.digest(timing_path) == auth['original7_timing_summary_sha256']
    assert rr.digest(timing_audit_path) == auth['original7_timing_audit_sha256']
    return plan, {'status':'AUTHORIZED_EXTENSION_EXECUTION','authorization_sha256':rr.digest(authorization_path)}


def train_one(plan_path, authorization_path, protocol, seed):
    plan, gate = execution_gate(plan_path, authorization_path)
    if gate['status'] != 'AUTHORIZED_EXTENSION_EXECUTION': raise RuntimeError(gate['status'])
    if protocol not in plan['protocols'] or seed not in plan['seeds']: raise ValueError('Unregistered cell')
    config = plan['training_configuration']; manifest, data = rr.load_dataset(plan['manifest_path'])
    split_path = plan['protocols'][protocol]['split_path']; split = read(split_path); rr.check_split(split, manifest)
    output = safe_output(Path(plan['runs_root']) / protocol / MODEL_NAME / ('seed_' + str(seed)))
    if output.exists(): raise ValueError('No existing training attempt can be overwritten or retried automatically')
    output.mkdir(parents=True)
    rr.configure(seed, config['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    model = SetAttentionSurrogate().to(config['device'])
    assert parameter_count(model) == plan['parameters'] == 194273
    metadata = {'purpose':PURPOSE,'epistemic_status':plan['epistemic_status'],'model_version':MODEL_VERSION,
                'mode':MODEL_NAME,'protocol':protocol,'seed':seed,'plan_sha256':rr.digest(plan_path),
                'original_plan_sha256':ORIGINAL_PLAN_SHA,'dataset_sha256':plan['manifest_sha256'],
                'split_sha256':rr.digest(split_path),'target_version':rr.TARGET_VERSION,'feature_version':rr.FEATURE_VERSION,
                'feature_tensor_sha256':manifest['feature_tensor_sha256'],'target_tensor_sha256':manifest['target_tensor_sha256'],
                'configuration':config,'architecture':ARCHITECTURE,'parameters':parameter_count(model),
                'original_source_sha256':plan['original_source_sha256'],'extension_source_sha256':plan['extension_source_sha256'],
                'execution_authorization_sha256':gate['authorization_sha256'],
                'backend_configuration':plan['backend_configuration'],
                'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
    write_json(output/'run_metadata.json', metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs_max'])
    rng = np.random.default_rng(seed); best, stale, best_epoch = math.inf, 0, 0
    rows=[]; start=time.perf_counter(); graphs,targets=data['graphs'],data['targets']
    for epoch in range(1,config['epochs_max']+1):
        model.train(); ids=list(rng.permutation(split['train'])); total_loss=0.
        for k in range(0,len(ids),config['batch_size']):
            group=ids[k:k+config['batch_size']]; optimizer.zero_grad(set_to_none=True)
            prediction=model(*rr.collate([graphs[s] for s in group],config['device'])).squeeze(-1)
            truth=torch.stack([targets[s] for s in group]).to(config['device'])
            loss=torch.nn.functional.mse_loss(prediction,truth)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip_norm'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            optimizer.step(); total_loss+=float(loss.detach())*len(group)
        vp=rr.predict(model,graphs,split['validation'],config['batch_size'],config['device'])
        vy=np.stack([targets[s].numpy() for s in split['validation']]); val_mse=float(np.square(vp-vy).mean())
        improved=new_best(val_mse,best)
        rows.append({'epoch':epoch,'train_mse':total_loss/len(ids),'validation_mse':val_mse,
                     'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_s':time.perf_counter()-start})
        scheduler.step()
        if improved:
            best,stale,best_epoch=val_mse,0,epoch
            torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'best_validation_mse':best,
                        'dataset_sha256':plan['manifest_sha256'],'split_sha256':rr.digest(split_path),
                        'plan_sha256':rr.digest(plan_path)},output/'best.pt')
        else:stale+=1
        with (output/'learning_curve.csv').open('w',newline='',encoding='utf-8') as stream:
            w=csv.DictWriter(stream,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
        write_json(output/'status.json',{'status':'RUNNING_EXTENSION','epoch':epoch,'best_epoch':best_epoch,'elapsed_s':time.perf_counter()-start})
        if stale>=config['patience']:break
    model.load_state_dict(torch.load(output/'best.pt',map_location=config['device'],weights_only=True)['model_state_dict'])
    training_s=time.perf_counter()-start
    test_ids=split['test'];truth=np.stack([targets[s].numpy() for s in test_ids])
    prediction=rr.predict(model,graphs,test_ids,config['batch_size'],config['device'])
    np.savez_compressed(output/'test_predictions.npz',sample_ids=np.asarray(test_ids),time_s=data['time_s'].numpy(),truth=truth,predictions=prediction)
    result=rr.metrics(truth,prediction)
    result.update(purpose=PURPOSE,epistemic_status=plan['epistemic_status'],mode=MODEL_NAME,protocol=protocol,seed=seed,
                  best_epoch=best_epoch,epochs_run=epoch,best_validation_mse=best,training_wall_s=training_s,
                  checkpoint_sha256=rr.digest(output/'best.pt'),topology_results={},
                  edge_view_scope='Raw edge tensors ignored by this set architecture; edge-removal invariance checked in software.')
    write_json(output/'result.json',result)
    write_json(output/'status.json',{'status':'COMPLETE_EXPLORATORY_EXTENSION','best_epoch':best_epoch,'epochs_run':epoch,'training_wall_s':training_s})


def queue(plan_path, authorization_path):
    plan,gate=execution_gate(plan_path,authorization_path)
    if gate['status']!='AUTHORIZED_EXTENSION_EXECUTION':print(json.dumps(gate));return 2
    output=safe_output(plan['runs_root'])
    if output.exists():raise ValueError('New queue root must not already exist')
    output.mkdir()
    status={'status':'RUNNING_EXPLORATORY_EXTENSION','pid':os.getpid(),'plan_sha256':rr.digest(plan_path),
            'planned_runs':60,'completed':[],'active':None,'epistemic_status':plan['epistemic_status']}
    try:
        for protocol in plan['protocols']:
            for seed in plan['seeds']:
                status['active']={'protocol':protocol,'model':MODEL_NAME,'seed':seed};write_json(output/'queue_status.json',status)
                command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'train','--plan',str(plan_path),
                         '--authorization',str(authorization_path),'--protocol',protocol,'--seed',str(seed)]
                with (output/'queue_console.log').open('a',encoding='utf-8') as stream:
                    run=subprocess.run(command,stdout=stream,stderr=stream,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if run.returncode:raise RuntimeError('Training cell failed; no automatic retry: '+str(status['active']))
                status['completed'].append(status['active']);write_json(output/'queue_status.json',status)
        status['status']='COMPLETE_60_EXPLORATORY_EXTENSION';status['active']=None
    except BaseException as exc:
        status['status']='ERROR_STOPPED_EXTENSION';status['error']=repr(exc);raise
    finally:
        status['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());write_json(output/'queue_status.json',status)
    return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check-ready','train','queue']);p.add_argument('--plan',required=True)
    p.add_argument('--authorization');p.add_argument('--protocol');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.command=='check-ready':
        _,gate=execution_gate(a.plan,a.authorization);print(json.dumps(gate));return 0 if gate['status']=='AUTHORIZED_EXTENSION_EXECUTION' else 2
    if a.command=='train':train_one(a.plan,a.authorization,a.protocol,a.seed);return 0
    return queue(a.plan,a.authorization)


if __name__=='__main__':raise SystemExit(main())
