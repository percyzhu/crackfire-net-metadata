"""Complete-protocol same-weight graph-view arithmetic audit; no refitting."""
from pathlib import Path
import argparse,csv,json,hashlib,collections
from datetime import datetime,timezone
import numpy as np
from audit_completed_protocol import BASE,HERE,read,sha,close

def distances(case):
    b=case['beam'];points=[]
    for c in case['cracks']:
        f,z,h,d=int(c['face']),c['z'],c['h'],c['d'];q=-h if f in (0,3) else h
        points.append([q,b['H']/2-d/2,z] if f==0 else [q,-b['H']/2+d/2,z] if f==1 else [b['W']/2-d/2,q,z] if f==2 else [-b['W']/2+d/2,q,z])
    p=np.asarray(points)
    return np.linalg.norm(p[:,None,:]-p[None,:,:],axis=2)/np.linalg.norm([b[k] for k in ('L','W','H')])

def counts(case,radius,k):
    d=distances(case);n=len(d);off=~np.eye(n,dtype=bool)
    r=(d<=radius)&off
    neighbors=np.zeros((n,n),bool)
    if n>1:
        for i in range(n):
            permitted=[j for j in range(n) if i!=j]
            cut=sorted(d[i,j] for j in permitted)[min(k,n-1)-1]
            neighbors[i]=[(j!=i and d[i,j]<=cut) for j in range(n)]
    knn=neighbors|neighbors.T
    return {rule:{'directed_edges':int(a.sum()),'isolated_nodes':int((a.sum(1)==0).sum()),'min_degree':int(a.sum(1).min()),'max_degree':int(a.sum(1).max())} for rule,a in [('complete',off),('radius',r),('symmetric_knn',knn)]}

def main(protocol,output):
    plan_path=BASE/'comparison_plan_v08_420.json';plan=read(plan_path)
    assert plan['seeds']==[42,43,44,45,46] and len(plan['models'])==7
    required=[Path(plan['runs_root'])/protocol/m/f'seed_{s}' for m in plan['models'] for s in plan['seeds']]
    pending=[p for p in required if not (p/'status.json').exists() or read(p/'status.json').get('status')!='COMPLETE_ARCHIVAL_RESEARCH']
    if pending:print(json.dumps({'status':'WAITING_FOR_COMPLETE35','pending':len(pending),'scores_read':False}));return
    table_path=BASE/'evaluation'/protocol/'topology_by_seed.csv'
    if not table_path.exists():print(json.dumps({'status':'WAITING_FOR_AGGREGATED_TOPOLOGY','scores_read':False}));return
    output=Path(output).resolve() if output else HERE/f'{protocol}_same_weight_topology_audit.json'
    if not output.is_relative_to(HERE) or output.exists():raise ValueError('Exclusive-create reviewer output required')
    manifest_path=Path(plan['manifest_path']);manifest=read(manifest_path);assert sha(manifest_path)==plan['manifest_sha256']
    split_path=Path(plan['protocols'][protocol]['split_path']);split=read(split_path);assert sha(split_path)==plan['protocols'][protocol]['split_sha256']
    cases={c['sample_id']:c for c in manifest['cases']};ids=split['test'];n=len(ids)
    # Independent training-only radius calculation; no fit on validation/test data.
    medians=[]
    for sid in split['train']:
        d=distances(cases[sid])
        if len(d)>1:medians.append(float(np.median(d[np.triu_indices(len(d),1)])))
    radius=float(np.median(medians));assert radius==split['topology_recipe']['radius']
    k=split['topology_recipe']['symmetric_knn_k'];structures={sid:counts(cases[sid],radius,k) for sid in ids}
    topology_counts={r:{'directed_edges':sum(v[r]['directed_edges'] for v in structures.values()),
      'isolated_nodes':sum(v[r]['isolated_nodes'] for v in structures.values()),
      'min_degree':min(v[r]['min_degree'] for v in structures.values()),'max_degree':max(v[r]['max_degree'] for v in structures.values())} for r in ('complete','radius','symmetric_knn')}
    with table_path.open(encoding='utf-8') as f:table=list(csv.DictReader(f))
    byrow={(r['model'],int(r['seed']),r['rule']):r for r in table};assert len(table)==len(byrow)==30
    hashes={};evaluated=[];truth=None
    for mode in ('gnn','gnn_zero_edge_features','gnn_mean'):
        for seed in plan['seeds']:
            folder=Path(plan['runs_root'])/protocol/mode/f'seed_{seed}'
            result=read(folder/'result.json');meta=read(folder/'run_metadata.json');checkpoint=sha(folder/'best.pt')
            assert meta['plan_sha256']==sha(plan_path) and meta['protocol']==protocol and meta['mode']==mode and meta['seed']==seed
            assert result['checkpoint_sha256']==checkpoint
            with np.load(folder/'test_predictions.npz',allow_pickle=False) as z:
                assert z['sample_ids'].tolist()==ids and np.array_equal(z['time_s'],np.arange(61)*60)
                y=z['truth'].astype(float);complete32=z['predictions'].copy();complete=complete32.astype(float)
            if truth is None:truth=y
            assert np.array_equal(y,truth)
            base_mae=float(np.abs(complete-y).mean());close(base_mae,result['equal_case_MAE'])
            for rule in ('radius','symmetric_knn'):
                path=folder/f'topology_{rule}.npz';hashes[str(path)]=sha(path)
                with np.load(path,allow_pickle=False) as z:
                    assert z['sample_ids'].tolist()==ids and np.array_equal(z['time_s'],np.arange(61)*60) and np.array_equal(z['truth'],truth)
                    sparse32=z['predictions'].copy();p=sparse32.astype(float)
                assert p.shape==(n,61) and np.isfinite(p).all()
                declared=result['topology_results'][rule];row=byrow[(mode,seed,rule)]
                assert declared['same_trained_checkpoint'] is True and declared['refitting'] is False and declared['checkpoint_sha256']==checkpoint
                assert row['same_trained_checkpoint']=='True' and row['refitting']=='False' and row['checkpoint_sha256']==checkpoint
                assert declared['total_edges']==int(row['directed_edge_total'])==topology_counts[rule]['directed_edges']
                e=p-y;mae=float(np.abs(e).mean());change=float(np.abs(sparse32-complete32).mean())
                close(mae,declared['equal_case_MAE']);close(mae,float(row['equal_case_MAE']))
                close(change,declared['mean_abs_prediction_change']);close(change,float(row['mean_abs_prediction_change']))
                close(mae-base_mae,float(row['MAE_change_vs_complete']))
                close(float(np.sqrt(np.mean(e**2))),float(row['equal_case_RMSE']))
                evaluated.append({'model':mode,'seed':seed,'rule':rule,'complete_MAE':base_mae,'view_MAE':mae,
                  'MAE_change_vs_complete':mae-base_mae,'mean_absolute_prediction_change_float32':change,
                  'max_absolute_prediction_change':float(np.abs(p-complete).max()),'checkpoint_sha256':checkpoint,
                  'same_checkpoint_binding':True,'refitting':False})
    summaries=[]
    for mode in ('gnn','gnn_zero_edge_features','gnn_mean'):
        for rule in ('radius','symmetric_knn'):
            r=[x for x in evaluated if x['model']==mode and x['rule']==rule]
            s={'model':mode,'rule':rule,'seeds':plan['seeds'],'seed_count':5,'test_cases':n}
            for metric in ('complete_MAE','view_MAE','MAE_change_vs_complete','mean_absolute_prediction_change_float32'):
                v=[x[metric] for x in r];s[metric]={'mean':float(np.mean(v)),'seed_sample_sd':float(np.std(v,ddof=1))}
            s['view_lower_MAE_seeds']=sum(x['MAE_change_vs_complete']<0 for x in r);summaries.append(s)
    report={'status':'COMPLETE_PROTOCOL_TOPOLOGY_BINDINGS_AND_ARITHMETIC_CHECKED','protocol':protocol,'test_cases':n,
      'created_utc':datetime.now(timezone.utc).isoformat(),'graph_runs':15,'paired_sparse_views':30,
      'radius_training_only_independent':radius,'symmetric_knn_k':k,'topology_counts':topology_counts,
      'per_seed':evaluated,'five_seed_summaries':summaries,'prediction_hashes':hashes,
      'source_hashes':{str(p):sha(p) for p in (plan_path,manifest_path,split_path,table_path,Path(__file__))},
      'proof_scope':'Independent source geometry/edge counts and artifact/CSV arithmetic; same-checkpoint hash declarations plus inspected frozen runner code. CPU/GPU prediction replay remains the upstream aggregate --replay audit responsibility.',
      'interpretation':'Same learned weights on complete/radius/symmetric-kNN representations of identical physical cases. This is representation-connectivity stress, not independent FE mesh/discretization transfer or altered physical crack interactions.',
      'source_confound':'LCRO high-N cases are all batch005; count shift is not isolated from source-batch/exposure mixture.' if protocol=='lcro_9_15' else None,
      'other_protocol_predictions_read':False,'model_refitted':False,'journal_or_submission_pass':False}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','protocol','test_cases','topology_counts','five_seed_summaries')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output');a=p.parse_args();main(a.protocol,a.output)
