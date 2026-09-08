"""Independent arithmetic audit of the completed IID protocol, not model selection."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import csv,json,hashlib,collections
from datetime import datetime,timezone
import numpy as np
import torch
from scipy.stats import rankdata

HERE=Path(__file__).resolve().parent
SCI=HERE.parents[2]
BASE=SCI/'experiments/research_v08'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):
    if a is None or b is None:assert a is None and b is None
    else:assert np.allclose(a,b,atol=2e-13,rtol=2e-12),(a,b)
def pair_rank(a,b):
    ra,rb=rankdata(a),rankdata(b)
    rho=None if np.ptp(ra)==0 or np.ptp(rb)==0 else float(np.corrcoef(ra,rb)[0,1])
    i,j=np.triu_indices(len(a),1); da=a[i]-a[j];db=b[i]-b[j]; keep=da!=0
    agreement=np.mean(np.where(db[keep]==0,.5,(np.sign(da[keep])==np.sign(db[keep])).astype(float))) if keep.any() else None
    return rho,agreement,int(keep.sum())

def main():
    plan_path=BASE/'comparison_plan_v08_420.json'; plan=read(plan_path)
    manifest_path=Path(plan['manifest_path']); manifest=read(manifest_path)
    split=read(Path(plan['protocols']['iid997']['split_path']))
    ids=split['test'];cases={c['sample_id']:c for c in manifest['cases']}
    assert len(ids)==len({cases[i]['geometry_id'] for i in ids})==160
    assert plan['seeds']==[42,43,44,45,46] and len(plan['models'])==7
    summary_path=BASE/'evaluation/iid997/summary.json'; summary=read(summary_path)
    eng_path=HERE/'iid997_engineering_proxy_ci.json'; eng=read(eng_path)
    assert summary['status']=='COMPLETE_PROTOCOL_35_RUNS'
    assert eng['status']=='COMPLETE_35_RUN_PAIRED_ENGINEERING_PROXY_CI'
    assert summary['test_case_count']==eng['test_case_count']==160 and eng['completed_runs_checked']==35
    tensors=torch.load(manifest_path.parent/manifest['tensor_path'],map_location='cpu',weights_only=True)
    truth=np.stack([tensors['targets'][i].numpy() for i in ids]).astype(float)
    times=tensors['time_s'].numpy(); assert np.array_equal(times,np.arange(61)*60)
    with (BASE/'evaluation/iid997/per_case_metrics.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    case_rows={(r['model'],int(r['seed']),r['sample_id']):r for r in rows}
    assert len(case_rows)==len(rows)==7*5*160
    per_seed={(r['model'],r['seed']):r for r in eng['per_seed_event_counts_and_metrics']}
    assert len(per_seed)==35
    predictions={};maes={};proxy_errors={};hashes={};models=[];class_counts=[]
    families=np.asarray([cases[i]['fire_family'] for i in ids])
    for mode in plan['models']:
        collected=[];metric_rows=[]
        for seed in plan['seeds']:
            folder=Path(plan['runs_root'])/'iid997'/mode/f'seed_{seed}'
            assert read(folder/'status.json')['status']=='COMPLETE_ARCHIVAL_RESEARCH'
            meta=read(folder/'run_metadata.json')
            assert meta['plan_sha256']==sha(plan_path) and meta['mode']==mode and meta['seed']==seed and meta['protocol']=='iid997'
            path=folder/'test_predictions.npz'
            hashes[str(path)]=sha(path); assert hashes[str(path)]==eng['source_prediction_sha256'][str(path)]
            with np.load(path,allow_pickle=False) as z:
                assert z['sample_ids'].tolist()==ids and np.array_equal(z['time_s'],times) and np.array_equal(z['truth'],truth)
                p=z['predictions'].astype(float)
            assert p.shape==(160,61) and np.isfinite(p).all();collected.append(p)
            error=p-truth; cm=np.abs(error).mean(1); maximum=np.maximum(error,0).max(1)
            metrics={'equal_case_MAE':float(cm.mean()),'equal_case_RMSE':float(np.sqrt(np.mean(error**2))),
              'mean_case_RMSE':float(np.mean(np.sqrt(np.mean(error**2,axis=1)))),
              'R2_pooled':float(1-np.sum(error**2)/np.sum((truth-truth.mean())**2)),
              'maximum_positive_error':float(maximum.max()),'maximum_absolute_error':float(np.abs(error).max()),
              'case_time_positive_error_q95':float(np.quantile(np.maximum(error,0),.95)),
              'case_time_positive_error_q99':float(np.quantile(np.maximum(error,0),.99))}
            for q in (50,90,95,99):
                metrics[f'per_case_MAE_q{q}']=float(np.quantile(cm,q/100))
                metrics[f'per_case_max_positive_error_q{q}']=float(np.quantile(maximum,q/100))
            metric_rows.append(metrics)
            for j,sid in enumerate(ids):
                row=case_rows[(mode,seed,sid)]
                close(float(row['MAE']),cm[j]);close(float(row['RMSE']),np.sqrt(np.mean(error[j]**2)));close(float(row['maximum_overprediction']),maximum[j])
                assert row['geometry_id']==cases[sid]['geometry_id']
            record=per_seed[(mode,seed)]
            targets={'final':(truth[:,-1],p[:,-1]),'time_average':(np.trapezoid(truth,times,axis=1)/3600,np.trapezoid(p,times,axis=1)/3600)}
            for kind,(a,b) in targets.items():
                close(record['summaries'][kind]['MAE'],np.abs(b-a).mean());close(record['summaries'][kind]['bias'],(b-a).mean())
                rho,agreement,n=pair_rank(a,b)
                close(record['rankings'][kind]['spearman'],rho);close(record['rankings'][kind]['pairwise_order_agreement'],agreement)
                assert record['rankings'][kind]['eligible_pairs']==n
                for fam in sorted(set(families)):
                    mask=families==fam;rho,agreement,n=pair_rank(a[mask],b[mask])
                    close(record['within_family_rankings'][fam][kind]['spearman'],rho)
                    close(record['within_family_rankings'][fam][kind]['pairwise_order_agreement'],agreement)
            for q in (.8,.6):
                # Explicit first-hit scan independently handles right censoring.
                at=np.asarray([next((times[k] for k,v in enumerate(r) if v<=q),np.inf) for r in truth])
                pt=np.asarray([next((times[k] for k,v in enumerate(r) if v<=q),np.inf) for r in p])
                ae,pe=np.isfinite(at),np.isfinite(pt);both=ae&pe;d=pt[both]-at[both]
                count={'reference_events':int(ae.sum()),'reference_censored':int((~ae).sum()),'predicted_events':int(pe.sum()),'predicted_censored':int((~pe).sum()),
                  'true_positive':int((ae&pe).sum()),'false_negative':int((ae&~pe).sum()),'false_positive':int((~ae&pe).sum()),'true_negative':int((~ae&~pe).sum()),'both_event_count':int(both.sum())}
                recross=int(sum(np.any(r[times>first]>q) for r,first in zip(p,pt) if np.isfinite(first)))
                for key,value in count.items():assert record['crossings'][str(q)][key]==value
                assert record['crossings'][str(q)]['recross_above_threshold_count']==recross
                assert record['crossings'][str(q)]['optimistic_late_or_missed_count']==count['false_negative']+int((d>0).sum())
                close(record['crossings'][str(q)]['conditional_timing_MAE_s'],np.abs(d).mean() if len(d) else None)
                close(record['crossings'][str(q)]['conditional_timing_median_abs_s'],np.median(np.abs(d)) if len(d) else None)
                close(record['crossings'][str(q)]['conditional_timing_bias_s'],d.mean() if len(d) else None)
                class_counts.append(dict(model=mode,seed=seed,threshold=q,**count,late_or_missed=record['crossings'][str(q)]['optimistic_late_or_missed_count'],recross=recross))
            close(record['positive_step_fraction'],(np.diff(p,axis=1)>1e-6).mean())
        predictions[mode]=np.stack(collected);maes[mode]=np.abs(predictions[mode]-truth).mean(2)
        sm=next(r for r in summary['model_summaries'] if r['model']==mode)
        for key in metric_rows[0]:
            vv=[r[key] for r in metric_rows];close(sm['metrics'][key]['mean'],np.mean(vv));close(sm['metrics'][key]['sample_sd'],np.std(vv,ddof=1))
        proxy_errors[mode]={
          'final_MAE':np.abs(predictions[mode][:,:,-1]-truth[:,-1]),
          'time_average_MAE':np.abs(np.trapezoid(predictions[mode],times,axis=2)/3600-np.trapezoid(truth,times,axis=1)/3600)}
        models.append(dict(model=mode,MAE=sm['metrics']['equal_case_MAE'],per_case_MAE_q95=sm['metrics']['per_case_MAE_q95'],per_case_max_positive_error_q95=sm['metrics']['per_case_max_positive_error_q95']))
    contrasts={}
    geometry=np.asarray([cases[i]['geometry_id'] for i in ids]);order=np.argsort(geometry)
    for tag,a,b in [('graph_vs_capacity_matched','capacity_matched_deepsets','gnn'),('mean_vs_sum','gnn','gnn_mean')]:
        difference=maes[a]-maes[b];reported=summary['paired_contrasts'][tag]
        close(reported['point_estimate_equal_case_seed_mean'],difference.mean());close(reported['paired_effect_by_seed'],difference.mean(1))
        rng=np.random.default_rng(20260907);boot={k:[] for k in ('geometry_only','seed_only','two_way')}
        for _ in range(5000):
            sw=rng.multinomial(5,np.ones(5)/5);gw=rng.multinomial(160,np.ones(160)/160)
            ordered=difference[:,order]
            boot['geometry_only'].append(float(np.sum(ordered*gw)/(5*160)))
            boot['seed_only'].append(float(np.sum(difference.mean(1)*sw)/5))
            boot['two_way'].append(float(np.sum(ordered*sw[:,None]*gw[None,:])/(5*160)))
        for key,vals in boot.items():close(reported['intervals'][key]['percentile_95'],np.quantile(vals,[.025,.975]));close(reported['intervals'][key]['bootstrap_sd'],np.std(vals,ddof=1))
        for metric in ('final_MAE','time_average_MAE'):
            d=proxy_errors[a][metric]-proxy_errors[b][metric];rng=np.random.default_rng(20260907);rep=[]
            for _ in range(1000):
                ix=rng.integers(0,160,160);si=rng.integers(0,5,5);rep.append(d[si[:,None],ix[None,:]].mean())
            stated=eng['paired_contrasts'][tag]['metrics'][metric]
            close(stated['point_baseline_minus_candidate'],d.mean());close(stated['percentile_95_paired_difference'],np.quantile(rep,[.025,.975]))
        contrasts[tag]={'point':reported['point_estimate_equal_case_seed_mean'],'seed_effects':reported['paired_effect_by_seed'],
          'two_way95':reported['intervals']['two_way']['percentile_95'],'candidate_better_seeds':int((difference.mean(1)>0).sum()),
          'relative_candidate_MAE_reduction':float(difference.mean()/maes[a].mean())}
    with eng_path.with_suffix('.csv').open(encoding='utf-8') as f:ci_rows=list(csv.DictReader(f))
    assert len(ci_rows)==48 and len({(r['contrast'],r['metric']) for r in ci_rows})==48
    for r in ci_rows:
        v=eng['paired_contrasts'][r['contrast']]['metrics'][r['metric']]
        close(float(r['effect']) if r['effect'] else None,v['point_baseline_minus_candidate'])
        close([float(r['ci95_low']),float(r['ci95_high'])] if r['ci95_low'] else None,v['percentile_95_paired_difference'])
    report={'status':'IID_ARITHMETIC_DEVELOPMENT_AUDIT_PASSED_NOT_FULL_MATRIX_OR_SUBMISSION_REVIEW',
      'created_utc':datetime.now(timezone.utc).isoformat(),'runs':35,'seeds':plan['seeds'],'independent_geometries':160,'case_time_grid':61,
      'per_case_CSV_rows_checked':len(rows),'model_metric_fields_checked':16,
      'primary_bootstrap_independently_recomputed':{'contrasts':2,'methods':3,'repeats':5000},
      'secondary_checks':'All 35 per-seed final/J errors, full/within-family rank point estimates, first-grid-event counts, censoring/timing, recross and monotonicity; 4 final/J paired CIs independently recomputed; all48 CSV rows reconcile. Rank/timing bootstrap algorithms reviewed previously with analytic checks, not all48 repeated here.',
      'models':models,'primary_contrasts':contrasts,'threshold_class_counts':class_counts,
      'source_hashes':{str(p):sha(p) for p in (plan_path,summary_path,eng_path,Path(__file__))},'prediction_hashes':hashes,
      'future_protocols_read':False,'model_selection_changes':False,'submission_pass':False}
    (HERE/'iid997_independent_results_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','runs','independent_geometries','per_case_CSV_rows_checked','primary_contrasts')},indent=2))

if __name__=='__main__':main()
