"""Re-score every dense prediction and reference-grid effect from the arrays."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
def mean(v):return float(np.mean(v)) if np.size(v) else None
def score(p,y,z,L,mask=None):
    ym=y.min(1);pm=p.min(1);mask=(ym>=.05)&(ym<=.95) if mask is None else mask
    ix=p.argmin(1);selected=np.take_along_axis(y,ix[:,None,:],1).squeeze(1);regret=(selected-ym)*100
    near_y=y<=ym[:,None,:]+.01;near_p=p<=pm[:,None,:]+.01;con=(p-p.mean(1,keepdims=True))-(y-y.mean(1,keepdims=True))
    dist=np.min(np.where(near_y,abs(z[None,:,None]-z[ix][:,None,:]),np.inf),axis=1)
    return dict(section_mae_pp=mean(abs(p-y))*100,member_mae_pp=mean(abs(pm-ym))*100,
      active_time_directions=int(mask.sum()),active_member_mae_pp=mean(abs(pm-ym)[mask]*100),
      active_contrast_mae_pp=mean(abs(con)[np.broadcast_to(mask[:,None,:],con.shape)]*100),
      active_regret_pp=mean(regret[mask]),active_regret_p95_pp=float(np.quantile(regret[mask],.95)) if mask.any() else None,
      active_distance_mm=mean(dist[mask]*L*1000),active_near_iou=mean(((near_y&near_p).sum(1)/(near_y|near_p).sum(1))[mask]))
def profiles(z):return dict(constant=np.ones_like(z),udl=4*z*(1-z),eccentric=np.where(z<=.35,z/.35,(1-z)/.65),two_point=np.minimum(np.minimum(3*z,3*(1-z)),1))
def event(v,t):
    hit=(v<=0).any(0);return hit,np.where(hit,t[(v<=0).argmax(0)],np.nan)
def main():
    d=R/'dense_queries';expected=pd.read_csv(d/'combined_dense_metrics.csv');cases={r['case_id']:r for r in map(json.loads,(R/'data/case_geometry_manifest.jsonl').read_text(encoding='utf-8').splitlines())};report=[];maximum=0
    for cid,part in expected.groupby('case_id'):
        arrays={}
        for grid in ['31','61','121','augmented']:
            with np.load(d/'reference_sections'/cid/f'{grid}_xy2mm.npz') as f:arrays[grid]={k:f[k] for k in ['eta','section_z_over_L','time_s']}
        ym=arrays['121']['eta'].min(1);active=(ym>=.05)&(ym<=.95)
        for _,row in part.iterrows():
            grid=str(row.grid);source=d/'predictions'/f'{cid}_{row.model}_{int(row.seed)}_{grid}.npz'
            with np.load(source) as f:p=f['predicted']
            a=arrays[grid];s=score(p,a['eta'],a['section_z_over_L'],cases[cid]['beam_LWH_m'][0]);s.update({'common_mask_'+k:v for k,v in score(p,a['eta'],a['section_z_over_L'],cases[cid]['beam_LWH_m'][0],active).items() if k.startswith('active_')})
            for k,v in s.items():
                if v is None:assert pd.isna(row[k])
                else:err=abs(v-row[k]);maximum=max(maximum,err);assert err<3e-5,(cid,row.model,grid,k,err)
            report.append(dict(case_id=cid,model=row.model,seed=int(row.seed),grid=grid,**s))
    sampling=[];delays=[];extras=[]
    for cid in expected.case_id.unique():
        pair=[]
        for grid in [31,121]:
            with np.load(d/'reference_sections'/cid/f'{grid}_xy2mm.npz') as f:pair.append({k:f[k] for k in ['eta','section_z_over_L','time_s']})
        a,b=pair;loss=(a['eta'].min(1)-b['eta'].min(1))*100;sampling.append(dict(case_id=cid,p95_pp=float(np.quantile(loss,.95)),max_pp=float(loss.max())))
        for name in profiles(a['section_z_over_L']):
            for lam in [.2,.4,.6]:
                ae,at=event((a['eta']-lam*profiles(a['section_z_over_L'])[name][None,:,None]).min(1),a['time_s']);be,bt=event((b['eta']-lam*profiles(b['section_z_over_L'])[name][None,:,None]).min(1),b['time_s'])
                delays.extend((at-bt)[ae&be].tolist());extras.extend(dict(case_id=cid,profile=name,lambda_=lam,direction=int(k)) for k in np.flatnonzero(~ae&be))
    out=R/'reproduced';out.mkdir(exist_ok=True);pd.DataFrame(report).to_csv(out/'dense_recomputed.csv',index=False)
    summary=dict(predictions=len(report),score_max_abs_difference=maximum,sampling=sampling,paired_events=len(delays),event_delay_p95_s=float(np.quantile(delays,.95)),event_delay_max_s=float(max(delays)),extra_events=extras,pass_=True)
    assert summary['paired_events']==278 and len(extras)==3 and abs(summary['event_delay_p95_s']-71.5)<1e-8 and summary['event_delay_max_s']==190
    (out/'dense_reproduction_report.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
