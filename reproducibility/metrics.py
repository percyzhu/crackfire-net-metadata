"""Frozen spatial/engineering metrics. Prepare on fixtures/validation, then evaluate fixed checkpoints."""
from pathlib import Path
import argparse,json,csv,sys
import numpy as np
import torch
R=Path(__file__).resolve().parent
GROUPS=('validation','count','section','length','joint','reserved')
EPS=(.005,.01,.02)
LAM=(.2,.4,.6)

def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def caseavg(a,mask):
    a=np.asarray(a);mask=np.broadcast_to(mask,a.shape)
    n=mask.reshape(len(a),-1).sum(1)
    s=np.where(mask,a,0).reshape(len(a),-1).sum(1)
    return np.divide(s,n,out=np.full(len(a),np.nan),where=n>0)
def finiteavg(a):
    a=np.asarray(a);return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else None
def take(y,idx):return np.take_along_axis(y,idx[:,:,None,:],axis=2).squeeze(2)
def profiles(z):
    a=.35
    return dict(constant=np.ones_like(z),udl=4*z*(1-z),eccentric=np.where(z<=a,z/a,(1-z)/(1-a)),two_point=np.minimum(np.minimum(3*z,3*(1-z)),1))
def events(g,t):
    occurs=(g<=0).any(1);idx=(g<=0).argmax(1)
    return occurs,np.where(occurs,t[idx],np.nan)

def metrics(p,y,ids,t):
    z=np.linspace(1/60,59/60,31);n=len(y);out=[];pc=[];ym=y.min(2);pm=p.min(2)
    c=(p-p.mean(2,keepdims=True))-(y-y.mean(2,keepdims=True))
    idx=p.argmin(2);regret=take(y,idx)-ym
    mid=np.full(idx.shape,15);mr=take(y,mid)-ym
    uniform=np.broadcast_to(p.mean(2,keepdims=True),p.shape)
    masks={'all':np.ones_like(ym,dtype=bool),'active':(ym>=.05)&(ym<=.95)}
    for stratum,mask in masks.items():
        fields={
          'section_mae_pp':caseavg(abs(p-y)*100,mask[:,:,None,:]),
          'section_mse_pp2':caseavg((p-y)**2*10000,mask[:,:,None,:]),
          'member_mae_pp':caseavg(abs(pm-ym)*100,mask),
          'member_mse_pp2':caseavg((pm-ym)**2*10000,mask),
          'contrast_mae_pp':caseavg(abs(c)*100,mask[:,:,None,:]),
          'regret_pp':caseavg(regret*100,mask),
          'midspan_regret_pp':caseavg(mr*100,mask),
          'uniform_section_mae_pp':caseavg(abs(uniform-y)*100,mask[:,:,None,:]),
          'uniform_member_mae_pp':caseavg(abs(p.mean(2)-ym)*100,mask),
          'uniform_contrast_mae_pp':caseavg(abs(y-y.mean(2,keepdims=True))*100,mask[:,:,None,:]),
        }
        for e in EPS:
            a=p<=pm[:,:,None,:]+e;b=y<=ym[:,:,None,:]+e
            inter=(a&b).sum(2);union=(a|b).sum(2)
            # Distance to the reference near-minimum set; exact argmin ties need no arbitrary class.
            dist=np.min(np.where(b,abs(z[None,None,:,None]-z[idx][:,:,None,:]),np.inf),2)
            for name,v in dict(precision=inter/a.sum(2),recall=inter/b.sum(2),iou=inter/union,selected_fraction=a.mean(2),reference_fraction=b.mean(2),distance_over_L=dist,selection_hit=take(b,idx)).items():
                fields[f'{name}_eps{e:g}']=caseavg(v,mask)
        row=dict(stratum=stratum,cases=n,contributing_cases=int(np.isfinite(fields['regret_pp']).sum()),case_time_directions=int(mask.sum()),fraction=float(mask.mean()))
        row.update({k:finiteavg(v) for k,v in fields.items()})
        row['section_rmse_pp']=float(np.sqrt(row['section_mse_pp2'])) if row['section_mse_pp2'] is not None else None
        row['member_rmse_pp']=float(np.sqrt(row['member_mse_pp2'])) if row['member_mse_pp2'] is not None else None
        out.append(row)
        for i,id_ in enumerate(ids):pc.append(dict(case_id=str(id_),stratum=stratum,**{k:float(v[i]) if np.isfinite(v[i]) else None for k,v in fields.items()}))
    demand=[];dpc=[]
    for name,m in profiles(z).items():
        for lam in LAM:
            gp=p-lam*m[None,None,:,None];gy=y-lam*m[None,None,:,None]
            gpm=gp.min(2);gym=gy.min(2);ix=gp.argmin(2)
            dr=take(gy,ix)-gym
            pe,pt=events(gpm,t);ye,yt=events(gym,t);both=pe&ye
            # Per-case, then macro-average; never-events remain censored, never assigned 3600 s.
            fields=dict(margin_mae_pp=caseavg(abs(gpm-gym)*100,np.ones_like(gym,dtype=bool)),control_regret_pp=caseavg(dr*100,np.ones_like(gym,dtype=bool)),
              event_time_mae_s=caseavg(np.nan_to_num(abs(pt-yt)),both),
              event_false_negative_rate=caseavg(~pe,ye),event_false_positive_rate=caseavg(pe,~ye),event_agreement=caseavg(pe==ye,np.ones_like(ye,dtype=bool)),
              unreported_exceedance_duration_s=caseavg(np.maximum(np.where(pe,pt,t[-1])-np.nan_to_num(yt,nan=t[-1]),0),ye))
            row=dict(profile=name,lambda_=lam,cases=n,true_events=int(ye.sum()),predicted_events=int(pe.sum()),both_events=int(both.sum()),missed_events=int((ye&~pe).sum()),false_events=int((~ye&pe).sum()),right_censored_reference=int((~ye).sum()),**{k:finiteavg(v) for k,v in fields.items()})
            demand.append(row)
            for i,id_ in enumerate(ids):dpc.append(dict(case_id=str(id_),profile=name,lambda_=lam,**{k:float(v[i]) if np.isfinite(v[i]) else None for k,v in fields.items()}))
    return out,pc,demand,dpc

def checks():
    z=np.linspace(1/60,59/60,31);t=np.array([0.,10.,20.]);y=np.ones((2,3,31,4),np.float64)
    y[0,1:,5,:]=.4;y[1,1:,7,:]=.3
    rows,pc,dem,_=metrics(y.copy(),y,np.array(['a','b']),t)
    assert all(r['regret_pp']==0 and r['contrast_mae_pp']==0 and r['iou_eps0.01']==1 for r in rows)
    assert all(r['margin_mae_pp']==0 and r['control_regret_pp']==0 and r['event_agreement']==1 for r in dem)
    occ,tt=events(np.array([[[1.],[.1],[-.1]],[[1.],[.5],[.2]]]),t)
    assert occ[:,0].tolist()==[True,False] and tt[0,0]==20 and np.isnan(tt[1,0])
    assert caseavg(np.array([[1.,3.],[5.,99.]]),np.array([[1,1],[1,0]],bool)).tolist()==[2.,5.]
    assert np.allclose(profiles(z)['udl'],profiles(z)['udl'][::-1])
    return dict(exact_prediction=True,censoring=True,equal_case_weight=True,symmetric_moment=True)
