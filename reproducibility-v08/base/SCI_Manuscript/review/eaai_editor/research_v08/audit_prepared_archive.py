"""Independent read-only v08 archive/partition audit; no model results read."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,csv,collections,math
from datetime import datetime,timezone
import numpy as np
import torch
torch.set_num_threads(1)

HERE=Path(__file__).resolve().parent
SCI=HERE.parents[2]
ROOT=SCI.parent
DATA=SCI/'experiments/research_v08/archive997'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def canonical_geometry(c):
    rows=sorted(tuple(float(v[k]) for k in ('face','z','h','w','l','d')) for v in c['cracks'])
    return tuple(float(c['beam'][k]) for k in ('L','W','H')),tuple(rows)

def center(c,b):
    f,z,d=int(c['face']),float(c['z']),float(c['d'])
    q=float(c['h'])*(-1 if f in (0,3) else 1)
    return np.array([q,b['H']/2-d/2,z] if f==0 else [q,-b['H']/2+d/2,z] if f==1
                    else [b['W']/2-d/2,q,z] if f==2 else [-b['W']/2+d/2,q,z])

def expected_features(c):
    b=c['beam'];L,W,H=[b[k] for k in ('L','W','H')]
    cracks=c['cracks'];nodes=[]
    for v in cracks:
        f,z,w,l,d=int(v['face']),float(v['z']),float(v['w']),float(v['l']),float(v['d'])
        q=float(v['h'])*(-1 if f in (0,3) else 1)
        th=H if f in (0,1) else W;hs=max((W if f in (0,1) else H)/2-.05,.01)
        nodes.append([float(f==j) for j in range(4)]+[z/L,q/hs,w/W,l/L,d/th,l*d/(W*H),d/th])
    edges=[];features=[];centers=[center(v,b) for v in cracks]
    for i,a in enumerate(cracks):
        for j,d in enumerate(cracks):
            if i==j:continue
            fi,fj=int(a['face']),int(d['face'])
            edges.append([i,j])
            adjacent=((fi<2) != (fj<2))
            features.append([float(fi==fj),float(adjacent),abs(a['z']-d['z'])/L,
                float(np.linalg.norm(centers[i]-centers[j]))/math.sqrt(L*L+W*W+H*H),
                (a['d']+d['d'])/max(W,H)])
    g=[[len(cracks)/10,sum(v['l']*v['w'] for v in cracks)/(2*(W+H)*L),max(v['d'] for v in cracks)/max(W,H)]]
    return [np.asarray(nodes,np.float32),np.asarray(edges,np.int64).reshape(-1,2).T,
            np.asarray(features,np.float32).reshape(-1,5),np.asarray(g,np.float32)]

def describe(cases):
    return dict(count=len(cases),batch=dict(collections.Counter(c['source_batch'] for c in cases)),
      family=dict(collections.Counter(c['fire_family'] for c in cases)),
      N=dict(sorted(collections.Counter(c['num_cracks'] for c in cases).items())),
      restored=sum(c['legacy_final_id'] is None for c in cases),
      flat=sum(c['flat_response'] for c in cases),
      native_frames=dict(sorted(collections.Counter(c['native_time_count'] for c in cases).items())))

def main():
    start=datetime.now(timezone.utc).isoformat()
    m=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    assert sha(DATA/'tensors.pt')==m['tensor_sha256']
    data=torch.load(DATA/'tensors.pt',map_location='cpu',weights_only=True)
    cases=m['cases'];byid={c['sample_id']:c for c in cases}
    assert len(byid)==len(cases)==997 and set(byid)==set(data['graphs'])==set(data['targets'])
    assert np.array_equal(data['time_s'].numpy(),np.arange(61)*60)
    groups=collections.defaultdict(list)
    for c in cases:groups[canonical_geometry(c)].append(c['sample_id'])
    assert len(groups)==997
    files_ok={p:sha(ROOT/p)==h for p,h in m['metadata_source_sha256'].items()}
    assert all(files_ok.values())
    checked=0;legacy=0;max_diff=0.;max_feature=0.;allflat=[]
    for c in cases:
        sid=c['sample_id'];p=ROOT/c['source_npz'];g=data['graphs'][sid]
        with np.load(p,allow_pickle=False) as z:
            t,y=z['time_steps'],z['charring_ratios'];cracks=z['crack_params'];dims=z['beam_dims']
        assert hashlib.sha256(t.tobytes()+y.tobytes()).hexdigest()==c['source_scalar_arrays_sha256_current']
        assert t.ndim==1 and y.shape==t.shape and len(t)>=2 and np.isfinite(y).all()
        assert (np.diff(t)>0).all() and t[0]==0 and t[-1]==3600
        assert y.min()>=0 and y.max()<=1+1e-8
        assert np.array_equal(cracks,[[float(v[k]) for k in ('face','z','h','w','l','d')] for v in c['cracks']])
        assert np.array_equal(dims.ravel(),[1.5,.14,.2])
        yy=np.interp(np.arange(61)*60,t,np.minimum.accumulate(y)).astype(np.float32)
        diff=float(abs(yy-data['targets'][sid].numpy()).max());max_diff=max(max_diff,diff);assert diff==0
        exp=expected_features(c)
        for a,b in zip(exp,g[:4]):
            assert a.shape==tuple(b.shape)
            err=float(abs(a-b.numpy()).max()) if a.size else 0
            max_feature=max(max_feature,err);assert err<2e-7
        tf=g[4];T=tf[:,1]*1200
        # Descriptor construction check only; not a check against realized FE amplitude.
        assert torch.allclose(tf[:,0],torch.arange(61)/60,atol=1e-7,rtol=0)
        assert torch.allclose(tf[:,2],torch.cumsum(T*60,0)/(1000*3600),atol=2e-7,rtol=0)
        rate=torch.diff(T)/60/20
        assert torch.allclose(tf[1:,3],rate,atol=2e-7,rtol=0)
        assert torch.equal(tf[0,3],tf[1,3])
        assert all(torch.isfinite(a).all() for a in g)
        if c['legacy_final_id']:
            old=ROOT/'Abaqus/processed_pilot/raw'/(c['legacy_final_id']+'.npz')
            with np.load(old,allow_pickle=False) as z:
                previous=np.interp(np.arange(61)*60,z['time_steps'],np.minimum.accumulate(z['charring_ratios'])).astype(np.float32)
            assert np.array_equal(previous,yy);legacy+=1
        elif np.ptp(y)==0:allflat.append(c)
        checked+=1
    summaries={};cross=[];radii=[]
    for path in sorted((DATA/'splits').glob('*.json')):
        s=json.loads(path.read_text(encoding='utf-8'))
        assert s['dataset_sha256']==sha(DATA/'manifest.json')
        roles={k:set(s[k]) for k in ('train','validation','test')}
        assert set.union(*roles.values())==set(byid)
        assert not(roles['train']&roles['validation'] or roles['test']&roles['validation'] or roles['test']&roles['train'])
        for role,ids in roles.items():assert len(ids)==len(s[role])
        if path.stem=='lcro_9_15':
            assert max(byid[i]['num_cracks'] for i in roles['train']|roles['validation'])<=8
            assert min(byid[i]['num_cracks'] for i in roles['test'])>=9
        if path.stem.startswith('loco_'):
            family=path.stem[5:]
            assert {byid[i]['fire_family'] for i in roles['test']}=={family}
            assert all(byid[i]['fire_family']!=family for i in roles['train']|roles['validation'])
        distances=[]
        for sid in s['train']:
            c=byid[sid];centers=np.asarray([center(v,c['beam']) for v in c['cracks']]);n=len(centers)
            if n<2:continue
            d=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)/np.linalg.norm([c['beam'][k] for k in ('L','W','H')])
            distances.append(float(np.quantile(d[np.triu_indices(n,1)],.5)))
        radius=float(np.median(distances));assert abs(radius-s['topology_recipe']['radius'])<1e-14
        radii.append(dict(protocol=path.stem,independent_training_only_radius=radius,
                          non_singleton_training_cases=len(distances)))
        summaries[path.stem]={r:describe([byid[i] for i in sorted(ids)]) for r,ids in roles.items()}
        for role,ids in roles.items():
            count=collections.Counter((byid[i]['source_batch'],byid[i]['fire_family'],byid[i]['num_cracks'],
                                       byid[i]['legacy_final_id'] is None) for i in ids)
            for (batch,family,n,restored),v in sorted(count.items()):
                cross.append(dict(protocol=path.stem,role=role,batch=batch,family=family,N=n,restored=restored,cases=v))
    report=dict(status='NO_STOP_TRAINING_BUG_FOUND_IN_PREPARED_DATA_AUDIT',submission_pass=False,
       started_utc=start,completed_utc=datetime.now(timezone.utc).isoformat(),
       trained_predictions_or_scores_read=False,full_NT11_fields_read=False,
       source_scalar_cases_rechecked=checked,legacy911_reconciled=legacy,
       restored_flat=describe(allflat),independent_canonical_geometry_count=len(groups),
       prepared_target_max_difference=max_diff,independent_geometry_feature_max_difference=max_feature,
       metadata_files_current_hash_verified=len(files_ok),dataset=describe(cases),
       protocols=summaries,radius_checks=radii,
       hashes={str(p):sha(p) for p in [Path(__file__),DATA/'manifest.json',DATA/'readiness.json',DATA/'tensors.pt']+
               sorted((DATA/'splits').glob('*.json'))})
    (HERE/'prepared_archive_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with (HERE/'prepared_split_batch_family_count.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(cross[0]));w.writeheader();w.writerows(cross)
    print(json.dumps({k:report[k] for k in ['status','source_scalar_cases_rechecked','legacy911_reconciled','restored_flat','dataset']},indent=2))

if __name__=='__main__':main()
