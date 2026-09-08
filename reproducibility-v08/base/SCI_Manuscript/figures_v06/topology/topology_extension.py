"""Standalone prospective topology views; frozen production files are untouched.

The self-check uses archived geometry and random weights solely as software
evidence. Registering a scientific recipe requires the existing qualified-data
loader and a dataset-bound split. Registration does not train any model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
SCI=HERE.parents[1]
sys.path.insert(0,str(SCI/'experiments'))
import physical_features as pf
from models import Surrogate
from run_readiness import collate

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def pair_distances(case):
    centers=np.asarray([pf.crack_center_3d(c,case['beam']) for c in case['cracks']])
    diagonal=np.linalg.norm([case['beam'][key] for key in ('L','W','H')])
    return np.linalg.norm(centers[:,None,:]-centers[None,:,:],axis=-1)/diagonal

def fit_radius_on_training_geometries(cases, quantile=.5):
    """Equal weight per unique training geometry, excluding the zero diagonal."""
    if not 0 < quantile < 1: raise ValueError('quantile must lie strictly within (0,1)')
    if not cases or any(c.get('role')!='train' for c in cases):
        raise ValueError('Radius fitting accepts only explicitly assigned training cases')
    unique={}
    for c in cases:
        payload=json.dumps({'beam':c['beam'],'cracks':sorted(c['cracks'],key=lambda x:json.dumps(x,sort_keys=True))},sort_keys=True)
        key=hashlib.sha256(payload.encode()).hexdigest()
        unique.setdefault(key,c)
    per_case=[]
    for key,c in unique.items():
        dist=pair_distances(c); n=len(dist)
        if n>1:
            per_case.append({'geometry_sha256':key,'sample_id':c['sample_id'],
                'distance_quantile':float(np.quantile(dist[np.triu_indices(n,1)],quantile))})
    if not per_case: raise ValueError('No training geometry has a distinct crack pair')
    return {'normalized_radius':float(np.median([p['distance_quantile'] for p in per_case])),
        'fixed_quantile':quantile,'unique_training_geometries':len(unique),
        'non_singleton_training_geometries':len(per_case),'per_geometry_values':per_case,
        'fit_rule':'Median of per-geometry off-diagonal distance quantiles; unique training geometries only.',
        'validation_or_test_geometry_used':False}

def adjacency(case,rule='complete',radius=None,k=3):
    dist=pair_distances(case); n=len(dist)
    if rule=='complete': return ~np.eye(n,dtype=bool)
    if rule=='radius':
        if radius is None or radius<=0: raise ValueError('A positive training-bound radius is required')
        return (dist<=radius)&~np.eye(n,dtype=bool)
    if rule!='symmetric_knn': raise ValueError(rule)
    if not isinstance(k,int) or k<1: raise ValueError('k must be a positive integer')
    if n==1: return np.zeros((1,1),dtype=bool)
    # Include every tie at the kth distance to preserve relabelling symmetry.
    masked=dist.copy();np.fill_diagonal(masked,np.inf)
    cutoff=np.sort(masked,axis=1)[:,min(k,n-1)-1]
    directed=masked<=cutoff[:,None]
    result=directed|directed.T
    np.fill_diagonal(result,False)
    return result

def graph_view(case,rule,radius=None,k=3,tf=None):
    edges,features=pf.build_crack_edges(case['cracks'],case['beam'])
    adj=adjacency(case,rule,radius,k)
    keep=torch.from_numpy(adj[edges[0].numpy(),edges[1].numpy()])
    if tf is None: tf=torch.zeros((7,4),dtype=torch.float32)
    return (pf.encode_crack_node_features(case['cracks'],case['beam']),edges[:,keep],
        features[keep],pf.encode_global_features(case['cracks'],case['beam']),tf)

def self_check():
    import draw_topology as source
    _,original,_=source.build_data()
    cases=[{'sample_id':c['sample_id'],'beam':dict(pf.DEFAULT_BEAM),'cracks':c['cracks'],
            'role':'test' if c['N']==8 else 'train'} for c in original]
    fitted=fit_radius_on_training_geometries([c for c in cases if c['role']=='train'])
    rejected=False
    try:fit_radius_on_training_geometries(cases)
    except ValueError:rejected=True
    assert rejected
    torch.set_num_threads(1);torch.manual_seed(20260907);model=Surrogate().eval()
    initial=source.state_sha(model)
    output=[]
    for c in cases:
        for rule in ('complete','radius','symmetric_knn'):
            g=graph_view(c,rule,fitted['normalized_radius'],k=3)
            perm=np.arange(len(c['cracks']))[::-1]
            permuted=dict(c,cracks=[c['cracks'][i] for i in perm])
            a=adjacency(c,rule,fitted['normalized_radius'],3)
            ap=adjacency(permuted,rule,fitted['normalized_radius'],3)
            assert np.array_equal(ap,a[perm][:,perm])
            with torch.inference_mode():
                y=model(*collate([g]));yp=model(*collate([graph_view(permuted,rule,fitted['normalized_radius'],3)]))
            error=float(torch.max(torch.abs(y-yp)))
            assert error<2e-6 and torch.isfinite(y).all()
            assert not np.diag(a).any() and np.array_equal(a,a.T)
            assert torch.equal(g[2],pf.build_crack_edges(c['cracks'],c['beam'])[1][
                torch.from_numpy(a[pf.build_crack_edges(c['cracks'],c['beam'])[0][0].numpy(),
                                   pf.build_crack_edges(c['cracks'],c['beam'])[0][1].numpy()])])
            output.append({'sample_id':c['sample_id'],'N':len(c['cracks']),'rule':rule,
                'directed_edges':g[1].shape[1],'minimum_in_degree':int(a.sum(0).min()),
                'maximum_in_degree':int(a.sum(0).max()),'isolated_nodes':int((a.sum(0)==0).sum()),
                'permutation_max_abs_difference':error,'finite_output':True})
    assert initial==source.state_sha(model)
    result={'purpose':'SOFTWARE_ONLY_NOT_A_SCIENTIFIC_TRAIN_TEST_SPLIT','status':'PASSED_INTERFACE_CHECKS',
        'weights':'one random initial state, seed 20260907, no updates',
        'state_sha256':initial,'fit':fitted,'mixed_train_test_radius_fit_rejected':rejected,
        'checks':output,'source_sha256':{str(p.relative_to(SCI)):digest(p) for p in [Path(__file__),SCI/'experiments/models.py',SCI/'experiments/physical_features.py']},
        'scientific_limitation':'These calls show tensor and relabelling compatibility only; no accuracy, adaptation, or physical topology-transfer result is measured.'}
    (HERE/'topology_extension_software_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

def register(manifest,split,output,quantile,k):
    from qualified_data import QualifiedDataset
    ds=QualifiedDataset(manifest,allow_software_fixture=False)
    split_data=ds.check_split(split)
    training=[dict(ds.by_id[sid],role='train') for sid in split_data['train']]
    fitted=fit_radius_on_training_geometries(training,quantile)
    recipe={'status':'REGISTERED_TOPOLOGY_RECIPE_NOT_TRAINED','purpose':'Prospective paired connectivity transfer',
        'dataset_sha256':digest(manifest),'split_sha256':digest(split),'radius_fit':fitted,
        'symmetric_knn_k':k,'knn_tie_policy':'include_all_at_kth_distance_then_symmetric_union',
        'views':['complete','radius','symmetric_knn'],'training_view':'complete',
        'test_rule':'Freeze each trained seed checkpoint once. Evaluate the same held-out geometries and fire histories under all three graph views without adaptation.',
        'feature_sha256':digest(SCI/'experiments/physical_features.py'),
        'model_sha256':digest(SCI/'experiments/models.py'),'topology_extension_sha256':digest(__file__),
        'plan_sha256':digest(HERE/'topology_transfer_plan_zh.md'),
        'no_model_or_existing_protocol_edits':True,'weights_fitted':False,'scientific_results':False}
    with Path(output).open('x',encoding='utf-8') as f:json.dump(recipe,f,indent=2)
    print(str(Path(output).resolve()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('self-check')
    r=sub.add_parser('register');r.add_argument('--manifest',required=True);r.add_argument('--split',required=True)
    r.add_argument('--output',required=True);r.add_argument('--quantile',type=float,default=.5);r.add_argument('--k',type=int,default=3)
    args=p.parse_args()
    if args.command=='self-check':self_check()
    else:
        if args.k<1:raise ValueError('k must be >=1')
        register(args.manifest,args.split,args.output,args.quantile,args.k)
