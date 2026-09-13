"""Reconstruct crack, edge and temporal inputs from raw geometry/fire metadata."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent
def box(c,beam):
    L,b,h=beam;f=c['face'];q=(-1 if f in [0,3] else 1)*c['h'];w=c['w'];d=c['d'];z=c['z'];l=c['l']
    return np.array([[q-w/2,q+w/2],[h/2-d,h/2] if f==0 else [-h/2,-h/2+d],[z-l/2,z+l/2]] if f<2 else [[b/2-d,b/2] if f==2 else [-b/2,-b/2+d],[q-w/2,q+w/2],[z-l/2,z+l/2]])
def main():
    with np.load(R/'data/raw_inputs.npz') as f:raw=f['crack_node_features'];mask=f['crack_node_mask'].astype(bool);curves=f['fire_curves_61pt'];beam=f['beam_Lbh_m'];ids=f['case_ids']
    with np.load(R/'data/learning_inputs.npz') as f:reference={k:f[k] for k in f.files}
    rows={r['case_id']:r for r in map(json.loads,(R/'data/case_geometry_manifest.jsonl').read_text(encoding='utf-8').splitlines())}
    norm=json.loads((R/'data/normalization.json').read_text(encoding='utf-8'));face=np.where(mask,raw[...,0],0).astype(int)
    L,b,h=(beam[:,i,None] for i in range(3));span=np.where(face<2,b,h);normal=np.where(face<2,h,b)
    relative=raw[...,5:]/np.stack([np.broadcast_to(L,face.shape),span/2,span,np.broadcast_to(L,face.shape),normal],-1)
    nodes=np.concatenate([raw[...,1:5],relative,raw[...,5:]/[1.5,.1,.01,.4,.1]],-1).astype('float32');nodes[~mask]=0
    edges=np.zeros((len(ids),15,15,4),dtype='float32')
    for i,cid in enumerate(ids):
        # Production dimensions are the nominal millimetre values; tensor dimensions are float32.
        bm=np.array(rows[str(cid)]['beam_LWH_m']);bm[1:]=np.round(bm[1:]*1000)/1000
        boxes=np.array([box(c,bm) for c in rows[str(cid)]['cracks']]);n=len(boxes);centers=boxes.mean(-1)
        delta=centers[None]-centers[:,None];gap=np.maximum(np.maximum(boxes[:,None,:,0]-boxes[None,:,:,1],boxes[None,:,:,0]-boxes[:,None,:,1]),0)
        edges[i,:n,:n,:3]=delta/[.14,.2,1.5];edges[i,:n,:n,3]=np.linalg.norm(gap,axis=-1)/.2
    times=reference['time_s'];gas=np.array([np.interp(times,c[:,0],c[:,1]) for c in curves],np.float32)
    fire=np.stack([gas,np.maximum.accumulate(gas,1),np.broadcast_to(times,gas.shape)],-1)
    # Saved statistics were fitted exclusively on the final training role.
    rebuilt=dict(nodes=nodes,mask=mask,edges=edges,fire=((fire-norm['fire_mean'])/norm['fire_std']).astype('float32'),beam=((beam-norm['beam_mean'])/norm['beam_std']).astype('float32'))
    errors={}
    for name,a in rebuilt.items():
        if a.dtype==bool:assert np.array_equal(a,reference[name]);errors[name]=0
        else:
            errors[name]=float(abs(a-reference[name]).max());assert errors[name]<3e-6,(name,errors[name])
    out=R/'reproduced';out.mkdir(exist_ok=True);np.savez_compressed(out/'rebuilt_inputs.npz',case_ids=ids,**rebuilt)
    (out/'input_preprocessing_report.json').write_text(json.dumps(dict(cases=len(ids),maximum_absolute_differences=errors,pass_=True),indent=2));print(errors)
if __name__=='__main__':main()
