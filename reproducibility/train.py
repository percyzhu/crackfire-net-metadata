"""Portable re-training, explicit opt-in. Does not evaluate test roles."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from models import SpatialModel
R=Path(__file__).resolve().parent
KINDS=['global_graph','global_contrast','local_graph','local_set','local_no_contrast']
def batches(ix):return np.array_split(ix,max(1,int(np.ceil(len(ix)/16))))
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=KINDS,required=True);p.add_argument('--seed',type=int,choices=[2601,2602,2603],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a fresh output directory; archived weights are immutable.')
    a.output.mkdir(parents=True);torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    with np.load(R/'data/learning_inputs.npz',allow_pickle=False) as z:
        ts=[torch.from_numpy(z[k]).to(device) for k in ['nodes','mask','edges','fire','beam']];roles=z['roles'];ids=z['case_ids']
    truth=torch.from_numpy(np.load(R/'data/section_eta.npy')).to(device)
    tr=np.flatnonzero(roles=='train');va=np.flatnonzero(roles=='validation');assert len(tr)==860 and len(va)==27
    norm=json.loads((R/'data/normalization.json').read_text(encoding='utf-8'));torch.manual_seed(a.seed);np.random.seed(a.seed);rng=np.random.default_rng(a.seed)
    m=SpatialModel(a.model,norm).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4);best=float('inf');history=[];start=time.perf_counter()
    for ep in range(100):
        m.train();losses=[]
        for ix in batches(rng.permutation(tr)):
            pred=m(*(v[ix] for v in ts));y=truth[ix];loss=(pred-y).square().mean()+.5*(pred.min(2).values-y.min(2).values).square().mean()
            if a.model not in ['global_graph','local_no_contrast']:loss+=4*((pred-pred.mean(2,keepdim=True))-(y-y.mean(2,keepdim=True))).square().mean()
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();losses.append(loss.item())
        m.eval()
        with torch.no_grad():
            pred=torch.cat([m(*(v[ix] for v in ts)) for ix in batches(va)]);y=truth[va]
            c=((pred-pred.mean(2,keepdim=True))-(y-y.mean(2,keepdim=True))).abs().mean().item();member=(pred.min(2).values-y.min(2).values).abs().mean().item();score=c+.5*member
        history.append(dict(epoch=ep+1,loss=float(np.mean(losses)),validation_contrast_mae=c,validation_member_mae=member,selection_score=score))
        if score<best:best=score;bestep=ep+1;torch.save(m.state_dict(),a.output/'best.pt')
        if (ep+1)%25==0:print(ep+1,score,flush=True)
    (a.output/'history.json').write_text(json.dumps(history,indent=2))
    (a.output/'completed.json').write_text(json.dumps(dict(model=a.model,seed=a.seed,best_epoch=bestep,selection_score=best,seconds=time.perf_counter()-start,test_evaluated=False),indent=2))
if __name__=='__main__':main()
