"""One command: audit data, re-score every saved prediction, reproduce tables/figures."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
import numpy as np
import torch
from models import SpatialModel
from metrics import metrics,checks
from reconstruct_labels import integrate
R=Path(__file__).resolve().parent;OUT=R/'reproduced'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip-figures',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4);report={'metric_fixtures':checks()}
    subprocess.run([sys.executable,str(R/'prepare_inputs.py')],cwd=R,check=True)
    if (R/'SHA256SUMS.txt').exists():
        n=0
        for line in (R/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            expected,name=line.split('  ',1)
            with (R/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            assert actual==expected,name;n+=1
        report['hashed_files']=n
    with np.load(R/'data/learning_inputs.npz') as f:d={k:f[k] for k in f.files}
    ids=d['case_ids'];y=np.load(R/'data/section_eta.npy',mmap_mode='r');lookup={str(cid):i for i,cid in enumerate(ids)}
    assert len(ids)==1482 and len(lookup)==1482 and y.shape==(1482,361,31,4)
    for role in ['validation','count','section','length','joint','reserved']:assert not set(d['families'][d['roles']=='train'])&set(d['families'][d['roles']==role])
    report['cases']=len(ids);report['checkpoints']=[];report['label_reconstruction']=[]
    norm=json.loads((R/'data/normalization.json').read_text(encoding='utf-8'));freeze=json.loads((R/'checkpoints/validation_freeze.json').read_text(encoding='utf-8'));forward_max=0
    for key in sorted(freeze):
        kind,seed=key.rsplit('_',1);path=R/'checkpoints'/f'{key}.pt'
        with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==freeze[key]['checkpoint_sha256']
        m=SpatialModel(kind,norm).eval();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        with np.load(R/'predictions'/key/'validation.npz') as saved:vids=saved['case_ids'][:2];expected=saved['predicted'][:2]
        ix=np.array([lookup[str(v)] for v in vids]);inputs=[torch.from_numpy(d[k][ix]) for k in ['nodes','mask','edges','fire','beam']]
        with torch.no_grad():pred=m(*inputs).numpy()
        error=float(abs(pred-expected).max());assert error<1e-5,(key,error);forward_max=max(forward_max,error)
        report['checkpoints'].append({'name':key,'forward_max_abs_difference':error})
    report['forward_max_abs_difference']=forward_max
    for cid in ['b004_0000','b005_0033','ms500_0001','ms500_0064','ms500_0496']:
        ix=lookup[cid];frames=np.array([0,90,180,270,360]);eta,area=integrate(R/'data/first300'/f'{cid}.first300.npz',d['beam_Lbh_m'][ix],frames)
        err=float(abs(eta-y[ix,frames]).max());assert err<2e-6;assert (np.diff(area,axis=0)<=1e-10).all()
        report['label_reconstruction'].append(dict(case=cid,max_abs_eta_difference=err))
    # Full score recovery from arrays: same per-case definition, not transcription of summary JSON.
    recovered=[];demands=[]
    for path in sorted((R/'predictions').glob('*/*.npz')):
        key=path.parent.name;kind,seed=key.rsplit('_',1);group=path.stem
        with np.load(path) as a:pid=a['case_ids'];p=a['predicted'];t=a['time_s']
        ix=np.array([lookup[str(v)] for v in pid]);assert (d['roles'][ix]==group).all();assert np.isfinite(p).all()
        rows,_,dm,_=metrics(p,y[ix],pid,t)
        recovered.extend(dict(model=kind,seed=int(seed),group=group,**r) for r in rows);demands.extend(dict(model=kind,seed=int(seed),group=group,**r) for r in dm)
        print(key,group,'re-scored',flush=True)
    maximum=0
    for name,rec,keys in [('combined_results',recovered,['model','seed','group','stratum']),('combined_demand_results',demands,['model','seed','group','profile','lambda_'])]:
        expected=json.loads((R/'metrics'/f'{name}.json').read_text(encoding='utf-8'));emap={tuple(r[k] for k in keys):r for r in expected};assert len(emap)==len(rec)
        for row in rec:
            ref=emap[tuple(row[k] for k in keys)]
            for k,v in row.items():
                if isinstance(v,(float,int)) and k not in keys:
                    assert ref[k] is not None;diff=abs(v-ref[k]);maximum=max(maximum,diff);assert diff<1e-8,(name,k,diff)
                else:assert v==ref[k],(name,k,v,ref[k])
        dump(OUT/f'{name}.json',rec)
    report['score_max_abs_difference']=maximum;report['prediction_files']=90
    if not args.skip_figures:
        import shutil
        # Build an output-only visualization workspace; manuscript sources are private.
        paper=OUT/'visualization';shutil.copytree(R/'visualization',paper,dirs_exist_ok=True)
        for source,dest in [('combined_results','results'),('combined_demand_results','demand_results')]:
            primary=[r for r in json.loads((OUT/f'{source}.json').read_text(encoding='utf-8')) if r['model']!='local_no_contrast'];dump(paper/'reproducibility'/f'{dest}.json',primary)
        for script in ['reproducibility/reproduce_tables.py','figure_scripts/rebuild_all_figures.py']:
            subprocess.run([sys.executable,str(paper/script)],cwd=paper,check=True,stdout=subprocess.DEVNULL)
        report['reproduced']='Table 4; weak-set/event supplement tables; all sixteen manuscript and supplement figures (fifteen redrawn, attributed photograph preserved)'
    subprocess.run([sys.executable,str(R/'reproduce_dense.py')],cwd=R,check=True,stdout=subprocess.DEVNULL)
    report['dense_prediction_files']=420
    subprocess.run([sys.executable,str(R/'reproduce_statistics.py')],cwd=R,check=True,stdout=subprocess.DEVNULL)
    report['paired_family_intervals_reproduced']=True
    report['pass']=True;dump(OUT/'reproduction_report.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
