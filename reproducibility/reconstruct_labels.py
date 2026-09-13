"""Reintegrate retained square cells; no thermal solve or nearest-node fill."""
from pathlib import Path
import argparse,json
import numpy as np
R=Path(__file__).resolve().parent
def integrate(path,beam,frames):
    with np.load(path,allow_pickle=False) as a:
        xy=a['xy_m'].astype(float);material=a['material_mask'];first=a['first_ge300_frame'];times=a['time_s'];counts=a['retained_point_count']
    delta=.002;b,h=np.round(np.asarray(beam)[1:]*1000)/1000;w0=np.array([b*h*h/6]*2+[h*b*b/6]*2)
    eta=np.zeros((len(frames),31,4));area=np.zeros((len(frames),31))
    for ti,k in enumerate(frames):
        for j in range(31):
            keep=material[j]&(first[j]>k);assert int(keep.sum())==counts[k,j]
            p=xy[keep];n=len(p)
            if not n:continue
            A=n*delta**2;cx,cy=p.mean(0);dx=p[:,0]-cx;dy=p[:,1]-cy
            Ix=delta**2*np.dot(dy,dy)+n*delta**4/12;Iy=delta**2*np.dot(dx,dx)+n*delta**4/12
            distances=np.array([p[:,1].max()+delta/2-cy,cy-(p[:,1].min()-delta/2),p[:,0].max()+delta/2-cx,cx-(p[:,0].min()-delta/2)])
            eta[ti,j]=np.array([Ix,Ix,Iy,Iy])/distances/w0;area[ti,j]=A
    return eta,area
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='ms500_0064');p.add_argument('--all-times',action='store_true');a=p.parse_args()
    with np.load(R/'data/learning_inputs.npz') as d:ids=d['case_ids'];beam=d['beam_Lbh_m']
    ix=int(np.flatnonzero(ids==a.case)[0]);frames=np.arange(361) if a.all_times else np.array([0,90,180,270,360])
    eta,area=integrate(R/'data/first300'/f'{a.case}.first300.npz',beam[ix],frames)
    ref=np.load(R/'data/section_eta.npy',mmap_mode='r')[ix,frames];err=float(abs(eta-ref).max());assert err<2e-6
    out=R/'reproduced';out.mkdir(exist_ok=True);np.savez_compressed(out/f'{a.case}.reconstructed.npz',frames=frames,section_eta=eta,area_m2=area,member_eta=eta.min(1))
    print(json.dumps(dict(case=a.case,frames=len(frames),max_absolute_eta_difference=err)))
if __name__=='__main__':main()
