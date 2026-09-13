"""Portable paired geometry-family bootstrap using the frozen resampling indices."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
def main():
    pc=pd.read_json(R/'metrics/combined_per_case.json');split=pd.read_csv(R/'data/split_manifest.csv').set_index('case_id')
    pc=pc.merge(split[['family','L_m','cracks']],left_on='case_id',right_index=True,validate='many_to_one')
    # Distance metrics stored in normalized coordinates are converted per member.
    for col in list(pc.columns):
        if col.startswith('distance_over_L_'):pc[col.replace('distance_over_L_','distance_mm_')]=pc[col]*pc.L_m*1000
    pc['distance_mm']=pc['distance_over_L_eps0.01']*pc.L_m*1000
    cache={};report={};maximum=0
    def calc(g,st,f,ka,kb=None):
        if (g,st) not in cache:
            with np.load(R/'bootstrap'/f'bootstrap_indices_{g}_{st}.npz') as z:cache[g,st]=(z['families'],z['indices'])
        families,draws=cache[g,st];models=[ka] if kb is None else [ka,kb]
        a=pc[(pc.group==g)&(pc.stratum==st)&pc.model.isin(models)].groupby(['case_id','family','model'])[f].mean().unstack('model').dropna()
        v=a[ka] if kb is None else a[ka]-a[kb]
        sums=v.groupby(level='family').sum().reindex(families,fill_value=0).to_numpy();n=v.groupby(level='family').count().reindex(families,fill_value=0).to_numpy();den=n[draws].sum(1)
        vals=np.divide(sums[draws].sum(1),den,out=np.full(len(draws),np.nan),where=den>0);vals=vals[np.isfinite(vals)]
        return dict(estimate=float(v.mean()),low=float(np.quantile(vals,.025)),high=float(np.quantile(vals,.975)),cases=len(a),families=int((n>0).sum()),valid_draws=len(vals))
    for name in ['ablation_bootstrap','paired_family_bootstrap','family_bootstrap_intervals']:
        original=json.loads((R/'metrics'/f'{name}.json').read_text(encoding='utf-8'));rebuilt=[]
        for row in original:
            if name=='ablation_bootstrap':ka,kb='local_graph','local_no_contrast';estimatekey='difference';drawkey='draws'
            elif name=='paired_family_bootstrap':ka,kb=row['model_a'],row['model_b'];estimatekey='difference_A_minus_B';drawkey='valid_draws'
            else:ka,kb=row['model'],None;estimatekey='estimate';drawkey='valid_draws'
            res=calc(row['group'],row['stratum'],row['metric'],ka,kb);update={estimatekey:res.pop('estimate'),drawkey:res.pop('valid_draws'),**res}
            for k,v in update.items():
                diff=abs(v-row[k]);maximum=max(maximum,diff);assert diff<1e-10,(name,k,diff)
            rebuilt.append({**row,**update})
        out=R/'reproduced';out.mkdir(exist_ok=True);(out/f'{name}.json').write_text(json.dumps(rebuilt,indent=2));report[name]=len(rebuilt)
    report.update(max_absolute_difference=maximum,bootstrap_index_files=len(cache),pass_=True)
    (R/'reproduced/statistics_reproduction_report.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
