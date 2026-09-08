"""Independent actual plot-data audit from retained complete source snapshots."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
provenance = read(HERE/'provenance.json')
sources = provenance['source_snapshot']
for row in sources: assert sha(HERE/row['snapshot_path']) == row['sha256']
def snapshot(protocol, name):
    matches = [x for x in sources if Path(x['original_path']).name == name and Path(x['original_path']).parent.name == protocol]
    assert len(matches) == 1
    return HERE/matches[0]['snapshot_path']
def same(a,b): assert np.allclose(a,b,atol=1e-13,rtol=1e-12)
families, models = provenance['family_order'], provenance['model_order']
seeds = pd.read_csv(HERE/'family_model_seed_mae.csv')
heat = pd.read_csv(HERE/'plotted_heatmap.csv')
paired = pd.read_csv(HERE/'paired_contrasts_source.csv')
aggregate = pd.read_csv(HERE/'cross_holdout_descriptive_seed_means.csv')
assert len(seeds)==350 and len(heat)==84 and len(paired)==20 and len(aggregate)==70
counts, all_ids = {}, []
for family in families:
    protocol='loco_'+family;frame=pd.read_csv(snapshot(protocol,'per_case_metrics.csv'))
    summary=read(snapshot(protocol,'summary.json')); counts[family]=summary['test_case_count']
    ids=set(frame.sample_id); assert len(ids)==counts[family];all_ids.extend(ids)
    assert len(frame)==35*counts[family] and not frame.duplicated(['model','seed','sample_id']).any()
    for model in models:
        values=frame[frame.model==model].groupby('seed').MAE.mean().reindex([42,43,44,45,46]).to_numpy()
        saved=seeds[(seeds.family==family)&(seeds.model==model)].sort_values('seed')
        same(saved.MAE,values);same(saved.MAE_percentage_points,values*100)
        cell=heat[(heat.row==family)&(heat.model==model)].iloc[0]
        same(cell.MAE_percentage_points,values.mean()*100)
    for tag,first,second in [('graph_vs_capacity_matched','capacity_matched_deepsets','gnn'),('mean_vs_sum','gnn','gnn_mean')]:
        src=summary['paired_contrasts'][tag];row=paired[(paired.family==family)&(paired.contrast==tag)].iloc[0]
        a=frame[frame.model==first].groupby('seed').MAE.mean().sort_index().to_numpy()
        b=frame[frame.model==second].groupby('seed').MAE.mean().sort_index().to_numpy()
        same(row.difference,(a-b).mean());same([row.percentile_95_lower,row.percentile_95_upper],src['intervals']['two_way']['percentile_95'])
        same([row.difference_percentage_points,row.lower_percentage_points,row.upper_percentage_points],np.asarray([row.difference,row.percentile_95_lower,row.percentile_95_upper])*100)
assert len(all_ids)==len(set(all_ids))==997
for model in models:
    frame=seeds[seeds.model==model].pivot(index='family',columns='seed',values='MAE').reindex(families)
    for name,value in [('case_weighted_cross_holdout',np.asarray([counts[f] for f in families])@frame.to_numpy()/997),('equal_family_cross_holdout',frame.to_numpy().mean(axis=0))]:
        rows=aggregate[(aggregate.estimand==name)&(aggregate.model==model)].sort_values('seed')
        same(rows.MAE,value)
        same(heat[(heat.row==name)&(heat.model==model)].iloc[0].MAE_percentage_points,value.mean()*100)
subgroups=pd.read_csv(HERE/'smoldering_subgroup_seed_mae.csv');assert len(subgroups)==70
sm=pd.read_csv(snapshot('loco_smoldering','per_case_metrics.csv'))
for r in subgroups.itertuples():
    block=sm[(sm.model==r.model)&(sm.seed==r.seed)]
    block=block[block.flat_response] if r.subgroup=='restored_flat' else block[~block.flat_response]
    assert len(block)==r.case_count==(86 if r.subgroup=='restored_flat' else 11)
    same(r.MAE,block.MAE.mean())
result={'status':'PASSED_INDEPENDENT_COMPLETE_PLOT_DATA_ARITHMETIC','source_per_case_rows':sum(counts.values())*35,
        'unique_cross_holdout_cases':997,'seed_MAE_values':350,'heatmap_cells':84,'paired_intervals':20,
        'descriptive_aggregate_seed_values':70,'smoldering_subgroup_seed_values':70,'percentage_point_scale':100,
        'primary_intervals_crossing_zero':int(((paired.percentile_95_lower<=0)&(paired.percentile_95_upper>=0)).sum()),
        'aggregate_CI_created':False,'prediction_arrays_read':0,'bootstrap_recomputed':False}
(HERE/'independent_arithmetic_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
