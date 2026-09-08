"""Correlated analytic example: paired CI must not use interval endpoints."""
import json
from pathlib import Path
import numpy as np
import engineering_proxy_ci as ci

times=np.array([0,60,120,180.])
truth=np.array([[.9,.8,.7,.4],[.8,.7,.6,.3],[.7,.6,.5,.2],[.6,.5,.4,.1]])
seed_offset=np.arange(5)[:,None,None]*.01
case_error=np.array([.05,.10,.15,.20])[None,:,None]
gnn=truth[None]+case_error+seed_offset
predictions={'gnn':gnn,'capacity_matched_deepsets':gnn+.05,'gnn_mean':gnn-.02}
summaries,contrasts=ci.paired_intervals(truth,predictions,times,['a','a','b','b'],[42,43,44,45,46],
                                      repetitions=40,random_seed=11)
for name,delta in [('graph_vs_capacity_matched',.05),('mean_vs_sum',.02)]:
    for key in ('final_MAE','time_average_MAE'):
        v=contrasts[name]['metrics'][key]
        assert np.isclose(v['point_baseline_minus_candidate'],delta)
        assert np.allclose(v['percentile_95_paired_difference'],[delta,delta],atol=1e-14)
        assert v['common_valid_bootstrap_replicates']==40
for model in predictions:
    low,high=summaries[model]['final_MAE']['paired_hierarchical_95_interval']
    assert high-low>.02
assert 'higher-is-better' in ci.preference('final_spearman')
assert 'Neither sign' in ci.preference('final_bias')
report={'purpose':'ANALYTIC_CHECK_ONLY_NOT_TRAINED_OUTCOMES','status':'PASS',
        'checks':['Individual modelMAE intervals are wide because case/seed errors vary together.',
                  'The paired graph-vs-set final/timeaverageMAE differences are exactly.05 in every resample.',
                  'The paired mean-vs-sum final/timeaverageMAE differences are exactly.02 in every resample.',
                  'Higher-is-better scores and signed biases do not inherit the MAE effect-direction interpretation.'],
        'real_predictions_read':False,
        'code_sha256':{str(p):ci.ep.sha(p) for p in [Path(__file__),Path(ci.__file__),Path(ci.ep.__file__)]}}
(ci.ep.HERE/'engineering_proxy_ci_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','analytic_paired_contrasts':2,'real_predictions_read':False}))
