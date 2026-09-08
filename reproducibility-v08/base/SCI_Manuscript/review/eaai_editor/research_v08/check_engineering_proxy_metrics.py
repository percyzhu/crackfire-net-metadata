"""Analytic hand-checks only. Never used as scientific model performance."""
from pathlib import Path
import json
import hashlib
import tempfile
import numpy as np
import engineering_proxy_metrics as e


def main():
    checks=[]
    t=np.array([0,60,120,180.])
    truth=np.array([[1,.9,.7,.5],[1,.95,.9,.85],[1,.7,.65,.5],[1,.9,.85,.81]])
    pred=np.array([[1,.8,.75,.5],[1,.95,.8,.7],[1,.9,.85,.81],[1,.9,.85,.81]])
    q=e.crossing_metrics(truth,pred,t,.8)
    assert [q[k] for k in ('true_positive','false_negative','false_positive','true_negative')]==[1,1,1,1]
    assert q['conditional_timing_MAE_s']==60 and q['conditional_timing_bias_s']==-60
    assert q['both_event_count']==1 and q['both_event_population_fraction']==.25
    assert q['recall']==q['false_positive_rate']==q['balanced_accuracy']==.5
    assert q['optimistic_late_or_missed_count']==1
    checks.append('q=.8 hand table: early event60s, one missed event, one false event, one true censor; all denominators explicit')
    q6=e.crossing_metrics(truth,pred,t,.6)
    assert [q6[k] for k in ('true_positive','false_negative','false_positive','true_negative')]==[1,1,0,2]
    assert q6['conditional_timing_MAE_s']==0 and q6['balanced_accuracy']==.75
    checks.append('Second threshold retained: q=.6 has different counts and zero joint timing error without hiding missed event')
    ae,at,_=e.event_arrays(np.array([[1,.9,.85,.81],[1,.9,.85,.8]]),t,.8)
    assert not ae[0] and np.isnan(at[0]) and ae[1] and at[1]==180
    checks.append('Censoring staysNaN; crossing exactlyat horizon180s is an event, not censoring')
    r=e.crossing_metrics(np.array([[1,.7,.6,.5]]),np.array([[1,.75,.9,.7]]),t,.8)
    assert r['recross_above_threshold_count']==1
    checks.append('Neural recrossing is detected and not silently monotonized')
    r=e.rank_metrics([.1,.1,.8],[.1,.2,.9])
    assert r['eligible_pairs']==2 and r['excluded_reference_tied_pairs']==1
    assert r['pairwise_order_agreement']==1 and np.isclose(r['spearman'],np.sqrt(3)/2)
    r=e.rank_metrics([0,1,2],[1,1,0])
    assert np.isclose(r['pairwise_order_agreement'],1/6) and np.isclose(r['spearman'],-np.sqrt(3)/2)
    assert e.rank_metrics([1,1],[.2,.8])['spearman'] is None
    assert e.rank_metrics([1,1],[.2,.8])['pairwise_order_agreement'] is None
    checks.append('Reference ties excluded, prediction tie earnshalf, exact hand correlations±sqrt(3)/2; constant target isNA')
    rank_truth=np.array([[1,.8],[1,.6],[1,.7],[1,.5]])
    rank_pred=np.array([[1,.8],[1,.6],[1,.6],[1,.6]])
    family_scores=e.evaluate_population(rank_truth,rank_pred,[0,60],['a','a','b','b'])
    eq=family_scores['equal_family_ranking_summaries']['final']['spearman']
    assert eq['reference_eligible_families']==2 and eq['defined_families']==1
    assert eq['mean_over_defined_families'] is None
    checks.append('Equal-family comparison cannot silently omit a supported family when a model predicts constant scores')
    a=np.array([[1,.5,0.]])
    v=e.evaluate_population(a,a+.1,[0,60,120])
    assert np.isclose(v['summaries']['final']['MAE'],.1)
    assert np.isclose(v['summaries']['time_average']['MAE'],.1)
    assert np.isclose(v['summaries']['time_average']['bias'],.1)
    checks.append('Analytic linear trajectory has timeaverage.5; constant+.1 prediction shift preserves exact.1 bias without clipping1.1')
    bad=pred.copy();bad[0,0]=np.nan
    try:e.evaluate_population(truth,bad,t)
    except ValueError:pass
    else:raise AssertionError('NaN not rejected')
    checks.append('Nonfinite prediction rejected, never imputed')
    _,ba=e.comparative_summary(truth,np.stack([pred]*5),t,['a']*4,repetitions=12,random_seed=17)
    _,bb=e.comparative_summary(truth,np.stack([pred]*5),t,['a']*4,repetitions=12,random_seed=17)
    assert all(np.array_equal(ba[k],bb[k],equal_nan=True) for k in ba)
    checks.append('Paired bootstrap exact reproducibility for identical models, including undefined-class replicate masking')
    with tempfile.TemporaryDirectory(prefix='proxy_gate_fixture_',dir=e.HERE) as temp:
        p=Path(temp)/'synthetic_plan.json'
        p.write_text(json.dumps(dict(status='FROZEN_BEFORE_TRAINING',purpose='ARCHIVAL_SURROGATE_RESEARCH_V08',
           models=list('abcdefg'),seeds=[42,43,44,45,46],protocols={'fixture':{}},runs_root=temp)),encoding='utf-8')
        gate,runs=e.complete_protocol(p,'fixture')
        assert runs is None and gate['pending_runs']==35 and not gate['predictions_or_scores_read']
    checks.append('Incomplete35-run protocol returnswaiting without reading any prediction file')
    report={'purpose':'ANALYTIC_HAND_CHECKS_ONLY_NOT_SCIENTIFIC_OUTCOMES','status':'PASS',
       'checks':checks,'count':len(checks),'training_results_read':False,
       'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(e.__file__)]}}
    (e.HERE/'engineering_proxy_metric_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','analytic_checks':len(checks),'training_results_read':False}))


if __name__=='__main__':main()
