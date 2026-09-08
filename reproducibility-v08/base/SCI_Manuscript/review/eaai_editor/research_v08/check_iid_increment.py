"""Bind the limited IID text/figure development review to exact file versions."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,re,collections
import numpy as np

HERE=Path(__file__).resolve().parent
SCI=HERE.parents[2]
FIG=SCI/'figures_v08/iid_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    text_path=SCI/'latex_v08/sections/05_results.tex'
    text=text_path.read_text(encoding='utf-8')
    summary=json.loads((FIG/'source_summary.json').read_text(encoding='utf-8'))
    audited=json.loads((HERE/'iid997_independent_results_audit.json').read_text(encoding='utf-8'))
    assert summary['status']=='COMPLETE_PROTOCOL_35_RUNS' and summary['test_case_count']==160
    for tag,c in summary['paired_contrasts'].items():
        a=audited['primary_contrasts'][tag]
        assert c['point_estimate_equal_case_seed_mean']==a['point']
        assert c['intervals']['two_way']['percentile_95']==a['two_way95']
    percase=rows(FIG/'source_per_case_metrics.csv')
    assert len(percase)==5600
    seed_values=collections.defaultdict(list);case_values=collections.defaultdict(list)
    for r in percase:
        seed_values[(r['model'],int(r['seed']))].append(float(r['MAE']))
        case_values[(r['model'],r['sample_id'])].append(float(r['MAE']))
    plotted=rows(FIG/'plotted_seed_mae.csv')
    assert len(plotted)==35
    for r in plotted:
        v=np.mean(seed_values[(r['model'],int(r['seed']))])
        assert np.isclose(float(r['MAE']),v,atol=1e-14,rtol=0)
        assert np.isclose(float(r['MAE_percentage_points']),100*v,atol=1e-12,rtol=0)
    ecdf=rows(FIG/'plotted_case_ecdf.csv');assert len(ecdf)==1120
    for model in summary['models']:
        group=[r for r in ecdf if r['model']==model];assert len(group)==160
        assert all(float(a['seed_mean_case_MAE'])<=float(b['seed_mean_case_MAE']) for a,b in zip(group[:-1],group[1:]))
        for rank,r in enumerate(group,1):
            v=np.mean(case_values[(model,r['sample_id'])]);assert len(case_values[(model,r['sample_id'])])==5
            assert np.isclose(float(r['seed_mean_case_MAE']),v,atol=1e-14,rtol=0)
            assert np.isclose(float(r['MAE_percentage_points']),100*v,atol=1e-12,rtol=0)
            assert float(r['cumulative_fraction'])==rank/160
    for r in rows(FIG/'plotted_paired_contrasts.csv'):
        a=audited['primary_contrasts'][r['contrast']]
        assert float(r['mean_difference'])==a['point']
        assert [float(r['percentile_95_lower']),float(r['percentile_95_upper'])]==a['two_way95']
    table=re.findall(r'^(.*?) & \$(.*?)\\pm(.*?)\$ & \$(.*?)\$\\\\$',text,re.MULTILINE)
    assert len(table)==7
    for row,model in zip(table,summary['model_summaries']):
        m=model['metrics']; expected=[f"{100*m['equal_case_MAE']['mean']:.3f}",f"{100*m['equal_case_MAE']['sample_sd']:.3f}",f"{100*m['equal_case_RMSE']['mean']:.3f}"]
        assert list(row[1:])==expected,(row,expected)
    figure_source=(FIG/'draw_iid_results_v08.py').read_text(encoding='utf-8')
    assert "point = c['mean_difference'] * 100; low = c['percentile_95_lower'] * 100; high = c['percentile_95_upper'] * 100" in figure_source
    recommendations=[]
    if 'Zero-edge graph &' in text:
        recommendations.append({'id':'IID-INCR-01','priority':'minor_method_label','status':'OPEN','location':'05_results.tex, Table iid, zero-edge graph row',
          'recommendation':'Rename Zero-edge graph to Graph, zero edge features (or Zero-edge-feature graph); adjacency and message passing remain present in this ablation.'})
    if 'In dimensionless index units' not in text:
        recommendations.append({'id':'IID-INCR-02','priority':'minor_unit_clarity','status':'OPEN','location':'05_results.tex, first primary paired-contrast sentence',
          'recommendation':'Explicitly label the prose contrasts as dimensionless index units because the preceding table and Figure 3 use 100-times percentage-point units. Existing numbers are correct.'})
    bib=SCI/'latex_v08/references.bib';bbl=SCI/'latex_v08/build/main.bbl';cover=SCI.parent/'3_最终成果/论文/LaTeX源码/cn/chapters/frontcover.tex'
    assert "type = {Bachelor's thesis}" in bib.read_text(encoding='utf-8')
    assert "Bachelor's thesis, Tongji University" in bbl.read_text(encoding='utf-8')
    assert 'CrackFireNet: Graph Neural Network-Based Fire Resistance Prediction for Cracked Timber Beams' in cover.read_text(encoding='utf-8')
    paths=[text_path,SCI/'latex_v08/sections/02_engineering_data.tex',bib,bbl,cover,FIG/'caption_en.tex',FIG/'iid_results_v08.pdf',FIG/'pdf_final_QA.png',FIG/'draw_iid_results_v08.py',FIG/'source_summary.json',FIG/'source_per_case_metrics.csv',HERE/'iid997_independent_results_audit.json',HERE/'iid997_engineering_proxy_ci.json']
    embedded=SCI/'latex_v08/figures/iid_results_v08.pdf'
    if embedded.exists():
        assert sha(embedded)==sha(FIG/'iid_results_v08.pdf');paths.append(embedded)
    result={'decision':'DEVELOPMENT_REVIEW_IID_INCREMENT_NUMERICALLY_CONSISTENT',
      'date_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'Only new IID Results prose/table/Figure 3/caption and Peng thesis attribution. Not the remaining protocols or full manuscript submission decision.',
      'submission_or_final_paper_pass':False,'reviewed_runs':35,'independent_geometries':160,'seeds':[42,43,44,45,46],
      'verified':{'all_seven_table_rows_and_mean_SD_RMSE_scaling':True,'plotted_seed_values':35,'plotted_case_ECDF_rows':1120,'ECDF_is_seed_mean_error_not_ensemble_prediction_error':True,
        'plotted_primary_contrasts_equal_audited_two_way_intervals':True,'table_and_all_figure_panels_error_units_times_100':True,
        'conditional_events_60_second_grid_and_censoring_explained':True,'primary_and_mean_vs_sum_superiority_not_claimed':True,
        'saturated_classification_endpoint_set_advantage_and_within_family_rank_limit_retained':True,'secondary_unadjusted_comparison_labeled':True,
        'bachelor_thesis_cover_title_and_rendered_bibliography_match':True,'prior_RMSE_source_explicitly_thesis_reported':True,'visual_PDF_preview_inspected':True},
      'recommendations':recommendations,
      'ongoing_evidence_limits':['Full count and fire-family matrix pending; no partial protocol inspected in this review.',
        'IID statistics do not establish graph advantage, physical-capacity validity, unrestricted topology transfer, or final EAAI suitability.',
        'No model selection or training changes authorized by the observed IID results.'],
      'source_sha256':{str(p):sha(p) for p in paths}}
    (HERE/'decision_v08_iid_increment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'decision':result['decision'],'recommendations':recommendations,'source_files':len(paths)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
