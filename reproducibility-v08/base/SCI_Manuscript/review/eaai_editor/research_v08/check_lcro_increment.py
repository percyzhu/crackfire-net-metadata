"""Bind independent LCRO manuscript increment QA; no new model/FE computation."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,hashlib,re,collections
import numpy as np
HERE=Path(__file__).resolve().parent
SCI=HERE.parents[2]
FIG=SCI/'figures_v08/transfer_results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def close(a,b):assert np.isclose(a,b,atol=2e-13,rtol=2e-12),(a,b)
def main():
    text_path=SCI/'latex_v08/sections/05_results.tex';text=text_path.read_text(encoding='utf-8')
    main_path=SCI/'latex_v08/main.tex';abstract=main_path.read_text(encoding='utf-8')
    discussion=SCI/'latex_v08/sections/06_discussion.tex'
    count_summary=read(FIG/'source_summary.json');iid_summary=read(SCI/'figures_v08/iid_results/source_summary.json')
    audit_path=HERE/'lcro_9_15_independent_results_audit_v2.json';audit=read(audit_path)
    topology_path=HERE/'lcro_9_15_same_weight_topology_audit.json';topology=read(topology_path)
    assert count_summary['status']=='COMPLETE_PROTOCOL_35_RUNS' and count_summary['test_case_count']==264
    table=re.findall(r'^(.*?) & \$(.*?)\\pm(.*?)\$ & \$(.*?)\$\\\\$',text,re.MULTILINE)
    assert len(table)==14
    for row,model in zip(table,iid_summary['model_summaries']+count_summary['model_summaries']):
        m=model['metrics'];expected=[f"{100*m['equal_case_MAE']['mean']:.3f}",f"{100*m['equal_case_MAE']['sample_sd']:.3f}",f"{100*m['equal_case_RMSE']['mean']:.3f}"]
        assert list(row[1:])==expected,(row,expected)
    for tag,c in count_summary['paired_contrasts'].items():
        a=audit['primary_contrasts'][tag];assert c['point_estimate_equal_case_seed_mean']==a['point'];assert c['intervals']['two_way']['percentile_95']==a['two_way95']
    reduction=audit['primary_contrasts']['graph_vs_capacity_matched']['relative_candidate_MAE_reduction']
    assert f'{100*reduction:.2f}\\%' in text and f'{100*reduction:.2f}\\%' in abstract
    assert '$[1.59,11.94]\\times10^{-4}$' in text and '$[1.59,11.94]\\times10^{-4}$' in abstract
    source=rows(FIG/'source_per_case_metrics.csv');assert len(source)==9240
    bygroup=collections.defaultdict(list)
    for r in source:bygroup[(r['model'],int(r['seed']),int(r['crack_count']))].append(float(r['MAE']))
    plotted=rows(FIG/'plotted_crack_count_seed_mae.csv');assert len(plotted)==245
    for r in plotted:
        v=bygroup[(r['model'],int(r['seed']),int(r['crack_count']))]
        assert len(v)==int(r['case_count']);close(float(r['MAE']),np.mean(v));close(float(r['MAE_percentage_points']),100*np.mean(v))
    actual={(r['model'],r['seed'],r['rule']):r for r in topology['per_seed']}
    graph_rows=rows(FIG/'plotted_same_checkpoint_connectivity.csv');assert len(graph_rows)==45
    for r in graph_rows:
        mode,seed,view=r['model'],int(r['seed']),r['view'];a=actual[(mode,seed,'radius' if view=='complete' else view)]
        value=a['complete_MAE'] if view=='complete' else a['view_MAE'];delta=0 if view=='complete' else a['MAE_change_vs_complete']
        close(float(r['MAE']),value);close(float(r['MAE_change_vs_complete']),delta);close(float(r['change_percentage_points']),100*delta)
        assert r['checkpoint_sha256']==a['checkpoint_sha256']
    assert 'compound count-and-source shift' in text and 'not a tested protocol interaction' in text
    assert '218 or 219 cases' in text and 'occurring in different seeds' in text
    assert 'source composition' in abstract and 'fixed archived numerical-response convention' in abstract
    assert 'share an exposure-history encoder and comprise a fire-only baseline' in abstract
    assert 'On the count-extrapolation test, the sum graph also improves' in text
    source_figure=FIG/'transfer_results_v08.pdf';embedded=SCI/'latex_v08/figures/transfer_results_v08.pdf'
    assert sha(source_figure)==sha(embedded)
    paths=[main_path,text_path,discussion,source_figure,embedded,FIG/'caption_en.tex',FIG/'pdf_final_QA.png',FIG/'source_summary.json',FIG/'source_per_case_metrics.csv',FIG/'plotted_crack_count_seed_mae.csv',FIG/'plotted_same_checkpoint_connectivity.csv',audit_path,topology_path,HERE/'lcro_9_15_engineering_proxy_ci.json',Path(__file__)]
    decision={'decision':'DEVELOPMENT_REVIEW_LCRO_INCREMENT_READY_FOR_WORKING_DRAFT',
      'review_date_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'New abstract/LCRO Results/Table 3/Figure 4/Discussion and their IID context. Not LOCO predictions or final full-paper/journal review.',
      'submission_pass':False,'full_matrix_pass':False,'reviewed_protocols':['iid997','lcro_9_15'],
      'verified':{'seven_LCRO_table_rows_MAE_SD_RMSE_times100':True,'IID_table_preserved':True,'245_count_plot_points':True,
        '45_same_checkpoint_view_plot_points':True,'relative_MAE_reduction_5_48_percent_vs_absolute_index_CI_distinct':True,
        'five_seed_primary_effect_and_two_way_interval_match':True,'same_weight_view_deterioration_and_zero_edge_exception_retained':True,
        'source_batch_compound_shift_and_no_interaction_test_disclosed':True,'conditional_event_coverage_censoring_recross_and60s_grid_correct':True,
        'secondary_final_vs_time_average_differentiation_correct':True,'abstract_target_scope_restricted_to_archive':True,
        'figure_PDF_visual_inspection':True,'embedded_figure_is_identical':True},
      'minor_text_corrections_applied':[{'file':str(main_path),'correction':'Explicitly include the fire-only baseline in the seven-representation description; previous phrasing could imply every model included geometry.'},
        {'file':str(text_path),'correction':'Attribute the time-average improvement to sum-graph versus matched-set prediction on the count test; avoid suggesting extrapolation itself improves accuracy.'}],
      'open_recommendations':[],
      'compilation_note':'Parent notified to rebuild the manuscript after the two small text changes. Figure PDF itself was unchanged and visually checked; no claim that the prior full-manuscript PDF already includes these wording corrections.',
      'development_conclusion':'The reviewed increment can enter the working manuscript. It supports a bounded graph benefit on the prespecified N1–8 to N9–15/source005 compound shift, with explicit null IID, unstable degree-normalization gains and sparse-connectivity limitations. No old numerical or arbitrary relative-gain threshold was used to discard the observed effect.',
      'source_sha256':{str(p):sha(p) for p in paths}}
    (HERE/'decision_v08_lcro_increment.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'decision':decision['decision'],'open_recommendations':[],'minor_text_corrections':2,'hashed_files':len(paths)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
