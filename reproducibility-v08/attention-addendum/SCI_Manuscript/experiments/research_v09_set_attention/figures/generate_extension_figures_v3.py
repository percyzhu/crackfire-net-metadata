"""Gated exploratory figures: metadata first, no partial-result plotting."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
EXT=HERE.parent
PLAN=EXT/'plan_set_attention_60_v1.json'
PLAN_SHA='8d21e13af116dc97556b37060daaf86afed5f7ba645a5bdae18e3d4ffb3b05a9'
AGGREGATOR_SHA='aa06a3280fed95be571f1b39677c84c20d193804897d4f4d55ec7a99053a2e7a'
REVIEW_STATUS='PASS_INDEPENDENT_NUMERICAL_REVIEW_60_NEW_PLUS120_REUSED'
MODELS=('capacity_matched_deepsets','gnn','set_attention')
MODEL_LABELS={'capacity_matched_deepsets':'Matched set','gnn':'Sum GNN','set_attention':'Set attention'}
COLORS={'capacity_matched_deepsets':'#71818C','gnn':'#315D80','set_attention':'#BB7439'}
CONTRASTS=('attention_vs_matched_set','attention_vs_sum_GNN')
CONTRAST_LABELS=('Matched set − attention','Sum GNN − attention')
BASELINES=('capacity_matched_deepsets','gnn')
LABELS={'iid997':'IID','lcro_9_15':'LCRO (9–15)', 'loco_iso834':'ISO 834',
    'loco_astm_e119':'ASTM E119','loco_perturbed_iso':'Perturbed ISO', 'loco_plateau':'Plateau',
    'loco_decay':'Decay','loco_log_variant':'Log variant','loco_external_fire':'External fire',
    'loco_bilinear':'Bilinear','loco_linear':'Linear','loco_smoldering':'Smoldering'}


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def path_inside(path,root):
    path=Path(path).resolve();require(path.is_relative_to(Path(root).resolve()),'Path outside declared directory');return path


def metadata_gate(audit_dir=None,review_path=None):
    """All pre-review operations are metadata reads and source hashes only."""
    require(sha(PLAN)==PLAN_SHA,'Extension plan SHA changed');plan=read_json(PLAN)
    require(plan['epistemic_status']=='AFTER_ORIGINAL_TEST_RESULTS_REVIEW_EXPLORATORY','Wrong epistemic scope')
    require(plan['new_run_count']==60 and len(plan['protocols'])==12 and plan['seeds']==[42,43,44,45,46],'Wrong plan matrix')
    require(set(plan['protocols'])==set(LABELS),'Unrecognized or omitted protocol')
    for group in ('original_source_sha256','extension_source_sha256'):
        for path,digest in plan[group].items():require(sha(path)==digest,'Frozen source changed: '+path)
    root=Path(plan['runs_root']);queue_path=root/'queue_status.json'
    queue=read_json(queue_path) if queue_path.exists() else None
    state={'plan_sha256':PLAN_SHA,'expected_new_runs':60,'queue_completed_metadata_count':len(queue['completed']) if queue else 0,
           'performance_payloads_read':0,'prediction_arrays_read':0,'plotting_backend_loaded':False,'figures_created':0}
    if queue and queue['status']=='ERROR_STOPPED_EXTENSION':raise ValueError('Training queue failed; preserve the failed attempt')
    if not queue or queue['status']!='COMPLETE_60_EXPLORATORY_EXTENSION':
        return plan,{**state,'status':'WAITING_ALL60_COMPLETE_QUEUE'}
    expected={(p,'set_attention',s) for p in plan['protocols'] for s in plan['seeds']}
    actual=[(v['protocol'],v['model'],v['seed']) for v in queue['completed']]
    require(len(actual)==len(set(actual))==60 and set(actual)==expected,'Queue identity matrix differs')
    require(queue['active'] is None and queue['plan_sha256']==PLAN_SHA,'Active/wrong complete queue')
    allowed={str((root/p/m/('seed_'+str(s))/'result.json').resolve()) for p,m,s in expected}
    require(not [p for p in root.glob('*/*/*/result.json') if str(p.resolve()) not in allowed],'Extra result paths')
    for protocol,mode,seed in expected:
        folder=root/protocol/mode/('seed_'+str(seed))
        require(all((folder/f).is_file() for f in ('result.json','best.pt','test_predictions.npz','learning_curve.csv')),'Missing completed artifacts')
        status=read_json(folder/'status.json');meta=read_json(folder/'run_metadata.json')
        require(status['status']=='COMPLETE_EXPLORATORY_EXTENSION','Incomplete cell')
        require((meta['protocol'],meta['mode'],meta['seed'])==(protocol,mode,seed) and meta['plan_sha256']==PLAN_SHA,'Cell identity differs')
    if audit_dir is None or not (Path(audit_dir)/'report.json').is_file():
        return plan,{**state,'status':'WAITING_FINAL60_AGGREGATE'}
    audit_dir=path_inside(audit_dir,EXT);report_path=audit_dir/'report.json';report=read_json(report_path)
    require(not (audit_dir/'failure.json').exists(),'Aggregate failure artifact exists')
    require(report['status']=='COMPLETE_EXPLORATORY_60_NEW_PLUS120_REUSED_MATRIX','Aggregate not complete')
    require(report['new_runs_audited']==60 and report['reused_comparator_runs']==120 and report['cpu_prediction_replay_enabled'] is True,'Aggregate matrix or replay incomplete')
    require(report['plan_sha256']==PLAN_SHA and report['epistemic_status']==plan['epistemic_status'],'Aggregate provenance differs')
    require(report['protocols']==list(plan['protocols']) and report['models']==list(MODELS) and report['seeds']==plan['seeds'],'Aggregate protocol/model/seed coverage differs')
    if review_path is None or not Path(review_path).is_file():
        return plan,{**state,'status':'WAITING_INDEPENDENT_NUMERICAL_RESULT_REVIEW','aggregate_report_sha256':sha(report_path)}
    review=read_json(review_path)
    if review.get('status')!=REVIEW_STATUS or review.get('numerical_review_passed') is not True:
        return plan,{**state,'status':'WAITING_INDEPENDENT_NUMERICAL_RESULT_REVIEW','review_status':review.get('status')}
    require(review['extension_plan_sha256']==PLAN_SHA and review['aggregate_report_sha256']==sha(report_path),'Independent review is not bound to this aggregate')
    require(review['aggregate_run_integrity_sha256']==sha(audit_dir/'run_integrity.json'),'Independent integrity SHA differs')
    require(review['completed_new_runs']==60 and review['reused_comparator_runs']==120,'Independent review matrix incomplete')
    require(review['protocols']==list(plan['protocols']) and review['models']==list(MODELS) and review['seeds']==plan['seeds'],'Independent review coverage differs')
    # Reading quantitative audit payloads is allowed only after the above review gate.
    require(sha(EXT/'aggregate_extension.py')==AGGREGATOR_SHA,'Candidate aggregator changed; requires a new reviewed generator version')
    require(report['analysis_source_sha256'][str(EXT/'aggregate_extension.py')]==AGGREGATOR_SHA,'Aggregate uses a different candidate')
    for path,digest in report['analysis_source_sha256'].items():require(sha(path)==digest,'Analysis source changed')
    for relative,digest in report['outputs_sha256'].items():
        require(sha(path_inside(audit_dir/relative,audit_dir))==digest,'Statistical output SHA changed: '+relative)
    integrity=read_json(audit_dir/'run_integrity.json');audits=integrity['new_run_audits']
    keys=[(v['protocol'],v['model'],v['seed']) for v in audits]
    require(len(audits)==len(set(keys))==60 and set(keys)==expected,'Integrity matrix incomplete')
    for audit in audits:
        require(audit['status']=='PASSED_EXTENSION_BINDINGS_AND_REPLAY' and audit['plan_sha256']==PLAN_SHA,'Unpassed new audit')
        replay=audit['computational_replay'];error=replay['cpu_max_abs_difference']
        require(math.isfinite(error) and error>=0,'Invalid replay difference')
        if error<3e-6:
            require(replay['cpu_within_original_tolerance'] is True and replay['status']=='CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE','Inconsistent CPU replay')
        else:
            require(replay['cpu_within_original_tolerance'] is False and replay['same_backend_gpu_bitwise_equal'] is True and replay['gpu_max_abs_difference']==0,
                    'CPU flag lacks exact original GPU replay')
        folder=root/audit['protocol']/audit['model']/('seed_'+str(audit['seed']))
        for name,digest in audit['source_artifact_sha256'].items():require(sha(folder/name)==digest,'New artifact changed after independent review')
    require(len(integrity['reused_original_comparators'])==120,'Original comparator audit count differs')
    for ref in plan['reused_comparator_artifacts']:
        for item in ref['artifacts'].values():require(sha(item['path'])==item['sha256'],'Reused original artifact changed')
    return plan,{**state,'status':'READY_COMPLETE_AND_INDEPENDENTLY_REVIEWED',
        'audit_dir':str(audit_dir),'aggregate_report_sha256':sha(report_path),'aggregate_run_integrity_sha256':sha(audit_dir/'run_integrity.json'),
        'independent_review_path':str(Path(review_path).resolve()),'independent_review_sha256':sha(review_path)}


def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-13)


def load_plot_data(plan,gate):
    import numpy as np
    root=Path(gate['audit_dir']);means=[];seeds=[];effects=[];count_map={}
    for protocol,spec in plan['protocols'].items():
        require(sha(spec['split_path'])==spec['split_sha256'],'Split changed')
        split=read_json(spec['split_path']);ids=split['test'];count_map[protocol]=len(ids)
        summary=read_json(root/protocol/'summary.json')
        require(summary['status']=='COMPLETE_EXPLORATORY_PROTOCOL_5_NEW_PLUS10_REUSED_RUNS','Incomplete summary')
        require(summary['models']==list(MODELS) and summary['seeds']==plan['seeds'],'Summary coverage differs')
        require(summary['protocol']==protocol and summary['test_case_count']==summary['independent_geometry_clusters']==len(ids),'Protocol geometry count differs')
        with (root/protocol/'per_case_metrics.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        require(len(rows)==15*len(ids),'Per-case table does not have 3 models by5 seeds byN')
        require({(r['model'],int(r['seed'])) for r in rows}=={(m,s) for m in MODELS for s in plan['seeds']},'Per-case identities differ')
        summaries={v['model']:v for v in summary['model_summaries']};require(set(summaries)==set(MODELS),'Missing model summaries')
        array={}
        for model in MODELS:
            array[model]=[]
            for seed in plan['seeds']:
                items=[r for r in rows if r['model']==model and int(r['seed'])==seed]
                require([r['sample_id'] for r in items]==ids and len({r['geometry_id'] for r in items})==len(ids),'Ordered sample/geometry identities differ')
                value=np.asarray([float(r['MAE']) for r in items]);require(np.isfinite(value).all() and np.all(value>=0),'Invalid per-case MAE')
                array[model].append(value);seeds.append({'protocol':protocol,'model':model,'seed':seed,'independent_test_geometries':len(ids),
                    'equal_case_MAE':float(value.mean()),'display_MAE_times1000':float(value.mean()*1000)})
            array[model]=np.stack(array[model]);seed_means=array[model].mean(1);mean=float(seed_means.mean());sd=float(seed_means.std(ddof=1))
            declared=summaries[model];metric=declared['metrics']['equal_case_MAE']
            require(declared['seed_count']==5 and declared['test_case_count']==len(ids),'Seed count differs')
            require(close(mean,metric['mean']) and close(sd,metric['sample_sd']),'Summary mean/SD differs from case CSV')
            means.append({'protocol':protocol,'model':model,'independent_test_geometries':len(ids),'seed_count':5,
                'mean_equal_case_MAE':mean,'sample_SD_over_seeds':sd,'display_mean_times1000':mean*1000,'display_SD_times1000':sd*1000})
        require(set(summary['paired_contrasts'])==set(CONTRASTS),'Unexpected contrasts')
        for tag,baseline in zip(CONTRASTS,BASELINES):
            ci=summary['paired_contrasts'][tag];point=float((array[baseline]-array['set_attention']).mean())
            require(close(point,ci['point_estimate_equal_case_seed_mean']),'Paired effect differs from case CSV')
            require(ci['bootstrap_repeats']==5000 and ci['bootstrap_random_seed']==20260907 and ci['seeds']==plan['seeds'],'Bootstrap recipe differs')
            require(ci['seed_count']==5 and ci['test_cases']==ci['independent_geometry_clusters']==len(ids),'Bootstrap sample unit differs')
            by_seed=(array[baseline]-array['set_attention']).mean(1)
            require(np.allclose(by_seed,ci['paired_effect_by_seed'],rtol=1e-11,atol=1e-13),'Paired seed effects differ')
            record={'protocol':protocol,'contrast':tag,'baseline':baseline,'candidate':'set_attention','independent_test_geometries':len(ids),
                'seed_count':5,'point_baseline_minus_attention':point,'display_point_times1000':point*1000,'positive_favors':'set_attention'}
            for kind in ('geometry_only','seed_only','two_way'):
                interval=ci['intervals'][kind];require(interval['status']=='ESTIMATED','Undefined primary interval must not be drawn as zero')
                lo,hi=interval['percentile_95'];require(math.isfinite(lo) and math.isfinite(hi) and lo<=hi,'Invalid interval')
                record[kind+'_CI95_low']=lo;record[kind+'_CI95_high']=hi
                record[kind+'_display_CI95_low_times1000']=lo*1000;record[kind+'_display_CI95_high_times1000']=hi*1000
            effects.append(record)
    require(len(means)==36 and len(seeds)==180 and len(effects)==24,'Figure data coverage incomplete')
    return means,seeds,effects,count_map


def csv_write(path,rows):
    with path.open('x',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def draw_figures(plan,means,seeds,effects,counts,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'axes.labelsize':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
    protocols=list(plan['protocols']);names=[f'{LABELS[p]}  (n={counts[p]})' for p in protocols];y=np.arange(12)
    lookup={(r['protocol'],r['model']):r for r in means}
    seed_values={(p,m):[r['display_MAE_times1000'] for r in seeds if r['protocol']==p and r['model']==m] for p in protocols for m in MODELS}
    upper=max([r['display_mean_times1000']+r['display_SD_times1000'] for r in means]+[r['display_MAE_times1000'] for r in seeds])
    upper=max(upper*1.08,1e-6)
    lower=min(0.,min(r['display_mean_times1000']-r['display_SD_times1000'] for r in means))*1.08
    fig,axes=plt.subplots(1,3,figsize=(180/25.4,136/25.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.235,right=.985,bottom=.16,top=.88,wspace=.12)
    for j,(ax,model) in enumerate(zip(axes,MODELS)):
        avg=[lookup[(p,model)]['display_mean_times1000'] for p in protocols]
        sd=[lookup[(p,model)]['display_SD_times1000'] for p in protocols]
        ax.errorbar(avg,y,xerr=sd,fmt='o',ms=3.5,lw=1.0,capsize=2,color=COLORS[model],zorder=4)
        for k,p in enumerate(protocols):ax.scatter(seed_values[(p,model)],y[k]+np.linspace(-.11,.11,5),s=7,color=COLORS[model],alpha=.40,zorder=3)
        ax.set_title(MODEL_LABELS[model],pad=9);ax.set_xlim(lower,upper);ax.set_ylim(11.55,-.55)
        ticks=MaxNLocator(nbins=3).tick_values(lower,upper)
        ax.set_xticks([t for t in ticks if lower<=t<=upper]);ax.set_xlim(lower,upper)
        ax.set_yticks(y);ax.set_yticklabels(names);ax.tick_params(axis='y',length=0,labelleft=j==0)
        ax.tick_params(axis='x',labelsize=7.5);ax.grid(axis='x',color='#E3E8EB',lw=.5);ax.set_axisbelow(True)
        ax.axhline(1.5,color='#ADBCC5',lw=.8);ax.text(-.06,1.10,chr(97+j),transform=ax.transAxes,weight='bold',fontsize=10)
    fig.text(.60,.095,r'MAE ($\times10^{-3}$ response-index units)',ha='center',fontsize=9)
    fig.text(.60,.045,'Five seeds: small points; mean: large point; whisker: sample SD',ha='center',fontsize=7.5)
    figures=[('extension_mae',fig)]
    points={(r['protocol'],r['contrast']):r for r in effects}
    limit=max(abs(r[k]) for r in effects for k in ('display_point_times1000','two_way_display_CI95_low_times1000','two_way_display_CI95_high_times1000'))
    limit=max(limit*1.10,1e-6)
    fig=plt.figure(figsize=(180/25.4,174/25.4))
    axes=[fig.add_axes([.235,.15,.34,.57]),fig.add_axes([.645,.15,.34,.57])]
    zoom_axes=[fig.add_axes([.235,.845,.34,.072]),fig.add_axes([.645,.845,.34,.072])]
    zoom_limit=max(abs(points[('lcro_9_15',tag)][k]) for tag in CONTRASTS for k in
        ('display_point_times1000','two_way_display_CI95_low_times1000','two_way_display_CI95_high_times1000'))*1.15
    zoom_limit=max(zoom_limit,1e-6)
    for j,(ax,tag,baseline) in enumerate(zip(zoom_axes,CONTRASTS,BASELINES)):
        row=points[('lcro_9_15',tag)];lo=row['two_way_display_CI95_low_times1000'];hi=row['two_way_display_CI95_high_times1000']
        ax.axvline(0,color='#8A959C',ls=(0,(3,2)),lw=.8,zorder=0)
        ax.plot([lo,hi],[0,0],color=COLORS[baseline],lw=1.2)
        ax.plot([lo,lo],[-.10,.10],color=COLORS[baseline],lw=.7);ax.plot([hi,hi],[-.10,.10],color=COLORS[baseline],lw=.7)
        ax.scatter([row['display_point_times1000']],[0],s=18,marker='D',color=COLORS[baseline],zorder=3)
        ax.set_xlim(-zoom_limit,zoom_limit);ax.set_ylim(-.8,.8);ax.set_yticks([])
        ticks=MaxNLocator(nbins=4).tick_values(-zoom_limit,zoom_limit)
        ax.set_xticks([t for t in ticks if -zoom_limit<=t<=zoom_limit]);ax.tick_params(axis='x',labelsize=7)
        ax.set_title(CONTRAST_LABELS[j],pad=8)
        ax.text(-.06,1.37,chr(97+j),transform=ax.transAxes,weight='bold',fontsize=10)
        ax.text(.5,.04,f"{row['display_point_times1000']:+.3f} [{lo:+.3f}, {hi:+.3f}]",transform=ax.transAxes,ha='center',fontsize=7)
    fig.text(.225,.88,'LCRO zoom\n1–8 → 9–15 cracks\nn=264',ha='right',va='center',fontsize=8)
    fig.text(.61,.795,r'LCRO: expanded linear scale ($\times10^{-3}$)',ha='center',fontsize=8)
    for j,(ax,tag,baseline) in enumerate(zip(axes,CONTRASTS,BASELINES)):
        ax.axvline(0,color='#8A959C',ls=(0,(3,2)),lw=.8,zorder=0)
        for i,p in enumerate(protocols):
            row=points[(p,tag)];lo=row['two_way_display_CI95_low_times1000'];hi=row['two_way_display_CI95_high_times1000']
            ax.plot([lo,hi],[i,i],color=COLORS[baseline],lw=1.2)
            ax.plot([lo,lo],[i-.08,i+.08],color=COLORS[baseline],lw=.7);ax.plot([hi,hi],[i-.08,i+.08],color=COLORS[baseline],lw=.7)
            ax.scatter([row['display_point_times1000']],[i],s=18,marker='D',color=COLORS[baseline],zorder=3)
        ax.set_title('All 12 protocols: full linear scale',pad=8,fontsize=8);ax.set_xlim(-limit,limit);ax.set_ylim(11.55,-.55)
        ticks=MaxNLocator(nbins=4).tick_values(-limit,limit)
        ax.set_xticks([t for t in ticks if -limit<=t<=limit]);ax.set_xlim(-limit,limit)
        ax.set_yticks(y);ax.set_yticklabels(names);ax.tick_params(axis='y',length=0,labelleft=j==0)
        ax.tick_params(axis='x',labelsize=7.5);ax.grid(axis='x',color='#E3E8EB',lw=.5);ax.set_axisbelow(True)
        ax.axhline(1.5,color='#ADBCC5',lw=.8);ax.text(-.06,1.05,chr(99+j),transform=ax.transAxes,weight='bold',fontsize=10)
    fig.text(.60,.095,r'Paired MAE difference ($\times10^{-3}$); 95% crossed-bootstrap interval',ha='center',fontsize=8.5)
    fig.text(.60,.052,'Negative: comparator favored     |     Positive: attention favored',ha='center',fontsize=7.7)
    figures.append(('extension_effects',fig))
    qa=[]
    for name,figure in figures:
        figure.canvas.draw();renderer=figure.canvas.get_renderer();outside=[]
        for artist in figure.findobj(matplotlib.text.Text):
            if not artist.get_visible() or not artist.get_text():continue
            box=artist.get_window_extent(renderer);canvas=figure.bbox
            if box.x0<canvas.x0-2 or box.y0<canvas.y0-2 or box.x1>canvas.x1+2 or box.y1>canvas.y1+2:outside.append(artist.get_text())
        require(not outside,'Text outside canvas: '+repr(outside))
        for ext in ('pdf','svg','png'):figure.savefig(output/(name+'.'+ext),dpi=300)
        qa.append({'figure':name,'canvas_width_mm':float(figure.get_size_inches()[0]*25.4),'canvas_height_mm':float(figure.get_size_inches()[1]*25.4),'text_outside_canvas':outside,
                   'all12_protocols_retained':True,'visual_inspection_still_required':True})
        plt.close(figure)
    return {'matplotlib_version':matplotlib.__version__,'numpy_version':np.__version__,'figures':qa,'visual_QA_pass':False,'LaTeX_compile_tested':False}


def write_tables(plan,means,effects,counts,output):
    mean={(r['protocol'],r['model']):r for r in means};effect={(r['protocol'],r['contrast']):r for r in effects}
    lines=[r'\begin{table}[t]',r'\centering\small',r'\caption{Exploratory three-model MAE ($\times10^{-3}$): five-seed mean $\pm$ sample SD. The independent test geometry count is $n$.}',
           r'\label{tab:exploratory_attention_mae}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Protocol & $n$ & Matched set & Sum GNN & Set attention \\',r'\midrule']
    paired=[r'\begin{table}[t]',r'\centering\small',r'\caption{Exploratory paired MAE differences ($\times10^{-3}$) with 95\% crossed geometry--seed bootstrap intervals. Positive values favor attention.}',
            r'\label{tab:exploratory_attention_effects}',r'\begin{tabular}{lrrr}',r'\toprule',r'Protocol & $n$ & Matched set $-$ attention & Sum GNN $-$ attention \\',r'\midrule']
    for i,protocol in enumerate(plan['protocols']):
        if i==2:lines.append(r'\midrule');paired.append(r'\midrule')
        label=LABELS[protocol].replace('–','--')
        values=[f"${mean[(protocol,m)]['display_mean_times1000']:.3f} \\pm {mean[(protocol,m)]['display_SD_times1000']:.3f}$" for m in MODELS]
        lines.append(' & '.join([label,str(counts[protocol]),*values])+r' \\')
        values=[]
        for tag in CONTRASTS:
            row=effect[(protocol,tag)]
            values.append(f"${row['display_point_times1000']:+.3f}\\;[{row['two_way_display_CI95_low_times1000']:+.3f}, {row['two_way_display_CI95_high_times1000']:+.3f}]$")
        paired.append(' & '.join([label,str(counts[protocol]),*values])+r' \\')
    tail=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
    for name,content in [('table_extension_mae.tex',lines+tail),('table_extension_effects.tex',paired+tail)]:
        with (output/name).open('x',encoding='utf-8') as f:f.write('\n'.join(content)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('command',choices=('check-ready','generate'));p.add_argument('--audit-dir');p.add_argument('--review',default=str(EXT/'independent_full_matrix_review.json'));p.add_argument('--output')
    args=p.parse_args(argv);plan,gate=metadata_gate(args.audit_dir,args.review)
    if args.command=='check-ready' or gate['status']!='READY_COMPLETE_AND_INDEPENDENTLY_REVIEWED':
        print(json.dumps(gate,indent=2));return 0 if gate['status'].startswith('READY') else 2
    require(args.output,'Specify a new figure output snapshot directory');output=path_inside(args.output,HERE)
    require(not output.exists(),'Refuse to overwrite an earlier figure snapshot')
    means,seeds,effects,counts=load_plot_data(plan,gate)
    output.mkdir(parents=True)
    try:
        csv_write(output/'plotted_model_MAE.csv',means);csv_write(output/'plotted_seed_MAE.csv',seeds);csv_write(output/'plotted_paired_effects.csv',effects)
        qa=draw_figures(plan,means,seeds,effects,counts,output);write_tables(plan,means,effects,counts,output)
        write_json(output/'render_QA.json',qa)
        sources=[Path(__file__),HERE/'FIGURE_CONTRACT.md',HERE/'captions_extension_v3_en.tex',HERE/'methods_extension_draft.tex',HERE/'render_revision_v3.md']
        manifest={'status':'GENERATED_FROM_COMPLETE_INDEPENDENTLY_REVIEWED_EXTENSION_VISUAL_QA_PENDING',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epistemic_status':plan['epistemic_status'],
            'gate':gate,'protocols':list(plan['protocols']),'models':list(MODELS),'seeds':plan['seeds'],
            'table_and_plot_units':'dimensionless archived-response MAE multiplied by1000; not percent improvement',
            'primary420_evidence_replaced':False,'new_training_runs':60,'reused_comparator_runs':120,'independent_geometries_not_multiplied_by_seeds':True,
            'source_sha256':{str(f):sha(f) for f in sources},'outputs_sha256':{f.name:sha(f) for f in output.iterdir() if f.is_file()},
            'visual_QA_pass':False,'LaTeX_compile_tested':False,'submission_pass':False}
        write_json(output/'artifact_manifest.json',manifest)
        print(json.dumps({'status':manifest['status'],'output':str(output),'artifact_manifest_sha256':sha(output/'artifact_manifest.json')},indent=2))
    except BaseException as error:
        write_json(output/'failure.json',{'status':'FIGURE_GENERATION_FAILED_NO_VISUAL_PASS','error':repr(error)});raise
    return 0


if __name__=='__main__':raise SystemExit(main())
