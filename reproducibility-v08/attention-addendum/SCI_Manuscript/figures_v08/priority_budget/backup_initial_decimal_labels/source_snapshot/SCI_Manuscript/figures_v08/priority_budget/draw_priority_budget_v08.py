"""Draw the fixed complete exploratory budget matrix; never execute models."""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
SCI = HERE.parents[1]
SOURCE = SCI/'experiments/research_v08/engineering_budget_priority_v1'
REVIEW = SCI/'review/eaai_editor/research_v08'
PROTOCOLS = ['iid997','lcro_9_15','loco_iso834','loco_astm_e119','loco_perturbed_iso','loco_plateau','loco_decay','loco_log_variant','loco_external_fire','loco_bilinear','loco_linear','loco_smoldering']
ROW_LABELS = ['IID','Count OOD','ISO 834','ASTM E119','Perturbed ISO','Plateau','Decay','Log variant','External fire','Bilinear','Linear','Smoldering']
MODELS = ['fire_only','capacity_matched_deepsets','gnn']
TITLES = ['a  Fire only','b  Matched DeepSets','c  Sum GNN']
BUDGETS = ['10%','25%','50%']
ALL_MODELS = ['fire_only','global_stats','deepsets','capacity_matched_deepsets','gnn','gnn_zero_edge_features','gnn_mean']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def rows(p):
    with Path(p).open(encoding='utf-8', newline='') as f: return list(csv.DictReader(f))
def write(p, x): Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def main():
    out = HERE/'rendered'
    assert not out.exists(), 'Refuse to overwrite an actual output snapshot.'
    report_path = SOURCE/'results/analysis_report.json'
    arithmetic_path = SOURCE/'results/independent_arithmetic_audit.json'
    code_review_path = SOURCE/'independent_code_review.json'
    required = [report_path,arithmetic_path,code_review_path,SOURCE/'budget_priority.py',
                SOURCE/'input_equivalence_gate.json',SOURCE/'adoption_record.json',
                SOURCE/'audit_budget_arithmetic.py',SOURCE/'independent_code_review.md',
                REVIEW/'engineering_budget_priority_protocol.md',REVIEW/'engineering_budget_priority_protocol.json']
    if not all(p.is_file() for p in required):
        write(HERE/'gate_status.json', {'status':'WAITING_FOR_COMPLETE_BUDGET_AUDITS','plot_values_read':0}); return 2
    audit, report, code = read(arithmetic_path),read(report_path),read(code_review_path)
    assert audit['status']=='COMPLETE_ALL_SEVEN_MODEL_BUDGET_ARITHMETIC_INDEPENDENTLY_RECONCILED_NOT_SUBMISSION_REVIEW'
    assert code['status']=='PASSED_BOUNDED_INDEPENDENT_CODE_AND_STATISTICAL_SEMANTICS_REVIEW'
    assert audit['analysis_report_sha256']==code['analysis_report_sha256']==sha(report_path)
    assert report['code_sha256']==code['implementation_sha256']==sha(SOURCE/'budget_priority.py')
    assert audit['audit_code_sha256']==sha(SOURCE/'audit_budget_arithmetic.py')
    assert report['input_gate_sha256']==code['input_equivalence_gate_sha256']==sha(SOURCE/'input_equivalence_gate.json')
    assert report['approval_record_sha256']==sha(SOURCE/'adoption_record.json')
    for ext, field in [('md','approved_protocol_md_sha256'),('json','approved_protocol_json_sha256')]:
        assert sha(REVIEW/('engineering_budget_priority_protocol.'+ext))==report[field]
    assert report['models']==7 and report['protocols']==12 and report['seeds']==5
    assert report['rows']['model_seed_utility.csv']==audit['model_seed_utility_rows_checked']==2520
    assert report['rows']['model_five_seed_summary.csv']==audit['seed_summary_rows_checked']==504
    for name,digest in report['output_sha256'].items(): assert sha(SOURCE/'results'/name)==digest
    source_files = required + [SOURCE/'results'/name for name in ['model_seed_utility.csv','model_five_seed_summary.csv','reference_budgets.csv','paired_seed_contrasts.csv','paired_five_seed_summary.csv','raw_uncanonicalized_baseline_diagnostics.csv','analytic_identity_checks.json']]
    source_files += [HERE/'figure_contract.md',HERE/'caption_en.tex',Path(__file__),HERE/'audit_priority_budget_figure.py']
    bound = {str(p):sha(p) for p in source_files}
    write(HERE/'gate_status.json', {'status':'READY_ALL12_PROTOCOLS_ALL7_MODELS_INDEPENDENTLY_AUDITED','source_sha256':bound})
    summary = rows(SOURCE/'results/model_five_seed_summary.csv')
    lookup = {(r['protocol'],r['model'],r['endpoint'],r['budget']):r for r in summary}
    expected = {(p,m,e,b) for p in PROTOCOLS for m in ALL_MODELS for e in ['terminal','J'] for b in BUDGETS}
    assert len(lookup)==len(summary)==504 and set(lookup)==expected
    cols = [(e,b) for e in ['terminal','J'] for b in BUDGETS]
    matrices=[]; plotted=[]; counts={}
    for model in MODELS:
        matrix=np.full((12,6),np.nan)
        for i,p in enumerate(PROTOCOLS):
            for j,(endpoint,budget) in enumerate(cols):
                r=lookup[p,model,endpoint,budget]; counts[p]=int(r['N'])
                defined=int(r['oracle_improvement_fraction_defined_seeds'])
                assert defined in (0,5)
                value=float(r['oracle_improvement_fraction_mean'])*100 if defined else np.nan
                matrix[i,j]=value
                label='NA' if not defined else f'{value:.1f}'
                if label=='-0.0': label='0.0'
                plotted.append({'protocol':p,'model':model,'endpoint':endpoint,'budget':budget,
                                'N':int(r['N']),'k':int(r['k']),'actual_budget_fraction':float(r['actual_budget_fraction']),
                                'defined_seeds':defined,'mean_L':float(r['oracle_improvement_fraction_mean']) if defined else None,
                                'display_percent_L':float(value) if defined else None,'cell_text':label})
        matrices.append(matrix)
    actual=np.concatenate([x.ravel() for x in matrices]); finite=actual[np.isfinite(actual)]
    lo,hi=min(0.,float(finite.min())),max(100.,float(finite.max()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap,Normalize,TwoSlopeNorm
    from matplotlib.text import Text
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    meaningful_negative=lo < -1e-8
    cmap=LinearSegmentedColormap.from_list('utility', ['#a54b42','#fafafa','#275c80'] if meaningful_negative else ['#fafafa','#c2d8e4','#275c80'])
    cmap.set_bad('#d5d8db')
    norm=TwoSlopeNorm(vmin=lo,vcenter=0,vmax=hi) if meaningful_negative else Normalize(lo,hi)
    fig=plt.figure(figsize=(183/25.4,150/25.4))
    for panel,(model,matrix,title) in enumerate(zip(MODELS,matrices,TITLES)):
        left=.18+panel*.272
        ax=fig.add_axes([left,.245,.25,.585])
        image=ax.pcolormesh(np.arange(-.5,6.5),np.arange(-.5,12.5),np.ma.masked_invalid(matrix),cmap=cmap,norm=norm,
                           edgecolors='#e0e5e8',linewidth=.35,shading='flat')
        ax.set(xlim=(-.5,5.5),ylim=(11.5,-.5),xticks=np.arange(6),xticklabels=BUDGETS*2,
               yticks=np.arange(12),yticklabels=[f'{label} ({counts[p]})' for p,label in zip(PROTOCOLS,ROW_LABELS)] if panel==0 else ['']*12)
        ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=4)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.axvline(2.5,color='#455e6c',lw=1)
        ax.axhline(1.5,color='#455e6c',lw=1)
        fig.text(left,.965,title,weight='bold',fontsize=9,va='top')
        fig.text(left+.0625,.904,'Terminal',ha='center',va='top')
        fig.text(left+.1875,.904,'Time average J',ha='center',va='top')
        for i in range(12):
            for j in range(6):
                val=matrix[i,j]
                r=plotted[panel*72+i*6+j]
                rgb=np.asarray(cmap(norm(val))[:3]) if np.isfinite(val) else np.array([.84]*3)
                linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                lum=float(linear @ [.2126,.7152,.0722])
                color='white' if lum < .30 else '#142b38'
                ax.text(j,i,r['cell_text'],ha='center',va='center',fontsize=8,color=color)
    fig.text(.18,.21,'Five-seed mean; all 12 protocols × 2 endpoints × 3 case-slot budgets',fontsize=8)
    cax=fig.add_axes([.30,.145,.48,.018])
    colorbar=fig.colorbar(image,cax=cax,orientation='horizontal')
    colorbar.solids.set_rasterized(False); colorbar.solids.set_edgecolor('face'); colorbar.outline.set_visible(False)
    colorbar.set_ticks([v for v in ([lo,0,25,50,75,100,hi] if meaningful_negative else [0,25,50,75,100]) if lo<=v<=hi])
    colorbar.set_label('Available oracle improvement captured (%)',labelpad=4)
    colorbar.ax.tick_params(length=2,pad=2)
    fig.text(.025,.059,'0 = uniform-selection expectation; 100 = retrospective oracle. Gray / NA = undefined ratio.',fontsize=8)
    fig.text(.025,.03,'Expected cutoff-tie selection. Post-review exploratory analysis; no risk or time-saving claim.',fontsize=8)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); text_objects=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside=[]
    for t in text_objects:
        box=t.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5: outside.append(t.get_text())
    assert min(t.get_fontsize() for t in text_objects)>=8 and not outside,outside
    assert bound=={p:sha(p) for p in bound},'Source changed before snapshot'
    out.mkdir(); snapshots=[]
    for original,digest in bound.items():
        path=Path(original); relative=path.relative_to(SCI.parent); target=out/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
        assert sha(target)==sha(path)==digest
        snapshots.append({'original_path':original,'snapshot_path':target.relative_to(out).as_posix(),'sha256':digest})
    for name in ['model_seed_utility.csv','model_five_seed_summary.csv','reference_budgets.csv','paired_seed_contrasts.csv','paired_five_seed_summary.csv','raw_uncanonicalized_baseline_diagnostics.csv']:
        shutil.copyfile(SOURCE/'results'/name,out/('all7_'+name))
    with (out/'plotted_values.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=plotted[0]);w.writeheader();w.writerows(plotted)
    for ext in ['pdf','svg','png']: fig.savefig(out/('priority_budget_v08.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    write(out/'provenance.json',{'status':'ACTUAL_FULL_MATRIX_RENDERED_PENDING_INDEPENDENT_FIGURE_AND_VISUAL_QA',
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_snapshot':snapshots,'source_sha256':bound,
          'model_order':MODELS,'protocol_order':PROTOCOLS,'column_order':cols,'cells':216,'actual_raw_percent_range':[float(finite.min()),float(finite.max())],
          'color_limits':[lo,hi],'undefined_cells':int(np.isnan(actual).sum()),'negative_raw_cells':int((finite<0).sum()),
          'meaningful_negative_cells':int((finite < -1e-8).sum()),'display_rounding':'one decimal; rounded negative zero printed 0.0; raw retained',
          'size_mm':[183,150],'minimum_font_pt':8,'raster_objects_expected':0,'out_of_canvas_text':outside,
          'timing_or_model_execution':False,'all7_companion_rows':504,'manuscript_or_submission_pass':False})
    print(json.dumps({'status':'RENDERED_REAL_COMPLETE_BUDGET_MATRIX','raw_percent_range':[float(finite.min()),float(finite.max())],'NA_cells':int(np.isnan(actual).sum()),'pdf':str(out/'priority_budget_v08.pdf')}))
    return 0

if __name__=='__main__': raise SystemExit(main())
