"""Independent source arithmetic and vector audit; no generator imports."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import fitz

HERE=Path(__file__).resolve().parent
OUT=HERE/'rendered'
PROTOCOLS=['iid997','lcro_9_15','loco_iso834','loco_astm_e119','loco_perturbed_iso','loco_plateau','loco_decay','loco_log_variant','loco_external_fire','loco_bilinear','loco_linear','loco_smoldering']
ALL_MODELS=['fire_only','global_stats','deepsets','capacity_matched_deepsets','gnn','gnn_zero_edge_features','gnn_mean']
SHOWN=['fire_only','capacity_matched_deepsets','gnn']
BUDGETS=['10%','25%','50%']
METRICS=['U','regret_raw','regret_display','improvement_over_random','oracle_improvement_fraction']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def number(x): return None if x=='' else float(x)
def equal(a,b):
    if a is None or b is None: assert a is None and b is None
    else: assert math.isclose(a,b,rel_tol=2e-12,abs_tol=2e-13),(a,b)

def main():
    assert (OUT/'provenance.json').is_file(),'No actual completed figure'
    p=read(OUT/'provenance.json')
    assert p['cells']==216 and p['model_order']==SHOWN and p['protocol_order']==PROTOCOLS
    for item in p['source_snapshot']:
        assert sha(OUT/item['snapshot_path'])==item['sha256']==p['source_sha256'][item['original_path']]
    def source(name):
        found=[x for x in p['source_snapshot'] if Path(x['original_path']).name==name]
        assert len(found)==1,name
        return OUT/found[0]['snapshot_path']
    budget_audit=read(source('independent_arithmetic_audit.json'))
    report=read(source('analysis_report.json'))
    assert budget_audit['analysis_report_sha256']==sha(source('analysis_report.json'))
    assert report['rows']['model_seed_utility.csv']==2520 and report['rows']['model_five_seed_summary.csv']==504
    for name in ['model_seed_utility.csv','model_five_seed_summary.csv','reference_budgets.csv','paired_seed_contrasts.csv','paired_five_seed_summary.csv','raw_uncanonicalized_baseline_diagnostics.csv']:
        assert sha(OUT/('all7_'+name))==sha(source(name))==report['output_sha256'][name]
    seeds=rows(source('model_seed_utility.csv')); summaries=rows(source('model_five_seed_summary.csv'))
    references=rows(source('reference_budgets.csv'))
    seedkeys={(r['protocol'],r['model'],r['endpoint'],r['budget'],int(r['seed'])) for r in seeds}
    expected={(pr,m,e,b,s) for pr in PROTOCOLS for m in ALL_MODELS for e in ['terminal','J'] for b in BUDGETS for s in range(42,47)}
    assert len(seeds)==len(seedkeys)==2520 and seedkeys==expected
    ref={(r['protocol'],r['endpoint'],r['budget']):r for r in references}
    assert len(ref)==len(references)==72
    for r in references:
        N,k=int(r['N']),int(r['k']); denominator={'10%':10,'25%':4,'50%':2}[r['budget']]
        assert k==N//denominator
        equal(float(r['actual_budget_fraction']),k/N)
        equal(float(r['available_oracle_improvement']),float(r['U_random'])-float(r['U_oracle']))
    groups={}
    for r in seeds:
        key=(r['protocol'],r['model'],r['endpoint'],r['budget']);groups.setdefault(key,[]).append(r)
        baseline=ref[r['protocol'],r['endpoint'],r['budget']]
        equal(float(r['regret_raw']),float(r['U'])-float(baseline['U_oracle']))
        equal(float(r['improvement_over_random']),float(baseline['U_random'])-float(r['U']))
        available=float(baseline['available_oracle_improvement'])
        equal(number(r['oracle_improvement_fraction']),float(r['improvement_over_random'])/available if available>1e-12 else None)
    summary_lookup={}
    for r in summaries:
        key=(r['protocol'],r['model'],r['endpoint'],r['budget']);summary_lookup[key]=r
        group=groups[key];assert sorted(int(x['seed']) for x in group)==list(range(42,47))
        for metric in METRICS:
            vals=[number(x[metric]) for x in group if x[metric]!='']
            assert int(r[metric+'_defined_seeds'])==len(vals)
            for stat,value in [('mean',statistics.mean(vals) if vals else None),('sample_sd',statistics.stdev(vals) if vals else None),('min',min(vals) if vals else None),('max',max(vals) if vals else None)]:
                equal(number(r[metric+'_'+stat]),value)
    assert len(summary_lookup)==len(summaries)==504
    plotted=rows(OUT/'plotted_values.csv')
    plotkeys=[(r['protocol'],r['model'],r['endpoint'],r['budget']) for r in plotted]
    expected_plot={(pr,m,e,b) for pr in PROTOCOLS for m in SHOWN for e in ['terminal','J'] for b in BUDGETS}
    assert len(plotkeys)==len(set(plotkeys))==216 and set(plotkeys)==expected_plot
    finite=[];undefined=0
    for r,key in zip(plotted,plotkeys):
        original=summary_lookup[key]
        assert int(r['N'])==int(original['N']) and int(r['k'])==int(original['k'])
        L=number(original['oracle_improvement_fraction_mean']);equal(number(r['mean_L']),L)
        equal(number(r['display_percent_L']),100*L if L is not None else None)
        if L is None:
            assert r['cell_text']=='NA';undefined+=1
        else:
            value=100*L;finite.append(value)
            assert p['color_limits'][0]<=value<=p['color_limits'][1]
            expected_text=f'{value:.1f}'
            if expected_text=='-0.0':expected_text='0.0'
            assert r['cell_text']==expected_text
    assert undefined==p['undefined_cells']
    equal(min(finite),p['actual_raw_percent_range'][0]);equal(max(finite),p['actual_raw_percent_range'][1])
    document=fitz.open(OUT/'priority_budget_v08.pdf');assert len(document)==1
    page=document[0];size=[page.rect.width*25.4/72,page.rect.height*25.4/72]
    assert abs(size[0]-183)<.001 and abs(size[1]-150)<.001
    spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
    minimum=min(s['size'] for s in spans);assert minimum>=7.99999 and not page.get_images(full=True)
    page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(OUT/'pdf_final_QA.png')
    result={'status':'PASSED_INDEPENDENT_COMPLETE_BUDGET_FIGURE_ARITHMETIC_AND_EXPORT_PENDING_VISUAL_QA',
            'source_seed_rows':2520,'all7_summary_rows':504,'summary_metrics_reconciled':5,'reference_budgets':72,
            'plotted_cells':216,'all12_protocols':True,'all_endpoints_and_budgets':True,'percentage_scale':100,
            'undefined_cells':undefined,'actual_raw_percent_range':[min(finite),max(finite)],'no_raw_value_clipping':True,
            'size_mm':size,'minimum_pdf_font_pt':minimum,'raster_objects':0,'pdf_sha256':sha(OUT/'priority_budget_v08.pdf'),
            'raw_prediction_arrays_read':0,'generator_functions_imported':False,'statistical_CI_created':False,
            'remaining':'Actual PDF visual review and manuscript placement review','submission_pass':False}
    (OUT/'independent_figure_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
