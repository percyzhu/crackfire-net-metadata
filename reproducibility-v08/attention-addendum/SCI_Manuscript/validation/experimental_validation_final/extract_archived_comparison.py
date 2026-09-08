"""Recover existing plotted curves and workbook observations; no new FE solve."""
from pathlib import Path
import csv,hashlib,json
import fitz,numpy as np,openpyxl

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PDF=ROOT/'3_最终成果/论文/LaTeX源码/cn/figures/mesh_sensitivity.pdf'
XLSX=ROOT/'1_代码/src/data/T-t_experimental_curve.xlsx'
SCRIPT=ROOT/'1_代码/scripts/figures/ch3/generate_fig_fem.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def out(name,rows):
    with (HERE/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

wb=openpyxl.load_workbook(XLSX,read_only=True,data_only=True)
raw={s.title:list(s.iter_rows(values_only=True)) for s in wb}
rows=raw['Sheet1'];headers=rows[0][:7]
obs=np.asarray([r[:7] for r in rows[1:] if isinstance(r[0],(int,float))],float)
assert obs.shape==(361,7) and np.all(np.diff(obs[:,0])>0) and np.isfinite(obs).all()
out('experimental_native.csv',[dict(zip(headers,r)) for r in obs])
t=np.r_[obs[obs[:,0]<3600,0],3600.]
observed={h:np.interp(t,obs[:,0],obs[:,j]) for j,h in enumerate(headers[1:],1)}
doc=fitz.open(PDF);page=doc[0];draw=page.get_drawings()
colours={50:(31/255,119/255,180/255),30:(1,127/255,14/255),20:(44/255,160/255,44/255),
         15:(214/255,39/255,40/255),10:(140/255,86/255,75/255)}
panels=[(15,48.521873474121094,386.67938232421875),
        (30,413.3393859863281,751.4968872070312),
        (40,778.1568603515625,1116.3143310546875)]
# Independent axis ticks (all five labelled 0,200,...800 C) define a shared y
# transform. These are PDF grid coordinates, not temperatures fitted to data.
grid_y=np.array([draw[i]['rect'].y0 for i in [20,22,24,26,28]])
coef=np.polyfit(grid_y,np.arange(0,801,200),1)
assert np.max(np.abs(np.polyval(coef,grid_y)-np.arange(0,801,200)))<1e-4
all_curves=[];recovered={};checks=[];candidate_metadata=[]
for depth,x0,x1 in panels:
    candidates=[(i,d) for i,d in enumerate(draw) if d['color'] and len(d['items'])>20
                and d['rect'].width>300 and abs(d['rect'].x0-x0)<.005]
    expected_count=8 if depth!=30 else 7
    assert len(candidates)==expected_count,(depth,len(candidates))
    greys=0
    for i,d in candidates:
        assert all(k[0]=='l' for k in d['items'])
        points=[tuple(d['items'][0][1])]
        for segment in d['items']:
            assert np.max(np.abs(np.array(points[-1])-np.array(segment[1])))<1e-4
            points.append(tuple(segment[2]))
        xy=np.asarray(points,float)
        tx=(xy[:,0]-x0)/(x1-x0)*3600;temp=np.polyval(coef,xy[:,1])
        assert np.all(np.diff(tx)>0)
        colour=np.asarray(d['color'])
        if np.max(colour)-np.min(colour)<.01:
            if d['width']>2: label='experimental_mean'
            else:
                label='experimental_a' if greys==0 else 'experimental_b';greys+=1
            case_obs=(observed[f'T{depth}a']+observed[f'T{depth}b'])/2 if label.endswith('mean') else observed[f'T{depth}'+label[-1]]
            restored=np.interp(t,tx,temp)
            checks.append({'depth_mm':depth,'curve':label,'PDF_index':i,
                'max_absolute_difference_to_workbook_C':float(np.abs(restored-case_obs).max()),
                'RMS_difference_to_workbook_C':float(np.sqrt(np.mean((restored-case_obs)**2)))})
        else:
            matches=[s for s,c in colours.items() if np.max(np.abs(colour-c))<1e-6]
            assert len(matches)==1;seed=matches[0];label=f'FE_{seed}mm'
            assert abs(tx[0])<1e-6 and abs(tx[-1]-3600)<1e-6
            recovered[depth,seed]=(tx,temp)
        candidate_metadata.append({'depth_mm':depth,'label':label,'drawing_index':i,'vertices':len(tx),
            'colour':d['color'],'dashes':d['dashes'],'x_axis_PDF_pt':[x0,x1]})
        for j,(tt,yy) in enumerate(zip(tx,temp)):
            all_curves.append({'depth_mm':depth,'curve':label,'vertex':j,'time_s':tt,'temperature_C':yy,'drawing_index':i})
assert len(recovered)==14 and len(checks)==9
# Matplotlib simplifies plotted polylines: bound that operation with nine
# independently available observation traces before using recovered FE curves.
assert max(c['max_absolute_difference_to_workbook_C'] for c in checks)<2.0
out('all_original_vector_curves.csv',all_curves)
out('PDF_experimental_curve_crosscheck.csv',checks)
metrics=[];paired=[]
for depth in [15,30,40]:
    for seed in [50,30,20,15,10]:
        if (depth,seed) not in recovered:continue
        tx,yy=recovered[depth,seed];fem=np.interp(t,tx,yy)
        for group in ['a','b','mean']:
            exp=(observed[f'T{depth}a']+observed[f'T{depth}b'])/2 if group=='mean' else observed[f'T{depth}{group}']
            e=fem-exp
            metrics.append({'depth_mm':depth,'FE_mesh_label_mm':seed,'comparison_group':group,
                'comparison_points':len(t),'sample_RMSE_C':float(np.sqrt(np.mean(e**2))),
                'time_weighted_RMS_C':float(np.sqrt(np.trapezoid(e**2,t)/3600)),
                'time_weighted_bias_C':float(np.trapezoid(e,t)/3600),'maximum_absolute_difference_C':float(np.max(np.abs(e))),
                'source':'Recovered existing vector plot, not new FE solver output'})
        if seed==20:
            for j,tt in enumerate(t):paired.append({'depth_mm':depth,'time_s':tt,'experiment_a_C':observed[f'T{depth}a'][j],
                'experiment_b_C':observed[f'T{depth}b'][j],'experiment_mean_C':(observed[f'T{depth}a'][j]+observed[f'T{depth}b'][j])/2,
                'archived_FE_20mm_C':fem[j]})
out('recovered_curve_error_metrics.csv',metrics);out('main_20mm_comparison.csv',paired)
workbook_FEM=[]
for name in ['exp1','exp2','exp3']:
    rows=raw[name];table=np.asarray([r[10:14] for r in rows[1:] if len(r)>13 and isinstance(r[10],(int,float))],float)
    assert np.isfinite(table).all() and table[0,0]==0 and table[-1,0]==3600 and np.all(np.diff(table[:,0])>0)
    for j,row in enumerate(table):
        for k in range(1,4):workbook_FEM.append({'sheet':name,'source_excel_row':j+2,'time_s':row[0],
            'column':chr(75+k),'original_header':rows[0][10+k],'temperature_C':row[k]})
out('all_workbook_Abaqus_native.csv',workbook_FEM)
report={'status':'EXISTING_VALIDATION_CURVES_RECOVERED_AND_OBSERVATION_IDENTITIES_CROSSCHECKED',
    'sources_sha256':{str(p):sha(p) for p in [PDF,XLSX,SCRIPT,Path(__file__)]},
    'axis_y_PDF_pt':grid_y.tolist(),'axis_y_temperature_C':list(range(0,801,200)),
    'temperature_from_PDF_y_coefficients':coef.tolist(),'curves':candidate_metadata,'observation_crosschecks':checks,
    'maximum_observation_vector_reconstruction_difference_C':max(c['max_absolute_difference_to_workbook_C'] for c in checks),
    'main_curve_selection':'20 mm was the explicitly stated validation mesh before extraction, not selected by recalculated error.',
    'all14_FE_curves_retained':True,'all9_workbook_Abaqus_curves_retained':True,
    'experimental_identity':'Both a and b are workbook-labelled experimental groups; neither is a model output.',
    'unresolved_identity':'Group-to-specimen and exact TC coordinates remain unavailable; exp1/2/3 Abaqus mesh labels remain unknown.',
    'endpoint_rule':'Retain native observation times below3600 and interpolate3600 inside the final observation segment; FE curves cover0..3600. No extrapolation.',
    'new_FE_solver_runs':0,'new_measurements':0,'physics_or_curve_calibration':False,
    'important_scope':'Recovered archived validation comparison, not regenerated FE labels or sensor-location-matched revalidation.',
    'main_20mm_metrics':[m for m in metrics if m['FE_mesh_label_mm']==20]}
(HERE/'extraction_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':report['status'],'max_obs_curve_reconstruction_C':report['maximum_observation_vector_reconstruction_difference_C'],
    '20mm_mean_metrics':[m for m in metrics if m['FE_mesh_label_mm']==20 and m['comparison_group']=='mean']},indent=2))
