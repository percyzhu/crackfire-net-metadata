"""Audit archived vector recovery against original PDF and workbook, without FE."""
from pathlib import Path
from collections import defaultdict
import csv,datetime,hashlib,json,math
import fitz,numpy as np,openpyxl

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(name):
    with (HERE/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def integral(v,t):return float(np.sum(np.diff(t)*(v[:-1]+v[1:])/2))
def close(a,b,tol=1e-9):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=tol),(a,b)

def main():
    target=HERE/'independent_extraction_audit.json';assert not target.exists()
    report=json.loads((HERE/'extraction_report.json').read_text(encoding='utf-8'))
    for p,h in report['sources_sha256'].items():assert sha(p)==h
    pdf=next(Path(p) for p in report['sources_sha256'] if p.endswith('.pdf'))
    xlsx=next(Path(p) for p in report['sources_sha256'] if p.endswith('.xlsx'))
    page=fitz.open(pdf)[0];draw=page.get_drawings();words=page.get_text('words')
    # Recover axes independently from long grey horizontal grids and printed ticks.
    grids=[d['rect'] for d in draw if d['rect'].width>300 and d['rect'].height<1e-5 and d['color'] and .68<float(d['color'][0])<.70]
    assert len(grids)==15
    axes=sorted({(r.x0,r.x1) for r in grids});assert len(axes)==3
    y=sorted({r.y0 for r in grids});assert len(y)==5
    zero_y=y[-1];eight_y=y[0]
    for val,yy in zip([800,600,400,200,0],y):
        assert any(w[4]==str(val) and w[2]<axes[0][0] and w[1]<yy<w[3] for w in words)
        close((zero_y-yy)*800/(zero_y-eight_y),val,1e-4)
    for x0,x1 in axes:
        assert any(w[4]=='0' and abs((w[0]+w[2])/2-x0)<.01 and w[1]>290 for w in words)
        assert any(w[4]=='60' and abs((w[0]+w[2])/2-x1)<.01 and w[1]>290 for w in words)
    wb=openpyxl.load_workbook(xlsx,read_only=True,data_only=True)
    native=list(wb['Sheet1'].iter_rows(values_only=True));headers=list(native[0][:7])
    obs=np.asarray([r[:7] for r in native[1:] if isinstance(r[0],(int,float))],float)
    assert headers==['t','T15a','T30a','T40a','T15b','T30b','T40b']
    assert obs.shape==(361,7) and np.isfinite(obs).all() and obs[0,0]==0 and obs[-1,0]>3600
    native_export=readcsv('experimental_native.csv')
    np.testing.assert_allclose(np.array([[float(r[h]) for h in headers] for r in native_export]),obs,atol=0,rtol=0)
    t=np.concatenate((obs[obs[:,0]<3600,0],[3600.]));assert len(t)==361
    measurements={depth:{g:np.interp(t,obs[:,0],obs[:,headers.index(f'T{depth}{g}')]) for g in ['a','b']} for depth in [15,30,40]}
    for depth in measurements:measurements[depth]['mean']=(measurements[depth]['a']+measurements[depth]['b'])/2
    curve_rows=readcsv('all_original_vector_curves.csv');curve_groups=defaultdict(list)
    for r in curve_rows:curve_groups[(int(r['depth_mm']),r['curve'])].append(r)
    assert len(curve_groups)==23 and sum(k[1].startswith('FE_') for k in curve_groups)==14
    ycoef=report['temperature_from_PDF_y_coefficients'];recovered={};crosschecks=[]
    metadata={(r['depth_mm'],r['label']):r for r in report['curves']};assert set(metadata)==set(curve_groups)
    for (depth,label),records in curve_groups.items():
        m=metadata[(depth,label)]; d=draw[m['drawing_index']];axis=axes[[15,30,40].index(depth)]
        assert all(p[0]=='l' for p in d['items']) and len(d['items'])+1==len(records)==m['vertices']
        assert list(axis)==m['x_axis_PDF_pt'] and np.max(np.abs(np.array(d['color'])-m['colour']))<1e-8
        points=np.array([tuple(d['items'][0][1])]+[tuple(a[2]) for a in d['items']])
        tt=(points[:,0]-axis[0])/(axis[1]-axis[0])*60*60
        independent_temp=(zero_y-points[:,1])*800/(zero_y-eight_y)
        listed_t=np.array([float(r['time_s']) for r in records]);listed_temp=np.array([float(r['temperature_C']) for r in records])
        np.testing.assert_allclose(listed_t,tt,atol=1e-9,rtol=0)
        np.testing.assert_allclose(listed_temp,independent_temp,atol=1e-4,rtol=0)
        assert np.all(np.diff(tt)>0)
        if label.startswith('FE_'):
            close(tt[0],0);close(tt[-1],3600);recovered[depth,int(label[3:-2])]=np.interp(t,listed_t,listed_temp)
            if label=='FE_20mm':
                np.testing.assert_allclose(d['color'],np.array([44,160,44])/255,atol=1e-7,rtol=0)
                assert len(d['dashes'].strip().split())>4 and d['dashes']=='[ 11.52 2.88 1.8 2.88 ] 0'
        else:
            group=label.removeprefix('experimental_');e=np.interp(t,listed_t,listed_temp)-measurements[depth][group]
            stated=next(r for r in report['observation_crosschecks'] if r['depth_mm']==depth and r['curve']==label)
            close(stated['max_absolute_difference_to_workbook_C'],abs(e).max());close(stated['RMS_difference_to_workbook_C'],np.sqrt(np.mean(e*e)))
            crosschecks.append(float(abs(e).max()))
    metrics=readcsv('recovered_curve_error_metrics.csv');assert len(metrics)==42
    for r in metrics:
        depth=int(r['depth_mm']);seed=int(r['FE_mesh_label_mm']);group=r['comparison_group'];e=recovered[depth,seed]-measurements[depth][group]
        assert int(r['comparison_points'])==361
        for k,v in [('sample_RMSE_C',np.sqrt(np.mean(e*e))),('time_weighted_RMS_C',math.sqrt(integral(e*e,t)/3600)),('time_weighted_bias_C',integral(e,t)/3600),('maximum_absolute_difference_C',max(abs(e)))]:close(r[k],v)
    main=readcsv('main_20mm_comparison.csv');assert len(main)==1083
    for depth in [15,30,40]:
        a=[r for r in main if int(r['depth_mm'])==depth];assert len(a)==361
        np.testing.assert_allclose([float(r['time_s']) for r in a],t,atol=0,rtol=0)
        for g in ['a','b','mean']:np.testing.assert_allclose([float(r[f'experiment_{g}_C']) for r in a],measurements[depth][g],atol=1e-10,rtol=0)
        np.testing.assert_allclose([float(r['archived_FE_20mm_C']) for r in a],recovered[depth,20],atol=1e-10,rtol=0)
    result=dict(status='PASS_INDEPENDENT_ARCHIVED_CURVE_EXTRACTION_AND_ARITHMETIC',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_visual_inspection='Source PDF preview confirms three nominal-depth panels, green dash-dot FEM20mm legend, both a/b experiments and their black mean.',
        independent_axis_method='All15 long horizontal grids with printed0/200/400/600/800 ticks; each panel0..60min; separate endpoint linear map agrees with reported least-squares axis to1e-4C.',
        original_vector_curves=23,FE_curves=14,observation_vector_curves=9,main20mm_rows=1083,native_observation_rows=361,metric_comparisons_checked=42,
        maximum_observation_vector_reconstruction_difference_C=max(crosschecks),
        main20mm_time_weighted_RMS_to_observation_mean_C={str(d):math.sqrt(integral((recovered[d,20]-measurements[d]['mean'])**2,t)/3600) for d in [15,30,40]},
        no_time_extrapolation=True,no_new_solver_or_measurement=True,no_curve_or_physics_calibration=True,
        scope_limitations=['Nominal20mm identity is supplied by the original vector legend and archived plotting source, not inferred by best fit.',
            'Both a/b are recorded observations; exact specimen/TC correspondence is unresolved.',
            'FE trajectories are recovered plotted polylines; observation recovery differences do not rigorously bound FE simplification error.',
            'The comparison is archival experimental-route evidence, not newly regenerated sensor-matched validation, no997target revalidation, no grid convergence claim.'],
        source_sha256={str(p):sha(p) for p in [Path(__file__),HERE/'extract_archived_comparison.py',HERE/'extraction_report.json',HERE/'main_20mm_comparison.csv',HERE/'recovered_curve_error_metrics.csv',HERE/'all_original_vector_curves.csv',pdf,xlsx]},
        submission_pass=False)
    with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    text='# 归档试验—有限元曲线恢复：独立审查\n\n原PDF的三处图例明确将绿色点划线标为FEM20mm，a/b均为观测组，黑线为其均值；已视觉核实并与原绘图源交叉核对。独立从15条网格线与数字刻度恢复0–60min/0–800°C轴，独立端点变换与提取器拟合变换在0.0001°C以内一致。原xlsx的361条时间/观测数据、23条PDF折线、主图1083行、14条FE曲线共42项误差指标已逐项核对。\n\n三个深度15/30/40mm的20mm曲线对观测均值的时间加权RMS分别为35.4708305、43.2743361、39.9860937°C。九条观测折线反取对原表最大差0.8518131°C；这是一项恢复一致性检查，不能作为所有FE折线误差的严格上界。\n\n可以将其作为既有试验比较路线在主文展示。应明确“归档FE图恢复”，a/b不能画成模型/试验两条线；不得称为本轮新求解、新热电偶坐标匹配验证、所有997标签的重新物理验证或网格收敛。没有发现阻断性的来源或算术问题。\n'
    with (HERE/'independent_extraction_review_zh.md').open('x',encoding='utf-8') as f:f.write(text)
    print(json.dumps({'status':result['status'],'audit_sha256':sha(target),'RMS_C':result['main20mm_time_weighted_RMS_to_observation_mean_C']},indent=2))

if __name__=='__main__':main()
