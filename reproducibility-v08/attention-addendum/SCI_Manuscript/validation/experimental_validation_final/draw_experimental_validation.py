"""Source-backed four-panel validation figure; no simulation or data fitting."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.text import Text

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (HERE/'main_20mm_comparison.csv').open() as f: rows=list(csv.DictReader(f))
report=json.loads((HERE/'extraction_report.json').read_text(encoding='utf-8'))
assert report['status']=='EXISTING_VALIDATION_CURVES_RECOVERED_AND_OBSERVATION_IDENTITIES_CROSSCHECKED'
assert report['new_FE_solver_runs']==0
for p,d in report['sources_sha256'].items():assert sha(p)==d
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(183/25.4,152/25.4))
ink='#203644';blue='#0072B2';heat='#bd5e2e'
sk=fig.add_axes([.045,.575,.44,.345]);sk.set(xlim=(0,100),ylim=(0,100));sk.axis('off')
sk.text(0,100,'a  Standard-fire experiment',fontweight='bold',fontsize=9,va='top',color=ink)
sk.text(0,89,'Douglas-fir glulam',color=ink)
sk.text(0,80,'ISO 834 · four faces · 60 min',color=ink)
# End-protected prism is illustrative; no purported sensor positions are drawn.
front=[(9,41),(78,41),(78,59),(9,59)];top=[(9,59),(78,59),(85,65),(16,65)];end=[(78,41),(85,47),(85,65),(78,59)]
for pts,col in [(front,'#e8ddc9'),(top,'#f3ecdf'),(end,'#d4c4a9')]:
    sk.add_patch(Polygon(pts,closed=True,facecolor=col,edgecolor=ink,lw=.7))
for x in [9,78]:
    sk.add_patch(Polygon([(x-1.2,40),(x+1.2,40),(x+1.2,60),(x-1.2,60)],closed=True,
                         facecolor='#b6bdc1',edgecolor=ink,lw=.55))
for x in [26,43,60]:
    sk.annotate('',xy=(x,60.5),xytext=(x,72),arrowprops={'arrowstyle':'-|>','color':heat,'lw':.9,'mutation_scale':8})
    sk.annotate('',xy=(x,40),xytext=(x,31),arrowprops={'arrowstyle':'-|>','color':heat,'lw':.9,'mutation_scale':8})
sk.annotate('',xy=(9,26),xytext=(78,26),arrowprops={'arrowstyle':'<->','lw':.7,'color':ink})
sk.text(43.5,20,'600 mm',ha='center',color=ink)
sk.annotate('Protected ends',xy=(79,46),xytext=(99,34),ha='right',va='center',
    arrowprops={'arrowstyle':'-','lw':.65,'color':ink},color=ink)
sk.text(0,10,'Section: 140 × 200 mm',color=ink)
sk.text(0,0,'Depth groups: 15, 30 and 40 mm',va='bottom',color=ink)

axes=[];rms=[]
for depth,letter,rect in [(15,'b',[.595,.61,.37,.31]),(30,'c',[.105,.205,.37,.29]),(40,'d',[.595,.205,.37,.29])]:
    ax=fig.add_axes(rect);axes.append(ax)
    group=[r for r in rows if int(r['depth_mm'])==depth]
    t=np.array([float(r['time_s'])/60 for r in group])
    a=np.array([float(r['experiment_a_C']) for r in group]);b=np.array([float(r['experiment_b_C']) for r in group])
    avg=np.array([float(r['experiment_mean_C']) for r in group]);fe=np.array([float(r['archived_FE_20mm_C']) for r in group])
    assert len(t)==361 and t[0]==0 and t[-1]==60
    assert np.max(np.abs(avg-(a+b)/2))<1e-10
    rm=np.sqrt(np.trapezoid((fe-avg)**2,t*60)/3600);rms.append(rm)
    ax.fill_between(t,np.minimum(a,b),np.maximum(a,b),color='#e8e8e8',lw=0,zorder=1)
    ax.plot(t,a,color='#777777',lw=.7,zorder=2)
    ax.plot(t,b,color='#777777',ls=(0,(1.5,1.5)),lw=.7,zorder=2)
    ax.plot(t,avg,color='#202020',lw=1.0,ls=(0,(4,2)),zorder=3)
    ax.plot(t,fe,color=blue,lw=1.65,zorder=4)
    ax.set(xlim=(0,60),ylim=(0,1000),xticks=[0,20,40,60],yticks=[0,200,400,600,800,1000],
           xlabel='Time (min)',ylabel='Temperature (°C)')
    ax.set_title(f'{letter}  Depth group: {depth} mm',loc='left',fontweight='bold',color=ink,pad=8)
    ax.text(.025,.955,f'Time-weighted RMS: {rm:.1f} °C',transform=ax.transAxes,va='top',color=ink)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#e4e7e9',lw=.5,zorder=0)

legend=[Line2D([0],[0],color='#777777',lw=.8,label='Experiment a'),
        Line2D([0],[0],color='#777777',ls=(0,(1.5,1.5)),lw=.8,label='Experiment b'),
        Line2D([0],[0],color='#202020',ls=(0,(4,2)),lw=1,label='Experimental mean'),
        Line2D([0],[0],color=blue,lw=1.65,label='Archived FE, 20 mm mesh')]
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.52,.056),ncol=2,frameon=False,
           handlelength=2.7,columnspacing=2.8,handletextpad=.7)
fig.text(.52,.020,'Shading spans the two experimental groups; it is not a confidence interval.',ha='center',fontsize=8,color=ink)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
outside=[]
for txt in fig.findobj(Text):
    if not txt.get_visible() or not txt.get_text():continue
    box=txt.get_window_extent(renderer)
    if box.width and box.height and (box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5):outside.append(txt.get_text())
assert not outside,outside
for ext in ['pdf','svg','png']:
    fig.savefig(HERE/f'experimental_validation_final.{ext}',dpi=300)
plt.close(fig)
output={p.name:sha(p) for p in [HERE/'experimental_validation_final.pdf',HERE/'experimental_validation_final.svg',HERE/'experimental_validation_final.png']}
manifest={'status':'RENDERED_AWAITING_INDEPENDENT_NUMERICAL_AND_ACTUAL_VISUAL_QA',
    'width_mm':183,'height_mm':152,'minimum_requested_font_pt':8,'outside_canvas_text':outside,
    'data_sha256':sha(HERE/'main_20mm_comparison.csv'),'extraction_report_sha256':sha(HERE/'extraction_report.json'),
    'generator_sha256':sha(__file__),'figure_contract_sha256':sha(HERE/'figure_contract.md'),
    'output_sha256':output,'time_weighted_RMS_C':rms,'new_FE_runs':0,
    'actual_temperature_group_identity':'Both a and b are experimental groups; FE curves are separately identified by original mesh legend.',
    'schematic_sensor_coordinates_assigned':False}
(HERE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(manifest,indent=2,ensure_ascii=False))
