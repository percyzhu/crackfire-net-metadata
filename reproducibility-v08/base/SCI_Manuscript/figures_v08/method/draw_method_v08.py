"""Application workflow with one exact archived case and the actual learning target."""
from pathlib import Path
import ast,csv,hashlib,json,shutil,importlib.util,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,Polygon
from matplotlib.lines import Line2D
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;SCI=HERE.parents[1];WORK=SCI.parent
OLD=SCI/'figures_v07/method';MAN=SCI/'experiments/research_v08/archive997/manifest.json';PLAN=SCI/'experiments/research_v08/comparison_plan_v08_420.json'
BLUE,TEAL,OCHRE='#276c91','#368478','#b47737';COLORS=[BLUE,TEAL,OCHRE]
INK,MUTED,LINE='#24333c','#596970','#c9d2d5';WOOD,WOOD_SIDE,WOOD_END='#eee6d9','#d4c5ae','#e2d5bf'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    oldtext=(OLD/'draw_method_v07.py').read_text(encoding='utf8');tree=ast.parse(oldtext)
    names=['arrow','project','face','surface','dim'];parts=[ast.get_source_segment(oldtext,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    helper='import numpy as np\nfrom matplotlib.path import Path as MplPath\nfrom matplotlib.patches import Polygon,FancyArrowPatch,PathPatch\nINK="'+INK+'"\nMUTED="'+MUTED+'"\n\n'+'\n\n'.join(parts)+'\n'
    (HERE/'beam_vector_helpers.py').write_text(helper,encoding='utf8')
    for n in ['geometry_source.csv','graph_nodes.csv','graph_edges.csv']:shutil.copy2(OLD/n,HERE/n)
    slots=pd.read_csv(HERE/'geometry_source.csv');m=json.loads(MAN.read_text(encoding='utf8'));plan=json.loads(PLAN.read_text(encoding='utf8'));assert sha(MAN)==plan['manifest_sha256']
    key=lambda c:(int(c['face']),round(float(c['z']),5),round(float(c['h']),5),round(float(c['w']),5),round(float(c['l']),5),round(float(c['d']),5))
    want=sorted((int(s.face),round(s.z_center_m,5),round(s.local_h_m,5),round(s.width_m,5),round(s.length_m,5),round(s.depth_m,5)) for s in slots.itertuples())
    matches=[c for c in m['cases'] if c['num_cracks']==3 and sorted(map(key,c['cracks']))==want];assert len(matches)==1
    c=matches[0];assert c['sample_id']=='b004_0006' and c['fire']=={'type':'iso834','params':{}}
    source=WORK/c['source_npz'];assert sha(source)==c['source_npz_sha256_prior_full_audit']
    with np.load(source,allow_pickle=False) as z:t=z['time_steps'];raw=z['charring_ratios'];cracks=z['crack_params'];dims=z['beam_dims']
    assert hashlib.sha256(t.tobytes()+raw.tobytes()).hexdigest()==c['source_scalar_arrays_sha256_current']
    expected=np.asarray([[r[k] for k in ['face','z','h','w','l','d']] for r in c['cracks']]);assert np.array_equal(cracks,expected)
    assert np.array_equal(np.ravel(dims),[1.5,.14,.2]);env=np.minimum.accumulate(raw);grid=np.arange(61)*60.;target=np.interp(grid,t,env).astype('float32')
    pd.DataFrame(dict(time_s=t,native_scalar=raw,running_minimum=env)).to_csv(HERE/'native_scalar_and_envelope.csv',index=False)
    pd.DataFrame(dict(time_s=grid,target_float32=target)).to_csv(HERE/'target_61.csv',index=False)
    pd.DataFrame(dict(time_s=grid,prescribed_gas_C=20+345*np.log10(8*grid/60+1))).to_csv(HERE/'prescribed_iso834.csv',index=False)
    prov=dict(sample_id=c['sample_id'],geometry_id=c['geometry_id'],selection='Unique exact geometry match to previous three-slot illustration; no outcome/prediction selection',same_case_in_all_panels=True,fire=c['fire'],native_points=len(t),target_points=len(grid),native_max_cumulative_minimum_change=float(max(raw-env)),native_max_change_time_s=float(t[np.argmax(raw-env)]),target_operation=m['target_formula'],scalar_member='charring_ratios',actual_read_arrays=['time_steps','charring_ratios','crack_params','beam_dims'],node_temperature_arrays_read=False,prediction_results_read=False,source_case=c,models=plan['models'],source_files=[dict(path=str(p.relative_to(WORK)),sha256=sha(p)) for p in [MAN,PLAN,source,OLD/'draw_method_v07.py',OLD/'geometry_source.csv',OLD/'graph_nodes.csv',OLD/'graph_edges.csv',SCI/'experiments/research_v08/prepare997.py']],scope='Archival scalar prediction; not correctedP1history/capacity labels',target_float32_SHA256=hashlib.sha256(target.tobytes()).hexdigest())
    (HERE/'source_provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf8');return slots.to_dict('records'),prov

def main():
    slots,prov=prepare();spec=importlib.util.spec_from_file_location('beam_helpers',HERE/'beam_vector_helpers.py');bh=importlib.util.module_from_spec(spec);spec.loader.exec_module(bh)
    arrow,project,face,surface,dim=[getattr(bh,n) for n in ['arrow','project','face','surface','dim']]
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK})
    fig=plt.figure(figsize=(183/25.4,170/25.4));cv=fig.add_axes([0,0,1,1]);cv.set(xlim=(0,1),ylim=(0,1));cv.axis('off')
    def txt(x,y,s,**kw):kw.setdefault('fontsize',8);kw.setdefault('color',INK);return cv.text(x,y,s,**kw)
    for y,label in [(.97,'a  From a slotted beam to a crack graph'),(.707,'b  Geometry, prescribed exposure and scalar prediction'),(.397,'c  Actual archival reference from the same three-slot case')]:txt(.03,y,label,fontweight='bold',va='top')
    for y in [.73,.421]:cv.plot([.03,.97],[y,y],color=LINE,lw=.55)
    beam=fig.add_axes([.025,.773,.70,.178]);beam.axis('off');beam.set_aspect('equal');beam.set(xlim=(-.18,1.62),ylim=(-.28,.48))
    for i,s in enumerate(slots):
        xa,xb,ya,yb,za,zb=[s[k] for k in ['x_min_m','x_max_m','y_min_m','y_max_m','z_min_m','z_max_m']]
        if s['face']==0:
            for xx in [xa,xb]:face(beam,[[xx,ya,za],[xx,yb,za],[xx,yb,zb],[xx,ya,zb]],COLORS[i],zo=2)
            face(beam,[[xa,ya,za],[xb,ya,za],[xb,ya,zb],[xa,ya,zb]],COLORS[i],zo=2)
        else:face(beam,[[xa,ya,za],[xa,yb,za],[xa,yb,zb],[xa,ya,zb]],COLORS[i],zo=2)
    surface(beam,[[-.1,.1],[0,1.5]],[[[s['y_min_m'],s['y_max_m']],[s['z_min_m'],s['z_max_m']]] for s in slots if s['face']==2],lambda y,z:[.07,y,z],WOOD_SIDE,3)
    surface(beam,[[-.07,.07],[0,1.5]],[[[s['x_min_m'],s['x_max_m']],[s['z_min_m'],s['z_max_m']]] for s in slots if s['face']==0],lambda x,z:[x,.1,z],WOOD,4)
    face(beam,[[-.07,-.1,0],[.07,-.1,0],[.07,.1,0],[-.07,.1,0]],WOOD_END,INK,.7,6)
    for i,s in enumerate(slots):
        xa,xb,ya,yb,za,zb=[s[k] for k in ['x_min_m','x_max_m','y_min_m','y_max_m','z_min_m','z_max_m']]
        points=[[xa,.1,za],[xb,.1,za],[xb,.1,zb],[xa,.1,zb]] if s['face']==0 else [[.07,ya,za],[.07,yb,za],[.07,yb,zb],[.07,ya,zb]]
        face(beam,points,'none',COLORS[i],.95,7);end=np.array([[1.22,.435],[.78,.002],[.78,.36]][i]);start=project(np.mean(points,axis=0))
        beam.plot([start[0],end[0]],[start[1],end[1]],color=COLORS[i],lw=.7,zorder=8);beam.add_patch(Circle(end,.037,fc=COLORS[i],ec='white',lw=.7,zorder=9));beam.text(*end,str(i+1),color='white',ha='center',va='center',fontweight='bold',fontsize=8,zorder=10)
    dim(beam,project([.07,-.1,0]),project([.07,-.1,1.5]),'L = 1.50 m',offset=(0,-.085),textoffset=(0,-.02),rotation=10)
    dim(beam,project([.07,-.1,0]),project([.07,.1,0]),'h = 0.20 m',offset=(-.12,0),textoffset=(-.035,0),rotation=90)
    txt(.072,.756,'Width: 0.14 m',color=MUTED);txt(.29,.756,'Three actual slots; schematic projection',color=MUTED)
    s=slots[0];xa,xb=s['x_min_m'],s['x_max_m'];floor=s['y_min_m'];detail=fig.add_axes([.79,.791,.17,.137]);detail.axis('off');detail.set_aspect('equal');detail.set(xlim=(-.006,.016),ylim=(.073,.105))
    detail.add_patch(Polygon([[-.006,.075],[.014,.075],[.014,.1],[xb,.1],[xb,floor],[xa,floor],[xa,.1],[-.006,.1]],fc=WOOD,ec=INK,lw=.7));detail.plot([xa,xa,xb,xb],[.1,floor,floor,.1],color=BLUE,lw=1)
    dim(detail,[xa,.1],[xb,.1],'2.21 mm',offset=(0,.002),textoffset=(0,.002));dim(detail,[xb,floor],[xb,.1],'14.4 mm',offset=(.005,0),textoffset=(.002,0),rotation=90)
    txt(.875,.953,'Slot 1: magnified section',ha='center');txt(.875,.767,'Length: 254.8 mm',ha='center');txt(.875,.750,'White = initial cavity',ha='center',color=MUTED)
    graph=fig.add_axes([.062,.588,.18,.098]);graph.axis('off');graph.set_aspect('equal');graph.set(xlim=(0,1),ylim=(0,1));layout=np.array([[.77,.76],[.51,.16],[.16,.70]])
    for e in pd.read_csv(HERE/'graph_edges.csv').to_dict('records'):arrow(graph,layout[int(e['source'])-1],layout[int(e['target'])-1],color='#8b999e',lw=.75,scale=6,connectionstyle='arc3,rad=0.10',shrinkA=6,shrinkB=6)
    for i,p in enumerate(layout):graph.add_patch(Circle(p,.11,fc=COLORS[i],ec='white',lw=.6,zorder=3));graph.text(*p,str(i+1),color='white',ha='center',va='center',fontsize=8,fontweight='bold',zorder=4)
    txt(.155,.579,'3 nodes · 6 directed edges',ha='center')
    def box(x,y,w,h,label,fc,ec):cv.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=.8));txt(x+w/2,y+h/2,label,ha='center',va='center')
    box(.305,.628,.20,.059,'Crack-graph encoder','#f2f6f8',BLUE);arrow(cv,(.245,.652),(.297,.652),color=BLUE)
    box(.605,.611,.16,.062,'Scalar decoder','#f2f6f8',BLUE);arrow(cv,(.506,.658),(.595,.642),color=BLUE)
    arrow(cv,(.766,.642),(.823,.642),color=BLUE);txt(.889,.655,'Prediction ŷ(k)',ha='center',fontweight='bold',color=BLUE);txt(.889,.633,'61 scalar outputs',ha='center',color=BLUE);txt(.889,.610,'Paired with reference y(k)',ha='center')
    gas=pd.read_csv(HERE/'prescribed_iso834.csv');gax=fig.add_axes([.080,.482,.157,.073]);gax.plot(gas.time_s/60,gas.prescribed_gas_C,color=OCHRE,lw=1.1);gax.set(xlim=(0,60),ylim=(0,1000),xticks=[0,60],yticks=[0,1000],ylabel='°C');gax.tick_params(length=2,pad=1);gax.yaxis.labelpad=1
    txt(.080,.562,'ISO 834, time (min)',color=OCHRE)
    box(.305,.487,.20,.055,'Exposure LSTM','#f7f4ef',OCHRE);arrow(cv,(.25,.514),(.297,.514),color=OCHRE);cv.plot([.507,.558,.558],[.514,.514,.624],color=MUTED,lw=.8);arrow(cv,(.558,.624),(.598,.624),color=MUTED)
    txt(.605,.578,'Seven comparison variants',fontweight='bold');txt(.605,.555,'Fire only · statistics · DeepSets');txt(.605,.534,'Matched DeepSets · GNN');txt(.605,.513,'Zero-edge-feature GNN · mean-message GNN')
    txt(.03,.446,'GNN transfer tests: same trained weights → unseen N / radius / symmetric kNN views',color=BLUE)
    txt(.055,.363,'1  Native scalar r(i)   →   2  Running minimum e(i)   →   3  Reference y(k), k = 0,…,60',fontweight='bold')
    txt(.055,.340,'e(i) = min[r(0), …, r(i)];   y(k) = linear interpolation of e at t = 60k s',color=MUTED)
    raw=pd.read_csv(HERE/'native_scalar_and_envelope.csv');target=pd.read_csv(HERE/'target_61.csv')
    left=fig.add_axes([.080,.105,.405,.175]);right=fig.add_axes([.622,.105,.330,.175]);native='#AB7955';envcol='#368478';marker='#276c91'
    for ax in [left,right]:
        ax.plot(raw.time_s/60,raw.native_scalar,color=native,lw=1.1);ax.plot(raw.time_s/60,raw.running_minimum,color=envcol,lw=1.1);ax.plot(target.time_s/60,target.target_float32,ls='',marker='o',ms=2.9,mfc='white',mec=marker,mew=.65,zorder=4)
        ax.set_xlabel('Time (min)',labelpad=2);ax.tick_params(length=2,pad=2)
    left.set(xlim=(0,60),ylim=(0,1.04),xticks=[0,30,60],yticks=[0,.5,1],ylabel='Archival scalar');left.yaxis.labelpad=3
    right.set(xlim=(20,23),ylim=(.35,.51),xticks=[20,21,22,23],yticks=[.36,.42,.48],ylabel='Local detail');right.yaxis.labelpad=3
    left.add_patch(Rectangle((20,.35),3,.16,fc='none',ec=MUTED,lw=.65,ls=(0,(2,2))))
    handles=[Line2D([0],[0],color=native,lw=1.1,label='Native scalar'),Line2D([0],[0],color=envcol,lw=1.1,label='Running minimum'),Line2D([0],[0],ls='',marker='o',ms=3,mfc='white',mec=marker,label='61 target points')]
    cv.legend(handles=handles,loc='center left',bbox_to_anchor=(.068,.303),frameon=False,ncol=3,handlelength=1.8,columnspacing=1.0,handletextpad=.5)
    txt(.79,.299,'20–23 min enlargement',ha='center',color=MUTED)
    txt(.055,.030,'997 cases · 10 prescribed-fire families · variable crack counts N = 1–15',color=MUTED)
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bad=[];sizes=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer);sizes.append(t.get_fontsize())
            if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:bad.append(t.get_text())
    for ext in ['pdf','svg','png']:fig.savefig(HERE/f'method_v08.{ext}',dpi=300,facecolor='white')
    plt.close(fig);qa=dict(size_mm=[183,170],min_font_pt=min(sizes),text_outside_canvas=bad,same_case_geometry_and_target=True,native_points=571,target_points=61,no_temperature_map=True,no_prediction_curve=True,seven_variants=prov['models']);(HERE/'figure_qa.json').write_text(json.dumps(qa,indent=2),encoding='utf8');assert not bad,bad;print(json.dumps(qa,indent=2))

if __name__=='__main__':main()
