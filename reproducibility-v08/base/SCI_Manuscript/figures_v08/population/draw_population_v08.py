"""Manifest-only997-case descriptive figure; no prediction/result/tensor reads."""
from pathlib import Path
import hashlib,json,csv,datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.patches import Patch
from matplotlib import patheffects

HERE=Path(__file__).resolve().parent
SCI=HERE.parents[1]
MANIFEST=SCI/'experiments/research_v08/archive997/manifest.json'
DESIGN=SCI/'experiments/research_v08/design_audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=json.loads(MANIFEST.read_text(encoding='utf8'));cases=manifest['cases']
    df=pd.DataFrame([dict(sample_id=c['sample_id'],geometry_id=c['geometry_id'],source_batch=c['source_batch'],fire_family=c['fire_family'],crack_count=c['num_cracks'],legacy911=c['legacy_final_id'] is not None,flat_response=c['flat_response']) for c in cases])
    assert len(df)==997 and df.sample_id.nunique()==997 and df.geometry_id.nunique()==997
    assert all(c['num_cracks']==len(c['cracks']) for c in cases)
    families=['iso834','astm_e119','external_fire','linear','bilinear','plateau','decay','perturbed_iso','log_variant','smoldering']
    labels={'iso834':'ISO 834','astm_e119':'ASTM E119','external_fire':'External fire','linear':'Linear','bilinear':'Bilinear','plateau':'Plateau','decay':'Decay','perturbed_iso':'Perturbed ISO','log_variant':'Log variant','smoldering':'Smoldering'}
    table=pd.crosstab(df.fire_family,df.crack_count).reindex(index=families,columns=range(1,16),fill_value=0)
    batch=pd.crosstab(df.source_batch,df.crack_count).reindex(index=['batch_004','batch_005'],columns=range(1,16),fill_value=0)
    design=pd.read_csv(DESIGN/'population_batch_family_count.csv')
    expected=design.groupby(['source_batch','fire_family','crack_count']).case_count.sum().sort_index()
    actual=df.groupby(['source_batch','fire_family','crack_count']).size().sort_index()
    assert expected.equals(actual)
    comps=json.loads((DESIGN/'split_composition.json').read_text(encoding='utf8'))
    limits=json.loads((DESIGN/'design_limitations.json').read_text(encoding='utf8'))
    ndev=int((df.crack_count<=8).sum());ntest=int((df.crack_count>=9).sum())
    assert ndev==comps['lcro_9_15']['train']['cases']+comps['lcro_9_15']['validation']['cases']
    assert ntest==comps['lcro_9_15']['test']['cases']
    restored=df[~df.legacy911];smolder=df[df.fire_family=='smoldering']
    assert len(restored)==86 and restored.flat_response.all() and set(restored.fire_family)=={'smoldering'}
    assert df[df.crack_count>=9].source_batch.nunique()==1 and set(df[df.crack_count>=9].source_batch)=={'batch_005'}
    records=[]
    for fam in families:
        for n in range(1,16):records.append(dict(fire_family=fam,crack_count=n,cases=int(table.loc[fam,n])))
    pd.DataFrame(records).to_csv(HERE/'fire_family_by_crack_count.csv',index=False)
    pd.DataFrame([dict(source_batch=b,crack_count=n,cases=int(batch.loc[b,n])) for b in batch.index for n in batch.columns]).to_csv(HERE/'source_batch_by_crack_count.csv',index=False)
    retention=[]
    for group,subset in [('All cases',df),('Smoldering',smolder)]:
        legacy=int(subset.legacy911.sum());total=len(subset)
        retention.append(dict(group=group,legacy_retained=legacy,restored=total-legacy,total=total,retained_fraction=legacy/total,restored_fraction=(total-legacy)/total))
    pd.DataFrame(retention).to_csv(HERE/'retained_and_restored_composition.csv',index=False)
    df.to_csv(HERE/'population_case_identifiers.csv',index=False)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig=plt.figure(figsize=(183/25.4,164/25.4))
    ax=fig.add_axes([.20,.54,.735,.38]);bx=fig.add_axes([.10,.16,.445,.20]);cx=fig.add_axes([.74,.175,.235,.17]);cb=fig.add_axes([.958,.54,.014,.38])
    cmap=LinearSegmentedColormap.from_list('counts',['#FFFFFF','#DEEBF1','#7DA9BD','#315D80'])
    mat=table.to_numpy();norm=Normalize(vmin=0,vmax=int(mat.max()))
    im=ax.pcolormesh(np.arange(-.5,15,1),np.arange(-.5,10,1),mat,cmap=cmap,norm=norm,shading='flat',edgecolors='#E8ECF0',linewidth=.35)
    ax.set_xlim(-.5,14.5);ax.set_ylim(9.5,-.5)
    ax.set_xticks(np.arange(15),[str(n) for n in range(1,16)]);ax.set_yticks(np.arange(10),[f'{labels[f]}  ({int(table.loc[f].sum())})' for f in families])
    ax.tick_params(axis='both',length=0,pad=5);ax.set_xlabel('Number of cracks, N',labelpad=5)
    for side in ax.spines.values():side.set_visible(False)
    ax.set_xticks(np.arange(-.5,15,1),minor=True);ax.set_yticks(np.arange(-.5,10,1),minor=True);ax.grid(which='minor',color='#E8ECF0',lw=.45);ax.tick_params(which='minor',bottom=False,left=False)
    for j in range(10):
        for k in range(15):
            v=int(mat[j,k]);rgb=np.array(cmap(norm(v))[:3]);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4);luminance=float(linear@[.2126,.7152,.0722]);white_contrast=1.05/(luminance+.05);dark_contrast=(luminance+.05)/.067
            ax.text(k,j,str(v),ha='center',va='center',fontsize=8,color='white' if white_contrast>dark_contrast else '#142634')
    boundary=ax.axvline(7.5,color='#4D5056',lw=1.1,ls=(0,(3,2)));boundary.set_path_effects([patheffects.Stroke(linewidth=2.4,foreground='white'),patheffects.Normal()])
    cbar=fig.colorbar(im,cax=cb);cbar.solids.set_rasterized(False);cbar.solids.set_edgecolor('face');cbar.ax.set_title('Cases',pad=6,fontsize=8);cbar.outline.set_visible(False);cbar.ax.tick_params(labelsize=8,length=2);cbar.set_ticks([0,7,14,21]);assert mat.max()==21
    fig.text(.04,.96,'a  Joint population coverage: 997 archived cases',fontweight='bold',fontsize=8,va='top')
    batchcolors={'batch_004':'#91A8B9','batch_005':'#315D80'};positions=np.arange(1,16)
    bx.axvspan(8.5,15.6,color='#E9F1EF',zorder=0)
    bx.bar(positions,batch.loc['batch_004'],color=batchcolors['batch_004'],width=.73,label=f'Source 004 ({int(batch.loc["batch_004"].sum())})')
    bx.bar(positions,batch.loc['batch_005'],bottom=batch.loc['batch_004'],color=batchcolors['batch_005'],width=.73,label=f'Source 005 ({int(batch.loc["batch_005"].sum())})')
    totals=batch.sum(axis=0)
    for n,v in totals.items():bx.text(n,float(v)+2,str(int(v)),ha='center',va='bottom',fontsize=8)
    bx.axvline(8.5,color='#4D5056',lw=.9,ls=(0,(3,2)));bx.set(xlim=(.35,15.65),ylim=(0,115),xticks=range(1,16),yticks=[0,50,100],xlabel='Number of cracks, N',ylabel='Cases')
    bx.tick_params(axis='x',length=2,pad=3)
    fig.text(.10,.44,'b  Graph-size split and source batch',fontweight='bold',fontsize=8,va='top')
    bx.legend(loc='lower left',bbox_to_anchor=(-.02,1.02),frameon=False,ncol=2,handlelength=1.1,columnspacing=.75,handletextpad=.35,borderaxespad=0)
    fig.text(.10,.09,f'N = 1–8: train + validation, {ndev} cases',fontsize=8,va='top')
    fig.text(.10,.062,f'N = 9–15: test, {ntest} cases; source 005 only',fontsize=8,va='top')
    retained_color='#CCD4DA';restored_color='#498D86'
    for j,r in enumerate(retention):
        f=r['retained_fraction'];cx.barh(j,f,height=.39,color=retained_color);cx.barh(j,1-f,left=f,height=.39,color=restored_color)
        cx.text(0,j-.32,f'{r["legacy_retained"]} + {r["restored"]} = {r["total"]}',fontsize=8,ha='left',va='bottom')
    cx.set(xlim=(0,1),ylim=(1.55,-.7),xticks=[0,.5,1],xticklabels=['0','50','100'],yticks=[0,1],yticklabels=['All cases','Smoldering'],xlabel='Share of cases (%)')
    cx.spines['left'].set_visible(False);cx.tick_params(axis='y',length=0,pad=5);cx.tick_params(axis='x',length=2,pad=3)
    fig.text(.635,.44,'c  Restored constant responses',fontweight='bold',fontsize=8,va='top')
    cx.legend(handles=[Patch(facecolor=retained_color,label='Retained'),Patch(facecolor=restored_color,label='Restored')],loc='lower left',bbox_to_anchor=(-.46,1.02),frameon=False,ncol=2,handlelength=1.1,columnspacing=.65,handletextpad=.35,borderaxespad=0)
    fig.text(.635,.09,'86 restored cases are all smoldering;',fontsize=8,va='top');fig.text(.635,.062,'constant responses remain in the benchmark.',fontsize=8,va='top')
    fig.canvas.draw();renderer=fig.canvas.get_renderer();overflow=[];sizes=[]
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text():
            box=t.get_window_extent(renderer);sizes.append(t.get_fontsize())
            if box.x0<-.25 or box.y0<-.25 or box.x1>fig.bbox.x1+.25 or box.y1>fig.bbox.y1+.25:overflow.append(t.get_text())
    assert min(sizes)>=8
    for ext in ['pdf','svg','png']:fig.savefig(HERE/f'population_v08.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    sources=[MANIFEST,DESIGN/'population_batch_family_count.csv',DESIGN/'split_composition.json',DESIGN/'design_limitations.json']
    prov=dict(created_at=datetime.datetime.now().astimezone().isoformat(),source_files=[dict(path=str(p.relative_to(SCI)),sha256=sha(p)) for p in sources],cases=len(df),unique_sample_ids=int(df.sample_id.nunique()),unique_geometry_groups=int(df.geometry_id.nunique()),legacy_cases=int(df.legacy911.sum()),restored_cases=len(restored),smoldering_cases=len(smolder),source_batches={k:int(v) for k,v in df.source_batch.value_counts().items()},development_N1_8=ndev,test_N9_15=ntest,all_N9_15_from_batch005=True,all_cells_annotated=True,design_audit_aggregation_exactly_matched=True,prediction_or_result_or_tensor_files_read=False,scope=limits,figure_size_mm=[183,164],font_sizes_pt=sorted(set(sizes)),text_outside_canvas=overflow,backend='Python/matplotlib')
    (HERE/'provenance.json').write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding='utf8')
    assert not overflow,overflow
    print(json.dumps({k:prov[k] for k in ['cases','legacy_cases','restored_cases','smoldering_cases','source_batches','development_N1_8','test_N9_15','font_sizes_pt','text_outside_canvas']},indent=2))

if __name__=='__main__':main()
