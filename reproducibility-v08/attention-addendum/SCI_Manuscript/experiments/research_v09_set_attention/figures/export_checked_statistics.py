"""Export already independently reviewed36 model/protocol summaries for writing."""
from pathlib import Path
import csv,json,subprocess,datetime
import generate_extension_figures_v3 as gen


def main():
    aggregate=gen.EXT/'evaluation_complete60_v1';review=gen.EXT/'independent_full_matrix_review.json'
    plan,gate=gen.metadata_gate(aggregate,review);gen.require(gate['status']=='READY_COMPLETE_AND_INDEPENDENTLY_REVIEWED','Review gate')
    out=gen.HERE/'final_validation';gen.require(not out.exists(),'Preserve existing validation attempt');out.mkdir()
    rows=[]
    for protocol in plan['protocols']:
        s=gen.read_json(aggregate/protocol/'summary.json')
        for item in s['model_summaries']:
            row={'protocol':protocol,'model':item['model'],'independent_geometries':s['test_case_count'],'seed_count':5}
            for metric in ('equal_case_MAE','equal_case_RMSE'):
                row[metric+'_mean']=item['metrics'][metric]['mean'];row[metric+'_SD']=item['metrics'][metric]['sample_sd']
                row[metric+'_mean_times1000']=row[metric+'_mean']*1000;row[metric+'_SD_times1000']=row[metric+'_SD']*1000
            rows.append(row)
    gen.require(len(rows)==36,'Expected12 by3')
    gen.csv_write(out/'three_model_12_protocol_statistics.csv',rows)
    text=['# 已独立复核的12协议三模型统计','',
          '以下数值均为响应指标误差×10³，展示五种子均值±样本SD；n是独立测试几何数。RMSE是每个种子先在所有case×time误差上求均方根，再对五种子求均值/SD，不是逐例RMSE的平均。','']
    for metric in ('equal_case_MAE','equal_case_RMSE'):
        text+=['## '+metric,'','| 协议 | n | Matched set | Sum GNN | Set attention |','|---|---:|---:|---:|---:|']
        for protocol in plan['protocols']:
            values=[next(r for r in rows if r['protocol']==protocol and r['model']==model) for model in gen.MODELS]
            formatted=[f"{r[metric+'_mean_times1000']:.3f} ± {r[metric+'_SD_times1000']:.3f}" for r in values]
            text.append('| '+' | '.join([gen.LABELS[protocol],str(values[0]['independent_geometries']),*formatted])+' |')
        text.append('')
    (out/'three_model_12_protocol_statistics.md').write_text('\n'.join(text),encoding='utf-8')
    tex=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{booktabs,amsmath}
\begin{document}
\input{../complete_reviewed_v3/table_extension_mae.tex}
\clearpage
\input{../complete_reviewed_v3/table_extension_effects.tex}
\end{document}
'''
    (out/'tables_compile.tex').write_text(tex,encoding='utf-8')
    for repeat in (1,2):
        completed=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','tables_compile.tex'],cwd=out,capture_output=True,text=True)
        (out/f'pdflatex_run{repeat}.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        gen.require(completed.returncode==0,'LaTeX compile failed')
    log=(out/'tables_compile.log').read_text(encoding='utf-8',errors='replace')
    gen.require('Overfull' not in log,'Table overfull box detected')
    import fitz
    pdf=fitz.open(out/'tables_compile.pdf');pages=[]
    for i,page in enumerate(pdf):
        page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/f'tables_page{i+1}.png')
        gen.require('Smoldering' in page.get_text(),'Last protocol absent in compiled table')
        pages.append({'page':i+1,'text_characters':len(page.get_text())})
    # Verify vector figure text and export one PDF render for direct visual QA.
    for name in ('extension_mae','extension_effects'):
        figure=fitz.open(gen.HERE/'complete_reviewed_v3'/f'{name}.pdf')
        gen.require(len(figure)==1 and 'Smoldering' in figure[0].get_text(),'Vector figure text missing')
        figure[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(out/f'{name}_PDF_preview.png')
    value={'status':'STATISTICS_EXPORTED_AND_ACTUAL_TABLES_COMPILED_VISUAL_REVIEW_PENDING',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gate':gate,'rows':36,'protocols':12,'models':list(gen.MODELS),
        'LaTeX_compile_passed':True,'overfull_boxes':0,'compiled_table_pages':pages,'visual_review_pending':True,
        'source_sha256':gen.sha(__file__),'outputs_sha256':{p.name:gen.sha(p) for p in out.iterdir() if p.is_file()}}
    gen.write_json(out/'validation_manifest.json',value);print(json.dumps({'status':value['status'],'output':str(out),'table_pages':len(pdf),'rows':36},indent=2))


if __name__=='__main__':main()
