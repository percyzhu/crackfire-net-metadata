"""Independent arithmetic/coverage audit; no benchmark imports or model execution."""
from pathlib import Path
from collections import defaultdict
import csv
import datetime
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
EXT = BENCH.parent
SCI = EXT.parents[1]
ROOT = SCI.parent
MEASURED = BENCH/'measured'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-12),(a,b)
def index(records,names):
    x={tuple(r[k] for k in names):r for r in records}
    assert len(x)==len(records)
    return x

def main():
    assert not (HERE/'independent_timing_audit.json').exists(), 'Preserve prior audit'
    assert not (MEASURED/'failure.json').exists()
    plan=read(BENCH/'timing_plan_v1.json'); summary=read(MEASURED/'summary.json')
    assert summary['status']=='COMPLETE_SEPARATE_THREE_MODEL_ATTENTION_TIMING'
    assert summary['rows']==6435 and summary['original7_14355_records_pooled_or_replaced'] is False
    assert sha(MEASURED/'summary.json')=='89d49c38d853081e69b89d7b1b5316194ee04de83d6ca092f22d1c3d9157771f'
    assert summary['timing_plan_sha256']==sha(BENCH/'timing_plan_v1.json')
    assert plan['benchmark_source_sha256']==sha(BENCH/'benchmark_attention_extension.py')
    assert plan['original_benchmark_helper_sha256']==sha(plan['original_benchmark_helper_path'])
    assert plan['extension_plan_sha256']==sha(EXT/'plan_set_attention_60_v1.json')
    assert plan['original_training_plan_sha256']==sha(SCI/'experiments/research_v08/comparison_plan_v08_420.json')
    for source,h in plan['extension_source_sha256'].items(): assert sha(source)==h
    for source,h in plan['original_source_sha256'].items(): assert sha(ROOT/source)==h
    assert plan['models']==summary['models']==['capacity_matched_deepsets','gnn','set_attention']
    assert plan['protocol']=='iid997' and plan['seed']==42 and plan['times_per_case']==61
    assert plan['devices']==['cpu','cuda'] and plan['batch_sizes']==[1,32] and plan['rounds']==3
    assert plan['CPU_threads']==4 and plan['warmup_full_population_passes']==1 and plan['warmup_batches_per_shape']==10
    assert plan['stages']==['forward_resident_inputs','common_input_to_host_output']
    assert sha(plan['old_timing_summary_path'])==plan['old_timing_summary_sha256']
    assert sha(plan['old_timing_review_path'])==plan['old_timing_review_sha256']
    assert sha(plan['manifest_path'])==plan['manifest_sha256'] and sha(plan['IID_split_path'])==plan['IID_split_sha256']
    manifest=read(plan['manifest_path']); ids=plan['sample_ids']
    assert ids==read(plan['IID_split_path'])['test'] and len(ids)==len(set(ids))==160
    counts={c['sample_id']:c['num_cracks'] for c in manifest['cases']}
    assert sorted({counts[s] for s in ids})==plan['observed_crack_counts']==list(range(1,16))
    audit=read(EXT/'independent_full_matrix_review.json')
    assert audit['status']=='PASS_INDEPENDENT_NUMERICAL_REVIEW_60_NEW_PLUS120_REUSED' and audit['numerical_review_passed']
    binding=read(MEASURED/'premeasurement_artifact_binding.json'); env=read(MEASURED/'environment.json')
    auth=read(BENCH/'execution_authorization_root.json'); gates=read(MEASURED/'readiness_checks.json')
    evidence={'aggregate_report_sha256':sha(EXT/'evaluation_complete60_v1/report.json'),
              'aggregate_run_integrity_sha256':sha(EXT/'evaluation_complete60_v1/run_integrity.json'),
              'independent_review_sha256':sha(EXT/'independent_full_matrix_review.json')}
    assert binding['evidence_hashes']==evidence
    assert binding['new_artifact_runs_checked']==60 and binding['reused_artifact_runs_checked']==120
    assert binding['status']=='ALL60_NEW_AND120_REUSED_ARTIFACTS_BOUND_BEFORE_SEPARATE_TIMING'
    assert binding['selected_checkpoints']==plan['checkpoints']
    assert auth['status']=='AUTHORIZED_SEPARATE_ATTENTION_TIMING_AFTER_COMPLETE60_REPLAY_AND_QUIET'
    assert auth['authorized_by']=='root' and auth['timing_plan_sha256']==summary['timing_plan_sha256']
    for key,val in evidence.items(): assert auth[key]==val
    assert len(gates)==23
    for gate in gates:
        assert gate['status']=='READY_AUTHORIZED_SEPARATE_TIMING' and gate['reasons']==[]
        assert gate['matching_project_processes']==[] and gate['completed_extension_metadata_runs']==60
        assert gate['timing_plan_sha256']==summary['timing_plan_sha256'] and gate['evidence_hashes']==evidence
    assert all(gates[i]['checked_utc']<=gates[i+1]['checked_utc'] for i in range(len(gates)-1))
    assert env['CPU_threads']==4 and env['old7_measurements_pooled'] is False
    assert env['matmul_allow_tf32'] is False and env['cudnn_allow_tf32'] is True and env['deterministic_algorithms'] is True
    for key,path in [('artifact_binding_sha256',MEASURED/'premeasurement_artifact_binding.json'),
                     ('raw_csv_sha256',MEASURED/'raw_timings.csv'),('environment_sha256',MEASURED/'environment.json'),
                     ('authorization_sha256',BENCH/'execution_authorization_root.json')]: assert summary[key]==sha(path)
    assert env['authorization_sha256']==summary['authorization_sha256']
    assert env['premeasurement_artifact_binding_sha256']==summary['artifact_binding_sha256']
    for ck in plan['checkpoints']:
        assert sha(ck['checkpoint'])==ck['checkpoint_sha256'] and sha(ck['run_metadata_path'])==ck['run_metadata_sha256']
        meta=read(ck['run_metadata_path'])
        assert meta['protocol']=='iid997' and meta['seed']==42 and meta['mode']==ck['model']
        assert meta['parameters']==ck['parameters'] and meta['plan_sha256']==ck['training_plan_sha256']
    with (MEASURED/'raw_timings.csv').open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    expected_order=[]
    for repeat in range(3):
        for batch in [1,32]:
            for start in range(0,160,batch): expected_order.append((repeat,'cpu','common_features','feature_preparation_only',batch,tuple(ids[start:start+batch])))
        for device in plan['device_orders_by_round'][repeat]:
            for model in plan['model_orders_by_round'][repeat]:
                for batch in [1,32]:
                    for stage in plan['stages']:
                        for start in range(0,160,batch): expected_order.append((repeat,device,model,stage,batch,tuple(ids[start:start+batch])))
    assert len(expected_order)==len(set(expected_order))==len(rows)==6435
    groups=defaultdict(list); roundgroups=defaultdict(list); ngroups=defaultdict(list)
    for row,expected in zip(rows,expected_order):
        samples=tuple(row['sample_ids'].split('|'))
        key=(int(row['round']),row['device'],row['model'],row['stage'],int(row['batch_size']),samples)
        assert key==expected and int(row['cases'])==len(samples)
        assert list(map(int,row['num_cracks'].split('|')))==[counts[s] for s in samples]
        value=float(row['elapsed_s']); assert math.isfinite(value) and value>0
        item=(value,len(samples),samples)
        groups[key[1:5]].append(item);roundgroups[key[:5]].append(item)
        if key[4]==1: ngroups[key[1:4]+(counts[samples[0]],)].append(item)
    assert len(groups)==26 and len(roundgroups)==78 and len(ngroups)==195
    mainrows=index(summary['summaries'],['device','model','stage','batch_size'])
    assert set(mainrows)==set(groups)
    for key,items in groups.items():
        t=np.array([x[0] for x in items]); n=sum(x[1] for x in items); result=mainrows[key]
        assert result['measurements']==len(t) and result['cases_total']==n==480
        for field,v in [('mean_batch_ms',t.mean()*1000),('median_batch_ms',np.median(t)*1000),
                        ('p95_batch_ms',np.percentile(t,95)*1000),('amortized_ms_per_case',t.sum()*1000/n),('cases_per_second',n/t.sum())]: close(result[field],v)
    rrows=index(summary['per_round_summaries'],['round','device','model','stage','batch_size'])
    assert set(rrows)==set(roundgroups)
    for key,items in roundgroups.items():
        total=math.fsum(x[0] for x in items);n=sum(x[1] for x in items);result=rrows[key]
        assert result['measurements']==len(items) and result['cases_total']==n==160
        for field,v in [('mean_batch_ms',total*1000/len(items)),('amortized_ms_per_case',total*1000/n),('cases_per_second',n/total)]:close(result[field],v)
    nrows=index(summary['single_query_by_crack_count'],['device','model','stage','num_cracks'])
    assert set(nrows)==set(ngroups)
    for key,items in ngroups.items():
        t=np.array([x[0] for x in items]);result=nrows[key]
        assert result['measurements']==len(t) and result['unique_cases']==len({x[2] for x in items})
        for field,v in [('mean_single_query_ms',t.mean()*1000),('median_single_query_ms',np.median(t)*1000),('p95_single_query_ms',np.percentile(t,95)*1000)]:close(result[field],v)
    table=[]
    for model in plan['models']:
        row={'model':model}
        for device in ['cpu','cuda']:
            single=mainrows[(device,model,'common_input_to_host_output',1)]
            batch=mainrows[(device,model,'common_input_to_host_output',32)]
            row.update({device+'_single_median_ms':single['median_batch_ms'],device+'_single_p95_ms':single['p95_batch_ms'],device+'_batch32_cases_per_second':batch['cases_per_second']})
        table.append(row)
    sourcepaths=[Path(__file__),BENCH/'timing_plan_v1.json',BENCH/'benchmark_attention_extension.py',BENCH/'execution_authorization_root.json',EXT/'independent_full_matrix_review.json']
    sourcepaths += list(MEASURED.glob('*.json'))+[MEASURED/'raw_timings.csv']
    report=dict(status='PASS_INDEPENDENT_6435_THREE_MODEL_TIMING_RECORDS',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        timing_summary_sha256=sha(MEASURED/'summary.json'),raw_timing_sha256=sha(MEASURED/'raw_timings.csv'),timing_plan_sha256=sha(BENCH/'timing_plan_v1.json'),
        raw_records=6435,model_records=5940,common_feature_records=495,overall_groups=26,round_groups=78,single_query_N_groups=195,
        fixed_IID_geometries=160,fixed_checkpoint_seed=42,whole_curve_points=61,three_model_orders_and_device_order_and_all_case_coverage_verified=True,
        selected_actual_checkpoints_and_metadata_verified=3,prior_independent_180_run_review_bound=True,quiet_gate_records_checked=len(gates),
        old7_14355_summary_and_review_hashes_unchanged=True,model_inference_or_new_latency_measurement_performed=False,
        arithmetic_tolerance=dict(relative=1e-11,absolute=1e-12),main_text_common_interface_table=table,
        CPU=env['CPU'],GPU=env['GPU'],CPU_threads=4,
        limitations=['Recorded project-process checks do not prove an idle operating system.','Three repetitions are timing variability, not independent accuracy samples or hardware-population uncertainty.',
            'One entire61-point response query; excludes import/checkpoint/disk/network/FE costs.','No pooling with old seven-model session and no FE speedup ratio.'],
        manuscript_submission_pass=False,source_sha256={str(p):sha(p) for p in sourcepaths})
    with (HERE/'independent_timing_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    with (HERE/'three_model_common_interface.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    labels={'capacity_matched_deepsets':'Matched set','gnn':'Sum GNN','set_attention':'Set attention'}
    latex=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{A separate common-interface timing session for three models. Each query returns the full 61-point response trajectory. Serial-query median and 95th-percentile latencies use 480 measurements (160 IID geometries, three rounds); throughput uses fifteen batches of 32. The session uses the same seed-42 checkpoints, four CPU threads and an RTX 3060 Ti. Common geometry/fire input preparation, collation, transfers and host output are included; checkpoint/disk I/O and FE computation are excluded. Original seven-model measurements are not pooled.}',
        r'\label{tab:attention_timing}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r' & \multicolumn{3}{c}{CPU} & \multicolumn{3}{c}{GPU} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
        r'Model & Median & p95 & Batch-32 & Median & p95 & Batch-32 \\',
        r' & ms & ms & cases/s & ms & ms & cases/s \\',r'\midrule']
    for row in table:
        values=[f"{row[d+'_single_median_ms']:.3f} & {row[d+'_single_p95_ms']:.3f} & {row[d+'_batch32_cases_per_second']:.1f}" for d in ['cpu','cuda']]
        latex.append(labels[row['model']]+' & '+' & '.join(values)+r' \\')
    latex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    with (HERE/'three_model_common_interface_table.tex').open('x',encoding='utf-8') as f:f.write('\n'.join(latex)+'\n')
    md='# 三模型统一计时：独立原始记录核查\n\n6435条原始计时全部通过覆盖、顺序与算术核对，包含5940条模型接口计时和495条公共特征准备计时。26组总体、78组分轮、195组单例节点数分层汇总均一致。全部模型轮换顺序、CPU/GPU轮换顺序、160测试ID和批次切分均与事前计划相符。\n\n| 模型 | CPU median / p95 (ms) | CPU batch32 (cases/s) | GPU median / p95 (ms) | GPU batch32 (cases/s) |\n|---|---:|---:|---:|---:|\n'
    for r in table:md+=f"| {labels[r['model']]} | {r['cpu_single_median_ms']:.3f} / {r['cpu_single_p95_ms']:.3f} | {r['cpu_batch32_cases_per_second']:.1f} | {r['cuda_single_median_ms']:.3f} / {r['cuda_single_p95_ms']:.3f} | {r['cuda_batch32_cases_per_second']:.1f} |\n"
    md+='\n每个病例返回完整61时刻曲线；吞吐为实际病例总数除以累计批次时间，不能用单例中位数倒数替代。这里只比较本次统一测量的三模型，原七模型14355条记录及其审查哈希均未改变。23次已记录进程门槛没有检出其他项目计算，但不能证明操作系统完全空闲。审阅只复核记录、来源与算术，没有重测、重新推理或产生FE加速比。\n\n可供主文采用的独立LaTeX表为`three_model_common_interface_table.tex`；排版由主文编译核查。结果核查通过不等于整稿送审通过。\n'
    with (HERE/'independent_timing_review_zh.md').open('x',encoding='utf-8') as f:f.write(md)
    print(json.dumps({'status':report['status'],'audit_sha256':sha(HERE/'independent_timing_audit.json'),'table':table},indent=2))

if __name__=='__main__':main()
