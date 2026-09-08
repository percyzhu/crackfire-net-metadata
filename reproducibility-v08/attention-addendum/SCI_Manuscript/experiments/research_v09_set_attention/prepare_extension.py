"""Create a source-bound single-architecture plan, with training disabled.

Architecture choice is analytical capacity matching, not a performance search.
Original420 results have already been reviewed; the extension is exploratory.
"""
from pathlib import Path
import argparse
import datetime
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_extension as run
from attention_model import ARCHITECTURE,MODEL_VERSION,MODEL_NAME,SetAttentionSurrogate,parameter_count


def prepare(output, smoke_path):
    output=run.safe_output(output)
    if output.exists():raise ValueError('Preserve existing extension plans; choose an explicit reviewed revision')
    old_path=run.V08/'comparison_plan_v08_420.json';old=run.read(old_path)
    assert run.rr.digest(old_path)==run.ORIGINAL_PLAN_SHA and old['source_sha256']==run.rr.sources()
    report_path=run.V08/'evaluation/report.json';report=run.read(report_path)
    assert report['status']=='COMPLETE_420_RUN_MATRIX' and report['audited_completed_runs']==420 and report['cpu_prediction_replay_enabled']
    assert report['plan_sha256']==run.ORIGINAL_PLAN_SHA
    integrity_path=run.V08/'evaluation/run_integrity.json';audits=run.read(integrity_path)['audits']
    indexed={(a['protocol'],a['model'],a['seed']):a for a in audits};assert len(audits)==len(indexed)==420
    smoke=run.read(smoke_path)
    assert smoke['finite_forward_backward'] and smoke['parameters']==194273 and smoke['optimizer_steps']==0
    assert smoke['matched_set_common_branch_initial_states_equal_seed42']
    assert all(smoke['matched_set_common_branch_initial_states_equal_seed42'].values()) and not smoke['GPU_initialized']
    for path,digest in smoke['source_sha256'].items():assert run.rr.digest(path)==digest
    model=SetAttentionSurrogate();assert parameter_count(model)==194273
    references=[]
    for protocol in old['protocols']:
        for mode in ('capacity_matched_deepsets','gnn'):
            for seed in old['seeds']:
                a=indexed[(protocol,mode,seed)];assert a['status']=='PASSED_BINDINGS'
                folder=Path(old['runs_root'])/protocol/mode/('seed_'+str(seed))
                item={'protocol':protocol,'model':mode,'seed':seed,'artifacts':{}}
                for name in ('result.json','run_metadata.json','best.pt','test_predictions.npz'):
                    p=folder/name;item['artifacts'][name]={'path':str(p),'sha256':run.rr.digest(p)}
                assert item['artifacts']['best.pt']['sha256']==a['checkpoint_sha256']
                assert item['artifacts']['test_predictions.npz']['sha256']==a['predictions_sha256']
                references.append(item)
    assert len(references)==120
    sources=[HERE/x for x in ('attention_model.py','run_extension.py','smoke_cpu.py','prepare_extension.py')]
    plan={'status':'FROZEN_BEFORE_EXTENSION_TRAINING_AWAITING_EXECUTION_AUTHORIZATION',
          'purpose':run.PURPOSE,'epistemic_status':'AFTER_ORIGINAL_TEST_RESULTS_REVIEW_EXPLORATORY',
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'rationale':'Resolve an interaction-capable set comparator gap identified after complete420 review; not a claim of mechanism causality or a preregistered original comparison.',
          'original_results_already_reviewed':True,'original_plan_path':str(old_path),'original_plan_sha256':run.ORIGINAL_PLAN_SHA,
          'original_source_sha256':old['source_sha256'],'original_completed_report_path':str(report_path),
          'original_completed_report_sha256':run.rr.digest(report_path),'original_completed_audit_path':str(integrity_path),
          'original_completed_audit_sha256':run.rr.digest(integrity_path),
          'manifest_path':old['manifest_path'],'manifest_sha256':old['manifest_sha256'],
          'target_version':old['target_version'],'feature_version':old['feature_version'],
          'protocols':old['protocols'],'seeds':old['seeds'],'new_model':MODEL_NAME,'model_version':MODEL_VERSION,
          'architecture':ARCHITECTURE,'parameters':194273,'original_graph_parameters':194177,
          'parameter_delta':96,'relative_parameter_delta':96/194177,
          'capacity_choice':'72,769 original small-set parameters + 4*(16,960 + 129*104) attention/FFN parameters = 194,273. One fixed width chosen for parameter count only; no accuracy or validation search.',
          'training_configuration':old['training_configuration'],
          'backend_configuration':{'float_dtype':'float32','cuda_matmul_allow_tf32':False,'cudnn_allow_tf32':True,'deterministic_algorithms':True},
          'new_run_count':60,'runs_root':str(HERE/'runs_set_attention_60'),
          'execution_order':'Original protocol insertion order, then seeds42..46; one GPU child process at a time',
          'required_execution_authorization':'Separate authorization JSON binding this plan and a passed independent software review; original7 real timing and independent timing audit must be complete first.',
          'extension_source_sha256':{str(p):run.rr.digest(p) for p in sources},
          'software_smoke_path':str(Path(smoke_path).resolve()),'software_smoke_sha256':run.rr.digest(smoke_path),
          'reused_comparator_models':['capacity_matched_deepsets','gnn'],'reused_comparator_run_count':120,
          'reused_comparator_artifacts':references,
          'comparison_contract':{'primary_metric':'Equal-case mean over61times of absolute response-index error',
              'paired_contrasts':['matched_set_MAE_minus_set_attention_MAE','sum_GNN_MAE_minus_set_attention_MAE'],
              'positive_favors':'set_attention','seeds':5,'bootstrap_repetitions':5000,'bootstrap_random_seed':20260907,
              'bootstrap_design':'Reuse the v08 geometry-only, seed-only and crossed geometry-by-seed paired percentile method; no time resampling.',
              'secondary_engineering_metrics':'Retain the original prespecified final/J errors and both .8/.6 event/censoring/ranking summaries with NA policy; reused baselines use original saved predictions.',
              'performance_summary_gate':'All60 new model runs and bindings/replay checked before reporting cross-protocol performance; no test-based architecture or threshold adaptation.',
              'all_original420_retained':True,'negative_results_retained':True,'mechanism_causality_claim':False,
              'family_averages_scope':'Twelve different trained models across protocols; any ten-LOCO average is descriptive, not one deployed checkpoint.',
              'attention_edge_view_scope':'This set control ignores raw edges by construction; no new adjacency-robustness success claim.'},
          'prohibited_changes':['No overwrite of original420 plan, data, sources or weights','No HPO or additional architecture selected by test performance','No relabeling the extension as original preregistration'],
          'timing_scope':'Original7 timing completes first. It does not estimate the new attention model latency; any later attention timing requires an explicit separate record.'}
    with output.open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2,ensure_ascii=False);f.write('\n')
    run.verify_plan(output)
    _,gate=run.execution_gate(output,None)
    assert gate['status']=='BLOCKED_PENDING_EXPLICIT_EXECUTION_AUTHORIZATION'
    print(json.dumps({'status':plan['status'],'epistemic_status':plan['epistemic_status'],'new_runs':60,'comparator_runs_reused':120,
                      'parameters':194273,'plan':str(output),'plan_sha256':run.rr.digest(output),'actual_execution_gate':gate},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(HERE/'plan_set_attention_60_v1.json'))
    p.add_argument('--smoke',default=str(HERE/'software_smoke_cpu_final.json'));a=p.parse_args();prepare(a.output,a.smoke)
