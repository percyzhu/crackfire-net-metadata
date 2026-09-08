"""Actual incomplete60 run-gate probe; never measure timings or read scores."""
from pathlib import Path
from unittest.mock import patch
import datetime
import json
import benchmark_attention_extension as bench


def main():
    counts={key:0 for key in ('feature_dataset_load','weights_or_tensors_load','npz_load','model_creation','artifact_payload_validation','timed_clock')}
    def forbid(key):
        def failure(*args,**kwargs):counts[key]+=1;raise AssertionError('Premature '+key)
        return failure
    target=bench.HERE/'measured';bench.require(not target.exists(),'Existing real measurement cannot be probed')
    captured={};real_readiness=bench.readiness
    def readiness(*args,**kwargs):
        result=real_readiness(*args,**kwargs);captured.update(result);return result
    with patch.object(bench,'readiness',readiness),patch.object(bench.bm,'load_sources',forbid('feature_dataset_load')),\
         patch.object(bench.torch,'load',forbid('weights_or_tensors_load')),patch.object(bench.np,'load',forbid('npz_load')),\
         patch.object(bench,'SetAttentionSurrogate',forbid('model_creation')),patch.object(bench.ext.rr,'make_model',forbid('model_creation')),\
         patch.object(bench,'validate_artifacts',forbid('artifact_payload_validation')),patch.object(bench.time,'perf_counter',forbid('timed_clock')):
        result=bench.main(['run','--audit-dir',str(bench.EXT/'evaluation_final60'),
            '--review',str(bench.EXT/'independent_full_matrix_review.json'),
            '--authorization',str(bench.HERE/'execution_authorization_timing_root.json')])
    bench.require(result==2 and not any(counts.values()) and not target.exists(),'Incomplete gate performed prohibited work')
    record={'status':'ACTUAL_INCOMPLETE60_TIMING_RUN_REFUSED_ZERO_MEASUREMENTS',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timing_plan_sha256':bench.sha(bench.PLAN),
        'benchmark_source_sha256':bench.sha(bench.__file__),'probe_source_sha256':bench.sha(__file__),
        'actual_gate':captured,'forbidden_call_counts':counts,'returncode':result,'measurement_directory_created':False,
        'old7_timing_modified':False,'latencies_measured':0,'complete_collection_branch_tested':False}
    bench.write(bench.HERE/'deferred_gate_probe_evidence.json',record)
    print(json.dumps({'status':record['status'],'completed_metadata_runs':captured['completed_extension_metadata_runs'],
        'reasons':captured['reasons'],'forbidden_call_counts':counts,'returncode':result,'timing_plan_sha256':bench.sha(bench.PLAN)},indent=2))


if __name__=='__main__':main()
