"""Verify all60 saved attention predictions, without model inference or bootstrap."""
import argparse, os, sys
import numpy as np
import addendum_lib as a

def main():
    p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--output',required=True);args=p.parse_args()
    manifest,lib,rr,plan,metadata,tensors,splits,registry,base=a.initialize(args.base)
    checks=[]
    for record in registry:
        folder,meta,result=a.check_run(record,plan,metadata)
        truth,pred=lib.saved_array(folder,'complete',splits[record['protocol']],tensors)
        metrics=rr.metrics(truth,pred)
        for key,value in metrics.items():
            if value is None: assert result[key] is None
            else: assert np.allclose(value,result[key],atol=1e-13,rtol=1e-13),key
        checks.append({'protocol':record['protocol'],'seed':record['seed'],'cases':len(truth),'times':61,
            'checkpoint_sha256':record['checkpoint_sha256'],'prediction_sha256':a.sha(folder/'test_predictions.npz')})
    report={'status':'PASSED_ADDENDUM_RELOCATION_AND_ALL60_SAVED_PREDICTION_METRICS',
        'addendum_manifest_sha256':a.sha(a.ROOT/'addendum_manifest.json'),
        'base_manifest_sha256':a.sha(base/'package_manifest.json'),
        'addendum_directory':str(a.ROOT),'base_directory':str(base),'working_directory':os.getcwd(),
        'original_historical_paths_used_for_loading':False,'checks':checks,
        'new_runs_verified':60,'test_occurrences':sum(c['cases'] for c in checks),
        'model_inference_calls':0,'bootstrap_recalculated':False,'budget_arithmetic_recalculated':False,
        'all_payload_hashes_verified':True,'scientific_submission_pass':False}
    a.write_report(args.output,report);print(report['status']);return 0

if __name__=='__main__': raise SystemExit(main())
