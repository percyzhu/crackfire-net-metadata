"""Preselected IID attention42 CPU software replay, no tuning or retraining."""
import argparse, os, sys
import numpy as np
import torch
import addendum_lib as a

def main():
    p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--output',required=True);args=p.parse_args()
    manifest,lib,rr,plan,metadata,tensors,splits,registry,base=a.initialize(args.base)
    record=next(r for r in registry if r['protocol']=='iid997' and r['seed']==42)
    folder,meta,result=a.check_run(record,plan,metadata)
    module=a.load_module('packaged_attention_model',a.local(a.EXT_REL+'/attention_model.py'))
    import models
    assert a.io(models.__file__).is_relative_to(a.io(base)), 'Model base class must come from the verified base package'
    rr.configure(42,4)
    model=module.SetAttentionSurrogate().to('cpu')
    checkpoint=torch.load(a.io(folder/'best.pt'),map_location='cpu',weights_only=True)
    for key in ['plan_sha256','dataset_sha256','split_sha256']: assert checkpoint[key]==meta[key]
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    assert module.parameter_count(model)==194273
    split=splits['iid997'];truth,saved=lib.saved_array(folder,'complete',split,tensors)
    actual=rr.predict(model,tensors['graphs'],split['test'],32,'cpu')
    difference=float(np.max(np.abs(actual.astype(float)-saved.astype(float))))
    passed=bool(np.isfinite(actual).all() and difference<3e-6)
    report={'status':'PASSED_FIXED_IID_ATTENTION42_PORTABLE_CPU_REPLAY' if passed else 'CPU_DIFFERENCE_RECORDED_NOT_A_REPLAY_PASS',
        'addendum_manifest_sha256':a.sha(a.ROOT/'addendum_manifest.json'),
        'base_manifest_sha256':a.sha(base/'package_manifest.json'),
        'addendum_directory':str(a.ROOT),'base_directory':str(base),'working_directory':os.getcwd(),
        'protocol':'iid997','model':'set_attention','seed':42,'device':'cpu','threads':4,
        'cases':len(saved),'times':61,'batch_size':32,'checkpoint_sha256':record['checkpoint_sha256'],
        'prediction_sha256':a.sha(folder/'test_predictions.npz'),'maximum_absolute_difference':difference,
        'CPU_tolerance_unchanged':3e-6,'criterion':'maximum absolute difference strictly less than 3e-6',
        'prediction_metrics_against_saved_truth':rr.metrics(truth,actual),
        'optimizer_updates':0,'all60_reinference_claim':False,'graph_adjacency_transfer_claim':False,
        'meaning':'Software replay difference, not physical experimental error.',
        'versions':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__},
        'scientific_submission_pass':False}
    a.write_report(args.output,report);print(report['status']);return 0 if passed else 2

if __name__=='__main__': raise SystemExit(main())
