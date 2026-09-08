"""Short CPU software checks only; no optimizer steps or scientific scores."""
from pathlib import Path
import datetime
import argparse
import json
import math
import sys
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_extension as run
from attention_model import SetAttentionSurrogate, parameter_count


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(HERE/'software_smoke_cpu_initial.json'))
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260907)
    original=run.read(run.V08/'comparison_plan_v08_420.json')
    assert run.rr.sources()==original['source_sha256']
    manifest,data=run.rr.load_dataset(original['manifest_path'])
    split=run.read(original['protocols']['iid997']['split_path']);train=set(split['train'])
    selected=[next(c['sample_id'] for c in manifest['cases'] if c['num_cracks']==n and c['sample_id'] in train) for n in (1,3,8,15)]
    graphs=[data['graphs'][sid] for sid in selected]
    model=SetAttentionSurrogate().eval();assert parameter_count(model)==194273
    def pred(gs):return model(*run.rr.collate(gs,'cpu')).squeeze(-1)
    with torch.no_grad():
        full=pred(graphs);alone=torch.cat([pred([g]) for g in graphs]);reverse=pred(graphs[::-1]).flip(0)
        node_reverse=pred([(g[0].flip(0),g[1],g[2],g[3],g[4]) for g in graphs])
        no_edges=pred([(g[0],torch.empty((2,0),dtype=torch.long),torch.empty((0,5)),g[3],g[4]) for g in graphs])
    errors={'batch_vs_single':float((full-alone).abs().max()),'batch_order':float((full-reverse).abs().max()),
            'node_permutation':float((full-node_reverse).abs().max()),'edge_removal':float((full-no_edges).abs().max())}
    assert full.shape==(4,61) and torch.isfinite(full).all()
    assert all(e<=1e-6 for e in errors.values())
    model.train();output=pred(graphs)
    synthetic=torch.linspace(1.,0.,61)[None,:].expand_as(output)
    loss=torch.nn.functional.mse_loss(output,synthetic);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    dense=torch.randn(1,15,64,requires_grad=True);valid=torch.arange(15)[None,:]<3
    block=model.structure_head.blocks[0];cross=block(dense,valid)[0,0].sum()
    gradient=torch.autograd.grad(cross,dense)[0]
    interaction=float(gradient[0,1].abs().max());assert interaction>1e-9
    with torch.no_grad():
        changed=dense.detach().clone();changed[:,3:]=1000.
        padding_error=float((block(dense.detach(),valid)[:,:3]-block(changed,valid)[:,:3]).abs().max())
    assert padding_error==0.
    rejected=[]
    for count in (0,16):
        g=graphs[-1]
        nodes=g[0][:0] if count==0 else torch.cat([g[0],g[0][:1]])
        altered=(nodes,torch.empty((2,0),dtype=torch.long),torch.empty((0,5)),g[3],g[4])
        try:pred([altered])
        except ValueError:rejected.append(count)
    assert rejected==[0,16]
    best=math.inf;chosen=None
    for epoch,v in enumerate([3.,2.,2.,4.],1):
        if run.new_best(v,best):best=v;chosen=epoch
    assert chosen==2 and not run.new_best(2.,2.)
    try:run.new_best(float('nan'),2.)
    except FloatingPointError:nonfinite_rejected=True
    else:nonfinite_rejected=False
    assert nonfinite_rejected
    outside_rejected=False
    try:run.safe_output(run.V08/'runs_frozen_v08_420')
    except ValueError:outside_rejected=True
    assert outside_rejected and not torch.cuda.is_initialized()
    # Same architecture/code for all common branches; this constructor also
    # preserves the matched-set control's original shared-branch initialization.
    torch.manual_seed(42);matched=run.rr.make_model('capacity_matched_deepsets')
    torch.manual_seed(42);attention=SetAttentionSurrogate()
    shared_initialization={}
    for prefix in ('temporal_head.','decoder.','structure_head.global_encoder.','structure_head.global_update.'):
        left={k:v for k,v in matched.state_dict().items() if k.startswith(prefix)}
        right={k:v for k,v in attention.state_dict().items() if k.startswith(prefix)}
        same=left.keys()==right.keys() and all(torch.equal(left[k],right[k]) for k in left)
        assert same;shared_initialization[prefix]=same
    report={'status':'SHORT_CPU_SOFTWARE_SMOKE_PASSED_NOT_PLAN_OR_TRAINING_AUTHORIZATION',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parameters':194273,
            'GNN_parameters':194177,'parameter_delta':96,'relative_parameter_delta':96/194177,
            'selected_training_geometry_ids':selected,'crack_counts':[1,3,8,15],'shape':[4,61,1],
            'invariance_tolerance':1e-6,'maximum_absolute_differences':errors,'padding_invariance_max_abs':padding_error,
            'nonzero_cross_node_attention_gradient':interaction,'finite_forward_backward':True,
            'all_trainable_parameters_received_gradients':True,'scope_rejections':rejected,
            'selection_earliest_tied_minimum_verified':chosen==2,'nonfinite_validation_rejected':nonfinite_rejected,
            'original_output_path_rejected':outside_rejected,'optimizer_steps':0,'scientific_prediction_scores_read':False,
            'matched_set_common_branch_initial_states_equal_seed42':shared_initialization,
            'GPU_initialized':False,'full_plan_and_authorization_gate_still_pending':True,
            'source_sha256':{str(p):run.rr.digest(p) for p in (HERE/'attention_model.py',HERE/'run_extension.py',Path(__file__))}}
    target=run.safe_output(args.output)
    with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
