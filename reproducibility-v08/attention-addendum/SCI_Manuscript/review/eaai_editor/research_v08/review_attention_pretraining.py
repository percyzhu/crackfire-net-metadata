"""Independent root review: frozen plan and GPU software checks, no training."""
from pathlib import Path
import datetime
import json
import sys
import torch

SCI = Path(__file__).resolve().parents[3]
EXT = SCI / 'experiments/research_v09_set_attention'
sys.path.insert(0, str(EXT))
import run_extension as ex
from attention_model import SetAttentionSurrogate, parameter_count

plan_path = EXT / 'plan_set_attention_60_v1.json'
plan = ex.verify_plan(plan_path)
assert ex.rr.digest(plan_path) == '8d21e13af116dc97556b37060daaf86afed5f7ba645a5bdae18e3d4ffb3b05a9'
_, gate = ex.execution_gate(plan_path, None)
assert gate['status'] == 'BLOCKED_PENDING_EXPLICIT_EXECUTION_AUTHORIZATION'
manifest, data = ex.rr.load_dataset(plan['manifest_path'])
split = ex.read(plan['protocols']['iid997']['split_path'])
ids = [next(c['sample_id'] for c in manifest['cases'] if c['num_cracks'] == n and c['sample_id'] in split['train']) for n in (1, 3, 8, 15)]
graphs = [data['graphs'][s] for s in ids]
ex.rr.configure(20260907, 4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True
model = SetAttentionSurrogate().cuda().eval()
assert parameter_count(model) == 194273

def pred(items):
    return model(*ex.rr.collate(items, 'cuda')).squeeze(-1)

with torch.no_grad():
    base = pred(graphs)
    solo = torch.cat([pred([g]) for g in graphs])
    reverse_batch = pred(graphs[::-1]).flip(0)
    reverse_nodes = pred([(g[0].flip(0), g[1], g[2], g[3], g[4]) for g in graphs])
    zero_edges = pred([(g[0], torch.empty((2, 0), dtype=torch.long), torch.empty((0, 5)), g[3], g[4]) for g in graphs])
checks = {key: float((base - value).abs().max()) for key, value in
          [('batch_single', solo), ('batch_permutation', reverse_batch), ('node_permutation', reverse_nodes), ('edges_ignored', zero_edges)]}
assert base.shape == (4, 61) and torch.isfinite(base).all()
assert max(checks.values()) < 1e-6
model.train()
out = pred(graphs)
synthetic = torch.linspace(1., 0., 61, device='cuda')[None, :].expand_as(out)
torch.nn.functional.mse_loss(out, synthetic).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
torch.cuda.synchronize()
report = {
    'status': 'INDEPENDENT_PRETRAINING_SOFTWARE_AND_PLAN_REVIEW_PASSED',
    'software_review_passed': True,
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'extension_plan_sha256': ex.rr.digest(plan_path),
    'parameters': 194273, 'parameter_difference_from_original_graph': 96,
    'new_training_cells': 60, 'original_comparator_runs_reused': 120,
    'GPU': torch.cuda.get_device_name(0), 'GPU_checks_maximum_absolute_difference': checks,
    'GPU_backward_all_parameters_finite': True, 'optimizer_updates': 0,
    'scientific_target_accuracy_read': False, 'examples_from_IID_training_only': ids,
    'manual_review': [
        'Four shared attention blocks; no position embeddings or raw edge inputs; padded keys masked before softmax and padded outputs excluded from valid-node means.',
        'Original 11-node/3-global/61x4 time interface, common branches and frozen targets/splits preserved.',
        'Training loop matches original optimizer/scheduler/validation selection/order/patience and maximum budget. No test results enter selection.',
        'Attention includes self keys and layer normalization. Capacity matching does not isolate an architectural mechanism causally.',
        'One-to-fifteen count scope and padding capacity are explicit. No claim of arbitrary graph topology transfer follows from this set control.',
        'Original420 files are read-only; new output must remain in extension directory and cannot overwrite an existing attempt.',
        'Plan and four extension sources are frozen; original plan/source/split/checkpoint identities are independently bound.',
        'Extension is explicitly after original test-results review, exploratory, and fixed to all12 protocols/5seeds before any training.'
    ],
    'remaining_research_work': 'Actual60 runs, full replay/statistics and independent scientific comparison; this is not manuscript acceptance.',
    'source_sha256': {str(p): ex.rr.digest(p) for p in [Path(__file__), plan_path, EXT/'software_smoke_cpu_final.json']}
}
dest = Path(__file__).with_name('set_attention_independent_pretraining_review.json')
with dest.open('x', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print(json.dumps({'status': report['status'], 'GPU_checks': checks, 'optimizer_updates': 0}))
