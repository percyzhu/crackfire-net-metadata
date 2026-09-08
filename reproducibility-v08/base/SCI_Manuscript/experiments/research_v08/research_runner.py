"""Versioned archival-target AI experiments; independent of qualified-data runs.

The learner emulates the saved scalar envelope, not a newly reconstructed
pointwise-history capacity. Preparing tensors never admits numerical fields.
Training requires an explicit frozen v08 plan that binds all source hashes.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
SCI = EXPERIMENTS.parent
WORKSPACE = SCI.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(SCI / 'figures_v06/topology'))
import physical_features as pf
from models import Surrogate
from capacity_matched_set import CapacityMatchedDeepSets
from run_readiness import legacy_functions, collate
from qualified_data import tensor_collection_digest, geometry_id
from topology_extension import fit_radius_on_training_geometries, graph_view

MODES = ['fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets',
         'gnn', 'gnn_zero_edge_features', 'gnn_mean']
TARGET_VERSION = 'archive-scalar-running-min-interp61-v1'
FEATURE_VERSION = 'fe-global-v2-prescribed-archive-time-v1'
PURPOSE = 'ARCHIVAL_SURROGATE_RESEARCH_V08'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def safe_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE):
        raise ValueError('All v08 writes must remain in research_v08')
    return path


def sources():
    paths = [Path(__file__), HERE / 'prepare997.py', HERE / 'make_plan.py', HERE / 'run_queue.py',
             EXPERIMENTS / 'models.py', EXPERIMENTS / 'physical_features.py',
             EXPERIMENTS / 'capacity_matched_set.py', EXPERIMENTS / 'run_readiness.py',
             EXPERIMENTS / 'qualified_data.py', SCI / 'figures_v06/topology/topology_extension.py',
             WORKSPACE / '1_代码/src/data/dataset.py', WORKSPACE / '1_代码/src/gnn/model.py']
    return {str(p.relative_to(WORKSPACE)).replace('\\', '/'): digest(p) for p in paths}


class MeanMessageLayer(nn.Module):
    """Same parameter objects/state names as the sum layer; degree normalized."""
    def __init__(self, original):
        super().__init__()
        self.edge_mlp = original.edge_mlp
        self.node_mlp = original.node_mlp
        self.isolated_node_policy = 'always_update'

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        edge_attr = edge_attr + self.edge_mlp(torch.cat([x[src], x[dst], edge_attr], -1))
        aggregate = torch.zeros_like(x).index_add(0, dst, edge_attr)
        degree = torch.bincount(dst, minlength=len(x)).clamp_min(1)
        aggregate = aggregate / degree[:, None]
        return x + self.node_mlp(torch.cat([x, aggregate], -1)), edge_attr


def make_model(mode):
    if mode == 'capacity_matched_deepsets':
        return CapacityMatchedDeepSets()
    model = Surrogate(mode='gnn' if mode == 'gnn_mean' else mode)
    if mode == 'gnn_mean':
        model.structure_head.processors = nn.ModuleList(
            [MeanMessageLayer(layer) for layer in model.structure_head.processors])
    return model


def prepare(output):
    """Read only small NPZ members; do not read nodes/elements/full fields."""
    output = safe_output(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refuse to overwrite a prepared tensor snapshot')
    output.mkdir(parents=True, exist_ok=True)
    raw = WORKSPACE / 'Abaqus/processed_pilot/raw'
    meta_path = raw / 'batch_run_log.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    audit_path = SCI / 'audit/dataset/dataset_manifest.csv'
    with audit_path.open(encoding='utf-8-sig', newline='') as stream:
        audit = {r['filename']: r for r in csv.DictReader(stream) if r['dataset'] == 'processed_pilot'}
    if not len(meta['results']) == len(meta['crack_params']) == len(meta['fire_curves']) == len(audit) == 911:
        raise ValueError('Unexpected archival population')
    ns = legacy_functions(WORKSPACE)
    graphs, targets, cases = {}, {}, []
    times = np.linspace(0., 3600., 61)
    for row, cracks, fire in zip(meta['results'], meta['crack_params'], meta['fire_curves']):
        sid = 'sample_%04d' % row['sample_id']
        path = raw / (sid + '.npz')
        ar = audit[path.name]
        if row['status'] != 'success' or sid in targets:
            raise ValueError('Duplicate or invalid archived record: ' + sid)
        with np.load(path, allow_pickle=False) as z:
            native_t = z['time_steps']
            native_y = z['charring_ratios']
            raw_cracks = z['crack_params']
            beam_dims = z['beam_dims']
        expected_cracks = np.asarray([[float(c[k]) for k in ('face', 'z', 'h', 'w', 'l', 'd')] for c in cracks])
        if not np.array_equal(raw_cracks, expected_cracks):
            raise ValueError('NPZ/metadata geometry mismatch: ' + sid)
        if not np.array_equal(np.asarray(beam_dims).ravel(), [1.5, .14, .2]):
            raise ValueError('Unexpected beam dimensions')
        if native_t.shape != native_y.shape or native_t.ndim != 1 or len(native_t) < 2:
            raise ValueError('Invalid scalar history: ' + sid)
        if not np.isfinite(native_t).all() or not np.isfinite(native_y).all():
            raise ValueError('Nonfinite history')
        if np.any(np.diff(native_t) <= 0) or native_t[0] != 0 or native_t[-1] != 3600:
            raise ValueError('Incomplete or unordered history')
        if np.any(native_y < 0) or np.any(native_y > 1 + 1e-8):
            raise ValueError('Out-of-range archive label; no clipping permitted')
        y64 = np.interp(times, native_t, np.minimum.accumulate(native_y))
        # Bind the actual small arrays read today, independent of old full-file SHA.
        scalar_hash = hashlib.sha256(native_t.tobytes() + native_y.tobytes()).hexdigest()
        beam = dict(pf.DEFAULT_BEAM)
        case = {'sample_id': sid, 'beam': beam, 'cracks': cracks, 'fire': fire,
                'num_cracks': len(cracks), 'fire_family': fire['type'],
                'source_npz': str(path.relative_to(WORKSPACE)).replace('\\', '/'),
                'source_npz_sha256_prior_full_audit': ar['file_sha256'],
                'source_npz_bytes': path.stat().st_size,
                'source_scalar_arrays_sha256_current': scalar_hash,
                'native_time_count': len(native_t),
                'legacy_envelope_change_max': float(np.max(np.abs(np.interp(times, native_t, native_y) - y64)))}
        case['geometry_id'] = geometry_id(case)
        tf = ns['make_time_features'](61, 3600., fire['type'], fire['params']).squeeze(0)
        edges, attrs = pf.build_crack_edges(cracks, beam)
        graphs[sid] = (pf.encode_crack_node_features(cracks, beam), edges, attrs,
                       pf.encode_global_features(cracks, beam), tf)
        targets[sid] = torch.tensor(y64, dtype=torch.float32)
        if not all(torch.isfinite(a).all() for a in graphs[sid]):
            raise ValueError('Nonfinite graph tensors')
        cases.append(case)
    if len({c['geometry_id'] for c in cases}) != len(cases):
        raise ValueError('Repeated geometry needs explicitly grouped splits')
    # Save only small graph/time/target tensors, never megabyte FE mesh members.
    torch.save({'graphs': graphs, 'targets': targets, 'time_s': torch.tensor(times)}, output / 'tensors.pt')
    manifest = {'status': 'PREPARED_ARCHIVAL_TARGET_NOT_NUMERICALLY_QUALIFIED', 'purpose': PURPOSE,
        'target_version': TARGET_VERSION, 'feature_version': FEATURE_VERSION,
        'target_interpretation': 'Historical scalar envelope only; not corrected pointwise irreversible timber capacity.',
        'target_formula': 'np.interp(0:60:3600, native_time, np.minimum.accumulate(charring_ratios)) -> float32',
        'temporal_interpretation': 'Known prescribed scenario generated by archived formulas; not per-case realized amplitude binding.',
        'tensor_path': 'tensors.pt', 'tensor_sha256': digest(output / 'tensors.pt'),
        'feature_tensor_sha256': tensor_collection_digest(graphs),
        'target_tensor_sha256': tensor_collection_digest(targets), 'cases': cases,
        'source_sha256': sources(), 'metadata_sha256': digest(meta_path),
        'prior_full_file_audit_sha256': digest(audit_path),
        'full_temperature_payload_read_this_run': False,
        'current_arrays_read': ['time_steps', 'charring_ratios', 'crack_params', 'beam_dims'],
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    write_json(output / 'manifest.json', manifest)
    split_rows = list(csv.DictReader((SCI / 'audit/dataset/reconstructed_split_membership.csv').open(encoding='utf-8-sig')))
    for protocol in sorted({r['protocol'] for r in split_rows}):
        split = {'status': 'DRAFT_PRETRAINING', 'purpose': PURPOSE, 'protocol': protocol,
                 'dataset_sha256': digest(output / 'manifest.json')}
        for source_role, role in [('train', 'train'), ('val', 'validation'), ('test', 'test')]:
            split[role] = [r['sample_id'] for r in split_rows if r['protocol'] == protocol and r['split'] == source_role]
        check_split(split, manifest)
        fitted = fit_radius_on_training_geometries([dict(c, role='train') for c in cases if c['sample_id'] in set(split['train'])])
        split['topology_recipe'] = {'radius': fitted['normalized_radius'], 'radius_fit_quantile': .5,
            'radius_fit_only_training': True, 'symmetric_knn_k': 3,
            'radius_fit_geometry_count': fitted['unique_training_geometries']}
        write_json(output / 'splits' / (protocol + '.json'), split)
    summary = {'status': manifest['status'], 'cases': len(cases), 'unique_geometries': len(cases),
        'graph_node_shape': '[N,11], N in 1..15', 'edge_shape': '[N*(N-1),5]',
        'global_shape': '[1,3]', 'time_feature_shape': '[61,4]', 'target_shape': '[61]',
        'families': dict(collections.Counter(c['fire_family'] for c in cases)),
        'counts': dict(sorted(collections.Counter(c['num_cracks'] for c in cases).items())),
        'tensor_bytes': (output / 'tensors.pt').stat().st_size,
        'manifest_sha256': digest(output / 'manifest.json'),
        'feature_tensor_sha256': manifest['feature_tensor_sha256'],
        'target_tensor_sha256': manifest['target_tensor_sha256'],
        'training_started': False}
    write_json(output / 'readiness.json', summary)
    print(json.dumps(summary, indent=2))


def check_split(split, manifest):
    cases = {c['sample_id']: c for c in manifest['cases']}
    role_sets = []
    for role in ('train', 'validation', 'test'):
        ids = split[role]
        if not ids or len(ids) != len(set(ids)) or any(s not in cases for s in ids):
            raise ValueError('Invalid split role ' + role)
        role_sets.append(set(ids))
    if set.union(*role_sets) != set(cases):
        raise ValueError('Every case must be explicitly assigned')
    for i, a in enumerate(role_sets):
        for b in role_sets[:i]:
            if a & b or {cases[s]['geometry_id'] for s in a} & {cases[s]['geometry_id'] for s in b}:
                raise ValueError('Case/geometry leakage')
    if split['protocol'] == 'lcro_9_15':
        if any(cases[s]['num_cracks'] > 8 for s in split['train'] + split['validation']):
            raise ValueError('OOD count used for development')
        if any(cases[s]['num_cracks'] < 9 for s in split['test']):
            raise ValueError('Incorrect OOD test count')


def load_dataset(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['purpose'] != PURPOSE or manifest['target_version'] != TARGET_VERSION:
        raise ValueError('Different task/target version')
    tp = manifest_path.parent / manifest['tensor_path']
    if digest(tp) != manifest['tensor_sha256']:
        raise ValueError('Tensor file hash mismatch')
    tensors = torch.load(tp, map_location='cpu', weights_only=True)
    if tensor_collection_digest(tensors['graphs']) != manifest['feature_tensor_sha256']:
        raise ValueError('Input tensor hash mismatch')
    if tensor_collection_digest(tensors['targets']) != manifest['target_tensor_sha256']:
        raise ValueError('Target tensor hash mismatch')
    return manifest, tensors


def configure(seed, threads=4):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def predict(model, graphs, ids, batch_size, device):
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            output.append(model(*collate([graphs[s] for s in ids[start:start + batch_size]], device)).squeeze(-1).cpu())
    return torch.cat(output).numpy()


def metrics(truth, pred):
    e = pred.astype(np.float64) - truth.astype(np.float64)
    if not np.isfinite(e).all():
        raise ValueError('Nonfinite predictions')
    sst = np.square(truth.astype(np.float64) - np.mean(truth, dtype=np.float64)).sum()
    return {'equal_case_MAE': float(np.abs(e).mean()), 'RMSE': float(np.sqrt(np.square(e).mean())),
        'pooled_R2': None if sst == 0 else float(1 - np.square(e).sum() / sst),
        'maximum_overprediction': float(np.maximum(e, 0).max()),
        'maximum_absolute_error': float(np.abs(e).max()),
        'per_case_MAE': np.abs(e).mean(axis=1).tolist(),
        'monotonicity_violation_fraction': float((np.diff(pred, axis=1) > 1e-6).mean())}


def smoke(manifest_path, output):
    output = safe_output(output)
    manifest, tensors = load_dataset(manifest_path)
    graphs, targets = tensors['graphs'], tensors['targets']
    cases = manifest['cases']
    configure(42)
    selected = [next(c['sample_id'] for c in cases if c['num_cracks'] == n) for n in (1, 3, 8, 15)]
    radius = fit_radius_on_training_geometries([dict(c, role='train') for c in cases if c['sample_id'] in selected[:2]])['normalized_radius']
    initial_states, reports = {}, []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for mode in MODES:
        configure(42)
        model = make_model(mode).to(device)
        initial_states[mode] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        batch = collate([graphs[s] for s in selected], device)
        model.train()
        p = model(*batch).squeeze(-1)
        y = torch.stack([targets[s] for s in selected]).to(device)
        torch.nn.functional.mse_loss(p, y).backward()
        if not torch.isfinite(p).all() or any(not torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):
            raise ValueError('Failed forward/backward')
        together = predict(model, graphs, selected, len(selected), device)
        singles = predict(model, graphs, selected, 1, device)
        batch_error = float(np.max(np.abs(together - singles)))
        if batch_error > 3e-6:
            raise ValueError('Batch composition dependence')
        view_checks = []
        for sid in selected:
            c = next(c for c in cases if c['sample_id'] == sid)
            permuted = dict(c, cracks=list(reversed(c['cracks'])))
            for rule in ('complete', 'radius', 'symmetric_knn'):
                g = graph_view(c, rule, radius, 3, graphs[sid][-1])
                gp = graph_view(permuted, rule, radius, 3, graphs[sid][-1])
                pp = predict(model, {'a': g, 'b': gp}, ['a', 'b'], 2, device)
                err = float(np.max(np.abs(pp[0] - pp[1])))
                if err > 3e-6:
                    raise ValueError('Permutation dependence')
                view_checks.append({'sample_id': sid, 'N': c['num_cracks'], 'rule': rule,
                    'edges': g[1].shape[1], 'permutation_max_abs': err})
        reports.append({'mode': mode, 'parameters': sum(p.numel() for p in model.parameters()),
            'batch_composition_max_abs': batch_error, 'topology_checks': view_checks,
            'finite_forward_backward': True})
    equal = all(torch.equal(v, initial_states['gnn_mean'][k]) for k, v in initial_states['gnn'].items())
    if not equal:
        raise ValueError('Sum/mean initialization mismatch')
    # Execution timing only: no optimizer step, no accuracy result.
    big_ids = [c['sample_id'] for c in sorted(cases, key=lambda c: (-c['num_cracks'], c['sample_id']))[:32]]
    timing = []
    for dev in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        configure(42)
        model = make_model('gnn').to(dev)
        b = collate([graphs[s] for s in big_ids], dev)
        y = torch.stack([targets[s] for s in big_ids]).to(dev)
        measured = []
        for j in range(5):
            model.zero_grad(set_to_none=True)
            if dev == 'cuda': torch.cuda.synchronize()
            start = time.perf_counter()
            p = model(*b).squeeze(-1)
            torch.nn.functional.mse_loss(p, y).backward()
            if dev == 'cuda': torch.cuda.synchronize()
            if j: measured.append(time.perf_counter() - start)
        timing.append({'device': dev, 'batch': 32, 'graph_N': 15,
            'forward_backward_median_s': float(np.median(measured)),
            'repeats_after_one_warmup': 4, 'optimizer_updates': 0,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if dev == 'cuda' else None})
    report = {'purpose': 'SOFTWARE_AND_EXECUTION_TIMING_ONLY', 'trained': False,
        'python': sys.executable, 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'threads': torch.get_num_threads(), 'sum_mean_initial_state_equal': equal,
        'models': reports, 'timing': timing, 'source_sha256': sources()}
    write_json(output, report)
    print(json.dumps({'smoke_models': len(reports), 'sum_mean_initial_state_equal': equal, 'timing': timing}, indent=2))


def train(args):
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('status') != 'FROZEN_BEFORE_TRAINING' or plan.get('purpose') != PURPOSE:
        raise ValueError('A frozen archival research plan is required')
    if not plan.get('acknowledge_legacy_target_not_corrected_capacity', False):
        raise ValueError('The target interpretation must be explicit')
    if args.mode not in plan['models'] or args.seed not in plan['seeds'] or args.protocol not in plan['protocols']:
        raise ValueError('Run not in predeclared matrix')
    if plan['source_sha256'] != sources():
        raise ValueError('Research code changed after plan freeze')
    manifest_path = Path(plan['manifest_path'])
    if digest(manifest_path) != plan['manifest_sha256']:
        raise ValueError('Manifest changed')
    manifest, data = load_dataset(manifest_path)
    split_path = Path(plan['protocols'][args.protocol]['split_path'])
    if digest(split_path) != plan['protocols'][args.protocol]['split_sha256']:
        raise ValueError('Split changed')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    check_split(split, manifest)
    if split['dataset_sha256'] != digest(manifest_path):
        raise ValueError('Split binds another dataset')
    config = plan['training_configuration']
    output = safe_output(Path(plan['runs_root']) / args.protocol / args.mode / ('seed_' + str(args.seed)))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Existing run cannot be overwritten; review failure before any new attempt')
    output.mkdir(parents=True, exist_ok=True)
    configure(args.seed, config['threads'])
    model = make_model(args.mode).to(config['device'])
    metadata = {'purpose': PURPOSE, 'target_version': TARGET_VERSION, 'feature_version': FEATURE_VERSION,
        'mode': args.mode, 'seed': args.seed, 'protocol': args.protocol,
        'plan_sha256': digest(plan_path), 'dataset_sha256': digest(manifest_path),
        'split_sha256': digest(split_path), 'source_sha256': sources(),
        'feature_tensor_sha256': manifest['feature_tensor_sha256'],
        'target_tensor_sha256': manifest['target_tensor_sha256'],
        'configuration': config, 'parameters': sum(p.numel() for p in model.parameters()),
        'python': platform.python_version(), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0) if config['device'].startswith('cuda') else None,
        'selection': 'Earliest lowest validation MSE; test first evaluated after restoring best epoch'}
    write_json(output / 'run_metadata.json', metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs_max'])
    rng = np.random.default_rng(args.seed)
    best, stale, best_epoch = math.inf, 0, 0
    rows = []
    start = time.perf_counter()
    graphs, targets = data['graphs'], data['targets']
    for epoch in range(1, config['epochs_max'] + 1):
        model.train()
        ids = list(rng.permutation(split['train']))
        total_loss = 0.
        for k in range(0, len(ids), config['batch_size']):
            group = ids[k:k + config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            p = model(*collate([graphs[s] for s in group], config['device'])).squeeze(-1)
            y = torch.stack([targets[s] for s in group]).to(config['device'])
            loss = torch.nn.functional.mse_loss(p, y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip_norm'])
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            total_loss += float(loss.detach()) * len(group)
        vp = predict(model, graphs, split['validation'], config['batch_size'], config['device'])
        vy = np.stack([targets[s].numpy() for s in split['validation']])
        val_mse = float(np.square(vp - vy).mean())
        if not math.isfinite(val_mse): raise FloatingPointError('Nonfinite validation')
        rows.append({'epoch': epoch, 'train_mse': total_loss / len(ids), 'validation_mse': val_mse,
            'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_s': time.perf_counter() - start})
        scheduler.step()
        if val_mse < best:
            best, stale, best_epoch = val_mse, 0, epoch
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                'best_validation_mse': best, 'dataset_sha256': digest(manifest_path),
                'split_sha256': digest(split_path), 'plan_sha256': digest(plan_path)}, output / 'best.pt')
        else:
            stale += 1
        with (output / 'learning_curve.csv').open('w', newline='', encoding='utf-8') as stream:
            w = csv.DictWriter(stream, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
        write_json(output / 'status.json', {'status': 'RUNNING', 'epoch': epoch,
            'best_epoch': best_epoch, 'elapsed_s': time.perf_counter() - start})
        if stale >= config['patience']: break
    model.load_state_dict(torch.load(output / 'best.pt', map_location=config['device'], weights_only=True)['model_state_dict'])
    training_s = time.perf_counter() - start
    test_ids = split['test']
    truth = np.stack([targets[s].numpy() for s in test_ids])
    prediction = predict(model, graphs, test_ids, config['batch_size'], config['device'])
    np.savez_compressed(output / 'test_predictions.npz', sample_ids=np.asarray(test_ids),
        time_s=data['time_s'].numpy(), truth=truth, predictions=prediction)
    result = metrics(truth, prediction)
    result.update(purpose=PURPOSE, mode=args.mode, seed=args.seed, protocol=args.protocol,
        best_epoch=best_epoch, epochs_run=epoch, best_validation_mse=best,
        training_wall_s=training_s, checkpoint_sha256=digest(output / 'best.pt'),
        predictive_claim_scope='Archive scalar-envelope target only', topology_results={})
    if args.mode in ('gnn', 'gnn_mean', 'gnn_zero_edge_features'):
        recipe = split['topology_recipe']
        by_id = {c['sample_id']: c for c in manifest['cases']}
        checkpoint_hash = digest(output / 'best.pt')
        for rule in ('radius', 'symmetric_knn'):
            changed = {s: graph_view(by_id[s], rule, recipe['radius'], recipe['symmetric_knn_k'], graphs[s][-1]) for s in test_ids}
            p = predict(model, changed, test_ids, config['batch_size'], config['device'])
            np.savez_compressed(output / ('topology_' + rule + '.npz'), sample_ids=np.asarray(test_ids),
                time_s=data['time_s'].numpy(), truth=truth, predictions=p)
            result['topology_results'][rule] = dict(metrics(truth, p),
                same_trained_checkpoint=True, checkpoint_sha256=checkpoint_hash,
                refitting=False, total_edges=sum(g[1].shape[1] for g in changed.values()),
                mean_abs_prediction_change=float(np.abs(p - prediction).mean()))
    write_json(output / 'result.json', result)
    write_json(output / 'status.json', {'status': 'COMPLETE_ARCHIVAL_RESEARCH',
        'best_epoch': best_epoch, 'epochs_run': epoch, 'training_wall_s': training_s})
    print(json.dumps({k: result[k] for k in ('mode', 'seed', 'protocol', 'equal_case_MAE', 'RMSE', 'pooled_R2', 'training_wall_s')}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--output', default=str(HERE / 'archive911'))
    p = sub.add_parser('smoke'); p.add_argument('--manifest', default=str(HERE / 'archive911/manifest.json'))
    p.add_argument('--output', default=str(HERE / 'software_smoke.json'))
    p = sub.add_parser('train'); p.add_argument('--plan', required=True)
    p.add_argument('--mode', choices=MODES, required=True); p.add_argument('--seed', type=int, required=True)
    p.add_argument('--protocol', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.output)
    elif args.command == 'smoke': smoke(args.manifest, args.output)
    else: train(args)


if __name__ == '__main__':
    main()
