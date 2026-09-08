"""Package the small frozen benchmark and one predetermined checkpoint locally."""
from pathlib import Path
import datetime
import hashlib
import json
import platform
import shutil
import sys
import zipfile
import numpy as np
import torch
import research_runner as rr


PORTABLE_INFER = r'''"""CPU-only portable replay of the first predeclared IID GNN seed (42)."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
RESEARCH = BASE / 'SCI_Manuscript/experiments/research_v08'
sys.path.insert(0, str(RESEARCH))
import research_runner as rr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(BASE / 'portable_verification.json'))
    args = parser.parse_args()
    # Resolve kit paths from this script, never from cwd or the original D drive.
    original_plan = RESEARCH / 'comparison_plan_v08_420.json'
    plan = json.loads(original_plan.read_text(encoding='utf-8'))
    if plan['source_sha256'] != rr.sources(): raise ValueError('Bundled frozen source mismatch')
    manifest_path = RESEARCH / 'archive997/manifest.json'
    if rr.digest(manifest_path) != plan['manifest_sha256']: raise ValueError('Dataset manifest mismatch')
    manifest, tensors = rr.load_dataset(manifest_path)
    selection = json.loads((BASE / 'selected_checkpoint.json').read_text(encoding='utf-8'))
    folder = BASE / selection['relative_directory']
    meta = json.loads((folder / 'run_metadata.json').read_text(encoding='utf-8'))
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    if meta['plan_sha256'] != rr.digest(original_plan): raise ValueError('Checkpoint plan mismatch')
    if meta['source_sha256'] != plan['source_sha256']: raise ValueError('Checkpoint source mismatch')
    if meta['dataset_sha256'] != rr.digest(manifest_path): raise ValueError('Checkpoint dataset mismatch')
    for field in ('feature_tensor_sha256', 'target_tensor_sha256'):
        if meta[field] != manifest[field]: raise ValueError('Checkpoint tensor mismatch')
    protocol = selection['protocol']
    split_path = RESEARCH / 'archive997/splits' / (protocol + '.json')
    if rr.digest(split_path) != plan['protocols'][protocol]['split_sha256'] or rr.digest(split_path) != meta['split_sha256']:
        raise ValueError('Split mismatch')
    split = json.loads(split_path.read_text(encoding='utf-8'))
    rr.check_split(split, manifest)
    if result['checkpoint_sha256'] != rr.digest(folder / 'best.pt'): raise ValueError('Weight mismatch')
    rr.configure(42, 4)
    model = rr.make_model(selection['model'])
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if sum(p.numel() for p in model.parameters()) != meta['parameters']: raise ValueError('Parameter count mismatch')
    expected_y = np.stack([tensors['targets'][sid].numpy() for sid in split['test']])
    by_id = {c['sample_id']: c for c in manifest['cases']}
    checks = []
    for rule in ('complete', 'radius', 'symmetric_knn'):
        filename = 'test_predictions.npz' if rule == 'complete' else 'topology_' + rule + '.npz'
        with np.load(folder / filename, allow_pickle=False) as z:
            if z['sample_ids'].tolist() != split['test']: raise ValueError('Ordered IDs mismatch')
            if not np.array_equal(z['truth'], expected_y): raise ValueError('Truth mismatch')
            if not np.array_equal(z['time_s'], tensors['time_s'].numpy()): raise ValueError('Times mismatch')
            saved = z['predictions'].copy()
        recipe = split['topology_recipe']
        graphs = tensors['graphs'] if rule == 'complete' else {
            sid: rr.graph_view(by_id[sid], rule, recipe['radius'], recipe['symmetric_knn_k'], tensors['graphs'][sid][-1])
            for sid in split['test']}
        prediction = rr.predict(model, graphs, split['test'], 32, 'cpu')
        difference = float(np.max(np.abs(prediction - saved)))
        if difference > 3e-6: raise ValueError('Portable CPU replay failed')
        checks.append({'graph_rule': rule, 'cases': len(split['test']), 'times_per_case': expected_y.shape[1],
            'maximum_absolute_difference_vs_original_GPU_predictions': difference,
            'same_checkpoint_sha256': result['checkpoint_sha256']})
    report = {'status': 'PASSED_PORTABLE_CPU_REPLAY', 'purpose': 'Local source-backed reproducibility check; not a complete model comparison.',
        'working_directory_at_execution': os.getcwd(), 'kit_root': str(BASE),
        'source_root_is_kit': rr.WORKSPACE.resolve() == BASE,
        'original_plan_sha256': rr.digest(original_plan), 'selected_model': selection['model'],
        'selected_seed': selection['seed'], 'selection_rule': selection['selection_rule'],
        'source_hash_count': len(rr.sources()), 'model_parameters': meta['parameters'],
        'device': 'cpu', 'GPU_model_execution': False, 'optimizer_updates': 0,
        'tolerance': 3e-6, 'checks': checks, 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__}
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'source_root_is_kit': report['source_root_is_kit'], 'checks': checks}, indent=2))

if __name__ == '__main__': main()
'''

PORTABLE_TRAIN_DRY_RUN = r'''"""Resolve relative training paths and validate semantics; never train."""
from pathlib import Path
import json
import os
import sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'SCI_Manuscript/experiments/research_v08'))
import research_runner as rr
os.chdir(BASE)
path=BASE/'relocated_training_plan.json'
plan=json.loads(path.read_text(encoding='utf-8'))
original=json.loads((BASE/'SCI_Manuscript/experiments/research_v08/comparison_plan_v08_420.json').read_text(encoding='utf-8'))
changes=[]
for key in ('manifest_path','runs_root'):
    changes.append({'field':key,'original':original[key],'relocated':plan[key]})
    original[key]=plan[key]
for protocol in original['protocols']:
    changes.append({'field':'protocols.'+protocol+'.split_path','original':original['protocols'][protocol]['split_path'],'relocated':plan['protocols'][protocol]['split_path']})
    original['protocols'][protocol]['split_path']=plan['protocols'][protocol]['split_path']
if original!=plan:raise ValueError('Relocated plan differs beyond path fields')
if plan['source_sha256']!=rr.sources():raise ValueError('Frozen source differs')
if rr.digest(plan['manifest_path'])!=plan['manifest_sha256']:raise ValueError('Manifest differs')
manifest,tensors=rr.load_dataset(plan['manifest_path'])
for protocol,record in plan['protocols'].items():
    if rr.digest(record['split_path'])!=record['split_sha256']:raise ValueError('Split differs')
    rr.check_split(json.loads(Path(record['split_path']).read_text(encoding='utf-8')),manifest)
rr.safe_output(plan['runs_root'])
report={'status':'PASSED_RELOCATION_SEMANTIC_DRY_RUN','optimizer_updates':0,
        'training_execution_verified':False,'path_only_changes':changes,
        'relative_path_root':str(BASE),'derived_plan_sha256':rr.digest(path)}
(BASE/'relocation_dry_run.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'status':report['status'],'path_fields_changed':len(changes),'training_execution_verified':False},indent=2))
'''


def main():
    kit = rr.SCI / 'deliverables/EAAI_reproducibility_v08_working'
    if kit.exists() and any(kit.iterdir()): raise ValueError('Refuse to overwrite working kit')
    kit.mkdir(parents=True, exist_ok=True)
    plan_path = rr.HERE / 'comparison_plan_v08_420.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    copies = list(plan['source_sha256'])
    copies += [str(p.relative_to(rr.WORKSPACE)).replace('\\', '/') for p in (rr.HERE / 'archive997').rglob('*') if p.is_file()]
    copies += ['SCI_Manuscript/experiments/research_v08/comparison_plan_v08_420.json',
        'SCI_Manuscript/experiments/research_v08/aggregate_results.py',
        'SCI_Manuscript/experiments/evaluation/evaluation_stats.py',
        'SCI_Manuscript/experiments/research_v08/verify_quantile_semantics.py',
        'SCI_Manuscript/experiments/research_v08/io_recovery_patch1.py',
        'SCI_Manuscript/experiments/research_v08/recovery_io_patch1/amendment.json',
        'SCI_Manuscript/experiments/research_v08/recovery_io_patch1/replayed_prefix_check.json',
        'SCI_Manuscript/experiments/research_v08/design_audit/design_limitations.json',
        'SCI_Manuscript/experiments/research_v08/design_audit/split_composition.json',
        'SCI_Manuscript/experiments/research_v08/design_audit/population_batch_family_count.csv',
        'SCI_Manuscript/experiments/research_v08/design_audit/split_batch_family_count.csv']
    original_run = rr.HERE / 'runs_frozen_v08_420/iid997/gnn/seed_42'
    if not (original_run / 'result.json').exists(): raise ValueError('Predetermined checkpoint not completed')
    copies += [str(p.relative_to(rr.WORKSPACE)).replace('\\', '/') for p in original_run.iterdir() if p.is_file() and p.suffix in ('.json', '.pt', '.npz', '.csv')]
    records = []
    for relative in sorted(set(copies)):
        source = rr.WORKSPACE / relative
        destination = kit / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if rr.digest(source) != rr.digest(destination): raise ValueError('Copy checksum mismatch')
        records.append({'relative_path': relative, 'sha256': rr.digest(source), 'bytes': source.stat().st_size,
                        'copied_without_changes': True})
    selection = {'protocol': 'iid997', 'model': 'gnn', 'seed': 42,
        'relative_directory': str(original_run.relative_to(rr.WORKSPACE)).replace('\\', '/'),
        'selection_rule': 'First prespecified GNN seed42 under IID; selected for portability, not by test performance.',
        'complete_matrix_included': False, 'checkpoint_count': 1,
        'future_full_matrix': 'All420 checkpoints may be packaged after completion; no selection by favorable outcomes.'}
    (kit / 'selected_checkpoint.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    relocated = json.loads(json.dumps(plan))
    prefix = 'SCI_Manuscript/experiments/research_v08/'
    relocated['manifest_path'] = prefix + 'archive997/manifest.json'
    relocated['runs_root'] = prefix + 'reproduction_runs'
    for protocol in relocated['protocols']:
        relocated['protocols'][protocol]['split_path'] = prefix + 'archive997/splits/' + protocol + '.json'
    (kit / 'relocated_training_plan.json').write_text(json.dumps(relocated, indent=2, ensure_ascii=False), encoding='utf-8')
    (kit / 'portable_infer.py').write_text(PORTABLE_INFER, encoding='utf-8')
    (kit / 'verify_relocation.py').write_text(PORTABLE_TRAIN_DRY_RUN, encoding='utf-8')
    environment = {'recorded_local_environment': {'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'platform': platform.platform()},
        'inference_requirements': ['Python3.12', 'PyTorch2.5.1', 'NumPy compatible with this PyTorch release'],
        'not_required_for_portable_inference': ['Abaqus', 'PyG', 'CUDA GPU', 'original D drive files'],
        'CPU_execution_tested': True, 'dependencies_bundled': False,
        'license': 'No new license granted or public release authorization inferred; local working research package.'}
    (kit / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    (kit / 'requirements_inference.txt').write_text(f'numpy=={np.__version__}\ntorch==2.5.1\n', encoding='utf-8')
    manifest = {'status': 'LOCAL_WORKING_REPRODUCIBILITY_KIT_NOT_PUBLIC_RELEASE',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original_plan_sha256': rr.digest(plan_path),
        'derived_relative_plan_sha256': rr.digest(kit / 'relocated_training_plan.json'),
        'derived_plan_changes': 'Only manifest_path, runs_root, and twelve protocol split_path fields; all science/data/config/source hashes unchanged.',
        'case_count': 997, 'complete_geometry_count': 997, 'selected_checkpoint': selection,
        'includes_large_FE_files': False, 'raw_FE_reconstruction_reproducible_from_this_kit': False,
        'inference_portability_test_pending_at_build': True, 'training_relocation_execution_tested': False,
        'copied_files': records}
    (kit / 'kit_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'kit': str(kit), 'copied_files': len(records), 'bytes': sum(r['bytes'] for r in records)}, indent=2))


if __name__ == '__main__': main()
