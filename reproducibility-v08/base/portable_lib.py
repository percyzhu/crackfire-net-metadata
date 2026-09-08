"""Package-relative v08 runtime support; original historical paths are never used."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np

BASE = Path(__file__).resolve().parent
if os.name == 'nt' and not str(BASE).startswith('\\\\?\\'):
    value = str(BASE)
    BASE = Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)
RESEARCH = BASE / 'SCI_Manuscript/experiments/research_v08'
sys.dont_write_bytecode = True


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def local(relative):
    path = (BASE / relative).resolve()
    if not path.is_relative_to(BASE): raise ValueError('Package path escapes its own root')
    return path


def verify_files():
    package = read(BASE / 'package_manifest.json')
    for relative, item in package['files'].items():
        path = local(relative)
        assert path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], relative
    return package


def initialize():
    package = verify_files()
    sys.path.insert(0, str(RESEARCH))
    import research_runner as rr
    assert rr.WORKSPACE.resolve() == BASE
    original = RESEARCH / 'comparison_plan_v08_420.json'; plan = read(original)
    mapping = read(BASE / 'relocation_map.json')
    assert sha(original) == mapping['original_plan_sha256'] == package['original_plan_sha256']
    assert mapping['original_plan_modified'] is False and len(mapping['entries']) == 14
    entries = {x['field']: x for x in mapping['entries']}; assert len(entries) == 14
    expected = {'manifest_path', 'runs_root'} | {'protocols.' + p + '.split_path' for p in plan['protocols']}
    assert set(entries) == expected
    paths = {}
    for field, item in entries.items():
        source = plan[field] if field in ('manifest_path', 'runs_root') else plan['protocols'][field.split('.')[1]]['split_path']
        assert source == item['original']; paths[field] = local(item['relative'])
    assert rr.sources() == plan['source_sha256']
    assert sha(paths['manifest_path']) == plan['manifest_sha256']
    manifest, tensors = rr.load_dataset(paths['manifest_path'])
    splits = {}
    for protocol, info in plan['protocols'].items():
        path = paths['protocols.' + protocol + '.split_path']
        assert sha(path) == info['split_sha256']; split = read(path); rr.check_split(split, manifest)
        assert split['dataset_sha256'] == plan['manifest_sha256'] and split['protocol'] == protocol
        splits[protocol] = split
    registry = read(BASE / 'checkpoint_registry.json')
    assert len({(r['protocol'], r['model'], r['seed']) for r in registry}) == len(registry) == package['checkpoints']
    for r in registry:
        assert local(r['relative_directory']) == paths['runs_root'] / r['protocol'] / r['model'] / ('seed_' + str(r['seed']))
    return package, rr, plan, manifest, tensors, splits, registry


def check_run(rr, plan, manifest, record):
    import torch
    folder = local(record['relative_directory']); meta = read(folder / 'run_metadata.json'); result = read(folder / 'result.json')
    for key, value in [('protocol', record['protocol']), ('mode', record['model']), ('seed', record['seed'])]:
        assert meta[key] == result[key] == value
    assert meta['plan_sha256'] == sha(RESEARCH / 'comparison_plan_v08_420.json')
    assert meta['dataset_sha256'] == plan['manifest_sha256'] and meta['source_sha256'] == plan['source_sha256']
    assert meta['split_sha256'] == plan['protocols'][record['protocol']]['split_sha256']
    assert meta['configuration'] == plan['training_configuration']
    for key in ('feature_tensor_sha256', 'target_tensor_sha256', 'target_version', 'feature_version'): assert meta[key] == manifest[key]
    assert sha(folder / 'best.pt') == result['checkpoint_sha256'] == record['checkpoint_sha256']
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    for key in ('plan_sha256', 'dataset_sha256', 'split_sha256'): assert checkpoint[key] == meta[key]
    return folder, meta, result, checkpoint


def saved_array(folder, rule, split, tensors):
    filename = 'test_predictions.npz' if rule == 'complete' else 'topology_' + rule + '.npz'
    truth = np.stack([tensors['targets'][sid].numpy() for sid in split['test']])
    with np.load(folder / filename, allow_pickle=False) as z:
        assert z['sample_ids'].tolist() == split['test'] and np.array_equal(z['time_s'], tensors['time_s'].numpy())
        assert np.array_equal(z['truth'], truth)
        pred = z['predictions'].copy()
    assert pred.shape == truth.shape and np.isfinite(pred).all()
    return truth, pred


def write_report(path, value):
    path = Path(path).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
