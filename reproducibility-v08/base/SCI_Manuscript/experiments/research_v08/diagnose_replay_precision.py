"""Separate numerical backend precision from checkpoint/artifact replay identity."""
import json
from pathlib import Path
import numpy as np
import torch
import research_runner as rr


def differences(reference, values):
    diff = values.astype(float) - reference.astype(float)
    return {'max_abs': float(np.abs(diff).max()), 'mean_abs': float(np.abs(diff).mean()),
        'rms': float(np.sqrt(np.square(diff).mean())), 'abs_q50_q90_q99_q999': np.quantile(np.abs(diff), [.5, .9, .99, .999]).tolist(),
        'exceeds_original_3e6': int(np.count_nonzero(np.abs(diff) >= 3e-6)),
        'bitwise_equal': bool(np.array_equal(values, reference)), 'dtype': str(values.dtype)}


def main():
    folder = rr.HERE / 'replay_precision_diagnostic'; folder.mkdir(exist_ok=True)
    plan = json.loads((rr.HERE / 'comparison_plan_v08_420.json').read_text(encoding='utf-8'))
    manifest, data = rr.load_dataset(plan['manifest_path'])
    protocol = 'lcro_9_15'
    split = json.loads(Path(plan['protocols'][protocol]['split_path']).read_text(encoding='utf-8'))
    cell = Path(plan['runs_root']) / protocol / 'fire_only/seed_42'
    meta = json.loads((cell / 'run_metadata.json').read_text(encoding='utf-8'))
    state = torch.load(cell / 'best.pt', map_location='cpu', weights_only=True)
    with np.load(cell / 'test_predictions.npz') as z:
        saved = z['predictions'].copy(); ids = z['sample_ids'].tolist(); time_s = z['time_s'].copy()
    assert ids == split['test']
    rr.configure(42, 4)
    model = rr.make_model('fire_only'); model.load_state_dict(state['model_state_dict'])
    cpu = rr.predict(model, data['graphs'], ids, 32, 'cpu')
    gpu = rr.predict(model.to('cuda'), data['graphs'], ids, 32, 'cuda')
    gpu_again = rr.predict(model, data['graphs'], ids, 32, 'cuda')
    model = model.cpu()
    with torch.backends.mkldnn.flags(enabled=False):
        cpu_no_mkldnn = rr.predict(model, data['graphs'], ids, 32, 'cpu')
    # Optional high precision reference for backend discrepancy only, not model replacement.
    model.double()
    graphs64 = {s: tuple(v if v.dtype == torch.long else v.double() for v in data['graphs'][s]) for s in ids}
    cpu64 = rr.predict(model, graphs64, ids, 32, 'cpu')
    worst = np.unravel_index(np.abs(cpu.astype(float) - saved.astype(float)).argmax(), saved.shape)
    case = next(c for c in manifest['cases'] if c['sample_id'] == ids[worst[0]])
    report = {'purpose': 'BACKEND_PRECISION_DIAGNOSTIC_NO_OPTIMIZER_OR_TARGET_CHANGES',
        'cell': str(cell), 'checkpoint_sha256': rr.digest(cell / 'best.pt'),
        'predictions_sha256': rr.digest(cell / 'test_predictions.npz'),
        'source_hashes_match_frozen': meta['source_sha256'] == plan['source_sha256'] == rr.sources(),
        'saved_prediction_dtype': str(saved.dtype), 'weights_original_dtype': str(next(iter(state['model_state_dict'].values())).dtype),
        'CPU32_vs_savedGPU32': differences(saved, cpu), 'GPU32_same_backend_vs_savedGPU32': differences(saved, gpu),
        'GPU32_repeated_vs_first': differences(gpu, gpu_again),
        'CPU32_noMKLDNN_vs_savedGPU32': differences(saved, cpu_no_mkldnn),
        'CPU64_vs_savedGPU32': differences(saved, cpu64), 'CPU32_vs_CPU64': differences(cpu64, cpu),
        'worst': {'sample_id': ids[worst[0]], 'family': case['fire_family'], 'N': case['num_cracks'],
            'time_s': float(time_s[worst[1]]), 'saved_gpu': float(saved[worst]), 'cpu32': float(cpu[worst]),
            'gpu_replay': float(gpu[worst]), 'cpu64': float(cpu64[worst])},
        'config': {'torch': torch.__version__, 'threads': torch.get_num_threads(), 'deterministic': torch.are_deterministic_algorithms_enabled(),
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'MKLDNN': torch.backends.mkldnn.enabled}, 'optimizer_updates': 0}
    rr.write_json(folder / 'diagnosis.json', report)
    np.savez_compressed(folder / 'backend_comparison.npz', saved=saved, cpu32=cpu, gpu32=gpu, cpu_no_mkldnn=cpu_no_mkldnn,
                        cpu64=cpu64, sample_ids=np.asarray(ids), time_s=time_s)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
