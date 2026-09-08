"""Read-only v08 binding and CPU replay audit; no model comparison or training."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import research_runner as rr


def main():
    p = argparse.ArgumentParser(); p.add_argument('--limit', type=int, default=5)
    args = p.parse_args()
    plan_path = rr.HERE / 'comparison_plan_v08_420.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    manifest, tensors = rr.load_dataset(plan['manifest_path'])
    rr.configure(42, 4)
    results = []
    for file in sorted(Path(plan['runs_root']).glob('*/*/*/result.json'))[:args.limit]:
        out = file.parent
        r = json.loads(file.read_text(encoding='utf-8'))
        meta = json.loads((out / 'run_metadata.json').read_text(encoding='utf-8'))
        split_path = Path(plan['protocols'][r['protocol']]['split_path'])
        split = json.loads(split_path.read_text(encoding='utf-8'))
        assert meta['plan_sha256'] == rr.digest(plan_path)
        assert meta['dataset_sha256'] == rr.digest(plan['manifest_path'])
        assert meta['split_sha256'] == rr.digest(split_path)
        assert meta['source_sha256'] == plan['source_sha256'] == rr.sources()
        assert r['checkpoint_sha256'] == rr.digest(out / 'best.pt')
        log = list(csv.DictReader((out / 'learning_curve.csv').open(encoding='utf-8')))
        values = [float(row['validation_mse']) for row in log]
        best_epoch = 1 + int(np.argmin(values))
        assert best_epoch == r['best_epoch']
        assert [int(row['epoch']) for row in log] == list(range(1, len(log) + 1))
        assert len(log) <= plan['training_configuration']['epochs_max']
        state = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
        assert state['epoch'] == best_epoch
        model = rr.make_model(r['mode'])
        model.load_state_dict(state['model_state_dict'], strict=True)
        assert sum(v.numel() for v in model.parameters()) == meta['parameters']
        with np.load(out / 'test_predictions.npz', allow_pickle=False) as z:
            assert z['sample_ids'].tolist() == split['test']
            assert np.array_equal(z['truth'], np.stack([tensors['targets'][s].numpy() for s in split['test']]))
            assert np.array_equal(z['time_s'], tensors['time_s'].numpy())
            gpu_pred = z['predictions'].copy()
            truth = z['truth'].copy()
        replay = rr.predict(model, tensors['graphs'], split['test'], 32, 'cpu')
        error = float(np.max(np.abs(replay - gpu_pred)))
        assert error < 3e-6, error
        metric = rr.metrics(truth, gpu_pred)
        for key in ('equal_case_MAE', 'RMSE', 'pooled_R2', 'maximum_overprediction', 'maximum_absolute_error'):
            assert metric[key] == r[key]
        results.append({'run': str(out.relative_to(rr.HERE)).replace('\\', '/'), 'status': 'PASSED',
            'best_epoch_is_earliest_minimum_validation': True, 'truth_ids_times_exact': True,
            'source_dataset_split_checkpoint_hashes_bound': True,
            'all_test_cases_cpu_replay_max_abs_vs_cuda_saved': error,
            'test_case_count': len(split['test']), 'software_tolerance': 3e-6})
    rr.write_json(rr.HERE / 'completed_run_audit.json', {'purpose': 'READ_ONLY_BINDING_AND_REPLAY_NOT_FULL_PEER_REVIEW',
        'audited_runs': results, 'audited_count': len(results), 'no_optimizer_updates': True})
    print(json.dumps(results, indent=2))


if __name__ == '__main__': main()
