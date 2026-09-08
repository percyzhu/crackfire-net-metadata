"""Replay a packaged checkpoint from package-relative inputs; never retrain."""
import argparse
import os
import sys
import numpy as np
import torch
import portable_lib as p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default='iid997'); parser.add_argument('--model', default='gnn')
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--rules', nargs='+', default=['complete', 'radius', 'symmetric_knn'])
    parser.add_argument('--output', default=str(p.BASE / 'verification/portable_inference.json'))
    args = parser.parse_args()
    package, rr, plan, manifest, tensors, splits, registry = p.initialize()
    matches = [r for r in registry if (r['protocol'], r['model'], r['seed']) == (args.protocol, args.model, args.seed)]
    assert len(matches) == 1; record = matches[0]
    assert set(args.rules) <= set(record['graph_rules']) and len(set(args.rules)) == len(args.rules)
    folder, meta, result, checkpoint = p.check_run(rr, plan, manifest, record)
    rr.configure(args.seed, 4)
    if args.device == 'cuda' and not torch.cuda.is_available(): raise ValueError('Requested CUDA is unavailable')
    model = rr.make_model(args.model).to(args.device); model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    assert sum(x.numel() for x in model.parameters()) == meta['parameters']
    split = splits[args.protocol]; by_id = {c['sample_id']: c for c in manifest['cases']}; checks = []
    for rule in args.rules:
        _, saved = p.saved_array(folder, rule, split, tensors)
        recipe = split['topology_recipe']
        graphs = tensors['graphs'] if rule == 'complete' else {
            sid: rr.graph_view(by_id[sid], rule, recipe['radius'], recipe['symmetric_knn_k'], tensors['graphs'][sid][-1])
            for sid in split['test']}
        if args.device == 'cuda':
            with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=True):
                actual = rr.predict(model, graphs, split['test'], 32, 'cuda')
        else: actual = rr.predict(model, graphs, split['test'], 32, 'cpu')
        difference = float(np.max(np.abs(actual.astype(float) - saved.astype(float))))
        passed = difference < 3e-6 if args.device == 'cpu' else bool(np.array_equal(actual, saved))
        checks.append({'graph_rule': rule, 'cases': len(saved), 'times': 61, 'maximum_absolute_difference': difference,
                       'CPU_within_original_3e_6': difference < 3e-6 if args.device == 'cpu' else None,
                       'GPU_bitwise_equal': bool(np.array_equal(actual, saved)) if args.device == 'cuda' else None,
                       'requested_backend_check_passed': passed, 'checkpoint_sha256': record['checkpoint_sha256']})
    passed = all(x['requested_backend_check_passed'] for x in checks)
    report = {'status': 'PASSED_SELECTED_PORTABLE_REPLAY' if passed else 'BACKEND_DIFFERENCE_RECORDED_NOT_A_REPLAY_PASS',
              'package_scope': package['status'], 'source_root_is_package': rr.WORKSPACE.resolve() == p.BASE,
              'working_directory': os.getcwd(), 'package_root': str(p.BASE), 'protocol': args.protocol,
              'model': args.model, 'seed': args.seed, 'device': args.device, 'checks': checks,
              'optimizer_updates': 0, 'all420_inference_validated': False, 'CPU_tolerance_unchanged': 3e-6,
              'meaning': 'Software replay difference, not prediction error relative to physical experiments.',
              'versions': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__}}
    p.write_report(args.output, report); print(report['status']); return 0 if passed else 2


if __name__ == '__main__': sys.exit(main())
