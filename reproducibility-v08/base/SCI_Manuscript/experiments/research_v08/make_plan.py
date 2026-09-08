"""Create a draft, or explicitly freeze an outcome-independent v08 run matrix."""
import argparse
import json
import time
from pathlib import Path
import research_runner as rr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default=str(rr.HERE / 'archive997/manifest.json'))
    p.add_argument('--output', default=str(rr.HERE / 'comparison_plan_draft.json'))
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--protocols', nargs='+', default=['iid997', 'lcro_9_15'])
    p.add_argument('--models', nargs='+', choices=rr.MODES, default=rr.MODES)
    p.add_argument('--runs-root', default=str(rr.HERE / 'runs_primary'))
    args = p.parse_args()
    output = rr.safe_output(args.output)
    if output.exists(): raise ValueError('Plans are exclusive-create artifacts')
    manifest_path = Path(args.manifest).resolve()
    manifest, tensors = rr.load_dataset(manifest_path)
    protocols = {}
    for name in args.protocols:
        split_path = manifest_path.parent / 'splits' / (name + '.json')
        split = json.loads(split_path.read_text(encoding='utf-8'))
        rr.check_split(split, manifest)
        protocols[name] = {'split_path': str(split_path), 'split_sha256': rr.digest(split_path),
                           'counts': {r: len(split[r]) for r in ('train', 'validation', 'test')}}
    plan = {'status': 'FROZEN_BEFORE_TRAINING' if args.freeze else 'DRAFT_NOT_AUTHORIZATION_TO_TRAIN',
        'purpose': rr.PURPOSE, 'acknowledge_legacy_target_not_corrected_capacity': True,
        'scientific_scope': 'Emulate an explicitly defined historical numerical scalar-envelope reference; no claim of independent physical-capacity qualification.',
        'target_version': rr.TARGET_VERSION, 'feature_version': rr.FEATURE_VERSION,
        'manifest_path': str(manifest_path), 'manifest_sha256': rr.digest(manifest_path),
        'protocols': protocols, 'models': args.models, 'seeds': [42, 43, 44, 45, 46],
        'runs_root': str(rr.safe_output(args.runs_root)), 'source_sha256': rr.sources(),
        'training_configuration': {'device': 'cuda', 'threads': 4, 'epochs_max': 300,
            'batch_size': 32, 'patience': 50, 'learning_rate': .001, 'weight_decay': .00001,
            'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingLR(T_max=epochs_max)',
            'grad_clip_norm': 1., 'loss': 'equal_case_time_mean_squared_error',
            'validation_selection': 'Earliest minimum validation MSE', 'deterministic_algorithms': True},
        'tuning_budget': {'trials_per_model_protocol': 1, 'parameter_search': 'None; architecture and training settings fixed before outcomes.',
            'shared_training_budget': True, 'test_based_tuning': False},
        'primary_metric': 'equal_case_MAE', 'primary_contrast': ['capacity_matched_deepsets', 'gnn'],
        'secondary_contrast': ['gnn', 'gnn_mean'],
        'claim_rule': 'Report paired effect and uncertainty whether favorable, inconclusive or unfavorable; no post-hoc pass threshold.',
        'statistics': 'Five matched seeds, per-case errors, geometry-cluster/seed paired bootstrap; no timepoint pseudoreplication.',
        'topology': {'training_rule': 'complete directed without loops',
            'same_checkpoint_test_views': ['complete', 'radius', 'symmetric_knn'],
            'sparse_view_refit': False, 'radius_fit': 'training geometries only; median within-geometry distance median',
            'knn_k': 3, 'interpretation': 'Representation-connectivity stress test distinct from new physical geometry or FE mesh transfer.'},
        'legacy_subset_policy': 'Report retained911/restored86 only as prespecified strata within the997 test split; not extra independent datasets.',
        'publication_interpretation': 'Software smoke establishes no accuracy. New predictions must carry this target/feature version. Neither FE convergence nor journal acceptance is inferred.',
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    rr.write_json(output, plan)
    print(json.dumps({'path': str(output), 'status': plan['status'],
        'planned_runs': len(args.models) * len(args.protocols) * 5, 'sha256': rr.digest(output)}, indent=2))


if __name__ == '__main__': main()
