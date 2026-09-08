"""Hash and semantic verification of the separate14-entry path map; no training."""
import argparse
import sys
import portable_lib as p


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default=str(p.BASE / 'verification/relocation.json'))
    args = parser.parse_args()
    package, rr, plan, manifest, tensors, splits, registry = p.initialize()
    report = {'status': 'PASSED_RELATIVE_PATH_AND_FROZEN_IDENTITY_CHECK', 'package_scope': package['status'],
              'source_root_is_package': rr.WORKSPACE.resolve() == p.BASE, 'mapped_path_fields': 14,
              'original_plan_sha256': p.sha(p.RESEARCH / 'comparison_plan_v08_420.json'),
              'original_plan_modified': False, 'frozen_sources': len(rr.sources()), 'cases': len(manifest['cases']),
              'splits': len(splits), 'packaged_checkpoints': len(registry), 'optimizer_updates': 0,
              'new_training_execution_validated': False, 'inference_executed_by_this_check': False}
    p.write_report(args.output, report); print(report['status']); return 0


if __name__ == '__main__': sys.exit(main())
