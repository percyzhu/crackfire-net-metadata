"""Small read-only tensor-manifest audit and split composition tables."""
import collections
import csv
import json
from pathlib import Path
import research_runner as rr


def main():
    folder = rr.HERE / 'archive997'
    m = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    plan = json.loads((rr.HERE / 'comparison_plan_v08_420.json').read_text(encoding='utf-8'))
    by_id = {c['sample_id']: c for c in m['cases']}
    output = rr.HERE / 'design_audit'; output.mkdir(exist_ok=True)
    population = collections.Counter((c['source_batch'], c['fire_family'], c['num_cracks'],
                                     c['flat_response'], bool(c['legacy_final_id'])) for c in m['cases'])
    keys = ['source_batch', 'fire_family', 'crack_count', 'flat_response', 'in_legacy911', 'case_count']
    with (output / 'population_batch_family_count.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(keys)
        for group, n in sorted(population.items()): w.writerow([*group, n])
    rows, summary = [], {}
    for name, source in plan['protocols'].items():
        s = json.loads(Path(source['split_path']).read_text(encoding='utf-8'))
        summary[name] = {}
        for role in ('train', 'validation', 'test'):
            cases = [by_id[sid] for sid in s[role]]
            counts = collections.Counter((c['source_batch'], c['fire_family'], c['num_cracks'],
                                          c['flat_response'], bool(c['legacy_final_id'])) for c in cases)
            for group, n in sorted(counts.items()): rows.append([name, role, *group, n])
            summary[name][role] = {'cases': len(cases),
                'by_batch': dict(collections.Counter(c['source_batch'] for c in cases)),
                'flat_cases': sum(c['flat_response'] for c in cases),
                'legacy911_cases': sum(bool(c['legacy_final_id']) for c in cases),
                'crack_counts': sorted({c['num_cracks'] for c in cases})}
    with (output / 'split_batch_family_count.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['protocol', 'role', *keys]); w.writerows(rows)
    rr.write_json(output / 'split_composition.json', summary)
    rr.write_json(output / 'plan_registry.json', {
        'active_plan': 'comparison_plan_v08_420.json', 'active_plan_sha256': rr.digest(rr.HERE / 'comparison_plan_v08_420.json'),
        'superseded_unexecuted': [{'path': 'comparison_plan_v08.json',
            'reason': 'Ten-family population requires smoldering LOCO too; corrected before first optimizer step.',
            'optimizer_updates': 0}], 'draft': 'comparison_plan_draft.json'})
    rr.write_json(output / 'design_limitations.json', {
        'cases': 997, 'exact_geometry_groups': len({c['geometry_id'] for c in m['cases']}),
        'provenance': 'Source batch/crack-count/fire-family strata fixed before outcomes.',
        'count_ood_confounding': 'All test cases N9–15 are batch005; batch004 contains N1–8 only. Evaluate batch005 subset within development/IID strata; count OOD is not isolated from source-batch effects.',
        'legacy_subset': '911 mapped retained cases plus86 restored constant responses; separately report strata, do not multiply sample size.',
        'target': 'One explicit archival envelope transformation across both native source versions. This is not proof that underlying FE/label-generating software was identical across production batches.',
        'scenario': 'The entire prescribed exposure is known as input; not online unknown-future prediction.',
        'topology': 'N change is complete-graph-family size transfer. Sparse-view same-weight tests measure representation-connectivity stress, not independently simulated physical adjacency changes.',
        'near_duplicate_scope': 'Exact permutation-invariant crack geometry verified; approximate/mirror geometries not collapsed.'})
    print(json.dumps({k: summary[k] for k in ('iid997', 'lcro_9_15')}, indent=2))


if __name__ == '__main__': main()
