"""All-ten-family figure with a strict completeness gate before performance reads.

Run --gate-only for metadata readiness, or --render after full evaluation. A
missing/incomplete family returns exit 2 without opening a performance source.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCI = HERE.parents[1]
EXP = SCI / 'experiments/research_v08'
EVAL = EXP / 'evaluation'
PLAN = EXP / 'comparison_plan_v08_420.json'
FAMILIES = ['iso834', 'astm_e119', 'external_fire', 'linear', 'bilinear',
            'plateau', 'decay', 'perturbed_iso', 'log_variant', 'smoldering']
LABELS = ['ISO 834', 'ASTM E119', 'External fire', 'Linear', 'Bilinear',
          'Plateau', 'Decay', 'Perturbed ISO', 'Log variant', 'Smoldering†']
MODELS = ['fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets',
          'gnn', 'gnn_zero_edge_features', 'gnn_mean']
MODEL_LABELS = ['Fire\nonly', 'Global\nstats', 'DeepSets', 'Matched\nDeepSets',
                'GNN\nsum', 'Zero edge\nfeatures', 'GNN\nmean']
SEEDS = [42, 43, 44, 45, 46]
PAIRS = [('graph_vs_capacity_matched', 'capacity_matched_deepsets', 'gnn'),
         ('mean_vs_sum', 'gnn', 'gnn_mean')]
READ_LOG = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    READ_LOG.append(str(Path(path).resolve()))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    READ_LOG.append(str(Path(path).resolve()))
    return pd.read_csv(path)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def performance_reads():
    return [p for p in READ_LOG if Path(p).parent.name.startswith('loco_') and
            Path(p).name in ('summary.json', 'per_case_metrics.csv', 'strata_by_seed.csv', 'topology_by_seed.csv')]


def replay_passed(item):
    assert item and item['cpu_diagnostic_tolerance'] == 3e-6
    if item['cpu_within_original_tolerance']:
        assert item['status'] == 'CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE'
        assert item['cpu_max_abs_difference'] < 3e-6
    else:
        assert item['status'] == 'GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED'
        assert item['same_backend_gpu_bitwise_equal'] and item['gpu_max_abs_difference'] == 0


def completeness_gate():
    """No per-family numerical performance source may be opened here."""
    report_path = EVAL / 'report.json'
    if not PLAN.exists() or not report_path.exists():
        return None, {'status': 'WAITING_FOR_ALL_TEN_FAMILY_EVALUATIONS',
            'reason': 'Frozen plan or authoritative evaluation report is absent.', 'performance_files_opened': 0}
    metadata_hashes = {str(PLAN): sha(PLAN), str(report_path): sha(report_path)}
    plan, report = read_json(PLAN), read_json(report_path)
    assert metadata_hashes == {str(PLAN): sha(PLAN), str(report_path): sha(report_path)}, 'Completeness metadata changed during read.'
    assert plan['status'] == 'FROZEN_BEFORE_TRAINING'
    assert plan['models'] == MODELS and plan['seeds'] == SEEDS
    expected, waiting, statuses = len(MODELS) * len(SEEDS), [], []
    valid_report = report.get('status') in ('PENDING_MATRIX', 'COMPLETE_420_RUN_MATRIX')
    for family in FAMILIES:
        protocol = 'loco_' + family
        item = report.get('protocols', {}).get(protocol, {})
        required = [EVAL / protocol / name for name in ('summary.json', 'per_case_metrics.csv')]
        ready = (valid_report and item.get('status') == 'COMPLETE' and
                 item.get('audited_run_count') == item.get('expected_run_count') == expected and
                 item.get('missing_cells') == [] and item.get('scientific_summary_published') is True and
                 all(p.exists() for p in required))
        statuses.append({'family': family, 'protocol': protocol,
                         'evaluation_status': item.get('status', 'UNAVAILABLE'),
                         'audited_runs': item.get('audited_run_count'), 'required_runs': expected, 'ready': ready})
        if not ready:
            waiting.append(protocol)
    ready = not waiting and report.get('cpu_prediction_replay_enabled') is True and (EVAL / 'run_integrity.json').exists()
    gate = {'status': 'ALL_TEN_PROTOCOLS_COMPLETE_READY_FOR_SOURCE_VALIDATION' if ready else 'WAITING_FOR_ALL_TEN_FAMILY_EVALUATIONS',
        'required_protocols': 10, 'required_runs': 350, 'complete_protocols': sum(s['ready'] for s in statuses),
        'waiting_protocols': waiting, 'protocol_statuses': statuses,
        'source_report_sha256': metadata_hashes[str(report_path)], 'source_plan_sha256': metadata_hashes[str(PLAN)],
        'performance_files_opened': len(performance_reads()), 'no_formal_figure_created': True}
    assert not performance_reads()
    if not ready:
        return None, gate
    assert report['purpose'] == plan['purpose'] and report['plan_sha256'] == sha(PLAN)
    return (plan, report, metadata_hashes), gate


def validate_sources(plan, report, metadata_hashes):
    """Called only after all ten families pass the joint completeness gate."""
    manifest_path = Path(plan['manifest_path'])
    assert sha(manifest_path) == plan['manifest_sha256']
    manifest = read_json(manifest_path)
    by_id = {c['sample_id']: c for c in manifest['cases']}
    assert len(by_id) == 997
    integrity_path = EVAL / 'run_integrity.json'
    integrity_hash = sha(integrity_path)
    integrity = read_json(integrity_path)
    assert sha(integrity_path) == integrity_hash
    evaluator = EXP / 'aggregate_results.py'
    stats_source = SCI / 'experiments/evaluation/evaluation_stats.py'
    for p in (evaluator, stats_source):
        assert report['audit_source_sha256'][p.name] == sha(p)
    files = [PLAN, EVAL / 'report.json', integrity_path, manifest_path, evaluator, stats_source]
    for family in FAMILIES:
        protocol = 'loco_' + family
        files.extend([EVAL / protocol / 'summary.json', EVAL / protocol / 'per_case_metrics.csv',
                      Path(plan['protocols'][protocol]['split_path'])])
    hashes = {str(p): sha(p) for p in files}
    assert all(hashes[path] == value for path, value in metadata_hashes.items())
    assert hashes[str(integrity_path)] == integrity_hash
    expected_cells = {(m, s) for m in MODELS for s in SEEDS}
    populations, family_seed_rows, contrasts, smolder_rows, audited = {}, [], [], [], []
    all_test_ids = []
    for family in FAMILIES:
        protocol = 'loco_' + family
        audits = [a for a in integrity['audits'] if a['protocol'] == protocol]
        assert len(audits) == 35 and {(a['model'], a['seed']) for a in audits} == expected_cells
        for a in audits:
            assert a['status'] == 'PASSED_BINDINGS' and a['ordered_ids_truth_times_exact']
            assert a['plan_dataset_feature_target_split_checkpoint_hashes']
            replay_passed(a['computational_replay'])
        audited.extend(audits)
        split_path = Path(plan['protocols'][protocol]['split_path'])
        assert sha(split_path) == plan['protocols'][protocol]['split_sha256']
        split = read_json(split_path)
        assert split['dataset_sha256'] == plan['manifest_sha256'] and split['protocol'] == protocol
        test_ids = set(split['test']); all_test_ids.extend(split['test'])
        n = len(test_ids); populations[family] = n
        assert {by_id[s]['fire_family'] for s in test_ids} == {family}
        assert test_ids == {sid for sid, c in by_id.items() if c['fire_family'] == family}
        assert test_ids.isdisjoint(split['train']) and test_ids.isdisjoint(split['validation'])
        summary = read_json(EVAL / protocol / 'summary.json')
        assert summary['status'] == 'COMPLETE_PROTOCOL_35_RUNS' and summary['protocol'] == protocol
        assert summary['purpose'] == plan['purpose'] and summary['models'] == MODELS and summary['seeds'] == SEEDS
        assert summary['test_case_count'] == summary['independent_geometry_clusters'] == n
        assert summary['target_interpretation'] == manifest['target_interpretation']
        frame = read_csv(EVAL / protocol / 'per_case_metrics.csv')
        assert len(frame) == 35 * n and set(frame.protocol) == {protocol}
        assert set(zip(frame.model, frame.seed)) == expected_cells
        assert not frame.duplicated(['model', 'seed', 'sample_id']).any()
        assert np.isfinite(frame[['MAE', 'RMSE']].to_numpy()).all() and frame.MAE.ge(0).all()
        for _, part in frame.groupby(['model', 'seed']):
            assert len(part) == n and set(part.sample_id) == test_ids
            assert all(by_id[r.sample_id]['geometry_id'] == r.geometry_id and by_id[r.sample_id]['fire_family'] == r.fire_family for r in part.itertuples())
        seed_mean = frame.groupby(['model', 'seed']).MAE.mean()
        recorded = {r['model']: r for r in summary['model_summaries']}
        for model in MODELS:
            values = seed_mean.loc[model].reindex(SEEDS).to_numpy()
            source = recorded[model]['metrics']['equal_case_MAE']
            assert np.isclose(values.mean(), source['mean'], atol=1e-13, rtol=1e-13)
            assert np.isclose(values.std(ddof=1), source['sample_sd'], atol=1e-13, rtol=1e-13)
            for seed, value in zip(SEEDS, values):
                family_seed_rows.append({'family': family, 'protocol': protocol, 'model': model, 'seed': seed,
                                         'test_cases': n, 'MAE': float(value), 'MAE_percentage_points': float(value * 100)})
        for tag, first, second in PAIRS:
            source = summary['paired_contrasts'][tag]
            effect = (seed_mean.loc[first].reindex(SEEDS) - seed_mean.loc[second].reindex(SEEDS)).to_numpy()
            assert source['bootstrap_repeats'] == 5000 and source['bootstrap_random_seed'] == 20260907
            assert source['seeds'] == SEEDS and source['test_cases'] == n and source['independent_geometry_clusters'] == n
            assert np.allclose(effect, source['paired_effect_by_seed'], atol=1e-13, rtol=1e-13)
            assert np.isclose(effect.mean(), source['point_estimate_equal_case_seed_mean'], atol=1e-13, rtol=1e-13)
            interval = source['intervals']['two_way']; assert interval['status'] == 'ESTIMATED'
            lo, hi = interval['percentile_95']; assert np.isfinite([lo, hi]).all() and lo <= hi
            contrasts.append({'family': family, 'contrast': tag, 'first_model': first, 'second_model': second,
                'difference': float(effect.mean()), 'percentile_95_lower': lo, 'percentile_95_upper': hi,
                'difference_percentage_points': float(effect.mean() * 100), 'lower_percentage_points': lo * 100,
                'upper_percentage_points': hi * 100, 'bootstrap_repeats': 5000, 'bootstrap_random_seed': 20260907,
                'interval_method': 'source_two_way_geometry_seed_percentile', 'multiplicity_adjusted': False})
        if family == 'smoldering':
            flat = {s for s in test_ids if by_id[s]['flat_response']}
            restored = {s for s in test_ids if by_id[s]['legacy_final_id'] is None}
            assert n == 97 and len(flat) == 86 and flat == restored
            for label, ids in [('restored_flat', flat), ('retained_nonflat', test_ids - flat)]:
                subset = frame[frame.sample_id.isin(ids)]
                for (model, seed), value in subset.groupby(['model', 'seed']).MAE.mean().items():
                    smolder_rows.append({'subgroup': label, 'model': model, 'seed': seed,
                                          'case_count': len(ids), 'MAE': float(value), 'MAE_percentage_points': float(value * 100)})
    assert len(audited) == 350 and len(all_test_ids) == len(set(all_test_ids)) == 997 and set(all_test_ids) == set(by_id)
    assert len(family_seed_rows) == 350 and len(contrasts) == 20 and len(smolder_rows) == 70
    assert hashes == {str(p): sha(p) for p in files}, 'Source changed during validation; retry after evaluation finishes.'
    return files, hashes, populations, pd.DataFrame(family_seed_rows), pd.DataFrame(contrasts), pd.DataFrame(smolder_rows)


def snapshot_sources(files, hashes):
    records = []
    for source in files:
        relative = source.relative_to(SCI)
        target = HERE / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha(target) == hashes[str(source)] and sha(source) == hashes[str(source)]
        records.append({'original_path': str(source), 'snapshot_path': str(target.relative_to(HERE)), 'sha256': hashes[str(source)]})
    return records


def render(plan, validated):
    # Plotting imports deliberately occur only after the complete data gate.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import MaxNLocator
    files, hashes, counts, seed_rows, contrasts, smolder = validated
    snapshots = snapshot_sources(files, hashes)
    seed_rows.to_csv(HERE / 'family_model_seed_mae.csv', index=False)
    contrasts.to_csv(HERE / 'paired_contrasts_source.csv', index=False)
    smolder.to_csv(HERE / 'smoldering_subgroup_seed_mae.csv', index=False)
    family_means = seed_rows.groupby(['family', 'model']).MAE.mean().unstack('model').reindex(index=FAMILIES, columns=MODELS)
    aggregate = []
    for model in MODELS:
        table = seed_rows[seed_rows.model == model].pivot(index='family', columns='seed', values='MAE').reindex(index=FAMILIES, columns=SEEDS)
        pooled = np.asarray([counts[f] for f in FAMILIES]) @ table.to_numpy() / 997
        macro = table.mean(axis=0).to_numpy()
        for kind, values in [('case_weighted_cross_holdout', pooled), ('equal_family_cross_holdout', macro)]:
            for seed, value in zip(SEEDS, values):
                aggregate.append({'estimand': kind, 'model': model, 'seed': seed, 'MAE': float(value),
                                  'MAE_percentage_points': float(value * 100), 'confidence_interval': 'not_estimated_descriptive_only'})
    aggregate = pd.DataFrame(aggregate); aggregate.to_csv(HERE / 'cross_holdout_descriptive_seed_means.csv', index=False)
    totals = aggregate.groupby(['estimand', 'model']).MAE.mean().unstack('model').reindex(
        index=['case_weighted_cross_holdout', 'equal_family_cross_holdout'], columns=MODELS)
    matrix = np.vstack([family_means.to_numpy(), totals.to_numpy()]) * 100
    matrix_rows = []
    for i, label in enumerate(FAMILIES + ['case_weighted_cross_holdout', 'equal_family_cross_holdout']):
        for j, model in enumerate(MODELS):
            matrix_rows.append({'row': label, 'model': model, 'MAE_percentage_points': float(matrix[i, j])})
    pd.DataFrame(matrix_rows).to_csv(HERE / 'plotted_heatmap.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(183 / 25.4, 190 / 25.4))
    ax = fig.add_axes([.22, .515, .67, .34]); cb = fig.add_axes([.911, .515, .014, .34])
    fig.text(.04, .980, 'a  Generalization to each completely held-out fire family', weight='bold', va='top')
    fig.text(.04, .952, 'All ten families · seven models · five seeds · 350 complete audited runs', va='top')
    cmap = LinearSegmentedColormap.from_list('errors', ['#FFFFFF', '#D6E5ED', '#81ADBF', '#315D80'])
    norm = Normalize(0, max(float(matrix.max()), 1e-12))
    plot = ax.pcolormesh(np.arange(-.5, 7.5), np.arange(-.5, 12.5), matrix,
        shading='flat', cmap=cmap, norm=norm, edgecolors='#E7EDF0', linewidth=.4)
    ax.set(xlim=(-.5, 6.5), ylim=(11.5, -.5), xticks=range(7), xticklabels=MODEL_LABELS,
           yticks=range(12), yticklabels=[f'{name} ({counts[f]})' for f, name in zip(FAMILIES, LABELS)] +
           ['Case-weighted (997)', 'Equal-family (10)'])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=6)
    for spine in ax.spines.values(): spine.set_visible(False)
    for tick in ax.get_yticklabels()[-2:]: tick.set_weight('bold')
    ax.axhline(9.5, color='#315D80', lw=1.2)
    for i in range(12):
        for j in range(7):
            rgb = np.asarray(cmap(norm(matrix[i, j]))[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = float(linear @ [.2126, .7152, .0722])
            color = 'white' if 1.05 / (luminance + .05) > (luminance + .05) / .067 else '#142634'
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color=color, fontsize=8)
    colorbar = fig.colorbar(plot, cax=cb, format='%.2g'); colorbar.solids.set_rasterized(False); colorbar.solids.set_edgecolor('face')
    colorbar.outline.set_visible(False); colorbar.ax.set_title('MAE\n(pp)', pad=7, fontsize=8)
    colorbar.set_ticks(np.linspace(0, norm.vmax, 5)); colorbar.ax.tick_params(length=2, labelsize=8)
    fig.text(.04, .470, 'b  Matched DeepSets − sum GNN', weight='bold', va='top')
    fig.text(.58, .470, 'c  Sum GNN − mean GNN', weight='bold', va='top')
    fig.text(.04, .443, 'Prespecified paired differences: 95% geometry-and-seed intervals; positive favors the second model', va='top')
    maxabs = float(np.abs(contrasts[['difference_percentage_points', 'lower_percentage_points', 'upper_percentage_points']].to_numpy()).max())
    extent = max(1e-6, maxabs * 1.1)
    for j, (tag, _, _) in enumerate(PAIRS):
        forest = fig.add_axes([.22 + .42 * j, .120, .30, .285])
        forest.axvline(0, color='#879AA6', ls=(0, (3, 2)), lw=.8, zorder=0)
        data = contrasts[contrasts.contrast == tag].set_index('family').reindex(FAMILIES)
        for i, row in enumerate(data.itertuples()):
            forest.plot([row.lower_percentage_points, row.upper_percentage_points], [i, i], color='#315D80', lw=1.2)
            forest.plot([row.lower_percentage_points] * 2, [i - .11, i + .11], color='#315D80', lw=.65)
            forest.plot([row.upper_percentage_points] * 2, [i - .11, i + .11], color='#315D80', lw=.65)
            forest.scatter([row.difference_percentage_points], [i], s=17, marker='D', color='#172C39', zorder=3)
        forest.set(xlim=(-extent, extent), ylim=(9.5, -.5), yticks=range(10),
                   yticklabels=LABELS if j == 0 else [''] * 10, xlabel='MAE difference\n(percentage points)')
        forest.tick_params(axis='y', length=0, pad=6); forest.spines['left'].set_visible(False)
        forest.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        forest.set_xticks([t for t in forest.get_xticks() if -extent <= t <= extent])
    fig.text(.04, .055, '† Smoldering: 86/97 cases are restored flat responses; 11 retained cases are nonflat.', va='top')
    fig.text(.04, .029, 'Intervals are unadjusted; the two cross-family averages are descriptive, with no pooled interval.', va='top')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); overflow = []; sizes = []
    for obj in fig.findobj(match=matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text(): continue
        sizes.append(obj.get_fontsize()); box = obj.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            overflow.append(obj.get_text())
    assert min(sizes) >= 8 and not overflow, overflow
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(HERE / ('fire_transfer_v08.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    write_json(HERE / 'provenance.json', {'status': 'GENERATED_COMPLETE_350_FAMILY_RUNS_PENDING_VISUAL_QA',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_snapshot': snapshots,
        'script_sha256': sha(__file__), 'target_version': plan['target_version'], 'feature_version': plan['feature_version'],
        'family_order': FAMILIES, 'model_order': MODELS, 'test_population': counts,
        'unit_conversion': 'Dimensionless MAE and paired differences multiplied by 100, i.e. percentage points.',
        'aggregate_estimands': {'case_weighted': 'sum_f n_f MAE_f / 997, then mean over five seeds',
            'equal_family': 'mean of ten family MAEs, then mean over five seeds'},
        'aggregate_scope': 'Ten separately trained holdout models per representation/seed; no common trained checkpoint, no aggregate CI.',
        'bootstrap': '20 existing two-way geometry/seed percentile 95% intervals, 5000 draws each, no timepoint resampling or multiplicity adjustment.',
        'smoldering_scope': '86 restored flat and 11 retained nonflat cases, all included; companion subgroup CSV retained.'})
    write_json(HERE / 'figure_qa.json', {'status': 'GENERATED_PENDING_VISUAL_QA', 'size_mm': [183, 190],
        'minimum_font_pt': min(sizes), 'overflow': overflow, 'complete_audited_family_runs': 350,
        'heatmap_cells': 84, 'paired_intervals': 20, 'family_seed_values': len(seed_rows),
        'no_incomplete_family_read_before_gate': True, 'no_aggregate_CI_fabricated': True})


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gate-only', action='store_true')
    group.add_argument('--render', action='store_true')
    args = parser.parse_args()
    ready, gate = completeness_gate()
    gate['requested_mode'] = 'render' if args.render else 'gate-only'
    gate['read_files'] = READ_LOG.copy()
    write_json(HERE / 'gate_status.json', gate)
    if ready is None:
        assert not performance_reads()
        print(json.dumps(gate, indent=2)); return 2
    if args.gate_only:
        print(json.dumps(gate, indent=2)); return 0
    plan, report, metadata_hashes = ready
    try:
        validated = validate_sources(plan, report, metadata_hashes)
        render(plan, validated)
    except Exception as exc:
        write_json(HERE / 'generation_status.json', {'status': 'STOPPED_VALIDATION_OR_EXPORT_FAILURE',
            'error': str(exc), 'read_files': READ_LOG, 'peer_review_pass': False})
        raise
    write_json(HERE / 'generation_status.json', {'status': 'GENERATED_COMPLETE_350_RUNS_PENDING_VISUAL_QA',
        'performance_files_opened': len(performance_reads()), 'read_files': READ_LOG, 'peer_review_pass': False})
    print(json.dumps({'status': 'GENERATED_COMPLETE_350_RUNS_PENDING_VISUAL_QA', 'pdf': str(HERE / 'fire_transfer_v08.pdf')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
