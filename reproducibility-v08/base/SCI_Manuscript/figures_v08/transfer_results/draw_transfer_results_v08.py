"""Complete LCRO35 size-transfer and same-checkpoint connectivity figure.

No raw prediction arrays or partial run results are read. Shared scales, model
order, seed display and all connectivity views are fixed before outcomes.
"""
from pathlib import Path
import hashlib
import json
import datetime
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
SCI = HERE.parents[1]
EXP = SCI / 'experiments/research_v08'
EVAL = EXP / 'evaluation'
SOURCE = EVAL / 'lcro_9_15'
PLAN = EXP / 'comparison_plan_v08_420.json'
MODELS = ['fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets',
          'gnn', 'gnn_zero_edge_features', 'gnn_mean']
LABELS = ['Fire only', 'Global statistics', 'DeepSets', 'Matched\nDeepSets',
          'GNN\n(sum messages)', 'GNN\n(zero edge features)', 'GNN\n(mean messages)']
COLORS = ['#707070', '#946700', '#A1518B', '#357D9B', '#0072B2', '#B94A00', '#007C5A']
GNNS = MODELS[-3:]
VIEWS = ['complete', 'radius', 'symmetric_knn']
SEEDS = [42, 43, 44, 45, 46]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def replay_passed(item):
    assert item and item['cpu_diagnostic_tolerance'] == 3e-6
    if item['cpu_within_original_tolerance']:
        assert item['cpu_max_abs_difference'] < 3e-6
        assert item['status'] == 'CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE'
    else:
        assert item['status'] == 'GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED'
        assert item['same_backend_gpu_bitwise_equal'] and item['gpu_max_abs_difference'] == 0


def verified_sources():
    files = [SOURCE / 'summary.json', SOURCE / 'per_case_metrics.csv', SOURCE / 'strata_by_seed.csv',
             SOURCE / 'topology_by_seed.csv', EVAL / 'report.json', EVAL / 'run_integrity.json', PLAN]
    if not all(p.exists() for p in files):
        raise SystemExit('WAITING_FOR_COMPLETE_LCRO35: summary/audit absent; no plot generated.')
    before = {str(p): sha(p) for p in files}
    summary, plan, report, integrity = read(files[0]), read(PLAN), read(EVAL / 'report.json'), read(EVAL / 'run_integrity.json')
    assert summary['status'] == 'COMPLETE_PROTOCOL_35_RUNS' and summary['protocol'] == 'lcro_9_15'
    assert plan['status'] == 'FROZEN_BEFORE_TRAINING'
    assert summary['purpose'] == plan['purpose'] == report['purpose']
    assert summary['models'] == plan['models'] == MODELS
    assert summary['seeds'] == plan['seeds'] == SEEDS
    assert report['plan_sha256'] == sha(PLAN) and report['cpu_prediction_replay_enabled']
    status = report['protocols']['lcro_9_15']
    assert status['status'] == 'COMPLETE' and status['audited_run_count'] == status['expected_run_count'] == 35
    assert not status['missing_cells'] and status['scientific_summary_published']
    expected = {(m, s) for m in MODELS for s in SEEDS}
    audits = [a for a in integrity['audits'] if a['protocol'] == 'lcro_9_15']
    assert len(audits) == 35 and {(a['model'], a['seed']) for a in audits} == expected
    audit_by_cell = {(a['model'], a['seed']): a for a in audits}
    for a in audits:
        assert a['status'] == 'PASSED_BINDINGS' and a['ordered_ids_truth_times_exact']
        assert a['plan_dataset_feature_target_split_checkpoint_hashes']
        replay_passed(a['computational_replay'])
        if a['model'] in GNNS:
            assert a['topology_files_bound_to_same_checkpoint'] == 2
            assert set(a['topology_computational_replay']) == set(VIEWS[1:])
            for replay in a['topology_computational_replay'].values():
                replay_passed(replay)
    split_path = Path(plan['protocols']['lcro_9_15']['split_path'])
    manifest_path = Path(plan['manifest_path'])
    assert sha(split_path) == plan['protocols']['lcro_9_15']['split_sha256']
    assert sha(manifest_path) == plan['manifest_sha256']
    split, manifest = read(split_path), read(manifest_path)
    by_id = {c['sample_id']: c for c in manifest['cases']}
    ids = set(split['test'])
    assert len(ids) == summary['test_case_count'] == 264
    assert max(by_id[s]['num_cracks'] for s in split['train'] + split['validation']) == 8
    assert {by_id[s]['num_cracks'] for s in ids} == set(range(9, 16))
    assert {by_id[s]['source_batch'] for s in ids} == {'batch_005'}
    frame = pd.read_csv(SOURCE / 'per_case_metrics.csv')
    strata = pd.read_csv(SOURCE / 'strata_by_seed.csv')
    topology = pd.read_csv(SOURCE / 'topology_by_seed.csv')
    assert len(frame) == 35 * 264 and set(frame.protocol) == {'lcro_9_15'}
    assert set(zip(frame.model, frame.seed)) == expected
    assert not frame.duplicated(['model', 'seed', 'sample_id']).any()
    assert np.isfinite(frame.MAE).all() and frame.MAE.ge(0).all()
    for _, group in frame.groupby(['model', 'seed']):
        assert len(group) == 264 and set(group.sample_id) == ids
        assert set(group.source_batch) == {'batch_005'}
        assert all(by_id[r.sample_id]['num_cracks'] == r.crack_count and by_id[r.sample_id]['geometry_id'] == r.geometry_id for r in group.itertuples())
    seed_metrics = frame.groupby(['model', 'seed']).MAE.mean()
    count_metrics = frame.groupby(['model', 'seed', 'crack_count']).MAE.mean()
    assert len(count_metrics) == 7 * 5 * 7
    summaries = {s['model']: s for s in summary['model_summaries']}
    for model in MODELS:
        values = seed_metrics.loc[model].reindex(SEEDS).to_numpy()
        recorded = summaries[model]['metrics']['equal_case_MAE']
        assert np.isclose(values.mean(), recorded['mean'], atol=1e-13, rtol=1e-13)
        assert np.isclose(values.std(ddof=1), recorded['sample_sd'], atol=1e-13, rtol=1e-13)
    count_strata = strata[(strata.protocol == 'lcro_9_15') & (strata.stratum == 'crack_count')]
    assert len(count_strata) == 245
    for r in count_strata.itertuples():
        assert np.isclose(count_metrics.loc[r.model, r.seed, int(r.group)], r.equal_case_MAE, atol=1e-13, rtol=1e-13)
        assert r.case_count == sum(by_id[s]['num_cracks'] == int(r.group) for s in ids)
    assert len(topology) == 30 and set(topology.protocol) == {'lcro_9_15'}
    assert set(zip(topology.model, topology.seed, topology.rule)) == {(m, s, r) for m in GNNS for s in SEEDS for r in VIEWS[1:]}
    assert topology.same_trained_checkpoint.eq(True).all() and topology.refitting.eq(False).all()
    assert np.isfinite(topology[['equal_case_MAE', 'MAE_change_vs_complete']].to_numpy()).all()
    for r in topology.itertuples():
        assert r.checkpoint_sha256 == audit_by_cell[(r.model, r.seed)]['checkpoint_sha256']
        assert np.isclose(r.equal_case_MAE - seed_metrics.loc[r.model, r.seed], r.MAE_change_vs_complete, atol=1e-13, rtol=1e-13)
    assert before == {str(p): sha(p) for p in files}, 'Aggregator files changed during read; retry after completion.'
    return plan, summary, frame, count_metrics, seed_metrics, topology, audits, files + [split_path]


def main():
    plan, summary, frame, count_metrics, seed_metrics, topology, audits, sources = verified_sources()
    for source in sources:
        shutil.copyfile(source, HERE / ('source_' + source.name))
    counts = frame[['sample_id', 'crack_count']].drop_duplicates().groupby('crack_count').size()
    assert counts.reindex(range(9, 16)).tolist() == [48, 45, 41, 40, 36, 29, 25]
    count_rows = []
    for (model, seed, n), value in count_metrics.items():
        count_rows.append(dict(model=model, seed=seed, crack_count=n, case_count=int(counts.loc[n]),
                               MAE=float(value), MAE_percentage_points=float(value * 100)))
    connectivity = []
    for model in GNNS:
        for seed in SEEDS:
            complete = float(seed_metrics.loc[model, seed])
            checkpoint = next(a['checkpoint_sha256'] for a in audits if a['model'] == model and a['seed'] == seed)
            connectivity.append(dict(model=model, seed=seed, view='complete', MAE=complete,
                MAE_change_vs_complete=0.0, change_percentage_points=0.0, checkpoint_sha256=checkpoint))
            for rule in VIEWS[1:]:
                row = topology[(topology.model == model) & (topology.seed == seed) & (topology.rule == rule)].iloc[0]
                connectivity.append(dict(model=model, seed=seed, view=rule, MAE=float(row.equal_case_MAE),
                    MAE_change_vs_complete=float(row.MAE_change_vs_complete),
                    change_percentage_points=float(row.MAE_change_vs_complete * 100), checkpoint_sha256=checkpoint))
    pd.DataFrame(count_rows).to_csv(HERE / 'plotted_crack_count_seed_mae.csv', index=False)
    connect = pd.DataFrame(connectivity)
    connect.to_csv(HERE / 'plotted_same_checkpoint_connectivity.csv', index=False)
    counts.rename('cases').to_csv(HERE / 'holdout_count_distribution.csv')
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(183 / 25.4, 174 / 25.4))
    fig.text(.04, .983, 'a  Transfer to unseen crack counts: all seven models', weight='bold', va='top')
    fig.text(.04, .953, 'Train/validation: N = 1–8 · test: N = 9–15 · five seeds per model', va='top')
    offsets = np.linspace(-.14, .14, 5)
    max_mae = float(count_metrics.max() * 100) * 1.12
    for i, (model, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
        row, col = divmod(i, 4)
        ax = fig.add_axes([.09 + col * .23, .70 - row * .275, .18, .175])
        group = count_metrics.loc[model].unstack('crack_count').reindex(index=SEEDS, columns=range(9, 16)) * 100
        for j, seed in enumerate(SEEDS):
            ax.scatter(np.arange(9, 16) + offsets[j], group.loc[seed], s=10, facecolor='white', edgecolor=color, linewidth=.65, zorder=3)
        ax.plot(range(9, 16), group.mean(axis=0), color=color, lw=1.1, marker='o', ms=2.6, zorder=4)
        ax.set(xlim=(8.55, 15.45), ylim=(0, max_mae), xticks=[9, 12, 15])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= max_mae])
        ax.grid(axis='y', color='#E8EDF0', lw=.5)
        ax.set_title(label, color=color, pad=7)
        if col == 0:
            ax.set_ylabel('MAE (percentage points)')
        else:
            ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(length=2)
    meta = fig.add_axes([.78, .425, .18, .175]); meta.set_axis_off()
    meta.text(0, 1.12, 'Holdout scope', weight='bold', va='bottom')
    meta.text(0, .95, '264 cases; source 005 only', va='top')
    meta.text(0, .72, 'N: 9, 10, 11, 12, 13, 14, 15', va='top')
    meta.text(0, .49, 'n: 48, 45, 41, 40,\n    36, 29, 25 (by N)', va='top', linespacing=1.3)
    meta.legend(handles=[Line2D([], [], color='none', marker='o', markeredgecolor='#647886', markerfacecolor='white', ms=3, label='Individual seed'),
                         Line2D([], [], color='#172C39', marker='o', ms=2.6, lw=1, label='Mean over seeds')],
                loc='lower left', bbox_to_anchor=(-.04, -.09), frameon=False, handletextpad=.4, borderaxespad=0)
    fig.text(.50, .375, 'Number of cracks, N', ha='center', va='top')
    fig.text(.04, .330, 'b  Connectivity stress: same trained weights, changed graph view', weight='bold', va='top')
    fig.text(.04, .302, 'Each thin line pairs one seed; dark ticks show means · no refitting', va='top')
    values = connect.change_percentage_points.to_numpy()
    vmin, vmax = min(0, float(values.min())), max(0, float(values.max()))
    span = max(vmax - vmin, 1e-6); ylim = (vmin - .12 * span, vmax + .12 * span)
    for i, model in enumerate(GNNS):
        color = COLORS[MODELS.index(model)]
        ax = fig.add_axes([.11 + i * .30, .105, .24, .15])
        local = connect[connect.model == model].pivot(index='seed', columns='view', values='change_percentage_points').reindex(index=SEEDS, columns=VIEWS)
        for j, seed in enumerate(SEEDS):
            xs = np.arange(3) + offsets[j] * .45
            ax.plot(xs, local.loc[seed], color=color, lw=.65, alpha=.55, zorder=2)
            ax.scatter(xs, local.loc[seed], s=14, facecolor='white', edgecolor=color, linewidth=.8, zorder=3)
        for x, mean in enumerate(local.mean(axis=0)):
            ax.plot([x], [mean], marker='_', color='#172C39', ms=8, mew=1.8, zorder=4)
        ax.axhline(0, color='#647886', ls=(0, (3, 2)), lw=.7, zorder=0)
        ax.set(xlim=(-.25, 2.25), ylim=ylim, xticks=[0, 1, 2], xticklabels=['Complete', 'Radius', 'kNN'])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.set_yticks([t for t in ax.get_yticks() if ylim[0] <= t <= ylim[1]])
        ax.set_title(['Sum messages', 'Zero edge features', 'Mean messages'][i], color=color, pad=7)
        ax.tick_params(length=2)
        if i == 0:
            ax.set_ylabel('ΔMAE (percentage points)')
        else:
            ax.tick_params(axis='y', labelleft=False)
    fig.text(.04, .022, 'Archived scalar target · N is confounded with source batch · connectivity views reuse the same cases', va='bottom')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    overflow, sizes = [], []
    for obj in fig.findobj(match=matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        sizes.append(obj.get_fontsize()); box = obj.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            overflow.append(obj.get_text())
    assert min(sizes) >= 8 and not overflow, overflow
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(HERE / ('transfer_results_v08.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    write(HERE / 'provenance.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_files': [{'path': str(p), 'sha256': sha(p)} for p in sources],
        'script_sha256': sha(__file__), 'manifest_sha256': plan['manifest_sha256'],
        'protocol': 'lcro_9_15', 'target_version': plan['target_version'], 'feature_version': plan['feature_version'],
        'source_batch_holdout': 'batch_005 only; source and high crack count are confounded',
        'connectivity_recipe': read(Path(plan['protocols']['lcro_9_15']['split_path']))['topology_recipe'],
        'uncertainty_display': 'Individual seed values only. No confidence interval; no timepoint resampling.',
        'normalization_interpretation': 'Comparison of prespecified GNN variants under identical changed-connectivity inputs; no causal degree-effect estimate.',
        'unit_conversion': 'Source dimensionless MAE multiplied by 100, i.e. percentage points.'})
    write(HERE / 'figure_qa.json', {'status': 'GENERATED_PENDING_VISUAL_QA', 'size_mm': [183, 174],
        'minimum_font_pt': min(sizes), 'overflow': overflow, 'verified_runs': 35, 'test_cases': 264,
        'count_seed_points': len(count_rows), 'connectivity_seed_points': len(connectivity),
        'same_checkpoint_views': True, 'confidence_intervals_not_invented': True})
    print(json.dumps({'status': 'GENERATED_COMPLETE_LCRO35_FIGURE', 'pdf': str(HERE / 'transfer_results_v08.pdf')}))


if __name__ == '__main__':
    main()
