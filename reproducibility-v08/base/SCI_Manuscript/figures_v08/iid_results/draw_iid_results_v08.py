"""Draw only the independently audited, complete 35-run IID protocol.

The layout, order and estimands were selected before reading complete results.
No raw prediction arrays, partial run outputs or other protocols are read.
"""
from pathlib import Path
import datetime
import hashlib
import json
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
PLAN = EXP / 'comparison_plan_v08_420.json'
SOURCE = EXP / 'replay_precision_diagnostic/sealed_iid35'
REVIEW = SCI / 'review/eaai_editor/research_v08/iid997_independent_results_audit.json'
MODELS = ['fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets',
          'gnn', 'gnn_zero_edge_features', 'gnn_mean']
LABELS = ['Fire only', 'Global statistics', 'DeepSets', 'Matched DeepSets',
          'GNN (sum messages)', 'GNN (zero edge features)', 'GNN (mean messages)']
COLORS = ['#707070', '#946700', '#A1518B', '#357D9B', '#0072B2', '#B94A00', '#007C5A']
CONTRASTS = ['graph_vs_capacity_matched', 'mean_vs_sum']
CONTRAST_LABELS = ['Matched DeepSets − GNN', 'GNN − mean-message GNN']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def verified_data():
    required = [SOURCE / 'summary.json', SOURCE / 'per_case_metrics.csv',
                SOURCE / 'seal_manifest.json', REVIEW, PLAN]
    if not all(p.exists() for p in required):
        raise SystemExit('WAITING_FOR_COMPLETE_PROTOCOL: source summary or audit is absent; no figure generated.')
    before = {str(p): sha(p) for p in required}
    plan, summary = read_json(PLAN), read_json(SOURCE / 'summary.json')
    seal, review = read_json(SOURCE / 'seal_manifest.json'), read_json(REVIEW)
    assert plan['status'] == 'FROZEN_BEFORE_TRAINING'
    assert summary['status'] == 'COMPLETE_PROTOCOL_35_RUNS'
    assert summary['protocol'] == 'iid997'
    assert summary['purpose'] == plan['purpose']
    assert seal['status'] == 'SEALED_COMPLETE_IID_35_RUNS'
    assert seal['plan_sha256'] == sha(PLAN)
    assert seal['CPU_tolerance'] == 3e-6
    for item in seal['files']:
        assert sha(SOURCE / item['file']) == item['sha256']
    assert review['status'] == 'IID_ARITHMETIC_DEVELOPMENT_AUDIT_PASSED_NOT_FULL_MATRIX_OR_SUBMISSION_REVIEW'
    assert review['runs'] == 35 and review['per_case_CSV_rows_checked'] == 5600
    assert review['source_hashes'][str(PLAN)] == sha(PLAN)
    assert review['source_hashes'][str(EVAL / 'iid997/summary.json')] == sha(SOURCE / 'summary.json')
    assert summary['models'] == plan['models'] == MODELS
    seeds = plan['seeds']
    assert summary['seeds'] == seeds == [42, 43, 44, 45, 46]
    expected_cells = {(m, s) for m in MODELS for s in seeds}
    audits = seal['audited_cells']
    assert len(audits) == 35 and {(a['model'], a['seed']) for a in audits} == expected_cells
    for a in audits:
        assert a['status'] == 'PASSED_BINDINGS'
        assert a['protocol'] == 'iid997'
        assert a['ordered_ids_truth_times_exact'] and a['plan_dataset_feature_target_split_checkpoint_hashes']
        assert a['cpu_replay_max_abs_difference'] < 3e-6
        assert a['computational_replay']['cpu_within_original_tolerance']
        prediction_path = Path(plan['runs_root']) / 'iid997' / a['model'] / ('seed_' + str(a['seed'])) / 'test_predictions.npz'
        assert review['prediction_hashes'][str(prediction_path)] == a['predictions_sha256']
    split_path = Path(plan['protocols']['iid997']['split_path'])
    assert sha(split_path) == plan['protocols']['iid997']['split_sha256']
    split = read_json(split_path)
    n = summary['test_case_count']
    assert n == len(split['test']) == 160
    frame = pd.read_csv(SOURCE / 'per_case_metrics.csv')
    assert len(frame) == 35 * n and set(frame.protocol) == {'iid997'}
    assert set(zip(frame.model, frame.seed)) == expected_cells
    assert not frame.duplicated(['model', 'seed', 'sample_id']).any()
    assert np.isfinite(frame[['MAE', 'RMSE', 'maximum_overprediction']].to_numpy()).all()
    assert (frame[['MAE', 'RMSE', 'maximum_overprediction']].to_numpy() >= 0).all()
    assert frame.groupby('sample_id').geometry_id.nunique().max() == 1
    assert frame.geometry_id.nunique() == summary['independent_geometry_clusters']
    for _, group in frame.groupby(['model', 'seed']):
        assert len(group) == n and set(group.sample_id) == set(split['test'])
    seed_means = frame.groupby(['model', 'seed'], sort=False).MAE.mean()
    by_model = {row['model']: row for row in summary['model_summaries']}
    assert set(by_model) == set(MODELS)
    for model in MODELS:
        values = seed_means.loc[model].reindex(seeds).to_numpy()
        src = by_model[model]['metrics']['equal_case_MAE']
        assert np.isclose(values.mean(), src['mean'], atol=1e-13, rtol=1e-13)
        assert np.isclose(values.std(ddof=1), src['sample_sd'], atol=1e-13, rtol=1e-13)
    for tag, (first, second) in zip(CONTRASTS, [('capacity_matched_deepsets', 'gnn'), ('gnn', 'gnn_mean')]):
        src = summary['paired_contrasts'][tag]
        expected = (seed_means.loc[first].reindex(seeds) - seed_means.loc[second].reindex(seeds)).to_numpy()
        assert src['bootstrap_repeats'] == 5000 and src['bootstrap_random_seed'] == 20260907
        assert src['intervals']['two_way']['status'] == 'ESTIMATED'
        assert src['test_cases'] == n and src['seed_count'] == 5
        assert np.allclose(expected, src['paired_effect_by_seed'], atol=1e-13, rtol=1e-13)
        assert np.isclose(expected.mean(), src['point_estimate_equal_case_seed_mean'], atol=1e-13, rtol=1e-13)
        lo, hi = src['intervals']['two_way']['percentile_95']
        assert np.isfinite([lo, hi]).all() and lo <= hi
    assert before == {str(p): sha(p) for p in required}, 'Sources changed during verification; rerun after aggregator completion.'
    return plan, summary, frame, seed_means, required + [split_path], audits


def main():
    plan, summary, frame, seed_means, sources, audits = verified_data()
    seeds = plan['seeds']
    for path in sources:
        shutil.copyfile(path, HERE / ('source_' + path.name))
    seed_rows, contrast_rows, ecdf_rows = [], [], []
    case_means = frame.groupby(['model', 'sample_id'], sort=False).MAE.mean()
    for model, label in zip(MODELS, LABELS):
        for seed in seeds:
            v = float(seed_means.loc[model, seed])
            seed_rows.append(dict(model=model, display_label=label, seed=seed, MAE=v, MAE_percentage_points=v * 100))
        values = case_means.loc[model].sort_values()
        for rank, (sid, value) in enumerate(values.items(), 1):
            ecdf_rows.append(dict(model=model, sample_id=sid, seed_mean_case_MAE=float(value),
                                  MAE_percentage_points=float(value * 100), cumulative_fraction=rank / len(values)))
    for tag in CONTRASTS:
        c = summary['paired_contrasts'][tag]
        lo, hi = c['intervals']['two_way']['percentile_95']
        contrast_rows.append(dict(contrast=tag, definition=c['contrast'], mean_difference=c['point_estimate_equal_case_seed_mean'],
                                  percentile_95_lower=lo, percentile_95_upper=hi, bootstrap_repeats=c['bootstrap_repeats'],
                                  bootstrap_random_seed=c['bootstrap_random_seed'], interval_method='two_way_geometry_seed_percentile'))
    pd.DataFrame(seed_rows).to_csv(HERE / 'plotted_seed_mae.csv', index=False)
    pd.DataFrame(contrast_rows).to_csv(HERE / 'plotted_paired_contrasts.csv', index=False)
    pd.DataFrame(ecdf_rows).to_csv(HERE / 'plotted_case_ecdf.csv', index=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(183 / 25.4, 146 / 25.4))
    ax = fig.add_axes([.29, .565, .66, .345])
    bx = fig.add_axes([.11, .12, .37, .285])
    cx = fig.add_axes([.63, .12, .32, .285])
    fig.text(.04, .972, 'a  IID error across all prespecified models', weight='bold', va='top')
    fig.text(.04, .938, '160 test cases · five training seeds', va='top')
    offsets = np.linspace(-.15, .15, 5)
    means, maxval = [], 0.0
    for i, (model, color) in enumerate(zip(MODELS, COLORS)):
        vals = seed_means.loc[model].reindex(seeds).to_numpy() * 100
        means.append(vals.mean()); maxval = max(maxval, vals.max())
        ax.scatter(vals, i + offsets, s=21, facecolor='white', edgecolor=color, linewidth=.9, zorder=3)
        ax.plot([vals.mean()], [i], marker='|', markersize=12, mew=2, color='#172C39', zorder=4)
        ax.axhline(i, color='#E8EDF0', lw=.55, zorder=0)
    ax.set(ylim=(6.6, -.6), xlim=(0, maxval * 1.12), yticks=range(7), yticklabels=LABELS,
           xlabel='Equal-case MAE (percentage points)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.tick_params(axis='y', length=0, pad=8)
    ax.spines['left'].set_visible(False)
    for tick, color in zip(ax.get_yticklabels(), COLORS):
        tick.set_color(color)
    ax.legend(handles=[Line2D([], [], marker='o', color='none', markeredgecolor='#647886', markerfacecolor='white', markersize=4, label='Individual seed'),
                       Line2D([], [], marker='|', color='none', markeredgecolor='#172C39', markeredgewidth=2, markersize=9, label='Mean')],
              loc='lower right', bbox_to_anchor=(1, 1.005), frameon=False, ncol=2, handletextpad=.35, columnspacing=.8, borderaxespad=0)
    fig.text(.04, .475, 'b  Prespecified paired contrasts', weight='bold', va='top')
    fig.text(.04, .446, '95% interval: geometry and seed resampling', va='top')
    radius = max(abs(v) for c in contrast_rows for v in [c['mean_difference'], c['percentile_95_lower'], c['percentile_95_upper']]) * 115
    radius = max(radius, 1e-6)
    bx.axvline(0, color='#9FAEB7', ls=(0, (3, 2)), lw=.8, zorder=0)
    for i, c in enumerate(contrast_rows):
        y = 1.15 - i * 1.1
        point = c['mean_difference'] * 100; low = c['percentile_95_lower'] * 100; high = c['percentile_95_upper'] * 100
        bx.plot([low, high], [y, y], color='#315D80', lw=1.8)
        bx.plot([low, low], [y - .08, y + .08], color='#315D80', lw=.8)
        bx.plot([high, high], [y - .08, y + .08], color='#315D80', lw=.8)
        bx.scatter([point], [y], color='#172C39', marker='D', s=22, zorder=3)
        bx.text(.0, (y + .35 + .4) / 2.3, CONTRAST_LABELS[i], transform=bx.transAxes, va='bottom')
    bx.set(xlim=(-radius, radius), ylim=(-.4, 1.9), yticks=[], xlabel='First − second MAE (percentage points)')
    bx.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    bx.spines['left'].set_visible(False)
    fig.text(.565, .475, 'c  Distribution across test cases', weight='bold', va='top')
    fig.text(.565, .446, 'Colors identify the models in a', va='top')
    styles = ['-', (0, (5, 2)), (0, (1, 1)), (0, (3, 1, 1, 1)), '-', (0, (5, 2)), (0, (1, 1))]
    xmax = 0.0
    for model, color, style in zip(MODELS, COLORS, styles):
        v = np.sort(case_means.loc[model].to_numpy() * 100)
        xmax = max(xmax, float(v.max()))
        cx.step(np.r_[0, v], np.r_[0, np.arange(1, len(v) + 1) / len(v)], where='post', color=color, ls=style, lw=1.1)
    cx.axhline(.95, color='#647886', ls=(0, (3, 2)), lw=.6, zorder=0)
    cx.text(.985, .95, '95%', transform=cx.get_yaxis_transform(), ha='right', va='top', color='#647886', fontsize=8)
    cx.set(xlim=(0, xmax * 1.04), ylim=(0, 1.04), yticks=[0, .5, 1], yticklabels=['0', '50', '100'],
           xlabel='Seed-mean case MAE\n(percentage points)', ylabel='Cumulative share of cases (%)')
    cx.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    for axis in (ax, bx, cx):
        lo, hi = axis.get_xlim()
        axis.set_xticks([tick for tick in axis.get_xticks() if lo <= tick <= hi])
    fig.text(.04, .025, 'Archived scalar target · all models use the same test cases · complete 35-run IID protocol', va='bottom')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow, textsizes = [], []
    for obj in fig.findobj(match=matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        textsizes.append(obj.get_fontsize())
        box = obj.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5:
            overflow.append(obj.get_text())
    assert min(textsizes) >= 8 and not overflow, overflow
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(HERE / ('iid_results_v08.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    write_json(HERE / 'figure_qa.json', {'status': 'GENERATED_PENDING_VISUAL_QA', 'size_mm': [183, 146],
        'minimum_font_pt': min(textsizes), 'figure_boundary_overflow': overflow,
        'source_protocol_complete': True, 'verified_run_count': len(audits), 'test_cases': summary['test_case_count'],
        'model_order': MODELS, 'seed_count': 5, 'source_results_not_reordered': True,
        'intervals_copied_from_existing_5000_resample_summary': True,
        'per_case_timepoints_not_resampled': True, 'complete_ecdf_range': True})
    write_json(HERE / 'provenance.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_files': [{'path': str(p), 'sha256': sha(p)} for p in sources],
        'script_sha256': sha(__file__), 'purpose': plan['purpose'], 'protocol': 'iid997',
        'target_version': plan['target_version'], 'feature_version': plan['feature_version'],
        'unit_conversion': 'All plotted MAE and contrast values are source dimensionless values multiplied by 100, i.e. percentage points.',
        'distribution_estimand': 'Across cases, empirical distribution of each case MAE averaged over five independently trained seeds; not ensemble-prediction error.',
        'bootstrap': 'Existing two-way geometry/seed percentile 95% intervals, 5000 resamples, random seed 20260907; no interval recomputed by this script.'})
    print(json.dumps({'status': 'GENERATED_COMPLETE_PROTOCOL_FIGURE', 'pdf': str(HERE / 'iid_results_v08.pdf'), 'models': 7, 'seeds': 5, 'cases': 160}))


if __name__ == '__main__':
    main()
