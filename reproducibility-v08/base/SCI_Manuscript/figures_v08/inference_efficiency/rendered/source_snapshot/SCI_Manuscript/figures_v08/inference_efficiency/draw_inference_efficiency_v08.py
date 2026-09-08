"""Render only independently audited real timing records; no model execution."""
from pathlib import Path
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import shutil

HERE = Path(__file__).resolve().parent
SCI = HERE.parents[1]
EXP = SCI / 'experiments/research_v08'
BENCH = EXP / 'inference_benchmark_v1'
MEASURED = BENCH / 'measured'
AUDIT = SCI / 'review/eaai_editor/research_v08/inference_timing_independent_audit.json'
MODELS = ['fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets',
          'gnn', 'gnn_zero_edge_features', 'gnn_mean']
LABELS = ['Fire only', 'Global statistics', 'DeepSets', 'Matched set',
          'Sum GNN', 'Zero edge features', 'Mean GNN']
FORWARD = 'forward_resident_inputs'
COMMON = 'common_input_to_host_output'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def gate():
    bound_hashes = {}
    missing = [str(p.relative_to(SCI)) for p in (AUDIT, MEASURED / 'summary.json') if not p.is_file()]
    reasons = ['Missing ' + p for p in missing]
    if (MEASURED / 'failure.json').exists():
        reasons.append('Retained failed measurement attempt')
    # Incomplete inputs never trigger a latency read or a placeholder drawing.
    if not reasons:
        audit_hash = sha(AUDIT)
        audit = read(AUDIT)
        assert sha(AUDIT) == audit_hash, 'Independent audit changed during read'
        if audit.get('status') != 'REAL_TIMING_RAW_RECORDS_COVERAGE_AND_ARITHMETIC_AUDIT_PASSED':
            reasons.append('Independent real-record audit is not complete')
        else:
            required = [MEASURED / 'summary.json', MEASURED / 'raw_timings.csv',
                        MEASURED / 'environment.json', BENCH / 'timing_plan.json',
                        EXP / 'evaluation/report.json', EXP / 'evaluation/run_integrity.json',
                        AUDIT.parent / 'audit_inference_timing.py', EXP / 'benchmark_inference.py',
                        MEASURED / 'training_costs_420.csv', MEASURED / 'premeasurement_artifact_binding.json',
                        MEASURED / 'readiness_at_completion.json']
            assert all(str(p) in audit['source_sha256'] for p in required)
            for path, expected in audit['source_sha256'].items():
                assert Path(path).is_file() and sha(path) == expected, 'Timing audit source changed: ' + path
            assert audit['raw_measurements_checked'] == 14355
            assert audit['actual_training_artifact_identities_checked'] == audit['training_runs'] == 420
            result = read(MEASURED / 'summary.json')
            assert result['status'] == 'COMPLETE_INFERENCE_TIMING' and result['rows'] == 14355
            timing = read(BENCH / 'timing_plan.json')
            assert sha(BENCH / 'timing_plan.json') == result['timing_plan_sha256']
            assert timing['protocol'] == 'iid997' and timing['seed'] == 42
            assert timing['times_per_case'] == 61 and timing['rounds'] == 3
            assert timing['devices'] == ['cpu', 'cuda'] and timing['batch_sizes'] == [1, 32]
            assert [x['model'] for x in timing['checkpoints']] == MODELS
            assert len(set(timing['sample_ids'])) == len(timing['sample_ids']) == 160
            assert sha(MEASURED / 'raw_timings.csv') == result['raw_csv_sha256']
            bound_hashes = {**audit['source_sha256'], str(AUDIT): audit_hash}
            assert bound_hashes == {p: sha(p) for p in bound_hashes}, 'Audited sources changed during gate'
    state = {'status': 'WAITING_FOR_REAL_AUDITED_TIMINGS' if reasons else 'READY_FOR_REAL_FIGURE',
             'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'reasons': reasons, 'raw_latency_records_read': 0, 'formal_figure_generated': False,
             'validated_source_sha256': bound_hashes}
    write(HERE / 'gate_status.json', state)
    return state


def checked_plot_values(bound_hashes):
    import numpy as np
    assert bound_hashes == {p: sha(p) for p in bound_hashes}, 'Audited sources changed before raw read'
    result = read(MEASURED / 'summary.json')
    with (MEASURED / 'raw_timings.csv').open(encoding='utf-8', newline='') as stream:
        raw = list(csv.DictReader(stream))
    assert len(raw) == 14355
    groups = collections.defaultdict(list)
    for row in raw:
        assert math.isfinite(float(row['elapsed_s'])) and float(row['elapsed_s']) > 0
        groups[(row['device'], row['model'], row['stage'], int(row['batch_size']))].append(row)
    summaries = {(r['device'], r['model'], r['stage'], r['batch_size']): r for r in result['summaries']}
    assert len(summaries) == len(result['summaries']) == len(groups) == 58
    plotted = []
    for device in ['cpu', 'cuda']:
        for model in MODELS:
            for stage, size in [(COMMON, 1), (FORWARD, 32), (COMMON, 32)]:
                key = (device, model, stage, size)
                rows = groups[key]
                assert len(rows) == (480 if size == 1 else 15)
                values = np.asarray([float(r['elapsed_s']) for r in rows])
                cases = sum(int(r['cases']) for r in rows)
                assert cases == 480
                metrics = {'median_batch_ms': float(np.median(values) * 1000),
                           'p95_batch_ms': float(np.quantile(values, .95) * 1000),
                           'cases_per_second': float(cases / values.sum())}
                for name, value in metrics.items():
                    assert math.isclose(value, summaries[key][name], rel_tol=1e-11, abs_tol=1e-12)
                selected = ['median_batch_ms', 'p95_batch_ms'] if size == 1 else ['cases_per_second']
                for name in selected:
                    plotted.append({'device': device, 'model': model, 'stage': stage, 'batch_size': size,
                                    'metric': name, 'value': metrics[name], 'cases_total': cases,
                                    'measurements': len(rows), 'unit': 'ms per 61-time query' if size == 1 else '61-time queries per second'})
    assert len(plotted) == 56
    assert bound_hashes == {p: sha(p) for p in bound_hashes}, 'Audited sources changed during calculation'
    return plotted, result


def render():
    state = gate()
    if state['status'] != 'READY_FOR_REAL_FIGURE':
        print('WAITING: no raw latency records read; no figure generated.')
        return 2
    target = HERE / 'rendered'
    assert not target.exists(), 'Preserve existing rendered snapshot; revise into a new version'
    bound_hashes = state['validated_source_sha256']
    plotted, result = checked_plot_values(bound_hashes)
    environment = read(MEASURED / 'environment.json')
    assert sha(MEASURED / 'environment.json') == bound_hashes[str(MEASURED / 'environment.json')]
    assert environment['threads'] == 4 and isinstance(environment['GPU'], str) and environment['GPU']
    gpu_label = environment['GPU'].removeprefix('NVIDIA GeForce ')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'axes.titlesize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .65})
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 154 / 25.4))
    fig.subplots_adjust(left=.205, right=.97, top=.89, bottom=.18, hspace=.68, wspace=.18)
    blue, teal = '#31688e', '#238b82'
    lookup = {(r['device'], r['model'], r['stage'], r['batch_size'], r['metric']): r['value'] for r in plotted}
    limits = {}
    for col, device in enumerate(['cpu', 'cuda']):
        for row in range(2):
            ax = axes[row, col]
            ax.set_yticks(range(7), LABELS if col == 0 else [''] * 7)
            ax.set_ylim(6.65, -.65)
            ax.tick_params(axis='y', length=0, pad=6)
            ax.grid(axis='x', color='#e5e7eb', linewidth=.5)
            ax.set_axisbelow(True)
            ax.set_title(('CPU (4 threads)' if device == 'cpu' else 'CUDA (' + gpu_label + ')'), loc='left', pad=7)
            ax.text(-.05, 1.13, 'abcd'[row * 2 + col], transform=ax.transAxes, weight='bold', fontsize=10)
        median = [lookup[(device, m, COMMON, 1, 'median_batch_ms')] for m in MODELS]
        p95 = [lookup[(device, m, COMMON, 1, 'p95_batch_ms')] for m in MODELS]
        axes[0, col].hlines(range(7), median, p95, color=blue, linewidth=1.2)
        axes[0, col].scatter(median, range(7), s=20, color=blue, zorder=3)
        axes[0, col].scatter(p95, range(7), s=22, facecolors='white', edgecolors=blue, marker='D', zorder=3)
        axes[0, col].set(xlabel='Single-query latency (ms)', xlim=(0, max(p95) * 1.13))
        all_values = []
        for offset, stage, color, marker in [(-.14, FORWARD, teal, 's'), (.14, COMMON, blue, 'o')]:
            values = [lookup[(device, m, stage, 32, 'cases_per_second')] for m in MODELS]
            all_values.extend(values)
            axes[1, col].scatter(values, np.arange(7) + offset, s=23, color=color, marker=marker, zorder=3)
        axes[1, col].set(xlabel='Batch-32 throughput (queries/s)', xlim=(0, max(all_values) * 1.13))
        limits[device] = {'single_latency_xlim': list(axes[0, col].get_xlim()),
                          'batch_throughput_xlim': list(axes[1, col].get_xlim())}
        # Automatic locators can place a tick outside the fixed data range.
        # Keep a concise set of ticks strictly within each declared axis.
        for row in range(2):
            ax = axes[row, col]
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
            lo, hi = ax.get_xlim()
            ax.set_xticks([v for v in ax.get_xticks() if lo <= v <= hi])
    fig.text(.205, .963, 'Single queries: common input to host output', fontsize=9, weight='bold')
    fig.text(.205, .495, 'Batch processing: two measured interfaces', fontsize=9, weight='bold')
    handles = [Line2D([], [], color=blue, marker='o', ls='', label='Single-query median'),
               Line2D([], [], color=blue, marker='D', mfc='white', ls='', label='Single-query p95'),
               Line2D([], [], color=teal, marker='s', ls='', label='Resident-input forward'),
               Line2D([], [], color=blue, marker='o', ls='', label='Common input to host output')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.57, .055), ncol=2,
               frameon=False, columnspacing=1.4, handletextpad=.5)
    fig.text(.205, .02, '160 geometries | 3 repeated rounds | seed 42 | 61-time trajectory per query', fontsize=8)
    fig.canvas.draw()
    from matplotlib.text import Text
    renderer = fig.canvas.get_renderer()
    texts = [a for a in fig.findobj(Text) if a.get_visible() and a.get_text()]
    assert all(a.get_fontsize() >= 8 for a in texts)
    outside = [a.get_text() for a in texts if not fig.bbox.expanded(1.002, 1.002).contains(*a.get_window_extent(renderer).get_points()[0])
               or not fig.bbox.expanded(1.002, 1.002).contains(*a.get_window_extent(renderer).get_points()[1])]
    assert not outside, ('Text outside canvas', outside)
    extra_sources = [HERE / 'figure_contract.md', HERE / 'caption_en.tex', Path(__file__), HERE / 'audit_inference_figure.py']
    source_hashes = {**bound_hashes, **{str(p): sha(p) for p in extra_sources}}
    assert source_hashes == {p: sha(p) for p in source_hashes}, 'Sources changed before final snapshot'
    target.mkdir()
    snapshot = target / 'source_snapshot'
    snapshot.mkdir()
    snapshot_records = []
    for original, digest in source_hashes.items():
        source = Path(original)
        copied = snapshot / source.relative_to(SCI.parent)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied)
        assert sha(source) == sha(copied) == digest, 'Snapshot differs from the validated source: ' + original
        snapshot_records.append({'original_path': original, 'snapshot_path': str(copied.relative_to(target)), 'sha256': digest})
    with (target / 'plotted_values.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=plotted[0]); writer.writeheader(); writer.writerows(plotted)
    # Companion records preserve common preparation, rounds, strata and historical costs.
    write(target / 'companion_summaries.json', result)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(target / ('inference_efficiency_v08.' + extension), dpi=220)
    plt.close(fig)
    write(target / 'figure_qa.json', {'status': 'RENDERED_PENDING_INDEPENDENT_AND_VISUAL_REVIEW',
          'dimensions_mm': [183, 154], 'minimum_text_pt': min(a.get_fontsize() for a in texts),
          'plotted_values': len(plotted), 'raw_latency_records': 14355, 'axes_limits': limits,
          'source_sha256': source_hashes, 'source_snapshot': snapshot_records,
          'measured_GPU': environment['GPU'], 'CPU_threads': environment['threads'],
          'derived_artifact_sha256': {name: sha(target / name) for name in ('plotted_values.csv', 'companion_summaries.json',
                                      'inference_efficiency_v08.pdf', 'inference_efficiency_v08.svg', 'inference_efficiency_v08.png')},
          'visual_QA_passed': False,
          'scientific_manuscript_or_submission_pass': False})
    print('Real timing figure rendered; independent arithmetic and visual review pending.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--gate-only', action='store_true')
    choice.add_argument('--render', action='store_true')
    args = parser.parse_args()
    raise SystemExit(render() if args.render else (0 if gate()['status'] == 'READY_FOR_REAL_FIGURE' else 2))
