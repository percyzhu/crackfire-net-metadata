"""Four deterministic LOCO examples; no predictions before the full review gate."""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCI = HERE.parents[1]
WORKSPACE = SCI.parent
EXP = SCI / 'experiments/research_v08'
EVAL = EXP / 'evaluation'
REVIEW = SCI / 'review/eaai_editor/research_v08'
SELECTION = HERE.parent / 'engineering_examples_selection_plan.json'
SELECTION_SHA = '1850acfce80828fa7db4cd7fc751e9c86b8c47f8366fa3c91cc24135ce5d8c1e'
HELPER = HERE.parent / 'fire_transfer/draw_fire_transfer_v08.py'
HELPER_SHA = '650a4e002615230ae3739ec084ac5ddab06aebc82bb113ad9f11bcd30ba2ef87'
PRIMARY = ['capacity_matched_deepsets', 'gnn']
NPZ_READS = []
OWN_READS = []


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
def read(path):
    OWN_READS.append(str(Path(path).resolve()))
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_helper():
    assert sha(HELPER) == HELPER_SHA and sha(SELECTION) == SELECTION_SHA
    spec = importlib.util.spec_from_file_location('declared_fire_transfer_helper', HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prerequisite_gate(fh):
    selection = read(SELECTION)
    assert selection['candidate_model'] == 'gnn' and selection['baseline_model'] == PRIMARY[0]
    assert selection['seeds'] == [42, 43, 44, 45, 46]
    assert [x['order_statistic_fraction'] for x in selection['quantile_examples']] == [.5, .95, 1.0]
    complete, base_gate = fh.completeness_gate()
    required, missing = [], []
    for family in fh.FAMILIES:
        protocol = 'loco_' + family
        current = [REVIEW / f'{protocol}_engineering_proxy_ci.json', REVIEW / f'{protocol}_engineering_proxy_ci.csv',
                   REVIEW / f'{protocol}_independent_results_audit_v2.json', REVIEW / f'{protocol}_results_development_review_zh.md']
        required.extend(current); missing.extend(str(p) for p in current if not p.exists())
    gate = {'status': 'READY_FOR_REVIEW_BINDING_CHECK' if complete and not missing else 'WAITING_FOR_ALL_FAMILY_RUNS_AND_REVIEWS',
        'complete_audited_family_protocols': base_gate.get('complete_protocols', 0), 'required_protocols': 10,
        'required_runs': 350, 'waiting_protocols': base_gate.get('waiting_protocols', []),
        'missing_review_or_engineering_files': missing, 'selection_plan_sha256': SELECTION_SHA,
        'helper_sha256': HELPER_SHA, 'prediction_files_opened': 0,
        'family_performance_files_opened': len(fh.performance_reads()),
        'review_performance_documents_opened': 0,
        'metadata_files_read': OWN_READS + fh.READ_LOG, 'no_formal_figure_created': True}
    if complete is None or missing:
        assert not NPZ_READS and not fh.performance_reads()
        return None, gate
    plan, report, metadata_hashes = complete
    assert selection['training_plan_sha256'] == sha(fh.PLAN)
    hashes = {str(p): sha(p) for p in required}
    records = {}
    for family in fh.FAMILIES:
        protocol = 'loco_' + family
        cp = REVIEW / f'{protocol}_engineering_proxy_ci.json'
        rp = REVIEW / f'{protocol}_independent_results_audit_v2.json'
        ci, review = read(cp), read(rp)
        assert ci['status'] == 'COMPLETE_35_RUN_PAIRED_ENGINEERING_PROXY_CI'
        assert review['status'] == 'COMPLETE_PROTOCOL_ARITHMETIC_DEVELOPMENT_AUDIT_PASSED_NOT_FULL_MATRIX_OR_SUBMISSION_REVIEW'
        assert ci['protocol'] == review['protocol'] == protocol
        assert ci['completed_runs_checked'] == review['runs'] == 35
        assert ci['seeds'] == review['seeds'] == fh.SEEDS
        assert ci['test_case_count'] == review['independent_geometries'] == plan['protocols'][protocol]['counts']['test']
        assert review['case_time_grid'] == 61 and review['per_case_CSV_rows_checked'] == 35 * ci['test_case_count']
        assert ci['thresholds'] == [.8, .6] and ci['secondary_bootstrap_repetitions'] == 1000
        assert ci['primary_MAE_bootstrap_repetitions_separately_implemented'] == 5000
        assert review['primary_bootstrap_independently_recomputed'] == {'contrasts': 2, 'methods': 3, 'repeats': 5000}
        assert ci['plan_sha256'] == review['source_hashes'][str(fh.PLAN)] == sha(fh.PLAN)
        assert ci['manifest_sha256'] == plan['manifest_sha256']
        assert review['source_hashes'][str(cp)] == hashes[str(cp)]
        assert review['source_hashes'][str(EVAL / protocol / 'summary.json')] == sha(EVAL / protocol / 'summary.json')
        for path, digest in ci['code_sha256'].items(): assert sha(path) == digest
        assert sha(REVIEW / 'engineering_proxy_evaluation_prespecification.md') == ci['prespecification_sha256']
        assert len(ci['source_prediction_sha256']) == len(review['prediction_hashes']) == 35
        assert ci['source_prediction_sha256'] == review['prediction_hashes']
        records[protocol] = {'ci': ci, 'review': review}
    assert hashes == {str(p): sha(p) for p in required}
    gate.update(status='ALL_350_RUNS_AND_TEN_REVIEWS_BOUND_READY_FOR_SOURCE_VALIDATION',
                review_performance_documents_opened=20, metadata_files_read=OWN_READS + fh.READ_LOG)
    return (selection, plan, report, metadata_hashes, records, required, hashes), gate


def open_predictions(fh, ready, validated):
    import torch
    selection, plan, report, metadata_hashes, reviews, review_files, review_hashes = ready
    source_files, source_hashes = validated[:2]
    manifest_path = Path(plan['manifest_path']); manifest = read(manifest_path)
    tensor_path = manifest_path.parent / manifest['tensor_path']
    assert sha(tensor_path) == manifest['tensor_sha256']
    tensors = torch.load(tensor_path, map_location='cpu', weights_only=True)
    times = tensors['time_s'].numpy(); assert np.array_equal(times, np.arange(61) * 60)
    cases = {c['sample_id']: c for c in manifest['cases']}
    integrity = read(EVAL / 'run_integrity.json')
    assert sha(EVAL / 'run_integrity.json') == source_hashes[str(EVAL / 'run_integrity.json')]
    audit = {(a['protocol'], a['model'], a['seed']): a for a in integrity['audits']}
    histories, components, candidates, prediction_sources = {}, [], [], []
    for family in fh.FAMILIES:
        protocol = 'loco_' + family
        split = read(Path(plan['protocols'][protocol]['split_path']))
        ids = split['test']; truth = np.stack([tensors['targets'][sid].numpy() for sid in ids]).astype(float)
        csv = pd.read_csv(EVAL / protocol / 'per_case_metrics.csv').set_index(['model', 'seed', 'sample_id'])
        pred = {}
        for model in PRIMARY:
            data = []
            for seed in fh.SEEDS:
                folder = Path(plan['runs_root']) / protocol / model / f'seed_{seed}'
                path, weights = folder / 'test_predictions.npz', folder / 'best.pt'
                row = audit[(protocol, model, seed)]
                digest = sha(path)
                assert digest == row['predictions_sha256'] == reviews[protocol]['ci']['source_prediction_sha256'][str(path)]
                assert digest == reviews[protocol]['review']['prediction_hashes'][str(path)]
                assert sha(weights) == row['checkpoint_sha256']
                NPZ_READS.append(str(path))
                with np.load(path, allow_pickle=False) as z:
                    assert z['sample_ids'].tolist() == ids and np.array_equal(z['time_s'], times)
                    assert np.array_equal(z['truth'], truth)
                    values = z['predictions'].astype(float)
                assert values.shape == truth.shape and np.isfinite(values).all() and sha(path) == digest
                error = values - truth
                for j, sid in enumerate(ids):
                    saved = csv.loc[(model, seed, sid)]
                    assert np.isclose(np.abs(error[j]).mean(), saved.MAE, atol=1e-13, rtol=1e-13)
                    assert np.isclose(np.maximum(error[j], 0).max(), saved.maximum_overprediction, atol=1e-13, rtol=1e-13)
                    if model == 'gnn':
                        components.append({'sample_id': sid, 'protocol': protocol, 'seed': seed,
                            'case_MAE': float(np.abs(error[j]).mean()), 'positive_peak': float(np.maximum(error[j], 0).max())})
                data.append(values)
                prediction_sources.append({'protocol': protocol, 'model': model, 'seed': seed,
                    'prediction_path': str(path), 'prediction_sha256': digest,
                    'checkpoint_path': str(weights), 'checkpoint_sha256': row['checkpoint_sha256']})
            pred[model] = np.stack(data)
        gnne = pred['gnn'] - truth[None]
        for j, sid in enumerate(ids):
            c = cases[sid]
            assert sid not in histories
            histories[sid] = {'reference': truth[j], **{m: pred[m][:, j] for m in PRIMARY}}
            candidates.append({'sample_id': sid, 'protocol': protocol, 'geometry_id': c['geometry_id'],
                'fire_family': family, 'source_batch': c['source_batch'], 'crack_count': c['num_cracks'],
                'flat_response': c['flat_response'], 'restored': c['legacy_final_id'] is None,
                'mean_case_MAE': float(np.abs(gnne[:, j]).mean(axis=1).mean()),
                'mean_positive_peak': float(np.maximum(gnne[:, j], 0).max(axis=1).mean())})
    assert len(candidates) == len(histories) == 997 and len(components) == 4985 and len(NPZ_READS) == 100
    assert source_hashes == {str(p): sha(p) for p in source_files}
    assert review_hashes == {str(p): sha(p) for p in review_files}
    return cases, tensors, times, pd.DataFrame(candidates), pd.DataFrame(components), histories, prediction_sources


def select_cases(scores, selection):
    sorted_values = np.sort(scores.mean_case_MAE.to_numpy())
    chosen, selected = [], []
    for item in selection['quantile_examples']:
        rank = int(np.floor(996 * item['order_statistic_fraction']))
        target = sorted_values[rank]
        candidates = scores[(scores.mean_case_MAE == target) & ~scores.sample_id.isin(chosen)].sort_values('sample_id')
        assert len(candidates)
        row = candidates.iloc[0].to_dict(); chosen.append(row['sample_id'])
        selected.append(dict(row, example_label=item['label'], selection_kind='mean_case_MAE', rank_zero_based=rank,
                             selection_score=float(target), no_remaining_positive_error=False))
    remaining = scores[~scores.sample_id.isin(chosen)].sort_values(['mean_positive_peak', 'sample_id'], ascending=[False, True])
    row = remaining.iloc[0].to_dict()
    selected.append(dict(row, example_label='Largest remaining mean positive peak', selection_kind='mean_positive_peak',
                         rank_zero_based=None, selection_score=float(row['mean_positive_peak']),
                         no_remaining_positive_error=bool(row['mean_positive_peak'] == 0)))
    assert len({r['sample_id'] for r in selected}) == 4
    return selected


def prescribed_functions(plan):
    import torch
    path = WORKSPACE / '1_代码/src/gnn/model.py'
    assert sha(path) == plan['source_sha256']['1_代码/src/gnn/model.py']
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('compute_fire_curve', 'make_time_features')]
    assert len(nodes) == 2
    ns = {'torch': torch, 'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns, path


def materialize(fh, ready, validated, loaded):
    import torch
    selection, plan = ready[:2]
    cases, tensors, times, scores, components, histories, prediction_sources = loaded
    selected = select_cases(scores, selection)
    ns, fire_source = prescribed_functions(plan)
    curves, all_seed, events, selected_rows, fire_parameters = [], [], [], [], []
    for number, selected_case in enumerate(selected, 1):
        sid = selected_case['sample_id']; case = cases[sid]; values = histories[sid]; fire = case['fire']
        computed_tf = ns['make_time_features'](61, 3600., fire['type'], fire['params']).squeeze(0)
        assert torch.equal(computed_tf, tensors['graphs'][sid][-1])
        temperature = ns['compute_fire_curve'](torch.tensor(times, dtype=torch.float32), fire['type'], fire['params']).numpy()
        assert np.isfinite(temperature).all()
        fire_parameters.append({'example': number, 'sample_id': sid, 'recorded_fire': fire,
                                'time_input_exactly_reproduced': True, 'function_source_sha256': sha(fire_source)})
        selected_case['example'] = number
        for model in PRIMARY:
            e = values[model][0] - values['reference']
            selected_case[model + '_seed42_MAE'] = float(np.abs(e).mean())
            selected_case[model + '_seed42_positive_peak'] = float(np.maximum(e, 0).max())
            for j, seed in enumerate(fh.SEEDS):
                for k, time in enumerate(times):
                    all_seed.append({'example': number, 'sample_id': sid, 'model': model, 'seed': seed,
                                     'time_s': float(time), 'prediction': float(values[model][j, k])})
        selected_rows.append(selected_case)
        for k, time in enumerate(times):
            curves.append({'example': number, 'sample_id': sid, 'time_s': float(time), 'reference': float(values['reference'][k]),
                'prescribed_temperature_C': float(temperature[k]),
                **{m + '_seed42': float(values[m][0, k]) for m in PRIMARY},
                **{m + '_min': float(values[m][:, k].min()) for m in PRIMARY},
                **{m + '_max': float(values[m][:, k].max()) for m in PRIMARY}})
        for model in ['reference'] + PRIMARY:
            series = values['reference'] if model == 'reference' else values[model][0]
            for q in (.8, .6):
                hits = np.flatnonzero(series <= q); has_event = bool(len(hits)); first = int(hits[0]) if has_event else None
                events.append({'example': number, 'sample_id': sid, 'series': model, 'threshold': q,
                    'event': has_event, 'first_grid_time_s': float(times[first]) if has_event else None,
                    'response_at_event': float(series[first]) if has_event else None,
                    'censor_label': '' if has_event else '>3600 s',
                    'recross': bool(np.any(series[first + 1:] > q)) if has_event else False})
    for name, frame in [('all_candidate_scores.csv', scores), ('selection_score_components.csv', components),
                        ('selected_cases.csv', pd.DataFrame(selected_rows)), ('curve_data.csv', pd.DataFrame(curves)),
                        ('all_seed_curves.csv', pd.DataFrame(all_seed)), ('threshold_events.csv', pd.DataFrame(events))]:
        frame.to_csv(HERE / name, index=False)
    write(HERE / 'selected_fire_parameters.json', fire_parameters)
    files, hashes = validated[:2]
    extra = ready[5] + [SELECTION, HELPER, fire_source, REVIEW / 'engineering_proxy_evaluation_prespecification.md',
                       Path(__file__), HERE / 'audit_engineering_examples.py', HERE / 'caption_en.tex']
    hashes = {**hashes, **ready[6], **{str(p): sha(p) for p in extra if p not in ready[5]}}
    assert hashes == {str(p): sha(p) for p in dict.fromkeys(files + extra)}, 'Sources changed before snapshot; retry after review finishes.'
    snapshots = []
    for path in dict.fromkeys(files + extra):
        target = HERE / 'source_snapshot' / path.relative_to(WORKSPACE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert sha(target) == sha(path) == hashes[str(path)]
        snapshots.append({'original_path': str(path), 'snapshot_path': str(target.relative_to(HERE)), 'sha256': hashes[str(path)]})
    write(HERE / 'provenance.json', {'status': 'COMPLETE_REVIEWED_SELECTION_PENDING_VISUAL_QA',
        'selection_plan_sha256': SELECTION_SHA, 'helper_sha256': HELPER_SHA, 'script_sha256': sha(__file__),
        'source_snapshot': snapshots, 'prediction_and_checkpoint_sources': prediction_sources,
        'target_version': plan['target_version'], 'feature_version': plan['feature_version'],
        'prescribed_fire_source_sha256': sha(fire_source), 'population_cases': 997, 'prediction_archives_read': 100,
        'full_temperature_payload_read': False, 'training_or_inference_run': False,
        'selection_ranks_zero_based': [498, 946, 996], 'band_definition': 'Pointwise observed minimum/maximum over seeds42–46; no calibrated interval.',
        'selection_scope': selection['selection_scope'],
        'derived_artifact_sha256': {name: sha(HERE / name) for name in (
            'all_candidate_scores.csv', 'selection_score_components.csv', 'selected_cases.csv', 'curve_data.csv',
            'all_seed_curves.csv', 'threshold_events.csv', 'selected_fire_parameters.json')}})
    return pd.DataFrame(selected_rows), pd.DataFrame(curves), pd.DataFrame(events)


def render(selected, curves, events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': .7, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(183 / 25.4, 200 / 25.4))
    colors = {'reference': '#172C39', PRIMARY[0]: '#B05B42', 'gnn': '#0072B2'}
    fig.text(.04, .984, 'GNN-selected engineering response examples from complete fire-family holdouts', weight='bold', va='top')
    fig.legend(handles=[Line2D([], [], color=colors['reference'], label='Reference'),
        Line2D([], [], color=colors['gnn'], label='GNN seed42'),
        Line2D([], [], color=colors[PRIMARY[0]], ls='--', label='Matched set seed42'),
        Patch(facecolor='#C6D5DE', label='Five-seed min–max'),
        Line2D([], [], color='none', marker='x', markeredgecolor='#172C39', label='First grid crossing')],
        loc='upper center', bbox_to_anchor=(.5, .958), ncol=3, frameon=False, columnspacing=1.0, handlelength=1.4)
    response_columns = curves.filter(regex='reference|_seed42|_min$|_max$')
    raw_min, raw_max = float(response_columns.min().min()), float(response_columns.max().max())
    ymax = max(raw_max, 1.)
    ymin = min(raw_min, 0.)
    pad = .04 * max(ymax - ymin, .1)
    tmin, tmax = min(0., float(curves.prescribed_temperature_C.min())), float(curves.prescribed_temperature_C.max())
    for i, row in enumerate(selected.to_dict('records')):
        rr, cc = divmod(i, 2); left = .10 + .46 * cc; offset = .45 * rr
        subset = curves[curves.example == row['example']].sort_values('time_s'); x = subset.time_s.to_numpy() / 60
        title = ['a  GNN median case MAE', 'b  GNN 95th-percentile case MAE',
                 'c  GNN maximum mean case MAE', 'd  Largest remaining mean positive peak'][i]
        fig.text(left - .025, .883 - offset, title, weight='bold', va='top')
        family = row['fire_family'].replace('_', ' ')
        fig.text(left - .025, .859 - offset, f'{row["sample_id"]} · {family} · N = {row["crack_count"]}', va='top')
        status = ('restored' if row['restored'] else 'retained') + (' / flat' if row['flat_response'] else ' / nonflat')
        fig.text(left - .025, .838 - offset, f'{row["source_batch"].replace("batch_", "Source ")} · {status}', va='top')
        score_name = 'Mean +peak' if i == 3 else 'Mean case MAE'
        fig.text(left - .025, .817 - offset, f'Selection: {score_name} = {row["selection_score"] * 100:.3f} pp', va='top')
        gas = fig.add_axes([left, .747 - offset, .36, .050])
        gas.plot(x, subset.prescribed_temperature_C, color='#697D89', lw=1.0)
        gas.set(xlim=(0, 60), ylim=(tmin, tmax * 1.08), ylabel='Fire (°C)')
        gas.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2)); gas.set_yticks([t for t in gas.get_yticks() if tmin <= t <= tmax * 1.08])
        gas.tick_params(axis='x', bottom=False, labelbottom=False); gas.tick_params(axis='y', length=2)
        ax = fig.add_axes([left, .550 - offset, .36, .174])
        for model in PRIMARY:
            ax.fill_between(x, subset[model + '_min'], subset[model + '_max'], color=colors[model], alpha=.14, linewidth=0)
            ax.plot(x, subset[model + '_seed42'], color=colors[model], ls='--' if model == PRIMARY[0] else '-', lw=1.15)
        ax.plot(x, subset.reference, color=colors['reference'], lw=1.0)
        for q in (.8, .6): ax.axhline(q, color='#9CA9B1', ls=(0, (2, 2)), lw=.6, zorder=0)
        for event in events[(events.example == row['example']) & events.event].itertuples():
            ax.scatter([event.first_grid_time_s / 60], [event.response_at_event], marker='x', color=colors[event.series], s=20, linewidth=.8, zorder=5)
        ax.set(xlim=(0, 60), ylim=(ymin - pad, ymax + pad), xticks=[0, 30, 60], xlabel='Time (min)', ylabel='Archived response')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3)); ax.set_yticks([t for t in ax.get_yticks() if ymin - pad <= t <= ymax + pad])
        ax.tick_params(length=2)
        fig.text(left - .025, .491 - offset, f'Seed42 MAE: GNN {row["gnn_seed42_MAE"] * 100:.3f}; set {row[PRIMARY[0] + "_seed42_MAE"] * 100:.3f} pp', va='top')
        fig.text(left - .025, .470 - offset, f'Seed42 +peak: GNN {row["gnn_seed42_positive_peak"] * 100:.3f}; set {row[PRIMARY[0] + "_seed42_positive_peak"] * 100:.3f} pp', va='top')
        if row['no_remaining_positive_error']:
            ax.text(.03, .06, 'No remaining positive-error case', transform=ax.transAxes, fontsize=8)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); overflow = []; sizes = []
    for obj in fig.findobj(match=matplotlib.text.Text):
        if not obj.get_visible() or not obj.get_text(): continue
        sizes.append(obj.get_fontsize()); box = obj.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5 or box.y1 > fig.bbox.height + .5: overflow.append(obj.get_text())
    assert min(sizes) >= 8 and not overflow, overflow
    for ext in ('pdf', 'svg', 'png'): fig.savefig(HERE / ('engineering_examples_v08.' + ext), dpi=300, facecolor='white')
    plt.close(fig)
    write(HERE / 'figure_qa.json', {'status': 'GENERATED_PENDING_INDEPENDENT_NUMERICAL_AND_VISUAL_QA',
        'size_mm': [183, 200], 'minimum_font_pt': min(sizes), 'overflow': overflow, 'selected_case_count': 4,
        'raw_response_extrema': [raw_min, raw_max], 'response_axis_limits': [ymin - pad, ymax + pad],
        'response_clipped_to_unit_interval': False})


def main():
    parser = argparse.ArgumentParser(); group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--gate-only', action='store_true'); group.add_argument('--render', action='store_true')
    args = parser.parse_args(); fh = load_helper(); ready, gate = prerequisite_gate(fh)
    write(HERE / 'gate_status.json', gate)
    if ready is None:
        print(json.dumps(gate, indent=2)); return 2
    if args.gate_only:
        print(json.dumps(gate, indent=2)); return 0
    try:
        validated = fh.validate_sources(ready[1], ready[2], ready[3])
        loaded = open_predictions(fh, ready, validated)
        selected, curves, events = materialize(fh, ready, validated, loaded)
        render(selected, curves, events)
    except Exception as exc:
        write(HERE / 'generation_status.json', {'status': 'STOPPED_VALIDATION_OR_EXPORT_FAILURE', 'error': str(exc),
            'prediction_files_opened': NPZ_READS, 'journal_pass': False}); raise
    write(HERE / 'generation_status.json', {'status': 'GENERATED_FULL_REVIEWED_POPULATION_PENDING_QA',
        'prediction_files_opened': NPZ_READS, 'journal_pass': False})
    print(json.dumps({'status': 'GENERATED_FULL_REVIEWED_POPULATION_PENDING_QA', 'pdf': str(HERE / 'engineering_examples_v08.pdf')})); return 0


if __name__ == '__main__': sys.exit(main())
