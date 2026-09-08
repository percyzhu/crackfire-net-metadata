"""Two compact engineering tables; no score payload before all12 review gates.

Only saved, independently checked engineering metrics are read. This program
does not train, predict, rebootstrap, modify labels, or measure runtime.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import math
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCI = HERE.parents[1]
WORKSPACE = SCI.parent
EXP = SCI / 'experiments/research_v08'
EVAL = EXP / 'evaluation'
REVIEW = SCI / 'review/eaai_editor/research_v08'
PLAN = EXP / 'comparison_plan_v08_420.json'
PLAN_SHA = 'fd9088427cbc9234191b71e619626b188a98c43c9af248c9dd3e435afdb8f068'
PROTOCOLS = ['iid997', 'lcro_9_15', 'loco_iso834', 'loco_astm_e119',
             'loco_perturbed_iso', 'loco_plateau', 'loco_decay', 'loco_log_variant',
             'loco_external_fire', 'loco_bilinear', 'loco_linear', 'loco_smoldering']
LABELS = ['IID', 'Count OOD', 'ISO 834', 'ASTM E119', 'Perturbed ISO', 'Plateau',
          'Decay', 'Log variant', 'External fire', 'Bilinear', 'Linear', 'Smoldering']
MODELS = ['fire_only', 'global_stats', 'deepsets', 'capacity_matched_deepsets',
          'gnn', 'gnn_zero_edge_features', 'gnn_mean']
PRIMARY = ['capacity_matched_deepsets', 'gnn']
SEEDS = [42, 43, 44, 45, 46]
THRESHOLDS = [.8, .6]
READ_LOG = []


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, role='metadata'):
    READ_LOG.append({'path': str(Path(path).resolve()), 'role': role})
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def score_reads(): return sum(x['role'] == 'score_payload' for x in READ_LOG)


def review_path(protocol):
    suffix = '_independent_results_audit.json' if protocol == 'iid997' else '_independent_results_audit_v2.json'
    return REVIEW / (protocol + suffix)


def metadata_header(path, stop_key):
    """Read pretty-printed top-level metadata only, stopping before metrics.

    A compact/reordered future source format fails closed rather than falling
    back to reading its performance payload before all reviews qualify.
    """
    READ_LOG.append({'path': str(Path(path).resolve()), 'role': 'review_metadata_header'})
    lines = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.startswith('  "' + stop_key + '":'):
                return json.loads(''.join(lines).rstrip().rstrip(',') + '\n}')
            if len(line) > 4096:
                raise ValueError('Unsupported metadata-header layout: ' + str(path))
            lines.append(line)
    raise ValueError('Expected metadata boundary missing: ' + str(path))


def prerequisites():
    contract = read(HERE / 'contract.json')
    assert contract['protocols'] == PROTOCOLS and contract['models'] == MODELS
    assert contract['seeds'] == SEEDS and contract['thresholds'] == THRESHOLDS
    assert sha(PLAN) == PLAN_SHA == contract['training_plan_sha256']
    plan = read(PLAN)
    assert plan['status'] == 'FROZEN_BEFORE_TRAINING' and plan['models'] == MODELS and plan['seeds'] == SEEDS
    assert list(plan['protocols']) == PROTOCOLS
    report_path = EVAL / 'report.json'
    report = read(report_path) if report_path.exists() else {}
    missing, statuses = [], []
    for protocol in PROTOCOLS:
        item = report.get('protocols', {}).get(protocol, {})
        paths = [EVAL / protocol / 'summary.json', review_path(protocol),
                 REVIEW / (protocol + '_results_development_review_zh.md'),
                 REVIEW / (protocol + '_engineering_proxy_ci.json'),
                 REVIEW / (protocol + '_engineering_proxy_ci.csv')]
        absent = [str(p) for p in paths if not p.exists()]
        required_results = [Path(plan['runs_root']) / protocol / model / ('seed_' + str(seed)) / 'result.json'
                            for model in MODELS for seed in SEEDS]
        actual_count = sum(p.exists() for p in required_results)
        complete = (item.get('status') == 'COMPLETE' and item.get('audited_run_count') == 35 and
                    item.get('expected_run_count') == 35 and item.get('missing_cells') == [] and
                    item.get('scientific_summary_published') is True and actual_count == 35)
        statuses.append({'protocol': protocol, 'complete_audited_runs': item.get('audited_run_count', 0),
                         'actual_result_file_count': actual_count, 'complete_35': complete,
                         'required_files_present': not absent})
        if not complete: missing.append(protocol + ': complete35 evaluation absent')
        missing.extend(absent)
    if (report.get('status') != 'COMPLETE_420_RUN_MATRIX' or report.get('audited_completed_runs') != 420 or
        report.get('cpu_prediction_replay_enabled') is not True or not (EVAL / 'run_integrity.json').exists()):
        missing.append('Final complete420 computational replay required')
    gate = {'status': 'WAITING_FOR_ALL12_COMPLETE_PROTOCOLS_CI_AND_INDEPENDENT_REVIEWS',
            'required_protocols': 12, 'required_runs': 420, 'protocol_statuses': statuses,
            'complete_audited_protocols': sum(x['complete_35'] for x in statuses),
            'missing_requirements': missing, 'score_payloads_read': score_reads(),
            'review_metadata_headers_read': 0, 'prediction_files_opened': 0,
            'formal_tables_created': False, 'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if missing: return None, gate
    assert report['plan_sha256'] == PLAN_SHA and report['purpose'] == plan['purpose']
    for protocol in PROTOCOLS:
        ci = metadata_header(REVIEW / (protocol + '_engineering_proxy_ci.json'), 'model_seed_summaries')
        review = metadata_header(review_path(protocol), 'models')
        expected_status = ('IID_ARITHMETIC_DEVELOPMENT_AUDIT_PASSED_NOT_FULL_MATRIX_OR_SUBMISSION_REVIEW'
                           if protocol == 'iid997' else
                           'COMPLETE_PROTOCOL_ARITHMETIC_DEVELOPMENT_AUDIT_PASSED_NOT_FULL_MATRIX_OR_SUBMISSION_REVIEW')
        assert ci['status'] == 'COMPLETE_35_RUN_PAIRED_ENGINEERING_PROXY_CI' and review['status'] == expected_status
        assert ci['protocol'] == protocol and review.get('protocol', protocol) == protocol
        assert ci['completed_runs_checked'] == review['runs'] == 35
        assert ci['seeds'] == review['seeds'] == SEEDS and ci['thresholds'] == THRESHOLDS
        assert ci['test_case_count'] == ci['unique_geometry_groups'] == review['independent_geometries']
        assert review['case_time_grid'] == 61 and review['per_case_CSV_rows_checked'] == 35 * ci['test_case_count']
        assert ci['secondary_bootstrap_repetitions'] == 1000 and ci['primary_MAE_bootstrap_repetitions_separately_implemented'] == 5000
        assert '4 final/J paired CIs independently recomputed' in review['secondary_checks']
    gate.update(status='ALL12_RUN_CI_REVIEW_METADATA_READY_FOR_SOURCE_BINDING', review_metadata_headers_read=24)
    assert score_reads() == 0
    return (plan, report, contract), gate


def replay_ok(r):
    assert r and r['cpu_diagnostic_tolerance'] == 3e-6
    if r['cpu_within_original_tolerance']:
        assert r['status'] == 'CPU_REPLAY_WITHIN_ORIGINAL_TOLERANCE'
        assert 0 <= r['cpu_max_abs_difference'] < 3e-6
    else:
        assert r['status'] == 'GPU_IDENTITY_EXACT_CPU_BACKEND_VARIATION_RECORDED'
        assert r['cpu_max_abs_difference'] >= 3e-6
        assert r['same_backend_gpu_bitwise_equal'] is True and r['gpu_max_abs_difference'] == 0


def path_hash(hashes, path):
    found = [h for p, h in hashes.items() if Path(p).resolve() == Path(path).resolve()]
    assert len(found) == 1
    return found[0]


def bind_and_load(ready):
    plan, report, contract = ready
    sources = [PLAN, EVAL / 'report.json', EVAL / 'run_integrity.json', HERE / 'contract.json', HERE / 'captions.tex', Path(__file__)]
    manifest_path = Path(plan['manifest_path']); assert sha(manifest_path) == plan['manifest_sha256']
    manifest = read(manifest_path); sources.append(manifest_path)
    by_id = {c['sample_id']: c for c in manifest['cases']}
    assert len(by_id) == 997
    for relative, digest in plan['source_sha256'].items():
        path = WORKSPACE / relative; assert sha(path) == digest; sources.append(path)
    for name, path in [('aggregate_results.py', EXP / 'aggregate_results.py'),
                       ('evaluation_stats.py', SCI / 'experiments/evaluation/evaluation_stats.py')]:
        assert sha(path) == report['audit_source_sha256'][name]; sources.append(path)
    integrity = read(EVAL / 'run_integrity.json')
    audit = {(x['protocol'], x['model'], x['seed']): x for x in integrity['audits']}
    expected = {(p,m,s) for p in PROTOCOLS for m in MODELS for s in SEEDS}
    assert len(integrity['audits']) == len(audit) == 420 and set(audit) == expected
    loaded, reference = {}, {}
    for protocol in PROTOCOLS:
        ci_path = REVIEW / (protocol + '_engineering_proxy_ci.json')
        rp = review_path(protocol); sp = Path(plan['protocols'][protocol]['split_path'])
        summary_path = EVAL / protocol / 'summary.json'
        csv_path = ci_path.with_suffix('.csv')
        assert sha(sp) == plan['protocols'][protocol]['split_sha256']
        split = read(sp); ids = split['test']
        assert split['dataset_sha256'] == plan['manifest_sha256']
        assert len(ids) == len(set(ids)) == len({by_id[s]['geometry_id'] for s in ids})
        ci, review = read(ci_path, 'score_payload'), read(rp, 'score_payload')
        assert ci['plan_sha256'] == path_hash(review['source_hashes'], PLAN) == PLAN_SHA
        assert ci['manifest_sha256'] == plan['manifest_sha256']
        assert path_hash(review['source_hashes'], ci_path) == sha(ci_path)
        assert path_hash(review['source_hashes'], summary_path) == sha(summary_path)
        assert ci['identities']['sample_ids'] == ids
        assert ci['identities']['geometry_ids'] == [by_id[s]['geometry_id'] for s in ids]
        assert ci['identities']['families'] == [by_id[s]['fire_family'] for s in ids]
        assert ci['test_case_count'] == len(ids) and len(ci['per_seed_event_counts_and_metrics']) == 35
        records = {(x['model'],x['seed']): x for x in ci['per_seed_event_counts_and_metrics']}
        assert len(records) == 35 and set(records) == {(m,s) for m in MODELS for s in SEEDS}
        assert len(ci['source_prediction_sha256']) == len(review['prediction_hashes']) == 35
        assert ci['source_prediction_sha256'] == review['prediction_hashes']
        for path, digest in ci['code_sha256'].items(): assert sha(path) == digest; sources.append(Path(path))
        ep = REVIEW / 'engineering_proxy_evaluation_prespecification.md'
        assert sha(ep) == ci['prespecification_sha256']; sources.append(ep)
        for model in MODELS:
            for seed in SEEDS:
                folder = Path(plan['runs_root']) / protocol / model / ('seed_' + str(seed))
                pp, wp = folder / 'test_predictions.npz', folder / 'best.pt'
                a = audit[(protocol,model,seed)]
                assert a['status'] == 'PASSED_BINDINGS' and a['ordered_ids_truth_times_exact'] is True
                assert a['plan_dataset_feature_target_split_checkpoint_hashes'] is True
                replay_ok(a['computational_replay'])
                assert sha(pp) == a['predictions_sha256'] == path_hash(ci['source_prediction_sha256'], pp)
                assert sha(wp) == a['checkpoint_sha256']
                sources.extend([pp, wp])
                assert records[(model,seed)]['cases'] == len(ids)
                assert records[(model,seed)]['grid_points'] == 61 and records[(model,seed)]['horizon_s'] == 3600
        # The independently reviewed CSV must also agree with the exact CI JSON.
        with csv_path.open(encoding='utf-8', newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 48
        for metric in ('final_MAE','time_average_MAE'):
            v = ci['paired_contrasts']['graph_vs_capacity_matched']['metrics'][metric]
            row = [x for x in csv_rows if x['contrast']=='graph_vs_capacity_matched' and x['metric']==metric]
            assert len(row)==1
            for name, value in [('effect',v['point_baseline_minus_candidate']),
                                ('ci95_low',v['percentile_95_paired_difference'][0]),
                                ('ci95_high',v['percentile_95_paired_difference'][1])]:
                close(float(row[0][name]), value)
        sources.extend([ci_path, csv_path, rp, sp, summary_path, REVIEW / (protocol + '_results_development_review_zh.md')])
        loaded[protocol] = (ci, records)
        reference[protocol] = len(ids)
    # Reading/hashing an NPZ binds provenance; no prediction arrays are opened.
    hashes = {str(path.resolve()): sha(path) for path in sources}
    return loaded, reference, hashes


def close(a,b):
    assert a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a), float(b), abs_tol=1e-13, rel_tol=1e-13), (a,b)


def valid_crossing(c,n):
    e, cens = c['reference_events'], c['reference_censored']
    tp, fn, fp, tn = [c[k] for k in ('true_positive','false_negative','false_positive','true_negative')]
    assert all(isinstance(x,int) and 0 <= x <= n for x in (e,cens,tp,fn,fp,tn))
    assert e+cens==n and tp+fn==e and fp+tn==cens
    assert c['predicted_events']==tp+fp and c['predicted_censored']==fn+tn
    assert c['both_event_count']==tp
    for key, value in [('recall',tp/e if e else None), ('false_positive_rate',fp/cens if cens else None),
                       ('balanced_accuracy', .5*(tp/e+tn/cens) if e and cens else None),
                       ('both_event_reference_event_fraction',tp/e if e else None), ('both_event_population_fraction',tp/n)]:
        if value is None: assert c[key] is None
        else: close(c[key],value)
    for key in ('conditional_timing_MAE_s','conditional_timing_median_abs_s','conditional_timing_bias_s'):
        if tp==0: assert c[key] is None
        else: assert c[key] is not None and math.isfinite(c[key])


def all_seed_mean(values):
    return float(np.mean(values)) if all(v is not None for v in values) else None


def derive(loaded, populations):
    scalar, screening, scalar_summary, screening_summary, effects = [], [], [], [], []
    for protocol in PROTOCOLS:
        ci, records = loaded[protocol]; n = populations[protocol]
        for model in MODELS:
            ss = {'protocol':protocol,'model':model,'test_geometries':n,'seeds':5}
            for target in ('final','time_average'):
                for metric in ('MAE','bias'):
                    values = [records[(model,s)]['summaries'][target][metric] for s in SEEDS]
                    assert all(v is not None and math.isfinite(v) for v in values)
                    ss[target+'_'+metric+'_mean_over_seeds'] = float(np.mean(values))
                    ss[target+'_'+metric+'_seed_SD'] = float(np.std(values,ddof=1))
            scalar_summary.append(ss)
            for seed in SEEDS:
                r = records[(model,seed)]
                scalar.append({'protocol':protocol,'model':model,'seed':seed,'test_geometries':n,
                               **{t+'_'+m:r['summaries'][t][m] for t in ('final','time_average') for m in ('MAE','bias')}})
                for q in THRESHOLDS:
                    c = r['crossings'][str(q)]; valid_crossing(c,n)
                    screening.append({'protocol':protocol,'model':model,'seed':seed,'test_geometries':n,**c})
            for q in THRESHOLDS:
                cs = [records[(model,s)]['crossings'][str(q)] for s in SEEDS]
                e, cens = cs[0]['reference_events'], cs[0]['reference_censored']
                assert all(c['reference_events']==e and c['reference_censored']==cens for c in cs)
                assert all(records[(m,s)]['crossings'][str(q)]['reference_events']==e for m in MODELS for s in SEEDS)
                row = {'protocol':protocol,'model':model,'threshold':q,'test_geometries':n,'seeds':5,
                       'reference_events':e,'reference_censored':cens,
                       'max_false_negative_over_seeds':max(c['false_negative'] for c in cs),
                       'max_false_positive_over_seeds':max(c['false_positive'] for c in cs),
                       'min_both_event_count_over_seeds':min(c['both_event_count'] for c in cs),
                       'min_both_event_reference_event_fraction_over_seeds':min(c['both_event_count'] for c in cs)/e if e else None,
                       'seed_extrema_can_come_from_different_checkpoints':True}
                for key in ('false_negative','false_positive','both_event_count','recall','false_positive_rate',
                            'balanced_accuracy','conditional_timing_MAE_s','conditional_timing_bias_s'):
                    vv = [c[key] for c in cs]
                    row[key+'_mean_over_all5_seeds'] = all_seed_mean(vv)
                    row[key+'_defined_seed_count'] = sum(v is not None for v in vv)
                screening_summary.append(row)
        contrast = ci['paired_contrasts']['graph_vs_capacity_matched']
        assert [contrast['baseline'],contrast['candidate']]==PRIMARY
        for target in ('final','time_average'):
            metric = target+'_MAE'; v = contrast['metrics'][metric]
            point = np.mean([records[(PRIMARY[0],s)]['summaries'][target]['MAE']-
                             records[(PRIMARY[1],s)]['summaries'][target]['MAE'] for s in SEEDS])
            assert v['status']=='ESTIMATED' and v['point_uses_all_paired_seeds'] is True
            assert v['baseline_defined_seeds']==v['candidate_defined_seeds']==v['required_seed_count']==5
            assert v['bootstrap_repetitions']==1000 and v['common_valid_bootstrap_replicates']==1000
            close(point,v['point_baseline_minus_candidate'])
            lower, upper = v['percentile_95_paired_difference']
            assert math.isfinite(lower) and math.isfinite(upper) and lower<=upper
            effects.append({'protocol':protocol,'target':target,'test_geometries':n,'seeds':5,
                            'baseline':PRIMARY[0],'candidate':PRIMARY[1],'baseline_minus_candidate':float(point),
                            'secondary_CI95_low':lower,'secondary_CI95_high':upper,
                            'secondary_bootstrap_repetitions':1000,'multiplicity_adjusted':False,
                            'positive_favors_sum_GNN':True})
    assert tuple(map(len,(scalar,screening,scalar_summary,screening_summary,effects)))==(420,840,84,168,24)
    return scalar,screening,scalar_summary,screening_summary,effects


def csv_write(path,rows):
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k:'NA' if v is None else v for k,v in row.items()} for row in rows)


def numeric(value):
    result=f'{value*1000:.3f}'
    return '0.000' if result=='-0.000' else result


def render_tables(target,populations,screening_summary,effects):
    effect={(r['protocol'],r['target']):r for r in effects}
    screen={(r['protocol'],r['model'],r['threshold']):r for r in screening_summary}
    a=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{\EngineeringScalarCaption}',
       r'\label{tab:engineering-secondary}',r'\setlength{\tabcolsep}{4pt}',
       r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}',r'\toprule',
       r'Protocol & $N$ & $\Delta\mathrm{MAE}_{60}$ [95\% CI] & $\Delta\mathrm{MAE}_{J}$ [95\% CI] \\',r'\midrule']
    b=[r'\begin{table}[!htbp]',r'\centering\small',r'\caption{\EngineeringScreeningCaption}',
       r'\label{tab:engineering-screening}',r'\setlength{\tabcolsep}{3pt}',
       r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',r'\toprule',
       r'& & & \multicolumn{2}{c}{Matched set} & \multicolumn{2}{c}{Sum GNN} \\',
       r'\cmidrule(lr){4-5}\cmidrule(l){6-7}',
       r'Protocol & $q$ & $E/C$ & Max FN/FP & Min $n_{\mathrm{com}}/E$ & Max FN/FP & Min $n_{\mathrm{com}}/E$ \\',r'\midrule']
    for protocol,label in zip(PROTOCOLS,LABELS):
        vals=[]
        for name in ('final','time_average'):
            r=effect[(protocol,name)]
            vals.append('$'+numeric(r['baseline_minus_candidate'])+r'\;['+numeric(r['secondary_CI95_low'])+','+numeric(r['secondary_CI95_high'])+']$')
        a.append(f'{label} & {populations[protocol]} & '+' & '.join(vals)+r' \\')
        for qi,q in enumerate(THRESHOLDS):
            rows=[screen[(protocol,m,q)] for m in PRIMARY]
            e,c=rows[0]['reference_events'],rows[0]['reference_censored']
            assert all(r['reference_events']==e and r['reference_censored']==c for r in rows)
            cells=[]
            for r in rows:
                cells.append(f"{r['max_false_negative_over_seeds']}/{r['max_false_positive_over_seeds']}")
                cells.append(f"{r['min_both_event_count_over_seeds']}/{e}" if e else r'\mathrm{NA}')
            b.append((label if qi==0 else '')+f' & {q:.1f} & {e}/{c} & '+ ' & '.join('$'+x+'$' for x in cells)+r' \\')
    ending=[r'\bottomrule',r'\end{tabular*}',r'\end{table}']
    (target/'scalar_effects.tex').write_text('\n'.join(a+ending)+'\n',encoding='utf-8')
    (target/'threshold_screening.tex').write_text('\n'.join(b+ending)+'\n',encoding='utf-8')
    (target/'captions.tex').write_text((HERE/'captions.tex').read_text(encoding='utf-8'),encoding='utf-8')
    preview=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage{booktabs,amsmath}
\usepackage[font=small,labelfont=bf]{caption}
\input{captions}
\begin{document}
\input{scalar_effects}
\input{threshold_screening}
\end{document}
'''
    (target/'preview.tex').write_text(preview,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--generate',action='store_true')
    args=parser.parse_args()
    ready,gate=prerequisites(); write(HERE/'gate_status.json',gate)
    if ready is None:
        print(json.dumps(gate,indent=2)); return 2
    if not args.generate:
        print(json.dumps(gate,indent=2)); return 0
    target=HERE/'generated'
    if target.exists(): raise ValueError('Preserve previous generated tables; use an explicit new version')
    loaded,populations,hashes=bind_and_load(ready)
    scalar,screening,sm,cm,effects=derive(loaded,populations)
    assert hashes=={p:sha(p) for p in hashes}, 'Source changed during table derivation'
    target.mkdir()
    try:
        for name,rows in [('scalar_by_seed',scalar),('screening_by_seed',screening),
                          ('scalar_model_summary',sm),('screening_model_summary',cm),
                          ('matched_set_vs_sum_gnn_secondary_ci',effects)]:
            csv_write(target/(name+'.csv'),rows)
        render_tables(target,populations,cm,effects)
        report={'status':'TABLES_GENERATED_PENDING_INDEPENDENT_AND_LATEX_QA','protocols':PROTOCOLS,
                'models_in_companion_CSV':MODELS,'seeds':SEEDS,'table_row_counts':[12,24],
                'CSV_row_counts':[420,840,84,168,24],'source_sha256':hashes,
                'generated_sha256':{p.name:sha(p) for p in target.iterdir() if p.is_file()},
                'score_payloads_read':score_reads(),'prediction_arrays_opened':0,
                'source_only_prediction_file_hash_checks':420,'ranking_metrics_exported':False,
                'NA_policy':'Undefined rates/coverage/conditional time remain NA; never zero-filled.',
                'precision_scope':'No ranking rho exported; archived tiny numerical differences do not create geometry information in fire-only inputs.',
                'latex_compilation_verified':False,'full_data_branch_requires_independent_check':True,
                'journal_or_submission_pass':False}
        write(target/'generation_manifest.json',report)
        print(json.dumps({'status':report['status'],'table_row_counts':[12,24],'CSV_row_counts':report['CSV_row_counts']}))
    except BaseException as exc:
        write(target/'failure.json',{'status':'FAILED_RETAIN_PARTIAL_DO_NOT_USE','error':str(exc)})
        raise
    return 0


if __name__=='__main__': raise SystemExit(main())
