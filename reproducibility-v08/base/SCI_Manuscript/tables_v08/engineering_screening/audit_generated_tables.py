"""Independent actual-table audit against completed source CIs; no generator import."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'generated'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(name):
    with (OUT / (name + '.csv')).open(encoding='utf-8', newline='') as stream: return list(csv.DictReader(stream))
def equals(actual, expected):
    if expected is None: assert actual == 'NA'
    elif isinstance(expected, bool): assert actual == str(expected)
    elif isinstance(expected, str): assert actual == expected
    else: assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-13), (actual, expected)
def formatted(x):
    value = f'{float(x) * 1000:.3f}'
    return '0.000' if value == '-0.000' else value


def main():
    if not (OUT / 'generation_manifest.json').exists():
        print('WAITING: no complete actual tables.'); return 2
    manifest = read(OUT / 'generation_manifest.json')
    assert manifest['status'] == 'TABLES_GENERATED_PENDING_INDEPENDENT_AND_LATEX_QA'
    for path, digest in manifest['source_sha256'].items(): assert sha(path) == digest
    for name, digest in manifest['generated_sha256'].items(): assert sha(OUT / name) == digest
    source_files = [Path(p) for p in manifest['source_sha256']]
    source = {}
    for protocol in manifest['protocols']:
        path = next(p for p in source_files if p.name == protocol + '_engineering_proxy_ci.json')
        ci = read(path); assert ci['status'] == 'COMPLETE_35_RUN_PAIRED_ENGINEERING_PROXY_CI'
        source[protocol] = ci, {(r['model'], r['seed']): r for r in ci['per_seed_event_counts_and_metrics']}
    scalar = rows('scalar_by_seed'); screening = rows('screening_by_seed')
    sm = rows('scalar_model_summary'); cm = rows('screening_model_summary'); effects = rows('matched_set_vs_sum_gnn_secondary_ci')
    assert list(map(len, (scalar, screening, sm, cm, effects))) == [420, 840, 84, 168, 24]
    for records, keys in [(scalar, ['protocol','model','seed']), (screening, ['protocol','model','seed','threshold']),
                          (sm, ['protocol','model']), (cm, ['protocol','model','threshold']), (effects, ['protocol','target'])]:
        assert len({tuple(r[k] for k in keys) for r in records}) == len(records)
    for row in scalar:
        ci, by = source[row['protocol']]; record = by[(row['model'], int(row['seed']))]
        equals(row['test_geometries'], ci['test_case_count'])
        for target in ('final','time_average'):
            for metric in ('MAE','bias'): equals(row[target + '_' + metric], record['summaries'][target][metric])
    for row in screening:
        ci, by = source[row['protocol']]; record = by[(row['model'], int(row['seed']))]['crossings'][row['threshold']]
        equals(row['test_geometries'], ci['test_case_count'])
        for key, value in record.items(): equals(row[key], value)
    for row in sm:
        ci, by = source[row['protocol']]
        equals(row['test_geometries'], ci['test_case_count']); equals(row['seeds'], 5)
        for target in ('final','time_average'):
            for metric in ('MAE','bias'):
                values = [by[(row['model'], s)]['summaries'][target][metric] for s in manifest['seeds']]
                equals(row[target+'_'+metric+'_mean_over_seeds'], float(np.mean(values)))
                equals(row[target+'_'+metric+'_seed_SD'], float(np.std(values, ddof=1)))
    for row in cm:
        ci, by = source[row['protocol']]; values = [by[(row['model'],s)]['crossings'][row['threshold']] for s in manifest['seeds']]
        equals(row['test_geometries'], ci['test_case_count']); equals(row['seeds'], 5)
        e, cens = values[0]['reference_events'], values[0]['reference_censored']
        equals(row['reference_events'], e); equals(row['reference_censored'], cens)
        common = min(x['both_event_count'] for x in values)
        for key, value in [('max_false_negative_over_seeds', max(x['false_negative'] for x in values)),
                           ('max_false_positive_over_seeds', max(x['false_positive'] for x in values)),
                           ('min_both_event_count_over_seeds', common),
                           ('min_both_event_reference_event_fraction_over_seeds', common/e if e else None)]: equals(row[key], value)
        assert row['seed_extrema_can_come_from_different_checkpoints'] == 'True'
        for key in ('false_negative','false_positive','both_event_count','recall','false_positive_rate','balanced_accuracy','conditional_timing_MAE_s','conditional_timing_bias_s'):
            vv = [v[key] for v in values]
            equals(row[key+'_defined_seed_count'], sum(v is not None for v in vv))
            equals(row[key+'_mean_over_all5_seeds'], float(np.mean(vv)) if all(v is not None for v in vv) else None)
    for row in effects:
        ci, by = source[row['protocol']]; target = row['target']; contrast = ci['paired_contrasts']['graph_vs_capacity_matched']['metrics'][target+'_MAE']
        point = np.mean([by[('capacity_matched_deepsets',s)]['summaries'][target]['MAE'] - by[('gnn',s)]['summaries'][target]['MAE'] for s in manifest['seeds']])
        equals(row['baseline_minus_candidate'], float(point))
        equals(row['secondary_CI95_low'], contrast['percentile_95_paired_difference'][0]); equals(row['secondary_CI95_high'], contrast['percentile_95_paired_difference'][1])
        assert row['secondary_bootstrap_repetitions'] == '1000' and row['positive_favors_sum_GNN'] == 'True'
    labels = ['IID','Count OOD','ISO 834','ASTM E119','Perturbed ISO','Plateau','Decay','Log variant','External fire','Bilinear','Linear','Smoldering']
    latex = (OUT / 'scalar_effects.tex').read_text(encoding='utf-8').splitlines()
    for protocol, label in zip(manifest['protocols'], labels):
        line = next(x for x in latex if x.startswith(label + ' &'))
        assert line.split('&')[1].strip() == str(source[protocol][0]['test_case_count'])
        expected = [r for r in effects if r['protocol'] == protocol]
        for target in ('final','time_average'):
            row = next(r for r in expected if r['target'] == target)
            block = formatted(row['baseline_minus_candidate']) + r'\;[' + formatted(row['secondary_CI95_low']) + ',' + formatted(row['secondary_CI95_high']) + ']'
            assert block in line
    body = [x for x in (OUT/'threshold_screening.tex').read_text(encoding='utf-8').splitlines() if '& 0.8 &' in x or '& 0.6 &' in x]
    assert len(body) == 24
    for i, protocol in enumerate(manifest['protocols']):
        for j, q in enumerate(['0.8','0.6']):
            line = body[2*i+j]; cells = [x.strip() for x in line.removesuffix(r'\\').split('&')]
            assert cells[0] == (labels[i] if j == 0 else '')
            expected = [next(r for r in cm if r['protocol']==protocol and r['model']==m and r['threshold']==q) for m in ['capacity_matched_deepsets','gnn']]
            e = int(expected[0]['reference_events']); cens = int(expected[0]['reference_censored'])
            assert cells[1:3] == [q, f'{e}/{cens}']
            for k, row in enumerate(expected):
                assert cells[3+2*k] == '$'+row['max_false_negative_over_seeds']+'/'+row['max_false_positive_over_seeds']+'$'
                assert cells[4+2*k] == ('$'+row['min_both_event_count_over_seeds']+'/'+str(e)+'$' if e else r'$\mathrm{NA}$')
    result = {'status':'PASSED_ACTUAL_TABLE_ARITHMETIC_AND_LATEX_VALUES_PENDING_VISUAL_QA', 'CSV_rows_checked':[420,840,84,168,24],
              'LaTeX_table_rows_checked':[12,24], 'scale_factor':1000, 'NA_preserved':True,
              'reference_counts_not_multiplied_by_seeds':True, 'seed_extrema_not_one_checkpoint':True,
              'prediction_arrays_read':0, 'bootstrap_recomputed':False, 'visual_QA_passed':False,
              'generated_file_hashes':manifest['generated_sha256']}
    (OUT/'independent_arithmetic_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2)); return 0


if __name__ == '__main__': sys.exit(main())
