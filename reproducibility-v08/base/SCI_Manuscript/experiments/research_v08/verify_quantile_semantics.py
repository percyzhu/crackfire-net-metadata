"""Known-answer software check: per-case and case-time quantiles differ."""
import numpy as np
import aggregate_results as ar

y = np.zeros((3, 2), dtype=np.float64)
p = np.array([[.1, 0], [.3, 0], [.9, 0]], dtype=np.float64)
m = ar.scalar_metrics(y, p)
expected = {'per_case_MAE_q95': .42, 'per_case_max_positive_error_q95': .84,
            'case_time_positive_error_q95': .75, 'equal_case_MAE': .65 / 3}
for key, value in expected.items():
    assert abs(m[key] - value) < 1e-12, (key, m[key], value)
report = {'purpose': 'SOFTWARE_ONLY_KNOWN_ANSWER_NO_FEM_OR_GNN_PERFORMANCE',
    'status': 'PASSED', 'truth_shape': [3, 2], 'per_case_MAE': [.05, .15, .45],
    'per_case_max_positive': [.1, .3, .9], 'case_time_positive': [.1, 0, .3, 0, .9, 0],
    'independent_expected_values': expected, 'tolerance': 1e-12,
    'source_sha256': ar.rr.digest(ar.__file__)}
ar.rr.write_json(ar.rr.HERE / 'quantile_semantics_software_check.json', report)
print('PASSED per-case MAE/max-positive versus case-time quantile distinction')
