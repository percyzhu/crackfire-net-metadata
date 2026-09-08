"""Exercise the real incomplete-matrix CLI while forbidding all performance work."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import contextlib
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import aggregate_extension as a

HERE = Path(__file__).resolve().parent
EXPECTED_SHA = 'aa06a3280fed95be571f1b39677c84c20d193804897d4f4d55ec7a99053a2e7a'


def main():
    candidate = HERE/'aggregate_extension.py'
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert actual == EXPECTED_SHA
    counts = {name: 0 for name in ('result_payload_reads', 'numpy_array_loads', 'torch_tensor_loads',
                                    'model_constructions', 'prediction_calls', 'dataset_loads')}
    original_read = a.run.read
    def guarded_read(path):
        if Path(path).name == 'result.json':
            counts['result_payload_reads'] += 1
            raise AssertionError('Performance result payload reached before all60')
        return original_read(path)
    def forbidden(name):
        def fail(*args, **kwargs):
            counts[name] += 1
            raise AssertionError('Forbidden incomplete-matrix operation: '+name)
        return fail
    a.run.read = guarded_read
    a.np.load = forbidden('numpy_array_loads')
    a.torch.load = forbidden('torch_tensor_loads')
    a.SetAttentionSurrogate = forbidden('model_constructions')
    a.run.rr.predict = forbidden('prediction_calls')
    a.run.rr.load_dataset = forbidden('dataset_loads')
    impossible_output = HERE/'independent_probe_must_not_create_output'
    assert not impossible_output.exists()
    # Supplying replay/authorization/output ensures this checks the real main
    # gate, not just the --check-ready convenience return branch.
    sys.argv = [str(candidate), '--replay', '--plan', str(HERE/'plan_set_attention_60_v1.json'),
                '--authorization', str(HERE/'execution_authorization_root.json'), '--output', str(impossible_output)]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exit_code = a.main()
    gate = json.loads(buffer.getvalue())
    assert exit_code == 2 and gate['status'] == 'WAITING_FOR_ALL60_COMPLETE'
    assert gate['completed_metadata_runs'] < 60 and all(v == 0 for v in counts.values())
    assert not impossible_output.exists()
    report = {'status': 'REAL_INCOMPLETE_MATRIX_MAIN_GUARD_PASSED_WITH_PERFORMANCE_PROBES',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'candidate_sha256': actual,
              'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'real_main_returncode': exit_code, 'supplied_replay_authorization_and_output': True,
              'prohibited_calls_during_main': counts, 'output_directory_created': False,
              'returned_gate': gate, 'full60_statistics_or_replay_executed': False,
              'submission_pass': False}
    with (HERE/'independent_missing_matrix_probe_evidence.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k: report[k] for k in ('status', 'real_main_returncode', 'prohibited_calls_during_main',
                                           'output_directory_created')}, indent=2))


if __name__ == '__main__':
    main()
