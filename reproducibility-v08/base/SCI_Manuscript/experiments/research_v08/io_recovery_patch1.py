"""Transport-only recovery for Windows sharing violations during JSON replace.

The frozen model, optimizer, loader, seeds, epochs and split files are unchanged.
This wrapper supplies retrying atomic JSON writes and records its own hash in
each new run. It never retries scientific training failures automatically.
"""
import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import research_runner as rr

HERE = rr.HERE
EVIDENCE = HERE / 'recovery_io_patch1'
PATCH_ID = 'windows-atomic-json-sharing-retry-v1'


def robust_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == 'run_metadata.json':
        value = dict(value, runtime_io_patch={'id': PATCH_ID, 'path': str(Path(__file__).resolve()),
            'sha256': rr.digest(__file__), 'amendment_path': str(EVIDENCE / 'amendment.json'),
            'amendment_sha256': rr.digest(EVIDENCE / 'amendment.json'),
            'scope': 'Atomic JSON write retry only; no numerical operation changed.'})
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    temp = path.with_name(path.name + f'.{os.getpid()}.atomic.tmp')
    start = time.monotonic()
    retries = 0
    while True:
        try:
            temp.write_text(payload, encoding='utf-8')
            temp.replace(path)
            break
        except PermissionError:
            retries += 1
            if time.monotonic() - start > 10:
                raise
            time.sleep(min(.01 * retries, .25))
    if retries:
        EVIDENCE.mkdir(exist_ok=True)
        with (EVIDENCE / 'sharing_retry_events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'target': str(path), 'retries': retries,
                'elapsed_s': time.monotonic() - start, 'pid': os.getpid(), 'status': 'SAME_JSON_WRITE_RECOVERED'}) + '\n')


def test():
    EVIDENCE.mkdir(exist_ok=True)
    target = EVIDENCE / 'windows_sharing_fixture.json'
    target.write_text('{"old": true}', encoding='utf-8')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                                  ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateFileW(str(target), 0x80000000, 1, None, 3, 0x80, None)
    if handle in (None, ctypes.c_void_p(-1).value): raise ctypes.WinError(ctypes.get_last_error())
    def release():
        time.sleep(.20)
        kernel.CloseHandle(handle)
    thread = threading.Thread(target=release); thread.start()
    start = time.monotonic()
    robust_json(target, {'new': True, 'numerical_updates': 0})
    thread.join()
    assert json.loads(target.read_text(encoding='utf-8')) == {'new': True, 'numerical_updates': 0}
    assert time.monotonic() - start >= .19
    robust_json(EVIDENCE / 'software_test.json', {'status': 'PASSED_REAL_WINDOWS_SHARING_VIOLATION',
        'held_target_handle_without_delete_sharing_s': .20, 'replacement_retried_same_bytes': True,
        'optimizer_updates': 0, 'patch_sha256': rr.digest(__file__)})
    print('PASSED real sharing-violation atomic-write recovery')


def check_prefix(path):
    prior = EVIDENCE / 'original_failed_seed44/learning_curve.csv'
    current = Path(path) / 'learning_curve.csv'
    old = list(csv.DictReader(prior.open(encoding='utf-8')))
    new = list(csv.DictReader(current.open(encoding='utf-8')))
    keys = ('epoch', 'train_mse', 'validation_mse', 'learning_rate')
    assert len(new) >= len(old)
    differences = []
    for a, b in zip(old, new):
        for key in keys:
            if a[key] != b[key]: differences.append({'epoch': a['epoch'], 'key': key, 'old': a[key], 'new': b[key]})
    report = {'status': 'IDENTICAL_NUMERICAL_TRAINING_PREFIX' if not differences else 'PREFIX_DIFFERENCE',
        'prior_epochs': len(old), 'compared_columns': list(keys), 'excluded_wall_clock_column': True,
        'differences': differences, 'recovery_reason': 'Windows JSON sharing violation at epoch156, not scientific outcome.',
        'same_seed_configuration_data_split_and_frozen_training_code': True}
    robust_json(EVIDENCE / 'replayed_prefix_check.json', report)
    if differences: raise ValueError('Recovery numerical prefix mismatch; stop for investigation')


def queue(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['status'] != 'FROZEN_BEFORE_TRAINING' or plan['source_sha256'] != rr.sources():
        raise ValueError('Frozen plan/code mismatch')
    root = rr.safe_output(plan['runs_root']); lock = root / 'queue.lock'
    with lock.open('x', encoding='utf-8') as stream: stream.write(str(os.getpid()))
    status = {'status': 'RUNNING', 'plan_sha256': rr.digest(plan_path), 'pid': os.getpid(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'completed': [],
        'planned_runs': len(plan['protocols']) * len(plan['models']) * len(plan['seeds']),
        'runtime_io_patch': PATCH_ID, 'runtime_io_patch_sha256': rr.digest(__file__),
        'runtime_amendment_sha256': rr.digest(EVIDENCE / 'amendment.json'),
        'prior_worker_failure_record': str(EVIDENCE / 'original_queue_status.json')}
    try:
        for protocol in plan['protocols']:
            for seed in plan['seeds']:
                for mode in plan['models']:
                    path = root / protocol / mode / ('seed_' + str(seed))
                    status['active'] = {'protocol': protocol, 'seed': seed, 'mode': mode}
                    robust_json(root / 'queue_status.json', status)
                    if (path / 'result.json').exists():
                        meta = json.loads((path / 'run_metadata.json').read_text(encoding='utf-8'))
                        result = json.loads((path / 'result.json').read_text(encoding='utf-8'))
                        if meta['plan_sha256'] != rr.digest(plan_path) or result['checkpoint_sha256'] != rr.digest(path / 'best.pt'):
                            raise ValueError('Completed run has incorrect binding')
                    else:
                        with (root / 'queue_console.log').open('a', encoding='utf-8') as log:
                            log.write('\n' + json.dumps(dict(status['active'], runtime_io_patch=PATCH_ID)) + '\n'); log.flush()
                            process = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                                'train-cell', '--plan', str(plan_path), '--protocol', protocol, '--mode', mode, '--seed', str(seed)],
                                cwd=str(rr.WORKSPACE), stdout=log, stderr=subprocess.STDOUT, check=False,
                                creationflags=subprocess.CREATE_NO_WINDOW)
                        if process.returncode: raise RuntimeError('Training failed; no automatic training retry: ' + str(status['active']))
                        if protocol == 'iid997' and seed == 44 and mode == 'gnn_zero_edge_features': check_prefix(path)
                    status['completed'].append(dict(status['active']))
                    robust_json(root / 'queue_status.json', status)
        status['status'] = 'COMPLETE_ARCHIVAL_AI_MATRIX'; status['active'] = None
    except BaseException as exc:
        status['status'] = 'ERROR_STOPPED'; status['error'] = str(exc)
        raise
    finally:
        status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        robust_json(root / 'queue_status.json', status)
        lock.unlink()


def main():
    command = sys.argv[1]
    if command == 'test': test()
    elif command == 'train-cell':
        # Bind transport change separately; imported frozen training equations remain exact.
        rr.write_json = robust_json
        sys.argv[1] = 'train'
        rr.main()
    elif command == 'queue':
        p = argparse.ArgumentParser(); p.add_argument('--plan', required=True)
        queue(p.parse_args(sys.argv[2:]).plan)
    else: raise ValueError(command)


if __name__ == '__main__': main()
