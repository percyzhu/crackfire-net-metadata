"""Sequential finite GPU matrix; skips verified complete cells, stops on errors."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import research_runner as rr


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True)
    args = p.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['status'] != 'FROZEN_BEFORE_TRAINING' or plan['purpose'] != rr.PURPOSE:
        raise ValueError('A frozen v08 plan is required')
    if plan['source_sha256'] != rr.sources(): raise ValueError('Code changed since freeze')
    plan_hash = rr.digest(plan_path)
    root = rr.safe_output(plan['runs_root']); root.mkdir(parents=True, exist_ok=True)
    lock = root / 'queue.lock'
    # Exclusive create prevents accidental simultaneous queues, never remove old lock automatically.
    with lock.open('x', encoding='utf-8') as stream: stream.write(str(__import__('os').getpid()))
    status = {'status': 'RUNNING', 'plan_sha256': plan_hash, 'pid': __import__('os').getpid(),
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'completed': [],
        'planned_runs': len(plan['protocols']) * len(plan['models']) * len(plan['seeds'])}
    try:
        for protocol in plan['protocols']:
            for seed in plan['seeds']:
                for mode in plan['models']:
                    cell = root / protocol / mode / ('seed_' + str(seed))
                    status['active'] = {'protocol': protocol, 'seed': seed, 'mode': mode}
                    rr.write_json(root / 'queue_status.json', status)
                    if (cell / 'result.json').exists():
                        meta = json.loads((cell / 'run_metadata.json').read_text(encoding='utf-8'))
                        result = json.loads((cell / 'result.json').read_text(encoding='utf-8'))
                        if meta['plan_sha256'] != plan_hash or result['checkpoint_sha256'] != rr.digest(cell / 'best.pt'):
                            raise ValueError('Existing run is not bound to this queue')
                    else:
                        with (root / 'queue_console.log').open('a', encoding='utf-8') as log:
                            log.write('\n' + json.dumps(status['active']) + '\n'); log.flush()
                            process = subprocess.run([sys.executable, '-X', 'utf8', str(rr.HERE / 'research_runner.py'),
                                'train', '--plan', str(plan_path), '--protocol', protocol,
                                '--mode', mode, '--seed', str(seed)], stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(rr.WORKSPACE), check=False, creationflags=subprocess.CREATE_NO_WINDOW)
                        if process.returncode:
                            raise RuntimeError('Training cell failed, no automatic retry: ' + str(status['active']))
                    status['completed'].append(dict(status['active']))
                    rr.write_json(root / 'queue_status.json', status)
        status['status'] = 'COMPLETE_ARCHIVAL_AI_MATRIX'; status['active'] = None
    except BaseException as exc:
        status['status'] = 'ERROR_STOPPED'; status['error'] = str(exc)
        raise
    finally:
        status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        rr.write_json(root / 'queue_status.json', status)
        lock.unlink()


if __name__ == '__main__': main()
