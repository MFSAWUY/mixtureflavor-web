"""Five-concurrent-request HTTP benchmark. Local mode uses its own database and server."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.python-deps')]
import psutil


def request(url, body=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', help='Existing API root, e.g. https://your-domain.com; creates benchmark submissions there')
    parser.add_argument('--pid', type=int, help='Optional local server PID for memory measurement in --url mode')
    parser.add_argument('--requests', type=int, default=50, help='Requests per scenario')
    parser.add_argument('--concurrency', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/verification/performance-local.json')
    args = parser.parse_args()
    if args.requests < 5 or not 1 <= args.concurrency <= 10:
        parser.error('Use at least 5 requests and concurrency between 1 and 10')
    process = None
    run_dir = ROOT / 'runtime' / ('benchmark-' + uuid.uuid4().hex[:10])
    run_dir.mkdir(parents=True)
    log = (run_dir / 'server.log').open('w', encoding='utf-8')
    start = time.perf_counter()
    try:
        if args.url:
            base = args.url.rstrip('/')
            pid = args.pid
        else:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            base = f'http://127.0.0.1:{port}'
            env = {**os.environ, 'FLAVOR_DATABASE': str(run_dir / 'records.sqlite3'), 'FLAVOR_BACKUP_DIR': str(run_dir / 'backups'), 'PYTHONDONTWRITEBYTECODE': '1'}
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_local.py'), '--port', str(port)], cwd=ROOT, env=env, stdout=log, stderr=log,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            pid = process.pid
        deadline = time.monotonic() + 60
        while True:
            try:
                health = request(base + '/api/health'); break
            except Exception:
                if (process and process.poll() is not None) or time.monotonic() > deadline:
                    raise RuntimeError(f'Service did not start; inspect {run_dir / "server.log"}')
                time.sleep(0.1)
        startup = round(time.perf_counter() - start, 3)
        memory, stop = [], threading.Event()
        def sample():
            while not stop.wait(0.01):
                if pid:
                    try:
                        memory.append(psutil.Process(pid).memory_info().rss)
                    except psutil.Error:
                        pass
        sampler = threading.Thread(target=sample, daemon=True); sampler.start()
        scenarios = [('single', ['CCO']), ('mixture', ['CCO', 'CC(=O)O', 'c1ccccc1O', 'CCOC(C)=O', 'CC(C)O'])]
        measurements = []
        try:
            for mode, smiles in scenarios:
                payload = {'mode': mode, 'molecules': [{'smiles': s, 'name': f'Benchmark {i + 1}'} for i, s in enumerate(smiles)], 'acknowledge_storage': True}
                request(base + '/api/predictions', payload)
                def timed(_):
                    before = time.perf_counter()
                    result = request(base + '/api/predictions', payload)
                    elapsed = (time.perf_counter() - before) * 1000
                    if not result.get('record_id') or result['model_version'] != health['model_version']:
                        raise RuntimeError('Unexpected prediction response')
                    return elapsed
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    times = list(pool.map(timed, range(args.requests)))
                p95 = statistics.quantiles(times, n=100, method='inclusive')[94]
                measurements.append({'mode': mode, 'components': len(smiles), 'requests': len(times), 'errors': 0, 'median_ms': round(statistics.median(times), 2), 'p95_ms': round(p95, 2), 'max_ms': round(max(times), 2), 'p95_under_3000ms': p95 <= 3000})
        finally:
            stop.set(); sampler.join(timeout=2)
        report = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'scope': 'existing_server' if args.url else 'local_windows_http', 'note': 'Local CPU inference plus HTTP, validation and SQLite persistence. Not a cloud or geographic network benchmark.', 'platform': platform.platform(), 'cpu': platform.processor(), 'logical_cpu_count': os.cpu_count(), 'model_threads': 2, 'concurrency': args.concurrency, 'model_version': health['model_version'], 'startup_to_health_seconds': startup if process else None, 'peak_sampled_rss_mb': round(max(memory) / 1024**2, 2) if memory else None, 'scenarios': measurements}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if process:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        log.close()


if __name__ == '__main__':
    main()
