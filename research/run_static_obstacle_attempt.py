"""Static-obstacle supervisor; exclusive registration permits one primary only."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.isaac_final_common import append_json, sha256, write_json
from research.static_obstacle_protocol import verify_static


def members(pgid):
    rows = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,stat=,args='], text=True)
    return [line.strip() for line in rows.splitlines()
            if len(line.split()) >= 3 and line.split()[1] == str(pgid)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', required=True)
    p.add_argument('--phase', choices=['infrastructure', 'scientific'], required=True)
    p.add_argument('--change', required=True)
    p.add_argument('--config', default='research/frozen-edge-static-obstacle.json')
    p.add_argument('--timeout', type=float, default=300)
    a = p.parse_args()
    assert Path(a.run_id).name == a.run_id
    if a.phase == 'scientific':
        config = json.loads(Path(a.config).read_text())
        verify_static(config, a.config)
        write_json(ROOT/'outputs/static-obstacle/scientific-registration.json',
                   dict(run_id=a.run_id, config_sha256=sha256(a.config),
                        goals=[g['name'] for g in config['goals']], registered_wall=time.time()))
    out = ROOT / 'outputs/static-obstacle' / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               XDG_CACHE_HOME=str(ROOT/'artifacts-local/isaac-cache-final'),
               XDG_DATA_HOME=str(ROOT/'artifacts-local/isaac-data-final'))
    socket_path = f'/tmp/omnivla-{os.getpid()}.sock'
    cmd = ['scripts/isaac6_python.sh', 'research/static_obstacle_run.py',
           '--phase', a.phase, '--output', str(out/'isaac'), '--config', a.config, '--socket', socket_path]
    service_cmd = None
    if a.phase == 'scientific':
        service_cmd = ['artifacts-local/envs/edge-blackwell/bin/python',
                       'research/edge_isaac_service.py', '--socket', socket_path,
                       '--output', str(out/'service')]
    attempt = dict(run_id=a.run_id, phase=a.phase, change=a.change,
                   command=shlex.join(cmd), service_command=shlex.join(service_cmd) if service_cmd else None,
                   supervisor_command=shlex.join([sys.executable, *sys.argv]),
                   environment={k:env[k] for k in ['PYTHONDONTWRITEBYTECODE','XDG_CACHE_HOME','XDG_DATA_HOME']},
                   started_wall=time.time(), git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                   config_sha256=sha256(a.config),
                   source_sha256={str(f.relative_to(ROOT)):sha256(f) for f in
                                  sorted(set((ROOT/'research').glob('*isaac*py')) | set((ROOT/'research').glob('*static_obstacle*py')))})
    write_json(out/'attempt.json', attempt)
    stop = threading.Event()
    samples = []
    def sample():
        while not stop.is_set():
            r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if r.returncode == 0:
                row = dict(wall=time.time(), used_MiB=int(r.stdout.splitlines()[0]))
                samples.append(row)
                append_json(out/'gpu-memory.jsonl', row)
            stop.wait(.25)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    service = child = None
    outcome = dict(status='FAIL', timed_out=False)
    try:
        if service_cmd:
            with (out/'service.log').open('x') as log:
                service_env = dict(env, PYTHONNOUSERSITE='1', MPLBACKEND='Agg',
                                   MPLCONFIGDIR=str(ROOT/'artifacts-local/edge-matplotlib'))
                for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONEXE'):
                    service_env.pop(key, None)
                service = subprocess.Popen(service_cmd, cwd=ROOT, env=service_env,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+180
            while not (out/'service/ready.json').exists():
                if service.poll() is not None:
                    raise RuntimeError(f'Edge service exited {service.returncode} before ready')
                if time.monotonic() > deadline:
                    raise TimeoutError('Edge service ready timeout')
                time.sleep(.1)
        with (out/'isaac.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        try:
            outcome['launcher_exit_code'] = child.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            outcome['timed_out'] = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            outcome['launcher_exit_code'] = child.returncode
        outcome['residual_isaac_processes'] = members(child.pid)
        log = (out/'isaac.log').read_text(errors='replace')
        outcome.update(segmentation_fault='Segmentation fault' in log,
                       cuda_symbol_error='undefined symbol:' in log,
                       log_sha256=sha256(out/'isaac.log'))
        result_path = out/'isaac/result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        outcome['pre_close_result'] = result
        valid = (result.get('status') == 'PRE_CLOSE_PASS' and result.get('constructor_returned')
                 and result.get('stage_opened') and result.get('camera_rendered')
                 and result.get('steps', 0) >= 60 and result.get('evidence_written_before_close')
                 and result.get('close_requested') and result.get('fast_shutdown') is True)
        if (valid and outcome['launcher_exit_code'] == 0 and not outcome['timed_out']
                and not outcome['residual_isaac_processes'] and not outcome['segmentation_fault']
                and not outcome['cuda_symbol_error']):
            outcome['status'] = 'PASS'
        else:
            outcome['failure_reason'] = result.get('traceback', 'Lifecycle/process criteria not met')
    except Exception as exc:
        outcome['failure_reason'] = repr(exc)
    finally:
        if child and members(child.pid):
            os.killpg(child.pid, signal.SIGTERM)
        if service:
            (out/'service-stop').touch()
            try:
                service.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(service.pid, signal.SIGTERM)
                service.wait(timeout=15)
            outcome['service_exit_code'] = service.returncode
            outcome['residual_service_processes'] = members(service.pid)
            if service.returncode or outcome['residual_service_processes']:
                outcome['status'] = 'FAIL'
                outcome.setdefault('failure_reason', 'Service termination failed')
        Path(socket_path).unlink(missing_ok=True)
        stop.set()
        thread.join()
        outcome.update(run_id=a.run_id, phase=a.phase, finished_wall=time.time(),
                       wall_seconds=time.time()-attempt['started_wall'],
                       device_memory_peak_MiB=max((v['used_MiB'] for v in samples), default=None),
                       device_memory_sample_count=len(samples), memory_sample_interval_s=.25)
        write_json(out/'outcome.json', outcome)
        print(json.dumps(outcome, indent=2), flush=True)
    return 0 if outcome['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
