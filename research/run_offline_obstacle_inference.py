"""Exactly one frozen batch, one model load, eight requests, no robot execution."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import sha256, write_json, append_json, sync_tree
from research.isaac_final_contracts import Client, action
from research.offline_obstacle_protocol import verify_protocol


def request_for(case, output):
    latest=case['history'][-1]
    return dict(op='infer',episode_id=case['case_id'],request_id=case['request_id'],
        observation_id=latest['frame_id'],observation_time_sim=latest['sim_time'],
        history=case['history'],robot_pose_observation=case['pose'],world_goal=case['world_goal'],
        local_goal=case['local_goal'],normalized_pose=case['normalized_pose'],
        action_path=str(output/'raw/actions'/f"{case['request_id']:06d}.npy"),
        request_send_wall_time=time.time(),observation_wall_time=latest['wall_time'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    args=p.parse_args()
    frozen,plan,manifest,config=verify_protocol(args.protocol)
    marker=ROOT/'artifacts-local/edge-obstacle-offline/scientific-batch-registration.json'
    output=ROOT/'outputs/edge-obstacle-offline/paired-001'
    # Exclusive global marker intentionally prevents an alternate output from retrying.
    write_json(marker,dict(protocol_sha256=sha256(args.protocol),start_wall_time=time.time(),
        planned_cases=[c['case_id'] for c in plan['cases']],output=str(output)))
    output.mkdir(parents=True,exist_ok=False);(output/'raw').mkdir()
    sock=f'/tmp/omnivla-offline-{os.getpid()}.sock'
    command=[str(ROOT/'artifacts-local/envs/edge-blackwell/bin/python'),
        str(ROOT/'research/edge_isaac_service.py'),'--socket',sock,'--output',str(output/'service')]
    env=os.environ.copy()
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONEXE'): env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',
        MPLCONFIGDIR=str(ROOT/'artifacts-local/edge-matplotlib'))
    write_json(output/'launch.json',dict(command=command,protocol_sha256=sha256(args.protocol),
        runtime_python=command[0],robot_execution=False,seed_policy=plan['seed_policy'],
        original_scheduler='Unmodified, not invoked for independent offline requests'))
    proc=None;client=None;error=None;records=[];code=None
    try:
        with (output/'service.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+300
            while not (output/'service/ready.json').exists():
                if proc.poll() is not None: raise RuntimeError(f'Service exited before ready: {proc.returncode}')
                if time.monotonic()>deadline: raise TimeoutError('Service load timeout')
                time.sleep(.1)
            client=Client(sock)
            for case in manifest['cases']:
                request=request_for(case,output)
                append_json(output/'raw/requests.jsonl',request)
                client.send(request);deadline=time.monotonic()+60
                response=None
                while response is None:
                    response=client.poll()
                    if time.monotonic()>deadline: raise TimeoutError('Offline response timeout; no retry')
                    if response is None: time.sleep(.001)
                assert response['rgb_hashes']==[f['rgb_sha256'] for f in case['history']]
                assert sha256(response['action_path'])==response['action_sha256']
                native=action(np.load(response['action_path'],allow_pickle=False))
                record=dict(**case,response=response,native_prediction=native.tolist(),
                    status='INFERRED',cold_inference=case['request_id']==0,robot_execution=False)
                append_json(output/'raw/cases.jsonl',record);records.append(record)
                print('INFERRED '+case['case_id'],flush=True)
    except BaseException:
        error=traceback.format_exc(); print(error,flush=True)
    finally:
        if client: client.close()
        (output/'service-stop').touch(exist_ok=False)
        if proc:
            try: code=proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,15)
                try: code=proc.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,9);code=proc.wait(timeout=10)
    inferred={r['case_id'] for r in records}
    requests=(output/'raw/requests.jsonl').read_text().splitlines() if (output/'raw/requests.jsonl').exists() else []
    sent={json.loads(r)['episode_id'] for r in requests}
    statuses=[dict(case_id=c['case_id'],status='INFERRED' if c['case_id'] in inferred else
        ('REQUEST_FAILED' if c['case_id'] in sent else 'NOT_RUN_AFTER_FAILURE'),
        reason=None if c['case_id'] in inferred else error) for c in plan['cases']]
    stopped=json.loads((output/'service/stopped.json').read_text()) if (output/'service/stopped.json').exists() else None
    passed=error is None and len(records)==8 and code==0 and stopped==dict(inference_count=8,load_count=1)
    write_json(output/'outcome.json',dict(status='PASS' if passed else 'FAILED',cases=statuses,error=error,
        inference_count=len(records),request_count=len(requests),service_exit_code=code,service_stopped=stopped,
        protocol_sha256=sha256(args.protocol),wall_time=time.time(),robot_execution=False))
    sync_tree(output)
    for path in (output/'raw').rglob('*'):
        if path.is_file(): path.chmod(0o444)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__': main()
