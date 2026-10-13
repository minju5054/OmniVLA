"""One-shot phase supervisor: 32 offline requests, then four independent trials."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import write_json,append_json,sha256,sync_tree
from research.isaac_final_contracts import Client
from research.pose_language_contracts import PLAN,PROTOCOL,instruction_fields,validate_response,verify_protocol
from research.run_static_obstacle_attempt import members


def register_once(path,phase,planned):
    write_json(path,dict(phase=phase,planned=planned,protocol_sha256=sha256(PROTOCOL),wall_time=time.time()))


def offline_requests(client,plan,out):
    manifest=json.loads((ROOT/plan['phase_a']['rgb_manifest']).read_text())
    inputs={c['case_id']:c for c in manifest['cases']}
    (out/'raw').mkdir()
    completed=[]
    for row in plan['offline_order']:
        case=inputs[row['rgb_case_id']];latest=case['history'][-1]
        req=dict(op='infer',episode_id=row['episode_id'],request_id=row['request_id'],
            observation_id=latest['frame_id'],observation_time_sim=latest['sim_time'],history=case['history'],
            robot_pose_observation=case['pose'],world_goal=case['world_goal'],local_goal=case['local_goal'],
            normalized_pose=case['normalized_pose'],request_send_wall_time=time.time(),observation_wall_time=latest['wall_time'],
            action_path=str(out/'raw/actions'/f"{row['request_id']:06d}.npy"),
            **instruction_fields(plan['instructions'][row['instruction_id']]))
        append_json(out/'raw/requests.jsonl',req);client.send(req)
        response=None;deadline=time.monotonic()+60
        while response is None:
            response=client.poll()
            if time.monotonic()>deadline:raise TimeoutError('Offline inference timeout; no retry')
            if response is None:time.sleep(.001)
        validate_response(response,req)
        record=dict(**row,request=req,response=response,
            native_prediction=np.load(response['action_path'],allow_pickle=False).tolist(),robot_execution=False)
        append_json(out/'raw/cases.jsonl',record);completed.append(row['episode_id'])
        print('INFERRED '+row['episode_id'],flush=True)
    return completed


def launch_episode(out,trial,socket_path,env,infrastructure=False):
    folder=out/'trials'/trial['instruction_id'];folder.parent.mkdir(exist_ok=True)
    command=['bash','scripts/isaac6_python.sh','research/pose_language_isaac.py',
        '--output',str(folder),'--socket',socket_path,'--instruction',trial['instruction_id']]
    if infrastructure:command.append('--infrastructure')
    record=dict(trial=trial,command=command,start_wall_time=time.time(),timed_out=False)
    write_json(folder.parent/(trial['instruction_id']+'-launch.json'),record)
    with (folder.parent/(trial['instruction_id']+'.log')).open('x') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            record['timed_out']=True;os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait(timeout=15)
    residual=members(proc.pid)
    if residual:os.killpg(proc.pid,signal.SIGTERM)
    result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
    valid=(code==0 and not residual and not record['timed_out'] and result.get('status')=='PRE_CLOSE_PASS'
        and result.get('fast_shutdown') is True and result.get('evidence_written_before_close')
        and result.get('close_requested') and result.get('constructor_returned'))
    episode=result.get('episode',{})
    valid=bool(valid and episode.get('status') not in (None,'IMPLEMENTATION_FAILURE','UPSTREAM_CONTROLLER_BUG'))
    record.update(exit_code=code,residual_processes=residual,status='PASS' if valid else 'IMPLEMENTATION_FAILURE',
        episode=episode,result=result,finish_wall_time=time.time())
    write_json(folder.parent/(trial['instruction_id']+'-outcome.json'),record)
    print('EPISODE '+trial['episode_id']+' '+str(episode.get('status')),flush=True)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['infrastructure','offline','navigation'],required=True)
    p.add_argument('--run-id',required=True);args=p.parse_args()
    assert Path(args.run_id).name==args.run_id
    scientific=args.phase!='infrastructure'
    if scientific:_,plan,_=verify_protocol()
    else:plan=json.loads(PLAN.read_text())
    planned=plan['offline_order'] if args.phase=='offline' else plan['navigation_order']
    if args.phase=='infrastructure':planned=planned[:1]
    if args.phase=='navigation':
        from research.pose_language_analysis import offline_validity
        gate=offline_validity(ROOT/'outputs/pose-language-obstacle/offline-001',plan)
        assert gate['status']=='MODALITY8_INPUT_VALID'
    if scientific:
        register_once(ROOT/f'outputs/pose-language-obstacle/{args.phase}-registration.json',args.phase,planned)
    out=ROOT/'outputs/pose-language-obstacle'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',MPLBACKEND='Agg',
        MPLCONFIGDIR=str(ROOT/'artifacts-local/edge-matplotlib'),XDG_CACHE_HOME=str(ROOT/'artifacts-local/isaac-cache-final'),
        XDG_DATA_HOME=str(ROOT/'artifacts-local/isaac-data-final'))
    for key in ('LD_LIBRARY_PATH','PYTHONPATH','PYTHONEXE'):env.pop(key,None)
    socket_path=f'/tmp/omnivla-language-{os.getpid()}.sock'
    command=['artifacts-local/envs/edge-blackwell/bin/python','research/pose_language_service.py',
        '--socket',socket_path,'--output',str(out/'service'),'--phase',args.phase]
    write_json(out/'launch.json',dict(phase=args.phase,service_command=command,runner_command=sys.argv,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        plan_sha256=sha256(PLAN),protocol_sha256=sha256(PROTOCOL) if scientific else None,
        starting_wall_time=time.time()))
    stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():
            p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if p.returncode==0:
                sample=dict(wall_time=time.time(),used_MiB=int(p.stdout.splitlines()[0]));samples.append(sample)
                append_json(out/'gpu-memory.jsonl',sample)
            stop.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    service=None;client=None;completed=[];trials=[];failure=None
    try:
        with (out/'service.log').open('x') as log:
            service=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline=time.monotonic()+300
        while not (out/'service/ready.json').exists():
            if service.poll() is not None:raise RuntimeError(f'Service exited before ready: {service.returncode}')
            if time.monotonic()>deadline:raise TimeoutError('Service startup timeout')
            time.sleep(.1)
        if args.phase=='offline':
            client=Client(socket_path);completed=offline_requests(client,plan,out);client.close();client=None
        else:
            for trial in planned[:1] if args.phase=='infrastructure' else planned:
                record=launch_episode(out,trial,socket_path,env,args.phase=='infrastructure');trials.append(record)
                if record['status']!='PASS':raise RuntimeError('Material episode implementation/runtime failure; no retry')
                completed.append(trial['episode_id'])
    except BaseException:
        failure=traceback.format_exc();print(failure,flush=True)
    finally:
        if client:client.close()
        (out/'service-stop').touch(exist_ok=False)
        if service:
            try:service.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(service.pid,signal.SIGTERM)
                try:service.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(service.pid,signal.SIGKILL);service.wait(timeout=10)
        stop.set();thread.join();Path(socket_path).unlink(missing_ok=True)
    if args.phase=='offline' and (out/'raw/cases.jsonl').exists():
        completed=[json.loads(line)['episode_id'] for line in (out/'raw/cases.jsonl').read_text().splitlines()]
    expected_count=32 if args.phase=='offline' else (1 if args.phase=='infrastructure' else 4)
    residual=members(service.pid) if service else []
    stopped=json.loads((out/'service/stopped.json').read_text()) if (out/'service/stopped.json').exists() else None
    passed=failure is None and len(completed)==expected_count and service and service.returncode==0 and not residual
    outcome=dict(status='PASS' if passed else 'IMPLEMENTATION_FAILURE',phase=args.phase,completed=completed,
        not_run=[dict(episode_id=r['episode_id'],reason=failure) for r in planned if r['episode_id'] not in completed],
        failure=failure,trials=[dict(instruction_id=r['trial']['instruction_id'],status=r['status'],episode=r['episode']) for r in trials],
        service_exit_code=service.returncode if service else None,residual_service_processes=residual,service_stopped=stopped,
        device_memory_peak_MiB=max((s['used_MiB'] for s in samples),default=None),finished_wall_time=time.time())
    if args.phase=='offline' and passed:
        from research.pose_language_analysis import offline_validity
        try:gate=offline_validity(out,plan,outcome)
        except Exception:
            gate=dict(status='IMPLEMENTATION_FAILURE',traceback=traceback.format_exc())
            passed=False;outcome.update(status='IMPLEMENTATION_FAILURE',failure=gate['traceback'])
        write_json(out/'technical-validity.json',gate)
    write_json(out/'outcome.json',outcome)
    sync_tree(out)
    for path in out.rglob('*'):
        if path.is_file():path.chmod(0o444)
    print(json.dumps(outcome,indent=2),flush=True)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
