"""Offline scientific evidence audit; never constructs Isaac or a model."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.edge_contracts import controller
from research.isaac_final_common import sha256, verify_frozen, write_json
from research.isaac_final_contracts import action, integrate, local_goal, pixel_hash, wrap


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def validate_episode(root, spec, goal):
    summary = read(root/'summary.json')
    history = rows(root/'raw/history.jsonl')
    requests = rows(root/'raw/requests.jsonl')
    predictions = rows(root/'raw/controller.jsonl')
    states = rows(root/'raw/robot_state.jsonl')
    events = rows(root/'raw/events.jsonl')
    assert summary['scientific'] is True
    assert summary['status'] in ('GOAL_REACHED','MODEL_NAVIGATION_FAILURE',
                                 'IMPLEMENTATION_FAILURE','UPSTREAM_CONTROLLER_BUG')
    frames = {f['frame_id']:f for f in history}
    assert len(frames) == len(history)
    for i, f in enumerate(history):
        assert f['frame_id'] == i and abs(f['sim_time']-i/3) < 1e-12
        assert sha256(f['path']) == f['png_sha256']
        image = np.asarray(Image.open(f['path']))
        assert image.shape == (480,640,3) and image.dtype == np.uint8
        assert pixel_hash(image) == f['rgb_sha256']
        if i < 6:
            assert f['pose'] == [0.,0.,0.]
    by_step = {s['step']:s for s in states}
    assert len(states) == summary['completed_steps']+1
    assert states[0]['pose'] == spec['start']
    for index, state in enumerate(states):
        assert state['step'] == index
        assert abs(state['elapsed_sim']-index/60) < 1e-12
        assert abs(state['sim_time']-(100+index)/60) < 1e-12
        assert math.isclose(state['goal_distance'], math.dist(state['pose'][:2],goal['pose'][:2]), abs_tol=1e-12)
        if index:
            np.testing.assert_allclose(state['pose'], integrate(states[index-1]['pose'],state['applied_vw']), atol=1e-12,rtol=0)
    reqs = {r['request_id']:r for r in requests}
    assert list(reqs) == list(range(len(reqs)))
    for req in requests:
        assert req['op'] == 'infer' and req['episode_id'] == goal['name']
        assert req['world_goal'] == goal['pose']
        obs = req['observation_time_sim']
        step = round(obs*60-100)
        assert step % 20 == 0
        np.testing.assert_array_equal(req['robot_pose_observation'],by_step[step]['pose'])
        local, normalized = local_goal(req['robot_pose_observation'],goal['pose'])
        np.testing.assert_allclose(req['local_goal'],local,atol=1e-12,rtol=0)
        np.testing.assert_allclose(req['normalized_pose'],normalized,atol=1e-12,rtol=0)
        frame_ids = [f['frame_id'] for f in req['history']]
        assert len(frame_ids)==6 and frame_ids == list(range(frame_ids[0],frame_ids[0]+6))
        assert req['observation_id'] == frame_ids[-1]
        assert req['history'] == [frames[i] for i in frame_ids]
        assert req['history'][-1]['sim_time'] == obs
        assert all(f['sim_time']<=obs for f in req['history'])
    response_times = {}
    for pred in predictions:
        req = reqs[pred['request_id']]
        assert pred['observation_id']==req['observation_id']
        assert pred['observation_time_sim']==req['observation_time_sim']
        assert pred['robot_pose_observation']==req['robot_pose_observation']
        assert pred['normalized_pose']==req['normalized_pose']
        assert pred['six_history_ids']==[f['frame_id'] for f in req['history']]
        assert pred['rgb_hashes']==[f['rgb_sha256'] for f in req['history']]
        assert pred['modality_id']==4 and pred['selected_index']==4
        assert sha256(pred['action_path'])==pred['action_sha256']
        raw=action(np.load(pred['action_path'],allow_pickle=False))
        np.testing.assert_array_equal(raw[0,4],np.array(pred['selected_native'],dtype=np.float32))
        metric=raw[0,4].copy();metric[:2]*=.1
        np.testing.assert_array_equal(metric,np.array(pred['selected_metric'],dtype=np.float32))
        assert controller(metric)==pred['controller']
        assert pred['applied_vw']==pred['controller']['final_vw']
        step=pred['application_physics_step']
        assert step==pred['response_poll_step']+1
        assert pred['application_sim_time']==by_step[step]['sim_time']
        assert by_step[step]['applied_vw']==pred['applied_vw']
        assert by_step[step]['active_request_id']==pred['request_id']
        assert req['request_send_wall_time']<=pred['ready_wall_time']<=pred['application_wall_time']
        response_times[pred['request_id']]=pred['ready_wall_time']
    for event in events:
        if 'response' in event:
            response=event['response']
            response_times[response['request_id']]=response['ready_wall_time']
            if 'action_path' in response:
                assert sha256(response['action_path'])==response['action_sha256']
                action(np.load(response['action_path'],allow_pickle=False))
    assert len(list((root/'raw/actions').glob('*.npy')))==len(requests)
    for previous,following in zip(requests,requests[1:]):
        assert previous['request_id'] in response_times
        assert following['request_send_wall_time']>=response_times[previous['request_id']]
    # Every physics step holds zero or the most recently applied prediction.
    applied={p['application_physics_step']:p for p in predictions}
    held=[0.,0.]
    for state in states:
        if state['step'] in applied:
            held=applied[state['step']]['applied_vw']
        assert state['applied_vw']==held
    assert summary['missed_slots']==sum(e.get('type')=='MISSED_INFERENCE_SLOT' for e in events)
    entries=[s for s in states if s['goal_distance']<=.5]
    if summary['status']=='GOAL_REACHED':
        assert entries and entries[0]['step']==states[-1]['step']
        assert summary['time_to_goal_sim_s']==entries[0]['elapsed_sim']<=40
    elif summary['status']=='MODEL_NAVIGATION_FAILURE':
        assert not entries and summary['elapsed_sim_s']==40.
    assert summary['final_distance_m']==states[-1]['goal_distance']
    assert summary['final_yaw_error_rad']==states[-1]['goal_yaw_error']
    manifest={p.relative_to(root).as_posix():sha256(p) for p in root.rglob('*') if p.is_file()}
    return dict(status='PASS', scientific_status=summary['status'],
                frames=len(history), requests=len(requests), applied_predictions=len(predictions),
                states=len(states), artifacts=manifest)


def validate(run, config_path):
    config=read(config_path)
    frozen=read(ROOT/'artifacts-local/edge-final/protocol-freeze.json')
    assert sha256(config_path)==frozen['sha256']
    verify_frozen(config)
    for gate in config['infrastructure_gates'].values():
        assert sha256(ROOT/gate['evidence'])==gate['sha256']
    audit=read(ROOT/'artifacts-local/edge-final/source-audit.json')
    for path,expected in audit['historical_reports'].items():
        assert sha256(ROOT/path)==expected
    registration=read(ROOT/'outputs/edge-final/scientific-registration.json')
    assert registration['run_id']==run.name and registration['config_sha256']==frozen['sha256']
    assert registration['registered_wall']>config['freeze_wall_time']
    outcome=read(run/'outcome.json')
    assert outcome['status']=='PASS' and outcome['launcher_exit_code']==0
    assert not outcome['residual_isaac_processes'] and not outcome['residual_service_processes']
    assert not outcome['segmentation_fault'] and not outcome['cuda_symbol_error']
    assert read(run/'service/strict-load.json')['status']=='STRICT_LOAD_PASS'
    assert read(run/'service/stopped.json')['load_count']==1
    episodes={}
    for goal in config['goals']:
        root=run/'isaac'/goal['name']
        if root.exists():
            episodes[goal['name']]=validate_episode(root,config,goal)
        else:
            episodes[goal['name']]=dict(status='NOT_RUN',scientific_status='INSUFFICIENT_EVIDENCE')
    assert read(run/'service/stopped.json')['inference_count']==sum(
        e.get('requests',0) for e in episodes.values())
    decision=('NO_OBSTACLE_GOAL_REACHING_PASS' if all(
        e['status']=='PASS' and e['scientific_status']=='GOAL_REACHED' for e in episodes.values())
        else 'NO_OBSTACLE_GOAL_REACHING_FAIL')
    return dict(validation='PASS',decision=decision,frozen_sha256=frozen['sha256'],episodes=episodes)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'research/frozen-edge-isaac-no-obstacle-final.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=validate(args.run,args.config)
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
    for name,episode in result['episodes'].items():
        print(name,{k:v for k,v in episode.items() if k!='artifacts'})
