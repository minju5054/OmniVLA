"""Frozen scheduler and logging; model commands are applied only in scientific mode."""
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
from research.isaac_scene import rgb, set_pose
from research.isaac_final_adapter import capture
from research.isaac_final_common import append_json, sha256, write_json
from research.isaac_final_contracts import Client, History, Scheduler, action, integrate, local_goal, stats, wrap


def request(root, episode_id, request_id, history, obs, pose, goal, op='infer'):
    local, normalized = local_goal(pose, goal)
    return dict(op=op, episode_id=episode_id, request_id=request_id,
                observation_id=history.frames[-1]['frame_id'], observation_time_sim=obs,
                history=history.snapshot(obs), robot_pose_observation=list(pose),
                world_goal=list(goal), local_goal=local, normalized_pose=normalized,
                action_path=str((root/'raw/actions'/f'{request_id:06d}.npy').resolve()),
                request_send_wall_time=time.time())


def validate_response(response, req):
    assert response['rgb_hashes'] == [f['rgb_sha256'] for f in req['history']]
    if req['op'] == 'infer':
        assert response['modality_id'] == 4 and response['selected_index'] == 4
        assert sha256(response['action_path']) == response['action_sha256']
        raw = action(np.load(response['action_path'], allow_pickle=False))
        np.testing.assert_array_equal(raw[0,4], np.asarray(response['selected_native'], dtype=np.float32))
        # Metric/controller rounding stays in the official Edge NumPy runtime.
        assert np.allclose(np.asarray(response['selected_metric'])[:2], raw[0,4,:2]*.1, atol=1e-7)
    else:
        assert response['dummy'] and response['lossless_roundtrip']


def bootstrap(world, camera, root):
    history = History()
    pose = [0.,0.,0.]
    set_pose(camera, pose)
    for _ in range(30):
        world.step(render=True)
    for step in range(101):
        if step:
            world.step(render=True)
        if step % 20 == 0:
            capture(camera, root, history, step//20, step/60, pose)
    return history, pose


def episode(world, camera, config, client, root, goal_spec, scientific):
    root.mkdir(parents=True, exist_ok=False)
    (root/'raw/actions').mkdir(parents=True)
    for name in ('events','requests','controller','robot_state'):
        (root/'raw'/f'{name}.jsonl').touch(exist_ok=False)
    history, pose = bootstrap(world, camera, root)
    goal = goal_spec['pose']
    scheduler = Scheduler()
    request_id = 0
    pending = None
    predictions = []
    sent = []
    frame_times = []
    wall_start = time.time()
    sim_start = float(world.current_time)
    status = 'MODEL_NAVIGATION_FAILURE' if scientific else 'INFRASTRUCTURE_PASS'
    reason = 'TIMEOUT' if scientific else None
    stop_steps = int(config['timeout_after_bootstrap_sim_s']*60) if scientific else 120
    completed = 0
    zero_xy = 0
    active_request = None
    append_json(root/'raw/robot_state.jsonl', dict(step=0, sim_time=100/60, elapsed_sim=0.,
                pose=pose, applied_vw=[0.,0.], active_request_id=None,
                goal_distance=math.dist(pose[:2],goal[:2]), goal_yaw_error=wrap(goal[2]-pose[2])))
    try:
        for k in range(stop_steps):
            ready = client.poll()
            apply_record = None
            if ready is not None:
                validate_response(ready, pending)
                if scientific and ready.get('controller_error'):
                    zero_xy += 1
                    status, reason = 'UPSTREAM_CONTROLLER_BUG', ready['controller_error']
                    append_json(root/'raw/events.jsonl', dict(type=status, response=ready))
                    break
                command = ready['controller']['final_vw'] if scientific else [0.,0.]
                scheduler.ready(ready['request_id'], command)
                active_request = ready['request_id']
                apply_record = dict(episode_id=goal_spec['name'], **ready,
                                    observation_time_sim=pending['observation_time_sim'],
                                    six_history_ids=[x['frame_id'] for x in pending['history']],
                                    six_history_sim_times=[x['sim_time'] for x in pending['history']],
                                    robot_pose_observation=pending['robot_pose_observation'],
                                    world_goal=goal, local_goal=pending['local_goal'],
                                    normalized_pose=pending['normalized_pose'],
                                    request_send_wall_time=pending['request_send_wall_time'],
                                    ready_observed_wall_time=time.time(), response_poll_step=k,
                                    application_physics_step=k+1,
                                    application_sim_time=(100+k+1)/60,
                                    application_wall_time=time.time(), applied_vw=command,
                                    application_pose_before_step=list(pose))
                pending = None
            if k % 20 == 0:
                obs = (100+k)/60
                if k:
                    capture(camera, root, history, 5+k//20, obs, pose)
                frame_times.append(obs)
                slot = scheduler.slot(k)
                if slot == 'MISSED_INFERENCE_SLOT':
                    append_json(root/'raw/events.jsonl', dict(type=slot, sim_time=obs, step=k))
                else:
                    # Dummy requests exercise the same scheduler/transport without model navigation.
                    pending = request(root, goal_spec['name'], request_id, history, obs, pose, goal,
                                      'infer' if scientific else 'dummy')
                    append_json(root/'raw/requests.jsonl', pending)
                    client.send(pending)
                    scheduler.sent(request_id)
                    sent.append(pending)
                    request_id += 1
            pose = integrate(pose, scheduler.command)
            set_pose(camera, pose)
            world.step(render=True)
            completed = k+1
            if apply_record is not None:
                apply_record['application_step_completed_wall_time'] = time.time()
                append_json(root/'raw/controller.jsonl', apply_record)
                predictions.append(apply_record)
            distance = math.dist(pose[:2], goal[:2])
            append_json(root/'raw/robot_state.jsonl', dict(step=completed, sim_time=(100+completed)/60,
                        elapsed_sim=completed/60, pose=pose, applied_vw=scheduler.command,
                        active_request_id=active_request, goal_distance=distance,
                        goal_yaw_error=wrap(goal[2]-pose[2])))
            if scientific and distance <= config['goal_radius_m']:
                status, reason = 'GOAL_REACHED', None
                break
        # Drain at rest without advancing simulation or applying an extra prediction.
        if client.pending is not None:
            deadline = time.monotonic()+30
            response = None
            while response is None and time.monotonic() < deadline:
                response = client.poll()
                if response is None:
                    time.sleep(.001)
            if response is None:
                raise TimeoutError('Outstanding response did not drain at episode end')
            validate_response(response, pending)
            append_json(root/'raw/events.jsonl', dict(type='UNAPPLIED_RESPONSE_AT_EPISODE_END', response=response))
        assert abs(float(world.current_time)-sim_start-completed/60) < 1e-3
    except Exception:
        status, reason = 'IMPLEMENTATION_FAILURE', traceback.format_exc()
        append_json(root/'raw/events.jsonl', dict(type=status, traceback=reason))
    wall_seconds = time.time()-wall_start
    elapsed = completed/60
    controls = [p['controller'] for p in predictions if 'controller' in p]
    result = dict(episode_id=goal_spec['name'], scientific=scientific, status=status, failure_reason=reason,
                  completed_steps=completed, elapsed_sim_s=elapsed, wall_seconds=wall_seconds,
                  sim_wall_ratio=elapsed/wall_seconds, final_pose=pose,
                  final_distance_m=math.dist(pose[:2],goal[:2]), final_yaw_error_rad=wrap(goal[2]-pose[2]),
                  goal_reached=status=='GOAL_REACHED', time_to_goal_sim_s=elapsed if status=='GOAL_REACHED' else None,
                  target_observation_hz=3, capture_interval_sim_s=stats(np.diff(frame_times).tolist()),
                  request_interval_sim_s=stats(np.diff([r['observation_time_sim'] for r in sent]).tolist()),
                  model_ms=stats([p['model_ms'] for p in predictions if 'model_ms' in p]),
                  inference_with_clip_ms=stats([p['forward_with_clip_ms'] for p in predictions if 'forward_with_clip_ms' in p]),
                  pipeline_ms=stats([p['pipeline_ms'] for p in predictions]),
                  observation_to_application_sim_s=stats([p['application_sim_time']-p['observation_time_sim'] for p in predictions]),
                  observation_to_application_wall_ms=stats([(p['application_wall_time']-p['request_send_wall_time'])*1000 for p in predictions]),
                  missed_slots=scheduler.missed, requests=len(sent), applied_predictions=len(predictions),
                  action_update_hz_sim=len(predictions)/elapsed if elapsed else None,
                  first_clip_saturation_count=sum(c['pre_clip_vw']!=c['clipped_vw'] for c in controls),
                  final_ratio_limit_count=sum(c['clipped_vw']!=c['final_vw'] for c in controls),
                  zero_xy_bug_occurrences=zero_xy, observation_pose_anchor=True)
    write_json(root/'summary.json', result)
    return result


def run_phase(world, camera, config, args):
    client = Client(args.socket)
    try:
        if args.phase == 'combined':
            sched = episode(world, camera, config, client, args.output/'dummy-scheduler',
                            dict(name='DUMMY_INFRASTRUCTURE', pose=[5.,0.,0.]), False)
            assert sched['status'] == 'INFRASTRUCTURE_PASS' and sched['applied_predictions'] > 0
            root = args.output/'combined-memory'
            (root/'raw/actions').mkdir(parents=True)
            history, pose = bootstrap(world, camera, root)
            observations = []
            for i in range(3):
                req = request(root, 'COMBINED_MEMORY', i, history, 100/60, pose, [5.,0.,0.])
                append_json(root/'raw/requests.jsonl', req)
                client.send(req)
                response = None
                steps = 0
                deadline = time.monotonic()+60
                while response is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Combined inference timeout')
                    world.step(render=True)
                    steps += 1
                    response = client.poll()
                validate_response(response, req)
                assert rgb(camera).std() > 1
                observations.append(dict(response=response, responsive_steps=steps))
            result = dict(status='PASS', finite_inferences=len(observations), observations=observations,
                          model_commands_applied=False)
            write_json(args.output/'combined-gate.json', result)
            return dict(ipc_gate='PASS', scheduler_gate=sched, combined_gate=result)
        results = {}
        for spec in config['goals']:
            result = episode(world, camera, config, client, args.output/spec['name'], spec, True)
            results[spec['name']] = result
            if result['status'] in ('IMPLEMENTATION_FAILURE','UPSTREAM_CONTROLLER_BUG'):
                break
        return dict(scientific_results=results)
    finally:
        client.close()
