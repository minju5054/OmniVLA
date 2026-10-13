"""Additive baseline episode with evaluation-only static collision termination.

Scheduler/IPC/integration/controller remain the baseline implementation. The
separate function preserves the old frozen source and raw evidence byte-for-byte.
"""
import math
import time
import traceback
import numpy as np
from research.isaac_scene import set_pose
from research.isaac_final_adapter import capture
from research.isaac_final_common import append_json, write_json
from research.isaac_final_contracts import Scheduler, integrate, stats, wrap
from research.isaac_final_navigation import bootstrap, request, validate_response
from research.static_obstacle_geometry import Rectangle, classify


def episode(world, camera, config, client, root, goal_spec, scientific):
    root.mkdir(parents=True, exist_ok=False)
    (root/'raw/actions').mkdir(parents=True)
    for name in ('events','requests','controller','robot_state'):
        (root/'raw'/f'{name}.jsonl').touch(exist_ok=False)
    history, pose = bootstrap(world, camera, root)
    goal = goal_spec['pose']
    rectangle = Rectangle.from_config(config['static_obstacle'])
    radius = config['evaluation']['agent_radius_m']
    margin = config['evaluation']['safety_margin_m']
    minimum_clearance = rectangle.clearance(pose, radius)
    first_collision_time = None
    path_length = 0.
    max_lateral = abs(pose[1])
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
                goal_distance=math.dist(pose[:2],goal[:2]), goal_yaw_error=wrap(goal[2]-pose[2]),
                obstacle_clearance=minimum_clearance, swept_minimum_clearance=minimum_clearance,
                collision=False, safety_margin_violation=minimum_clearance < margin))
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
                                    observation_wall_time=pending['history'][-1]['wall_time'],
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
                    pending['observation_wall_time'] = history.frames[-1]['wall_time']
                    append_json(root/'raw/requests.jsonl', pending)
                    client.send(pending)
                    scheduler.sent(request_id)
                    sent.append(pending)
                    request_id += 1
            previous_pose = list(pose)
            pose = integrate(pose, scheduler.command)
            sweep = rectangle.swept(previous_pose, pose, radius)
            minimum_clearance = min(minimum_clearance, sweep['minimum_clearance'])
            path_length += math.dist(previous_pose[:2], pose[:2])
            max_lateral = max(max_lateral, abs(pose[1]))
            set_pose(camera, pose)
            world.step(render=True)
            completed = k+1
            if apply_record is not None:
                apply_record['application_step_completed_wall_time'] = time.time()
                apply_record['application_pose_after_step'] = list(pose)
                append_json(root/'raw/controller.jsonl', apply_record)
                predictions.append(apply_record)
            distance = math.dist(pose[:2], goal[:2])
            append_json(root/'raw/robot_state.jsonl', dict(step=completed, sim_time=(100+completed)/60,
                        elapsed_sim=completed/60, pose=pose, applied_vw=scheduler.command,
                        active_request_id=active_request, goal_distance=distance,
                        goal_yaw_error=wrap(goal[2]-pose[2]),
                        obstacle_clearance=rectangle.clearance(pose, radius),
                        swept_minimum_clearance=sweep['minimum_clearance'],
                        collision=sweep['collision'],
                        safety_margin_violation=sweep['minimum_clearance'] < margin))
            if scientific:
                classification = classify(distance, completed/60, minimum_clearance, config)
                if sweep['collision']:
                    first_collision_time = (k+sweep['first_contact_fraction'])/60
                    append_json(root/'raw/events.jsonl', dict(type='COLLISION',
                        first_contact_elapsed_sim=first_collision_time,
                        first_contact_episode_sim=100/60+first_collision_time,
                        detection_elapsed_sim=completed/60, step=completed,
                        previous_pose=previous_pose, pose=pose, sweep=sweep))
                if classification:
                    status = classification
                    reason = 'TIMEOUT' if classification == 'MODEL_NAVIGATION_FAILURE' else None
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
                  goal_reached=status in ('SAFE_GOAL_REACHED','GOAL_REACHED_LOW_CLEARANCE'),
                  time_to_goal_sim_s=elapsed if status in ('SAFE_GOAL_REACHED','GOAL_REACHED_LOW_CLEARANCE') else None,
                  collision=first_collision_time is not None, first_collision_time_sim_s=first_collision_time,
                  minimum_signed_clearance_m=minimum_clearance,
                  safety_margin_violation=minimum_clearance < margin,
                  maximum_lateral_deviation_m=max_lateral, executed_path_length_m=path_length,
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

