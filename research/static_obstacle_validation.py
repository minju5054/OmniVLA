"""Model-free Isaac gates; temporary calibration objects never enter science."""
import json
import math
import socket
import threading
import time
import numpy as np
from PIL import Image
from pxr import UsdGeom
from research.isaac_scene import rgb, set_pose
from research.isaac_final_adapter import camera_gate, history_gate
from research.isaac_final_common import sha256, write_json
from research.isaac_final_contracts import Client, integrate, pixel_hash
from research.static_obstacle_geometry import Rectangle
from research.static_obstacle_scene import add_obstacle, geometry_record
from research.static_obstacle_navigation import episode


def qualify(world, camera, config, out, socket_path):
    calibration = camera_gate(world, camera, config, out)
    add_obstacle(world.stage, config)
    geometry = geometry_record(world.stage, config)
    rectangle = Rectangle.from_config(config['static_obstacle'])
    radius = config['evaluation']['agent_radius_m']
    margin = config['evaluation']['safety_margin_m']
    start, goal = config['start'], config['goals'][0]['pose']
    assert rectangle.clearance(start,radius) >= margin
    assert rectangle.clearance(goal,radius) >= margin + config['goal_radius_m']
    direct = rectangle.swept(start,goal,radius+margin)
    assert direct['collision']

    def image_at(pose, name):
        set_pose(camera,pose)
        for _ in range(30): world.step(render=True)
        value=rgb(camera)
        Image.fromarray(value).save(out/f'{name}.png')
        mask=(value[:,:,0].astype(float)>1.4*np.maximum(value[:,:,1],value[:,:,2])) & (value[:,:,0]>80)
        ys,xs=np.nonzero(mask)
        assert len(xs)>1000, 'Obstacle not visible as an opaque red object'
        return value,dict(pixels=len(xs),x=float(xs.mean()),y=float(ys.mean()))

    _, front=image_at(start,'obstacle-front')
    assert abs(front['x']-319.5)<4 and abs(front['y']-239.5)<4
    _, turned=image_at([0.,0.,math.radians(15)],'obstacle-yaw-positive')
    assert turned['x']>front['x']+30, 'Positive yaw must turn left, moving obstacle image right'
    before,_=image_at(start,'before-spectator')
    pose_before=tuple(v.copy() for v in camera.get_world_pose())
    observer=UsdGeom.Camera.Define(world.stage,'/World/ValidationSpectator')
    UsdGeom.Xformable(observer).AddTranslateOp().Set((2.6,-7.2,8.8))
    after,_=image_at(start,'after-spectator')
    spectator_difference=float(np.abs(before.astype(float)-after.astype(float)).mean())
    assert spectator_difference<=1., 'Observer camera unexpectedly changed model RGB'
    for a,b in zip(pose_before,camera.get_world_pose()): np.testing.assert_array_equal(a,b)
    assert not world.stage.GetPrimAtPath('/World/ReplayOverlays').IsValid()
    world.stage.RemovePrim('/World/ValidationSpectator')
    history=history_gate(world,camera,out)

    # Exercise the unchanged socket client with real saved RGB, but no model load.
    errors=[]
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    def serve():
        try:
            conn,_=server.accept()
            with conn,conn.makefile('rb') as stream:
                for line in stream:
                    request=json.loads(line)
                    assert request['op']=='dummy'
                    hashes=[]
                    for frame in request['history']:
                        assert sha256(frame['path'])==frame['png_sha256']
                        hashes.append(pixel_hash(np.asarray(Image.open(frame['path']).convert('RGB'))))
                    assert hashes==[f['rgb_sha256'] for f in request['history']]
                    response=dict(request_id=request['request_id'], observation_id=request['observation_id'],
                                  rgb_hashes=hashes,dummy=True,lossless_roundtrip=True,
                                  pipeline_ms=0.,ready_wall_time=time.time())
                    conn.sendall((json.dumps(response)+'\n').encode())
        except Exception as exc:
            errors.append(repr(exc))
    worker=threading.Thread(target=serve,daemon=True)
    worker.start()
    client=Client(socket_path)
    try:
        dummy=episode(world,camera,config,client,out/'dummy-scheduler',
                      dict(name='MODEL_FREE_DUMMY',pose=goal),False)
    finally:
        client.close()
        worker.join(timeout=10)
        server.close()
    assert not errors and not worker.is_alive(), errors
    assert dummy['status']=='INFRASTRUCTURE_PASS' and dummy['applied_predictions']>0
    assert dummy['final_pose']==start and not dummy['collision']
    # Straight scripted control must terminate on contact without steering repair.
    pose=list(start)
    for step in range(1,1001):
        next_pose=integrate(pose,[.3,0.])
        sweep=rectangle.swept(pose,next_pose,radius)
        pose=next_pose
        if sweep['collision']: break
    assert sweep['collision'] and 399<=step<=402
    result=dict(status='PASS',model_inference_count=0,geometry=geometry,
                start_clearance=rectangle.clearance(start,radius),
                goal_region_minimum_clearance=rectangle.clearance(goal,radius)-config['goal_radius_m'],
                direct_path_with_safety=direct,front_pixels=front,positive_yaw_pixels=turned,
                camera_gate=calibration['status'],history_gate=history['status'],
                spectator_rgb_mean_abs_difference=spectator_difference,
                scientific_stage_has_no_replay_markers=True,
                dummy_ipc_requests=dummy['requests'],dummy_ipc_applied=dummy['applied_predictions'],
                scripted_collision=dict(step=step,pose=pose,sweep=sweep))
    write_json(out/'static-gates.json',result)
    return result
