"""Isaac scene adapters; imported only after SimulationApp construction."""
import math
from pathlib import Path
import time
import numpy as np
from PIL import Image
from research.isaac_scene import box, rgb, set_pose
from research.isaac_final_common import append_json, sha256, write_json
from research.isaac_final_contracts import History, local_goal, pixel_hash


def capture(camera, root, history, frame_id, sim_time, pose):
    image = rgb(camera)
    assert np.isfinite(image).all() and image.std() > 1
    path = root/'raw/rgb'/f'{frame_id:06d}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        Image.fromarray(image).save(stream, format='PNG')
    path.chmod(0o444)
    frame = dict(frame_id=frame_id, sim_time=sim_time, wall_time=time.time(),
                 path=str(path.resolve()), rgb_sha256=pixel_hash(image),
                 png_sha256=sha256(path), pose=list(pose))
    history.append(frame)
    append_json(root/'raw/history.jsonl', frame)
    return frame


def camera_gate(world, camera, config, out):
    set_pose(camera, [0., 0., 0.])
    scene = config['scene']
    paths = []
    for name, position in scene['calibration_panels'].items():
        path = '/World/Calibration'+name
        box(world.stage, path, position, scene['calibration_panel_size'], scene['calibration_colors'][name])
        paths.append(path)
    for _ in range(30):
        world.step(render=True)
    image = rgb(camera)
    Image.fromarray(image).save(out/'calibration-rgb.png')
    values = image.astype(float)
    centroids = {}
    for name, channel in [('FRONT',0), ('LEFT',1), ('RIGHT',2)]:
        other = np.max(values[:, :, [k for k in range(3) if k != channel]], axis=2)
        mask = (values[:, :, channel] > 80) & (values[:, :, channel] > 1.4*other)
        ys, xs = np.nonzero(mask)
        centroids[name] = dict(pixels=len(xs), x=float(xs.mean()) if len(xs) else None,
                               y=float(ys.mean()) if len(ys) else None)
    fov = math.degrees(2*math.atan(camera.get_horizontal_aperture()/(2*camera.get_focal_length())))
    result = dict(centroids=centroids, horizontal_fov_deg=fov,
                  criterion='>=30 dominant-color pixels each; FRONT within 64px of center, LEFT < FRONT-60, RIGHT > FRONT+60',
                  resolution=list(camera.get_resolution()))
    good = all(c['pixels'] >= 30 for c in centroids.values())
    if good:
        center = centroids['FRONT']['x']
        good = (abs(center-320) <= 64 and centroids['LEFT']['x'] < center-60
                and centroids['RIGHT']['x'] > center+60 and abs(fov-90) < 1e-4)
    result['status'] = 'PASS' if good else 'FAIL'
    write_json(out/'camera-gate.json', result)
    for path in paths:
        world.stage.RemovePrim(path)
    for _ in range(30):
        world.step(render=True)
    assert good, f'Camera-axis validation failed: {result}'
    return result


def history_gate(world, camera, out):
    root = out/'history-gate'
    (root/'raw').mkdir(parents=True)
    h = History()
    start = float(world.current_time)
    for step in range(101):
        if step:
            world.step(render=True)
        if step % 20 == 0:
            capture(camera, root, h, step//20, step/60, [0.,0.,0.])
    stationary = h.snapshot(100/60)
    # Scripted, model-free translation proves rolling history uses past captures.
    for step in range(101,221):
        pose = [(step-100)/600, 0., 0.]
        set_pose(camera, pose)
        world.step(render=True)
        if step % 20 == 0:
            capture(camera, root, h, step//20, step/60, pose)
    moving = h.snapshot(220/60)
    assert len({f['frame_id'] for f in stationary}) == 6
    assert len({f['rgb_sha256'] for f in moving}) > 1
    actual_elapsed = float(world.current_time)-start
    assert abs(actual_elapsed-220/60) < 1e-4
    set_pose(camera, [0.,0.,0.])
    for _ in range(30):
        world.step(render=True)
    result = dict(status='PASS', stationary=stationary, moving=moving,
                  actual_sim_elapsed=actual_elapsed, adapter_interval_s=1/3)
    write_json(out/'history-gate.json', result)
    return result


def qualify(world, camera, config, args):
    if args.phase == 'camera':
        return dict(camera_gate=camera_gate(world, camera, config, args.output),
                    history_gate=history_gate(world, camera, args.output))
    from research.isaac_final_navigation import run_phase
    return run_phase(world, camera, config, args)
