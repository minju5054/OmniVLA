"""Isaac-only stationary input collection; imports no policy or IPC client."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.isaac_final_common import sha256, write_json, sync_tree
from research.static_obstacle_protocol import verify_static


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    args,_=p.parse_known_args()
    plan_path=ROOT/'research/edge-obstacle-offline-plan.json'
    plan=json.loads(plan_path.read_text())
    config_path=ROOT/'research/frozen-edge-static-obstacle.json'
    config=json.loads(config_path.read_text()); verify_static(config,config_path)
    assert sha256(config_path)==plan['source_static_config_sha256']
    a=ROOT/'outputs/edge-obstacle-offline/existing-chunks/derived-v2/complete.json'
    assert json.loads(a.read_text())['status']=='PASS'
    args.output.mkdir(parents=True,exist_ok=False)
    app=None; code=1
    result=dict(status='FAILED',part_a_complete_sha256=sha256(a),plan_sha256=sha256(plan_path),
        capture_source_sha256=sha256(__file__),robot_execution=False,cases=[])
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer=config['isaac']['renderer'],width=640,height=480,
            multi_gpu=False,active_gpu=0,enable_crashreporter=False,
            extra_args=['--/app/file/ignoreUnsavedOnExit=true','--/app/settings/persistent=false']))
        import numpy as np
        from PIL import Image
        from pxr import UsdGeom
        from research.isaac_scene import build, set_pose, rgb
        from research.static_obstacle_scene import add_obstacle, geometry_record
        from research.isaac_final_contracts import pixel_hash, local_goal
        world,camera=build(config)
        for case in plan['cases']:
            if world.stage.GetPrimAtPath(config['static_obstacle']['path']).IsValid():
                world.stage.RemovePrim(config['static_obstacle']['path'])
            present=case['condition']=='OBSTACLE_PRESENT'
            if present: add_obstacle(world.stage,config)
            set_pose(camera,case['pose'],config['camera']['height_m'])
            for _ in range(plan['history']['settling_steps_before_each_case']): world.step(render=True)
            assert not world.stage.GetPrimAtPath('/World/ReplayOverlays').IsValid()
            paths=[str(prim.GetPath()) for prim in world.stage.Traverse()]
            assert not any('Replay' in path or 'Calibration' in path or 'Spectator' in path for path in paths)
            prim=camera.prim
            cam=UsdGeom.Camera(prim)
            intrinsics={key:cam.GetPrim().GetAttribute(key).Get() for key in
                ('focalLength','horizontalAperture','verticalAperture','horizontalApertureOffset','verticalApertureOffset')}
            local,normalized=local_goal(case['pose'],plan['world_goal'])
            record=dict(**case,world_goal=plan['world_goal'],local_goal=local,normalized_pose=normalized,
                camera_intrinsics=intrinsics,camera_config=config['camera'],renderer=config['isaac']['renderer'],
                scene=config['scene'],stage_paths=paths,no_gui_markers=True,history=[],
                obstacle_prim_present=bool(world.stage.GetPrimAtPath(config['static_obstacle']['path']).IsValid()),
                obstacle_geometry=geometry_record(world.stage,config) if present else None)
            directory=args.output/case['case_id']/'raw/rgb'; directory.mkdir(parents=True)
            last=0
            for i,step in enumerate(plan['history']['frame_steps']):
                for _ in range(step-last): world.step(render=True)
                last=step
                position,orientation=camera.get_world_pose(camera_axes='world')
                expected=np.array([*case['pose'][:2],config['camera']['height_m']])
                assert np.max(np.abs(position-expected))<=plan['input_validity']['stationary_pose_max_abs_error']
                assert np.max(np.abs(orientation-np.array([1.,0,0,0])))<=1e-6
                before=time.time(); array=rgb(camera)
                path=directory/f'{i:06d}.png'
                with path.open('xb') as stream: Image.fromarray(array).save(stream,format='PNG')
                path.chmod(0o444)
                frame=dict(frame_id=case['request_id']*6+i,sim_time=step/60,wall_time=time.time(),
                    capture_start_wall_time=before,absolute_world_sim_time=float(world.current_time),
                    absolute_physics_step=int(world.current_time_step_index),case_step=step,
                    path=str(path.resolve()),png_sha256=sha256(path),rgb_sha256=pixel_hash(array),
                    rgb_shape=list(array.shape),pose=case['pose'],camera_position=position.tolist(),
                    camera_orientation_wxyz=orientation.tolist(),acquisition='independent camera.get_rgba after render')
                record['history'].append(frame)
            write_json(args.output/case['case_id']/'capture.json',record)
            result['cases'].append(record)
            print('CAPTURED '+case['case_id'],flush=True)
        result['status']='PRE_CLOSE_PASS'; code=0
    except BaseException:
        result['traceback']=traceback.format_exc(); print(result['traceback'],flush=True)
    result.update(requested_exit_code=code,close_requested=app is not None,wall_time=time.time())
    write_json(args.output/'result.json',result); sync_tree(args.output)
    if app is None: raise SystemExit(code)
    app.close(exit_code=code)


if __name__=='__main__': main()
