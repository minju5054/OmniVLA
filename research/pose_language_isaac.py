"""Independent Isaac episode process; original scene, bootstrap and episode code."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import write_json,sync_tree,sha256
from research.pose_language_contracts import PLAN,verify_protocol


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--socket',required=True);p.add_argument('--instruction',choices=['N','D','L','R'],required=True)
    p.add_argument('--infrastructure',action='store_true');args,_=p.parse_known_args()
    if args.infrastructure:
        from research.static_obstacle_protocol import verify_static
        plan=json.loads(PLAN.read_text());config=json.loads((ROOT/plan['static_config']).read_text())
        verify_static(config,ROOT/plan['static_config'])
    else:_,plan,config=verify_protocol()
    trial=next(t for t in plan['navigation_order'] if t['instruction_id']==args.instruction)
    spec=plan['instructions'][args.instruction]
    args.output.mkdir(parents=True,exist_ok=False)
    app=None;code=1;start=time.perf_counter()
    result=dict(status='FAILED',scientific=not args.infrastructure,evidence_written_before_close=True,
        instruction=spec,trial=trial,plan_sha256=sha256(PLAN))
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer=config['isaac']['renderer'],width=640,height=480,
            multi_gpu=False,active_gpu=0,enable_crashreporter=False,
            extra_args=['--/app/file/ignoreUnsavedOnExit=true','--/app/settings/persistent=false']))
        result.update(constructor_returned=True,constructor_seconds=time.perf_counter()-start,
            fast_shutdown=app.config['fast_shutdown'])
        from research.isaac_scene import build,rgb
        from research.static_obstacle_scene import add_obstacle,geometry_record
        from research.isaac_final_contracts import Client
        from research.pose_language_adapter import language_episode
        from PIL import Image
        world,camera=build(config);add_obstacle(world.stage,config)
        result['stage_opened']=True
        for _ in range(60):world.step(render=True)
        image=rgb(camera);assert image.std()>1
        Image.fromarray(image).save(args.output/'startup-rgb.png')
        result.update(steps=60,camera_rendered=True,geometry_before=geometry_record(world.stage,config))
        assert not world.stage.GetPrimAtPath('/World/ReplayOverlays').IsValid()
        client=Client(args.socket)
        try:
            result['episode']=language_episode(spec)(world,camera,config,client,args.output/trial['episode_id'],
                dict(name=trial['episode_id'],pose=plan['world_goal']),not args.infrastructure)
        finally:client.close()
        result['geometry_after']=geometry_record(world.stage,config)
        assert result['geometry_before']==result['geometry_after']
        result['status']='PRE_CLOSE_PASS';code=0
    except BaseException:
        result['traceback']=traceback.format_exc();print(result['traceback'],flush=True)
    result.update(requested_exit_code=code,close_requested=app is not None,wall_seconds=time.perf_counter()-start)
    write_json(args.output/'result.json',result);sync_tree(args.output)
    for path in args.output.rglob('*'):
        if path.is_file():path.chmod(0o444)
    if app is None:raise SystemExit(code)
    app.close(exit_code=code)


if __name__=='__main__':main()
