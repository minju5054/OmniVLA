"""Isaac-only static obstacle entry point, with the validated shutdown lifecycle."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import sha256, sync_tree, write_json
from research.static_obstacle_protocol import validate_scenario, verify_static


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['infrastructure','scientific'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--socket',required=True)
    args,_=parser.parse_known_args()
    config=json.loads(args.config.read_text())
    validate_scenario(config)
    if args.phase=='scientific': verify_static(config,args.config)
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(phase=args.phase,status='FAILED',constructor_returned=False,steps=0,
                evidence_written_before_close=True,config_sha256=sha256(args.config))
    app,code=None,1
    start=time.perf_counter()
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,renderer=config['isaac']['renderer'],
             width=640,height=480,multi_gpu=False,active_gpu=0,enable_crashreporter=False,
             extra_args=['--/app/file/ignoreUnsavedOnExit=true','--/app/settings/persistent=false']))
        result.update(constructor_returned=True,constructor_seconds=time.perf_counter()-start,
                      fast_shutdown=app.config['fast_shutdown'])
        from research.isaac_scene import build,rgb
        from research.static_obstacle_scene import add_obstacle,geometry_record
        from PIL import Image
        world,camera=build(config)
        result['stage_opened']=True
        if args.phase=='scientific': add_obstacle(world.stage,config)
        for _ in range(60): world.step(render=True)
        result['steps']=60
        image=rgb(camera)
        assert image.std()>1
        Image.fromarray(image).save(args.output/'startup-rgb.png')
        result.update(camera_rendered=True,rgb_shape=list(image.shape))
        if args.phase=='infrastructure':
            from research.static_obstacle_validation import qualify
            result['static_gates']=qualify(world,camera,config,args.output,args.socket)
        else:
            from research.static_obstacle_navigation import episode
            from research.isaac_final_contracts import Client
            result['geometry_before']=geometry_record(world.stage,config)
            assert not world.stage.GetPrimAtPath('/World/ReplayOverlays').IsValid()
            client=Client(args.socket)
            try:
                spec=config['goals'][0]
                result['scientific_results']={spec['name']:episode(world,camera,config,client,
                                                        args.output/spec['name'],spec,True)}
            finally:
                client.close()
            result['geometry_after']=geometry_record(world.stage,config)
            assert result['geometry_before']==result['geometry_after']
        result['status']='PRE_CLOSE_PASS'
        code=0
    except BaseException:
        result['traceback']=traceback.format_exc()
        print(result['traceback'],flush=True)
    result.update(pre_close_wall_seconds=time.perf_counter()-start,requested_exit_code=code,
                  close_requested=app is not None)
    write_json(args.output/'result.json',result)
    sync_tree(args.output)
    print(json.dumps(result),flush=True)
    if app is None: raise SystemExit(code)
    app.close(exit_code=code)


if __name__=='__main__': main()
