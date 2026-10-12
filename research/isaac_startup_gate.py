"""One simulator-only startup gate; never loads the Edge model."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args,_=parser.parse_known_args()
args.output.mkdir(parents=True,exist_ok=False)
result={'status':'ISAAC_STARTUP_FAILED','constructor_returned':False,'steps':0,'clean_shutdown':False}
app=None
start=time.perf_counter()
try:
    from isaacsim import SimulationApp
    config=json.loads((ROOT/'research/frozen-edge-isaac-goals.json').read_text())
    app=SimulationApp({'headless':True,'renderer':config['isaac']['renderer'],
                       'width':640,'height':480,'multi_gpu':False,'active_gpu':0,
                       'fast_shutdown':False,'enable_crashreporter':False,
                       'extra_args':['--/app/file/ignoreUnsavedOnExit=true', '--/app/settings/persistent=false']})
    result['constructor_returned']=True
    result['constructor_seconds']=time.perf_counter()-start
    from research.isaac_scene import build,rgb
    from PIL import Image
    import numpy as np
    world,camera=build(config)
    result['stage_opened']=world.stage is not None
    for i in range(60):
        world.step(render=True)
        result['steps']=i+1
    image=rgb(camera)
    assert float(image.std())>1., 'Rendered RGB is visually empty'
    Image.fromarray(image).save(args.output/'startup-rgb.png')
    result.update(rgb_shape=list(image.shape),rgb_std=float(image.std()),rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),simulation_time=float(world.current_time),isaac_version=Path('/home/gpuadmin/isaacsim/VERSION').read_text().strip())
    result['status']='ISAAC_STARTUP_PASS'
except BaseException:
    result['traceback']=traceback.format_exc()
    print(result['traceback'],flush=True)
finally:
    if app is not None:
        try:
            app.close()
            result['clean_shutdown']=True
        except BaseException:
            result['shutdown_traceback']=traceback.format_exc()
            result['status']='ISAAC_STARTUP_FAILED'
    result['wall_seconds']=time.perf_counter()-start
    with (args.output/'result.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
sys.exit(0 if result['status']=='ISAAC_STARTUP_PASS' else 1)
