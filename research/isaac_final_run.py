"""Isaac-only process. Default fast shutdown is the final lifecycle action."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.isaac_final_common import sha256, sync_tree, verify_frozen, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['lifecycle', 'camera', 'combined', 'scientific'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'research/frozen-edge-isaac-goals.json')
    parser.add_argument('--socket')
    args, _ = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    if args.phase == 'scientific':
        verify_frozen(config)
    result = dict(phase=args.phase, status='FAILED', constructor_returned=False,
                  steps=0, evidence_written_before_close=True,
                  config_sha256=sha256(args.config))
    app = None
    code = 1
    start = time.perf_counter()
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'renderer': config['isaac']['renderer'],
                             'width': 640, 'height': 480, 'multi_gpu': False, 'active_gpu': 0,
                             'enable_crashreporter': False,
                             'extra_args': ['--/app/file/ignoreUnsavedOnExit=true',
                                            '--/app/settings/persistent=false']})
        result.update(constructor_returned=True, constructor_seconds=time.perf_counter()-start,
                      fast_shutdown=app.config['fast_shutdown'])
        from research.isaac_scene import build, rgb
        from PIL import Image
        import numpy as np
        world, camera = build(config)
        result['stage_opened'] = world.stage is not None
        for i in range(60):
            world.step(render=True)
            result['steps'] = i + 1
        image = rgb(camera)
        assert np.isfinite(image).all() and image.std() > 1
        Image.fromarray(image).save(args.output / 'startup-rgb.png')
        result.update(camera_rendered=True, rgb_shape=list(image.shape), rgb_std=float(image.std()))
        if args.phase != 'lifecycle':
            from research.isaac_final_adapter import qualify
            result.update(qualify(world, camera, config, args))
        result['status'] = 'PRE_CLOSE_PASS'
        code = 0
    except BaseException:
        result['traceback'] = traceback.format_exc()
        print(result['traceback'], flush=True)
    result.update(pre_close_wall_seconds=time.perf_counter()-start, requested_exit_code=code,
                  close_requested=app is not None)
    write_json(args.output / 'result.json', result)
    sync_tree(args.output)
    print(json.dumps(result), flush=True)
    if app is None:
        raise SystemExit(code)
    app.close(exit_code=code)


if __name__ == '__main__':
    main()
