"""Interactive Isaac GUI replay of immutable recorded navigation evidence.

This viewer loads no policy and runs no navigation experiment. The arrow,
paths, and goal rings are visualization overlays; the original SE(2) experiment
used a kinematic camera. The inset displays the original recorded RGB files.
"""
import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.isaac_final_common import sha256, verify_frozen, write_json


def load_recordings(run, config):
    recordings = []
    for goal in config['goals']:
        folder = run / 'isaac' / goal['name']
        state_path = folder / 'raw/robot_state.jsonl'
        states = [json.loads(line) for line in state_path.read_text().splitlines()]
        frames = [json.loads(line) for line in (folder / 'raw/history.jsonl').read_text().splitlines()]
        summary = json.loads((folder / 'summary.json').read_text())
        assert states and summary['completed_steps'] == states[-1]['step']
        assert states[-1]['pose'] == summary['final_pose']
        assert all(math.isfinite(v) for row in states for v in row['pose'])
        assert all(a['elapsed_sim'] < b['elapsed_sim'] for a, b in zip(states, states[1:]))
        for frame in frames:
            # Resolve from the supplied run so copied evidence also works.
            frame['file'] = folder / 'raw/rgb' / Path(frame['path']).name
            if sha256(frame['file']) != frame['png_sha256']:
                raise ValueError(f"Recorded RGB hash mismatch: {frame['file']}")
        recordings.append(dict(goal=goal, states=states, frames=frames, summary=summary,
                               times=[row['elapsed_sim'] for row in states],
                               frame_times=[row['sim_time'] for row in frames],
                               state_sha256=sha256(state_path)))
    return recordings


def run_gui(app, recordings, config, output):
    import numpy as np
    import omni.ui as ui
    from pxr import Gf, UsdGeom
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    from isaacsim.core.utils.viewports import set_camera_view
    from research.isaac_scene import build, box, set_pose

    world, camera = build(config)
    stage = world.stage
    if 'static_obstacle' in config:
        from research.static_obstacle_scene import add_obstacle
        add_obstacle(stage, config)
    colors = [(0.15, 0.7, 1.0), (0.2, 1.0, 0.42), (1.0, 0.5, 0.12)]

    def curve(path, points, color, width):
        item = UsdGeom.BasisCurves.Define(stage, path)
        item.CreateTypeAttr('linear')
        item.CreateCurveVertexCountsAttr([len(points)])
        item.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        item.CreateWidthsAttr([width])
        item.SetWidthsInterpolation('constant')
        item.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return item

    UsdGeom.Xform.Define(stage, '/World/ReplayOverlays')
    for index, recording in enumerate(recordings):
        color = colors[index]
        name = recording['goal']['name']
        curve(f'/World/ReplayOverlays/Path{name}',
              [(row['pose'][0], row['pose'][1], 0.035) for row in recording['states']], color, 0.035)
        gx, gy, _ = recording['goal']['pose']
        curve(f'/World/ReplayOverlays/Goal{name}',
              [(gx + config['goal_radius_m'] * math.cos(t),
                gy + config['goal_radius_m'] * math.sin(t), 0.055)
               for t in np.linspace(0, 2 * math.pi, 65)], color, 0.045)
        box(stage, f'/World/ReplayOverlays/Center{name}', [gx, gy, 0.06], [0.13, 0.13, 0.12], color)

    agent = UsdGeom.Xform.Define(stage, '/World/ReplayOverlays/PoseArrow')
    translate = agent.AddTranslateOp()
    rotate = agent.AddRotateZOp()
    box(stage, '/World/ReplayOverlays/PoseArrow/Body', [-0.08, 0, 0.17],
        [0.48, 0.32, 0.22], [0.9, 0.93, 1.0])
    arrow = UsdGeom.Mesh.Define(stage, '/World/ReplayOverlays/PoseArrow/Heading')
    arrow.CreatePointsAttr([Gf.Vec3f(0.42, 0, 0.29), Gf.Vec3f(-0.14, 0.2, 0.29),
                           Gf.Vec3f(-0.14, -0.2, 0.29)])
    arrow.CreateFaceVertexCountsAttr([3])
    arrow.CreateFaceVertexIndicesAttr([0, 1, 2])
    arrow.CreateSubdivisionSchemeAttr('none')
    arrow.CreateDoubleSidedAttr(True)
    arrow.CreateDisplayColorAttr([Gf.Vec3f(*colors[0])])
    if 'evaluation' in config:
        radius = config['evaluation']['agent_radius_m']
        curve('/World/ReplayOverlays/PoseArrow/EvaluationFootprint',
              [(radius*math.cos(t), radius*math.sin(t), 0.07)
               for t in np.linspace(0, 2*math.pi, 65)], [1.,1.,1.], 0.02)

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError('No visible Isaac viewport')
    viewport.fill_frame = True

    def overview():
        viewport.camera_path = '/OmniverseKit_Persp'
        set_camera_view(np.array([2.6, -7.2, 8.8]), np.array([2.6, 0., 0.]),
                        viewport_api=viewport)

    overview()
    state = dict(index=0, elapsed=0., playing=True, image=None)

    def select(index):
        state.update(index=index, elapsed=0., playing=True, image=None)
        arrow.GetDisplayColorAttr().Set([Gf.Vec3f(*colors[index])])
        with (output / 'playback-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event='selected', goal=recordings[index]['goal']['name'],
                                         wall_time=time.time())) + '\n')

    def toggle():
        state['playing'] = not state['playing']

    window = ui.Window('OmniVLA | Recorded navigation', width=370, height=750)
    with window.frame:
        with ui.VStack(spacing=9, height=0):
            ui.Label('OmniVLA-edge', height=30, style={'font_size': 24})
            ui.Label('RECORDED RESULT REPLAY', height=22)
            ui.Label('White arrow = recorded SE(2) pose\nColored lines = saved paths\nRings = 0.50 m goal tolerance',
                     height=64, word_wrap=True)
            with ui.HStack(height=34, spacing=5):
                for index, item in enumerate(recordings):
                    ui.Button(item['goal']['name'], clicked_fn=lambda i=index: select(i))
            with ui.HStack(height=32, spacing=5):
                play_button = ui.Button('Pause', clicked_fn=toggle)
                ui.Button('Restart', clicked_fn=lambda: select(state['index']))
                ui.Button('Overview', clicked_fn=overview)
            with ui.HStack(height=24, spacing=7):
                auto_model = ui.SimpleBoolModel(True)
                ui.CheckBox(model=auto_model, width=20)
                ui.Label('Cycle recorded goals', width=145)
                ui.Label('Speed', width=45)
                speed_model = ui.SimpleFloatModel(1.)
                ui.FloatDrag(model=speed_model, min=0.25, max=4., step=0.25)
            title = ui.Label('', height=25, style={'font_size': 19})
            progress = ui.SimpleFloatModel(0.)
            ui.ProgressBar(model=progress, height=18)
            stats = ui.Label('', height=120, word_wrap=True)
            ui.Separator(height=2)
            ui.Label('Original camera recording (3 Hz)', height=24)
            recorded_image = ui.Image(str(recordings[0]['frames'][5]['file']), height=240,
                                      fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT)
            ui.Label('Playback: 1x = recorded simulation time.\nArrow and goal marks are viewer overlays.\nNo policy inference runs in this viewer.',
                     height=62, word_wrap=True)
            ui.Spacer(height=0)

    for _ in range(30):
        world.render()
    for name in ('Stage', 'Layer', 'Property', 'Content', 'Console', 'Render Settings'):
        panel = ui.Workspace.get_window(name)
        if panel:
            panel.visible = False
    window.dock_in(ui.Workspace.get_window('Viewport'), ui.DockPosition.RIGHT, 0.30)
    select(0)
    write_json(output / 'ready.json', dict(status='GUI_REPLAY_READY', scientific=False,
               source_run=str(recordings[0]['frames'][0]['file'].parents[4]), original_rgb=True,
               input_state_sha256={item['goal']['name']: item['state_sha256'] for item in recordings}))
    print('GUI_REPLAY_READY', flush=True)
    last = time.perf_counter()
    next_status = last
    capture = None
    frames_rendered = 0
    while app.is_running() and not (output / 'stop-requested').exists():
        now = time.perf_counter()
        delta, last = now - last, now
        if state['playing']:
            state['elapsed'] += delta * speed_model.as_float
        recording = recordings[state['index']]
        duration = recording['times'][-1]
        if state['elapsed'] > duration + 3 and auto_model.as_bool:
            select((state['index'] + 1) % len(recordings))
            recording = recordings[state['index']]
            duration = recording['times'][-1]
        elapsed = min(state['elapsed'], duration)
        index = max(0, bisect_right(recording['times'], elapsed) - 1)
        row = recording['states'][index]
        x, y, yaw = row['pose']
        translate.Set(Gf.Vec3d(x, y, 0.))
        rotate.Set(math.degrees(yaw))
        set_pose(camera, row['pose'], config['camera']['height_m'])
        frame_index = max(0, bisect_right(recording['frame_times'], row['sim_time'] + 1e-8) - 1)
        image_path = str(recording['frames'][frame_index]['file'])
        if state['image'] != image_path:
            recorded_image.source_url = image_path
            state['image'] = image_path
        finished = index == len(recording['states']) - 1
        title.text = recording['goal']['name'] + (' | FINISHED' if finished else ' | PLAYBACK')
        progress.set_value(elapsed / duration if duration else 1.)
        play_button.text = 'Pause' if state['playing'] else 'Play'
        stats.text = (f"Time  {elapsed:5.2f} / {duration:.2f} s\n"
                      f"Goal distance  {row['goal_distance']:.3f} m\n"
                      f"Pose  x {x:.2f}   y {y:.2f}   yaw {math.degrees(yaw):.1f} deg\n"
                      f"Recorded result: {recording['summary']['status']}")
        if 'obstacle_clearance' in row:
            stats.text += (f"\nClearance  {row['obstacle_clearance']:.3f} m  (required 0.10 m)"
                           f"\nCollision: {row['collision']}   Margin violation: {row['safety_margin_violation']}")
        world.render()
        frames_rendered += 1
        if frames_rendered == 120:
            capture = capture_viewport_to_file(viewport, str(output / 'overview.png'))
        if now >= next_status:
            (output / 'playback-status.json').write_text(json.dumps(dict(
                goal=recording['goal']['name'], elapsed_sim=elapsed, pose=row['pose'],
                playing=state['playing'], frame_index=frame_index, wall_time=time.time(),
                rendered_frames=frames_rendered), indent=2) + '\n')
            next_status = now + 2
        time.sleep(max(0., 1/60 - (time.perf_counter() - now)))
    # Keep references to the UI and capture alive for the lifetime of the loop.
    return dict(rendered_frames=frames_rendered, window_title=window.title,
                screenshot_requested=capture is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/edge-final/scientific-001')
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'research/frozen-edge-isaac-no-obstacle-final.json')
    parser.add_argument('--offline-protocol', type=Path,
                        help='Explicit stationary offline prediction viewer; no navigation playback')
    parser.add_argument('--output', type=Path, required=True,
                        help='New local viewer diagnostics directory; never an experiment directory')
    args, _ = parser.parse_known_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error('Viewer output must be outside the immutable experiment directory')
    config = json.loads(args.config.read_text())
    verify_frozen(config)
    if 'static_obstacle' in config:
        from research.static_obstacle_protocol import verify_static
        verify_static(config, args.config)
    if args.offline_protocol:
        from research.offline_obstacle_gui import load_offline, run_offline_gui
        recordings = load_offline(args.offline_protocol, config)
    else:
        recordings = load_recordings(args.run, config)
    args.output.mkdir(parents=True, exist_ok=False)
    app, code, result = None, 1, {}
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=False, renderer=config['isaac']['renderer'],
              width=1280, height=800, window_width=1600, window_height=1000,
              active_gpu=0, multi_gpu=False, enable_crashreporter=False,
              extra_args=['--/app/file/ignoreUnsavedOnExit=true',
                          '--/app/settings/persistent=false']))
        result = (run_offline_gui(app, recordings, config, args.output) if args.offline_protocol
                  else run_gui(app, recordings, config, args.output))
        code = 0
    except BaseException:
        result['traceback'] = traceback.format_exc()
        print(result['traceback'], flush=True)
    write_json(args.output / 'exit.json', dict(**result, exit_code=code, scientific=False))
    if app is None:
        raise SystemExit(code)
    app.close(exit_code=code)


if __name__ == '__main__':
    main()
