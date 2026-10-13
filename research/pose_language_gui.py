"""Saved Modality-8 navigation loader and overlays; never load a neural model."""
import json
import math
from pathlib import Path
import numpy as np
from research.isaac_final_common import sha256
from research.offline_obstacle_contracts import metric_world
from research.pose_language_contracts import verify_protocol, validate_instruction


def load_language(run, protocol, config):
    _, plan, frozen_config = verify_protocol(protocol)
    assert config == frozen_config
    derived = json.loads((run / 'derived/metrics.json').read_text())
    by_episode = {e['episode_id']: e for e in derived['episodes']}
    recordings = []
    for trial in plan['navigation_order']:
        folder = run / 'trials' / trial['instruction_id'] / trial['episode_id']
        def rows(name):
            return [json.loads(line) for line in (folder / 'raw' / name).read_text().splitlines()]
        states, frames, requests = rows('robot_state.jsonl'), rows('history.jsonl'), rows('requests.jsonl')
        summary = json.loads((folder / 'summary.json').read_text())
        assert states[-1]['pose'] == summary['final_pose']
        assert states[-1]['step'] == summary['completed_steps']
        assert all(a['elapsed_sim'] < b['elapsed_sim'] for a, b in zip(states, states[1:]))
        for frame in frames:
            frame['file'] = folder / 'raw/rgb' / Path(frame['path']).name
            assert sha256(frame['file']) == frame['png_sha256']
        native = {p['request_id']: p for p in by_episode[trial['episode_id']]['native_predictions']}
        for req in requests:
            validate_instruction(req, plan)
            assert req['instruction_id'] == trial['instruction_id']
            saved = native[req['request_id']]
            action_path = folder / 'raw/actions' / Path(req['action_path']).name
            assert sha256(action_path) == saved['action_sha256']
            raw = np.load(action_path, allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (1, 8, 4)
            assert saved['observation_pose'] == req['robot_pose_observation']
            # Float64 geometry is shared across NumPy versions. Do not recalculate
            # the float32 controller in Isaac's different NumPy runtime.
            _, world, _ = metric_world(raw, req['robot_pose_observation'])
            np.testing.assert_allclose(world, saved['world_xy'], atol=1e-12, rtol=0)
        assert all(s['active_request_id'] is None or s['active_request_id'] in native for s in states)
        recordings.append(dict(goal=dict(name=trial['episode_id'], pose=[5., 0., 0.]),
            instruction_id=trial['instruction_id'], instruction_text=plan['instructions'][trial['instruction_id']]['text'],
            states=states, frames=frames, summary=summary, native=native,
            times=[r['elapsed_sim'] for r in states], frame_times=[r['sim_time'] for r in frames],
            state_sha256=sha256(folder / 'raw/robot_state.jsonl'), source_run=str(run.resolve())))
    return recordings


def language_overlays(stage, config, curve, box):
    from pxr import Gf, UsdGeom
    from research.static_obstacle_geometry import Rectangle
    rect = Rectangle.from_config(config['static_obstacle'])
    radius = config['evaluation']['agent_radius_m'] + config['evaluation']['safety_margin_m']
    envelope = []
    for cx, cy, angle in [(rect.xmax, rect.ymax, 0), (rect.xmin, rect.ymax, 90),
                          (rect.xmin, rect.ymin, 180), (rect.xmax, rect.ymin, 270)]:
        for theta in np.linspace(math.radians(angle), math.radians(angle + 90), 17):
            envelope.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta), .08))
    curve('/World/ReplayOverlays/ClearanceEnvelope', envelope + envelope[:1], [1., .8, .1], .025)
    box(stage, '/World/ReplayOverlays/Start', [0., 0., .04], [.12, .12, .08], [1., 1., 1.])
    predicted = curve('/World/ReplayOverlays/NativePrediction', [(0., 0., 1.08)] * 9, [.85, .4, 1.], .022)
    points = UsdGeom.Points.Define(stage, '/World/ReplayOverlays/NativeWaypoints')
    points.CreateWidthsAttr([.055] * 8)
    points.CreateDisplayColorAttr([Gf.Vec3f(.85, .4, 1.)])
    selected = UsdGeom.Points.Define(stage, '/World/ReplayOverlays/SelectedWaypoint4')
    selected.CreateWidthsAttr([.11])
    selected.CreateDisplayColorAttr([Gf.Vec3f(1., 1., 0.)])
    anchor = curve('/World/ReplayOverlays/ObservationAnchor', [(0., 0., .08), (0., 0., 1.08)], [.85, .4, 1.], .015)
    previous = [None]

    def update(recording, row):
        request_id = row['active_request_id']
        key = (recording['instruction_id'], request_id)
        if key != previous[0]:
            previous[0] = key
            for item in (predicted, points, selected, anchor):
                item.GetVisibilityAttr().Set('invisible' if request_id is None else 'inherited')
            if request_id is not None:
                saved = recording['native'][request_id]
                x, y, _ = saved['observation_pose']
                xyz = [Gf.Vec3f(px, py, 1.08) for px, py in saved['world_xy']]
                predicted.GetPointsAttr().Set([Gf.Vec3f(x, y, 1.08)] + xyz)
                points.CreatePointsAttr(xyz)
                selected.CreatePointsAttr([xyz[4]])
                anchor.GetPointsAttr().Set([Gf.Vec3f(x, y, .08), Gf.Vec3f(x, y, 1.08)])
        if request_id is None:
            return 'Native: waiting for first recorded application'
        saved = recording['native'][request_id]
        x, y, yaw = saved['observation_pose']
        return (f"Active chunk #{request_id} | observation ({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f} deg)\n"
                f"{saved['polyline']['classification']} | min {saved['polyline']['minimum_clearance_m']:.3f} m")
    return update
