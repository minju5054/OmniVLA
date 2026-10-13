"""Reproducible offline geometry, command diagnostics, and publication plots."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.isaac_final_common import sha256, write_json
from research.offline_obstacle_contracts import evaluate_chunk, compare_pair, LABELS
from research.static_obstacle_protocol import verify_static


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def scene(ax, title):
    ax.add_patch(Patch((2.2, -.5), .6, 1, color='firebrick', alpha=.35))
    # Rounded rectangle gives the Euclidean radius+margin envelope.
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((1.9, -.8), 1.2, 1.6,
        boxstyle='round,pad=0,rounding_size=0.3', fill=False, ec='firebrick', ls='--'))
    ax.axhline(0, color='.8', lw=.6)
    ax.set(xlabel='World X (m)', ylabel='World Y (m)', title=title)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.2)


def span(values):
    a = np.asarray(values)
    return {'min': float(a.min()), 'median': float(np.median(a)), 'max': float(a.max())}


def analyze_existing(config):
    raw = ROOT/'outputs/static-obstacle/scientific-001/isaac/STATIC_CENTER/raw'
    requests, recorded = rows(raw/'requests.jsonl'), rows(raw/'controller.jsonl')
    assert len(requests) == len(recorded) == 21
    chunks = []
    for req, rec in zip(requests, recorded):
        assert req['request_id'] == rec['request_id']
        path = Path(rec['action_path'])
        assert sha256(path) == rec['action_sha256']
        native = np.load(path, allow_pickle=False)
        result = evaluate_chunk(native, req['robot_pose_observation'], config)
        selected = result['hypothetical_commands'][4]
        for key in ('pre_clip_vw', 'clipped_vw', 'final_vw'):
            assert selected[key] == rec['controller'][key], (req['request_id'], key)
        assert selected['final_vw'] == rec['applied_vw']
        assert selected['selected_metric'] == rec['selected_metric']
        chunks.append(dict(request_id=req['request_id'], observation_pose=req['robot_pose_observation'],
            observation_time_sim=req['observation_time_sim'], action_path=str(path.relative_to(ROOT)),
            action_sha256=sha256(path), native_shape=list(native.shape), native_dtype=str(native.dtype),
            native_prediction=native.tolist(), applied_vw=rec['applied_vw'], controller_parity=True, **result))
    counts = Counter(c['polyline']['classification'] for c in chunks)
    commands = [c['hypothetical_commands'][4] for c in chunks]
    summary = dict(count=len(chunks), classifications={k: dict(count=counts[k], fraction=counts[k]/21) for k in LABELS},
        controller_parity_count=21, heading_used_count=sum(c['heading_used'] for c in commands),
        initial_v_clipped_count=sum(c['pre_clip_vw'][0] != c['clipped_vw'][0] for c in commands),
        initial_w_clipped_count=sum(c['pre_clip_vw'][1] != c['clipped_vw'][1] for c in commands),
        ratio_limited_count=sum(c['clipped_vw'] != c['final_vw'] for c in commands),
        near_zero_w_zeroed_count=sum(c['clipped_vw'][1] != 0 and c['final_vw'][1] == 0 for c in commands),
        controller_bug_count=sum('error' in cmd for c in chunks for cmd in c['hypothetical_commands']),
        selected_local_lateral_m=span([c['selected_index4_local_xy'][1] for c in chunks]),
        all_waypoints_max_abs_local_lateral_m=max(c['maximum_absolute_local_lateral_m'] for c in chunks),
        selected_waypoint_collision_count=sum(c['selected_index4_clearance_m'] <= 0 for c in chunks),
        any_safe_passage_prefix_count=sum('PREDICTED_SAFE_PASSAGE' in c['prefix_classifications'] for c in chunks),
        waypoint_collision_counts=np.sum(np.array([c['waypoint_clearance_m'] for c in chunks]) <= 0, axis=0).tolist(),
        minimum_polyline_clearance_m=span([c['polyline']['minimum_clearance_m'] for c in chunks]),
        command_ranges={k: dict(v=span([c[k][0] for c in commands]),w=span([c[k][1] for c in commands]))
            for k in ('pre_clip_vw','clipped_vw','final_vw','initial_clip_delta_vw','ratio_limit_delta_vw','total_limit_delta_vw')},
        curvature_ranges={k: span([c['curvature_w_over_v'][k] for c in commands]) for k in ('pre_clip','clipped','final')})
    return dict(summary=summary, chunks=chunks)


def plot_existing(data, output):
    chunks = data['chunks']; ids = [c['request_id'] for c in chunks]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    ax = axes[0,0]; scene(ax, 'All 21 native spatial polylines')
    for i,c in enumerate(chunks):
        p = np.vstack([c['observation_pose'][:2], c['world_xy']])
        ax.plot(*p.T, color=plt.cm.viridis(i/20), lw=1.1)
    ax = axes[0,1]
    ax.plot(ids, [c['polyline']['minimum_clearance_m'] for c in chunks], 'o-')
    ax.axhline(0,color='firebrick'); ax.axhline(.1,color='orange',ls='--')
    ax.set(title='Swept clearance (observation included)',xlabel='Request ID',ylabel='Clearance (m)')
    ax = axes[0,2]
    ax.plot(ids,[c['selected_index4_local_xy'][1] for c in chunks],'o-',label='Row 4')
    ax.plot(ids,[c['local_lateral_min_m'] for c in chunks],ls='--',label='Min of 8')
    ax.plot(ids,[c['local_lateral_max_m'] for c in chunks],ls='--',label='Max of 8')
    ax.set(title='Local lateral displacement',xlabel='Request ID',ylabel='Y (m)'); ax.legend()
    ax = axes[1,0]
    ax.boxplot(np.array([c['waypoint_clearance_m'] for c in chunks]), positions=range(8))
    ax.axhline(0,color='firebrick'); ax.axhline(.1,color='orange',ls='--')
    ax.set(title='All-waypoint clearance distribution',xlabel='Waypoint index (not time)',ylabel='Clearance (m)')
    ax = axes[1,1]
    for key,label in [('pre_clip_vw','Pre-clip'),('clipped_vw','Initial clip'),('final_vw','Final / applied')]:
        ax.plot(ids,[c['hypothetical_commands'][4][key][1] for c in chunks],marker='.',label=label)
    ax.set(title='Selected row 4 angular commands',xlabel='Request ID',ylabel='w (rad/s)'); ax.legend()
    ax = axes[1,2]
    for k in ('pre_clip','clipped','final'):
        ax.plot(ids,[c['hypothetical_commands'][4]['curvature_w_over_v'][k] for c in chunks],label=k)
    ax.set(title='Curvature (near-zero w may be zeroed)',xlabel='Request ID',ylabel='w/v (1/m)'); ax.legend()
    fig.suptitle('Part A — immutable recorded predictions; spatial interpolation has no waypoint timing')
    fig.savefig(output/'all-21-diagnostics.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,c in zip(axes.flat,chunks[-6:]):
        p = np.array(c['world_xy']); start = c['observation_pose'][:2]
        scene(ax,f"Request {c['request_id']}: {c['polyline']['classification']}\nmin {c['polyline']['minimum_clearance_m']:.4f} m")
        ax.plot(*np.vstack([start,p]).T,'o-',ms=4)
        ax.scatter(*p[4],s=100,c='orange',marker='*',zorder=5)
        ax.scatter(*start,c='black',marker='s',s=30)
        for j,xy in enumerate(p): ax.annotate(str(j),xy,xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set_xlim(1,3.7); ax.set_ylim(-.95,.95)
    fig.suptitle('Final six predictions — square: observation; star: selected row 4; no executed alternatives')
    fig.savefig(output/'last-six-chunks.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--part',choices=['A'],required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    config_path=ROOT/'research/frozen-edge-static-obstacle.json'
    config=json.loads(config_path.read_text()); verify_static(config,config_path)
    args.output.mkdir(parents=True,exist_ok=False)
    if args.part == 'A':
        data=analyze_existing(config); write_json(args.output/'metrics.json',data)
        plot_existing(data,args.output)
        write_json(args.output/'complete.json',dict(status='PASS',wall_time=time.time(),
            metrics_sha256=sha256(args.output/'metrics.json'),source_sha256=sha256(__file__),summary=data['summary']))
        print(json.dumps(data['summary'],indent=2))


if __name__ == '__main__': main()
