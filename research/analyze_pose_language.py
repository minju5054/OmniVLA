"""Reproduce saved-data metrics and publication plots; no model or simulation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.isaac_final_common import write_json, sha256
from research.pose_language_contracts import verify_protocol
from research.pose_language_analysis import analyze_offline, analyze_navigation, rows


def compact(offline, navigation):
    predictions = []
    for p in offline['predictions']:
        predictions.append({k: p[k] for k in ('rgb_case_id', 'instruction_id', 'request_id', 'action_sha256',
            'clip_embedding_sha256', 'endpoint_local_displacement', 'local_lateral_min_m', 'local_lateral_max_m',
            'maximum_absolute_local_lateral_m', 'local_heading_vectors', 'latency_ms')} | dict(
                classification=p['polyline']['classification'], minimum_clearance_m=p['polyline']['minimum_clearance_m'],
                selected_command=p['hypothetical_commands'][4]))
    def pair(p):
        result = {k: v for k, v in p.items() if k not in ('present', 'absent')}
        result['first_endpoint_xy_m'] = p['present']['endpoint_local_displacement']
        result['second_endpoint_xy_m'] = p['absent']['endpoint_local_displacement']
        result['first_selected_command'] = p['present']['hypothetical_commands'][4]
        result['second_selected_command'] = p['absent']['hypothetical_commands'][4]
        return result
    return dict(offline=dict(summary=offline['summary'], predictions=predictions,
        directions=offline['directions'], prompt_comparisons=[pair(p) for p in offline['prompt_comparisons']],
        visual_pairs=[pair(p) for p in offline['visual_pairs']]),
        navigation=[{k: v for k, v in e.items() if k != 'native_predictions'} for e in navigation['episodes']])


def plots(output, offline, navigation, run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    colors = dict(N='#2695d0', D='#20843d', L='#df7b1b', R='#bf367c')
    def obstacle(ax):
        ax.add_patch(Rectangle((2.2, -.5), .6, 1., color='#c32929', alpha=.8))
        ax.add_patch(FancyBboxPatch((1.9, -.8), 1.2, 1.6, boxstyle='round,pad=0,rounding_size=0.3',
                                  fill=False, edgecolor='#987500', linestyle='--'))
        ax.set_aspect('equal'); ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y = left (m)')
        ax.grid(alpha=.2)
    fig, axes = plt.subplots(4, 2, figsize=(13, 15), constrained_layout=True)
    cases = list(dict.fromkeys(p['rgb_case_id'] for p in offline['predictions']))
    for ax, case in zip(axes.flat, cases):
        obstacle(ax)
        for p in [p for p in offline['predictions'] if p['rgb_case_id'] == case]:
            xy = np.array([p['pose'][:2]] + p['world_xy'])
            ax.plot(xy[:, 0], xy[:, 1], '.-', color=colors[p['instruction_id']], label=p['instruction_id'])
        ax.set_title(case.replace('OBSTACLE_', '')); ax.legend(ncol=4, fontsize=8)
        ax.set_xlim(-.1, 3.4); ax.set_ylim(-.9, .9)
    fig.suptitle('32 frozen ID-8 native predictions; counterfactual obstacle geometry also shown for ABSENT')
    fig.savefig(output / 'offline-native-matrix.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, e in zip(axes.flat, navigation['episodes']):
        key = e['instruction_id']; obstacle(ax)
        for p in e['native_predictions']:
            xy = np.array([p['observation_pose'][:2]] + p['world_xy'])
            ax.plot(xy[:, 0], xy[:, 1], color='#9461a7', alpha=.35, linewidth=.8)
        states = rows(run / 'trials' / key / e['episode_id'] / 'raw/robot_state.jsonl')
        xy = np.array([s['pose'][:2] for s in states])
        ax.plot(xy[:, 0], xy[:, 1], color=colors[key], linewidth=2, label='Executed')
        ax.scatter([0, 5], [0, 0], c=['black', 'green'], marker='x')
        ax.scatter(*xy[-1], c='red', s=55, marker='x', label='Collision stop')
        ax.set_xlim(-.2, 5.6); ax.set_ylim(-1., 1.)
        ax.set_title(f"{key}: {e['status']} at {e['first_collision_time_sim_s']:.3f} s"); ax.legend(fontsize=8)
    fig.suptitle('Actual paths and all 93 native predictions at their observation poses; no row timestamps')
    fig.savefig(output / 'navigation-paths-native.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12), constrained_layout=True)
    for axrow, e in zip(axes, navigation['episodes']):
        key = e['instruction_id']; folder = run / 'trials' / key / e['episode_id'] / 'raw'
        applied, states = rows(folder / 'controller.jsonl'), rows(folder / 'robot_state.jsonl')
        t = [s['elapsed_sim'] for s in states]
        for i, label in enumerate(('v (m/s)', 'w (rad/s)')):
            axrow[i].plot(t, [s['applied_vw'][i] for s in states], color=colors[key], label='Applied, including held commands')
            axrow[i].scatter([s['application_sim_time'] - 100/60 for s in applied],
                            [s['controller']['pre_clip_vw'][i] for s in applied], s=10, c='#555555', label='Pre-clip')
            axrow[i].set_ylabel(f'{key}: {label}'); axrow[i].set_xlabel('Elapsed simulation (s)')
            axrow[i].grid(alpha=.2)
        axrow[0].legend(fontsize=7)
        axrow[2].plot(t, [s['obstacle_clearance'] for s in states], color=colors[key])
        axrow[2].axhline(.1, c='#987500', linestyle='--'); axrow[2].axhline(0, c='red', linestyle=':')
        axrow[2].set_ylabel('Footprint clearance (m)'); axrow[2].set_xlabel('Elapsed simulation (s)'); axrow[2].grid(alpha=.2)
    fig.suptitle('Unchanged official controller: native selected-row conversion versus actual held commands')
    fig.savefig(output / 'navigation-commands-clearance.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for key in colors:
        pairs = [p for p in offline['visual_pairs'] if p['instruction_id'] == key]
        axes[0].plot([p['pose_id'] for p in pairs],
                     [p['present']['endpoint_local_displacement'][0] - p['absent']['endpoint_local_displacement'][0] for p in pairs],
                     '.-', color=colors[key], label=key)
    for label, color in [('L-N', colors['L']), ('R-N', colors['R']), ('L-R', '#70459e'), ('D-N', colors['D'])]:
        pairs = [p for p in offline['prompt_comparisons'] if p['comparison'] == label]
        axes[1].plot(range(8), [p['per_waypoint_delta_local_xy_m'][-1][1] for p in pairs], '.-', color=color, label=label)
    axes[0].set_ylabel('PRESENT minus ABSENT endpoint forward (m)'); axes[0].set_xlabel('Frozen distance')
    axes[1].set_xticks(range(8), [c.replace('_OBSTACLE_', '\n') for c in cases], fontsize=7)
    axes[1].set_ylabel('Prompt difference in endpoint left (m)')
    for ax in axes:
        ax.axhline(0, color='black', linewidth=.6); ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output / 'visual-versus-language-response.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New derived-only directory')
    args = parser.parse_args()
    _, plan, config = verify_protocol()
    a, b = ROOT / 'outputs/pose-language-obstacle/offline-001', ROOT / 'outputs/pose-language-obstacle/navigation-001'
    offline, navigation = analyze_offline(a, plan, config), analyze_navigation(b, plan, config)
    assert offline == json.loads((a / 'derived/metrics.json').read_text())
    assert navigation == json.loads((b / 'derived/metrics.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'measurements.json', compact(offline, navigation))
    plots(args.output, offline, navigation, b)
    write_json(args.output / 'reproduction.json', dict(status='EXACT_MATCH', live_inferences=0,
        input_metrics_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (a / 'derived/metrics.json', b / 'derived/metrics.json')},
        plots={p.name: sha256(p) for p in args.output.glob('*.png')}))
    print('EXACT_MATCH: 32 offline predictions, 93 navigation predictions, 1830 executed steps')


if __name__ == '__main__':
    main()
