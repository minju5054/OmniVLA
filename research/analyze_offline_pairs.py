"""Paired outputs against the same counterfactual geometry, with no execution."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import sha256, write_json
from research.isaac_final_contracts import stats
from research.offline_obstacle_contracts import compare_pair, evaluate_chunk
from research.offline_obstacle_protocol import verify_protocol
from research.analyze_offline_obstacle import rows, scene


def analyze(run, protocol):
    frozen,plan,manifest,config=verify_protocol(protocol)
    records=rows(run/'raw/cases.jsonl')
    outcome=json.loads((run/'outcome.json').read_text())
    native={};cases=[]
    for r in records:
        response=r['response'];path=Path(response['action_path'])
        assert sha256(path)==response['action_sha256']
        raw=np.load(path,allow_pickle=False)
        assert raw.tolist()==r['native_prediction']
        native[r['case_id']]=raw
        metrics=evaluate_chunk(raw,r['pose'],config)
        command=metrics['hypothetical_commands'][4]
        if 'controller' in response:
            for key in ('pre_clip_vw','clipped_vw','final_vw'): assert command[key]==response['controller'][key]
        else: assert command['error']=='UPSTREAM_CONTROLLER_BUG'
        cases.append(dict(case_id=r['case_id'],distance_to_front_m=r['distance_to_front_m'],pose=r['pose'],
            native_shape=list(raw.shape),native_dtype=str(raw.dtype),native_prediction=raw.tolist(),
            action_sha256=sha256(path),action_path=str(path.relative_to(ROOT)),cold_inference=r['cold_inference'],
            latency_ms={key:response[key] for key in ('model_ms','forward_with_clip_ms','pipeline_ms')},**metrics))
    pairs=[]
    for p,a in zip(plan['cases'][::2],plan['cases'][1::2]):
        if p['case_id'] not in native or a['case_id'] not in native:
            pairs.append(dict(pose_id=p['pose_id'],status='UNAVAILABLE',reason='See run outcome case statuses'));continue
        pair=compare_pair(native[p['case_id']],native[a['case_id']],p['pose'],config,
            plan['screening']['material_lateral_threshold_m'])
        pairs.append(dict(pose_id=p['pose_id'],pose=p['pose'],distance_to_front_m=p['distance_to_front_m'],status='COMPLETE',**pair))
    complete=[p for p in pairs if p['status']=='COMPLETE']
    summary=dict(status=outcome['status'],inference_count=len(records),model_load_count=outcome['service_stopped']['load_count'],
        material_lateral_response_pairs=sum(p['material_lateral_response'] for p in complete),
        complete_pairs=len(complete),robot_execution=False,
        cold_latency_ms=cases[0]['latency_ms'] if cases else None,
        subsequent_latency_ms={k:stats([c['latency_ms'][k] for c in cases[1:]]) for k in ('model_ms','forward_with_clip_ms','pipeline_ms')},
        pairs=[dict(pose_id=p['pose_id'],distance_to_front_m=p['distance_to_front_m'],
            maximum_absolute_lateral_difference_m=p['maximum_absolute_lateral_difference_m'],
            rms_position_difference_m=p['rms_position_difference_m'],
            selected_index4_displacement_difference_m=p['selected_index4_displacement_difference_m'],
            selected_index4_delta_local_xy_m=p['selected_index4_delta_local_xy_m'],
            selected_index4_command_delta=p['selected_index4_command_delta'],
            material_lateral_response=p['material_lateral_response'],
            present_classification=p['present']['polyline']['classification'],
            absent_counterfactual_classification=p['absent']['polyline']['classification'],
            present_minimum_clearance_m=p['present']['polyline']['minimum_clearance_m'],
            absent_counterfactual_minimum_clearance_m=p['absent']['polyline']['minimum_clearance_m']) for p in complete])
    return dict(summary=summary,cases=cases,pairs=pairs,input_validity=manifest['pair_metrics'],
        protocol_sha256=sha256(protocol),outcome_sha256=sha256(run/'outcome.json'))


def plots(data, manifest, output):
    pairs=[p for p in data['pairs'] if p['status']=='COMPLETE']
    fig,axes=plt.subplots(4,3,figsize=(16,15),constrained_layout=True)
    for i,(pair,p,a) in enumerate(zip(pairs,manifest['cases'][::2],manifest['cases'][1::2])):
        ax=axes[i,0];scene(ax,f"Front distance {pair['distance_to_front_m']:.1f} m\nmax |delta Y| {pair['maximum_absolute_lateral_difference_m']:.4f} m")
        for key,color in [('present','tab:red'),('absent','tab:blue')]:
            xy=np.array(pair[key]['world_xy'])
            ax.plot(*np.vstack([pair['pose'][:2],xy]).T,'o-',c=color,ms=3,label=key)
            ax.scatter(*xy[4],c=color,marker='*',s=110,zorder=5)
        ax.scatter(*pair['pose'][:2],marker='s',c='black');ax.set_xlim(0,3.6);ax.set_ylim(-.9,.9);ax.legend(fontsize=8)
        for j,case in enumerate((p,a),1):
            axes[i,j].imshow(Image.open(case['history'][-1]['path']));axes[i,j].axis('off')
            key='present' if j==1 else 'absent'
            axes[i,j].set_title(f"{case['condition']} | original sixth capture\n{pair[key]['polyline']['classification']}",fontsize=11)
    fig.suptitle('OFFLINE PREDICTION — NO ROBOT EXECUTION\nAbsent predictions use the same obstacle only for counterfactual clearance; stars select row 4',fontsize=15)
    fig.savefig(output/'paired-trajectories-and-rgb.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
    delta=np.array([p['per_waypoint_delta_local_xy_m'] for p in pairs])
    im=axes[0].imshow(delta[:,:,1],aspect='auto',cmap='coolwarm',vmin=-.2,vmax=.2)
    axes[0].set(xticks=range(8),yticks=range(len(pairs)),yticklabels=[p['pose_id'] for p in pairs],
        xlabel='Waypoint index (not time)',title='PRESENT minus ABSENT lateral Y (m)')
    fig.colorbar(im,ax=axes[0])
    ids=np.arange(len(pairs))
    axes[1].bar(ids,[p['maximum_absolute_lateral_difference_m'] for p in pairs],label='Max |delta Y|')
    axes[1].axhline(.2,c='red',ls='--',label='Predeclared material screen')
    axes[1].set(xticks=ids,xticklabels=[p['pose_id'] for p in pairs],ylabel='m',title='Lateral response screening');axes[1].legend(fontsize=8)
    for key in ('present','absent'):
        axes[2].plot(ids,[p[key]['polyline']['minimum_clearance_m'] for p in pairs],'o-',label=key)
    axes[2].axhline(0,c='red');axes[2].axhline(.1,c='orange',ls='--')
    axes[2].set(xticks=ids,xticklabels=[p['pose_id'] for p in pairs],ylabel='Clearance (m)',title='Same counterfactual obstacle');axes[2].legend()
    fig.savefig(output/'paired-differences.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'research/frozen-edge-obstacle-offline.json')
    args=p.parse_args();run=ROOT/'outputs/edge-obstacle-offline/paired-001'
    data=analyze(run,args.protocol)
    assert data==analyze(run,args.protocol), 'Derived metrics must reproduce exactly'
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'comparisons.json',data)
    _,_,manifest,_=verify_protocol(args.protocol);plots(data,manifest,args.output)
    print(json.dumps(data['summary'],indent=2))


if __name__=='__main__':main()
