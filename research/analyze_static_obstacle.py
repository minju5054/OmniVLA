"""Offline plots/metrics from immutable records. Never imports Isaac or a policy."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle as RectanglePatch, FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import sha256, write_json
from research.isaac_final_contracts import stats
from research.static_obstacle_geometry import Rectangle
from research.static_obstacle_protocol import verify_static


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def world_prediction(native, observation_pose):
    """Returned cumulative local XY, 0.1 m scaling, observation-time SE(2) anchor.

    Heading is the rotated local [cos(yaw),sin(yaw)] vector. No row times and
    no second cumsum. Training provenance limitations remain in the report.
    """
    value=np.asarray(native,dtype=np.float64)[0].copy()
    x,y,yaw=observation_pose
    rotation=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    return value[:,:2]*.1@rotation.T+np.array([x,y]),value[:,2:]@rotation.T


def analyze(run, config_path, output):
    config=json.loads(config_path.read_text())
    verify_static(config,config_path)
    root=run/'isaac/STATIC_CENTER'
    summary=json.loads((root/'summary.json').read_text())
    states=rows(root/'raw/robot_state.jsonl')
    requests=rows(root/'raw/requests.jsonl')
    controllers=rows(root/'raw/controller.jsonl')
    responses=rows(run/'service/responses.jsonl')
    baseline=rows(ROOT/'outputs/edge-final/scientific-001/isaac/STRAIGHT/raw/robot_state.jsonl')
    pose=np.array([row['pose'] for row in states]); times=np.array([row['elapsed_sim'] for row in states])
    commands=np.array([row['applied_vw'] for row in states])
    radius=config['evaluation']['agent_radius_m']; margin=config['evaluation']['safety_margin_m']
    rectangle=Rectangle.from_config(config['static_obstacle'])
    violation=next((i for i,row in enumerate(states) if row['safety_margin_violation']),None)
    boundary_time=None
    if violation:
        sweep=rectangle.swept(states[violation-1]['pose'],states[violation]['pose'],radius+margin)
        boundary_time=(violation-1+sweep['first_contact_fraction'])/60
    metrics=dict(**summary, all_response_model_ms=stats([r['model_ms'] for r in responses]),
                 all_response_forward_with_clip_ms=stats([r['forward_with_clip_ms'] for r in responses]),
                 all_response_pipeline_ms=stats([r['pipeline_ms'] for r in responses]),
                 recorded_frame_to_applied_step_wall_ms=stats([
                     (p['application_step_completed_wall_time']-p['observation_wall_time'])*1000 for p in controllers]),
                 recorded_frame_to_ready_wall_ms=stats([
                     (p['ready_wall_time']-p['observation_wall_time'])*1000 for p in controllers]),
                 first_safety_boundary_crossing_sim_s=boundary_time,
                 first_sampled_safety_violation_sim_s=states[violation]['elapsed_sim'] if violation is not None else None,
                 linear_velocity_range_m_s=[float(commands[:,0].min()),float(commands[:,0].max())],
                 angular_velocity_range_rad_s=[float(commands[:,1].min()),float(commands[:,1].max())],
                 maximum_absolute_heading_deg=float(np.max(np.abs(pose[:,2]))*180/math.pi),
                 unapplied_responses=len(responses)-len(controllers),frames=len(rows(root/'raw/history.jsonl')),
                 delay_definition='Latest RGB metadata wall time after PNG save to first completed application step; excludes acquisition/save before metadata timestamp',
                 raw_legacy_delay_definition='Raw summary observation_to_application_wall_ms is request-send to pre-step application marker')
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'savefig.facecolor':'white'})
    def obstacles(ax):
        for expansion,color,label in [(radius+margin,'#d89213','Required center envelope (r + margin = 0.30 m)'),
                                       (radius,'#e55b4c','Collision center envelope (r = 0.20 m)')]:
            patch=FancyBboxPatch((rectangle.xmin-expansion,rectangle.ymin-expansion),
                                rectangle.xmax-rectangle.xmin+2*expansion,
                                rectangle.ymax-rectangle.ymin+2*expansion,
                                boxstyle=f'round,pad=0,rounding_size={expansion}',
                                fill=False,edgecolor=color,linestyle='--',linewidth=1.4,label=label)
            ax.add_patch(patch)
        ax.add_patch(RectanglePatch((rectangle.xmin,rectangle.ymin),.6,1.,
                                   facecolor='#dc5656',edgecolor='#913131',alpha=.7,label='Static obstacle'))
        ax.add_patch(Circle((5,0),.5,facecolor='#cee8d7',edgecolor='#247b49',alpha=.7,label='Goal region'))
        ax.set(xlabel='World x [m]',ylabel='World y [m]',xlim=(-.35,5.65),ylim=(-1.15,1.15))
        ax.set_aspect('equal',adjustable='box')

    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    fig.suptitle(f"OmniVLA-edge | Static obstacle qualification: {summary['status']}\n"
                 'One frozen primary trial; headless science, recorded replay',fontsize=17,fontweight='bold')
    ax=axes[0,0]; obstacles(ax)
    ax.plot([r['pose'][0] for r in baseline],[r['pose'][1] for r in baseline],
            '--',color='#7a818b',label='Prior STRAIGHT (obstacle absent then)',linewidth=1.5)
    ax.plot(pose[:,0],pose[:,1],color='#1462ad',linewidth=2.7,label='Executed static trial')
    ax.scatter(*pose[0,:2],color='#1462ad',s=30,zorder=6)
    ax.scatter(*pose[-1,:2],color='#b22222',marker='x',s=90,zorder=6,label='Terminated pose')
    ax.add_patch(Circle(pose[-1,:2],radius,fill=False,edgecolor='#1462ad',linewidth=1.6))
    ax.set_title('Executed XY path and declared evaluation geometry')
    ax.legend(loc='upper left',bbox_to_anchor=(0.,-.23),fontsize=8,ncol=2,frameon=False)
    ax=axes[0,1]
    ax.plot(times,[r['goal_distance'] for r in states],color='#1462ad',label='Static trial',linewidth=2)
    ax.plot([r['elapsed_sim'] for r in baseline],[r['goal_distance'] for r in baseline],
            '--',color='#7a818b',label='Prior obstacle-free STRAIGHT')
    ax.axhline(.5,color='#247b49',linestyle=':',label='Goal threshold 0.50 m')
    ax.set(title='Distance to goal',xlabel='Elapsed simulation time [s]',ylabel='XY distance [m]')
    ax.legend(fontsize=9)
    ax=axes[1,0]
    ax.plot(times,[r['swept_minimum_clearance'] for r in states],color='#1462ad',linewidth=2,label='Swept minimum per step')
    ax.axhline(.1,color='#d89213',linestyle='--',label='Required clearance 0.10 m')
    ax.axhline(0,color='#b22222',linestyle=':',label='Contact threshold')
    ax.axvline(summary['first_collision_time_sim_s'],color='#b22222',linestyle=':',alpha=.7)
    ax.text(.04,.90,f"First continuous contact: {summary['first_collision_time_sim_s']:.4f} s\n"
            f"Minimum: {summary['minimum_signed_clearance_m']:.5f} m",transform=ax.transAxes,va='top')
    ax.set(title='Obstacle clearance',xlabel='Elapsed simulation time [s]',ylabel='Signed clearance [m]')
    ax.legend(fontsize=9,loc='center right')
    ax=axes[1,1]
    ax.step(times,commands[:,0],where='pre',color='#1462ad',label='Linear velocity [m/s]')
    ax.step(times,commands[:,1],where='pre',color='#ab4ea8',label='Angular velocity [rad/s]')
    ax.set(title='Applied official-controller commands',xlabel='Elapsed simulation time [s]',ylabel='Command value')
    ax.legend(fontsize=9,loc='center right')
    for extension in ('png','svg'): fig.savefig(output/f'navigation-summary.{extension}',dpi=170)
    plt.close(fig)

    # Fixed first/middle/last request selection, declared independently of shape.
    indices=sorted(set([0,len(requests)//2,len(requests)-1]))
    fig,axes=plt.subplots(1,len(indices),figsize=(15,5),layout='constrained',squeeze=False)
    comparison=[]
    for ax,index in zip(axes[0],indices):
        req=requests[index];raw=np.load(req['action_path'],allow_pickle=False)
        xy,heading=world_prediction(raw,req['robot_pose_observation'])
        obstacles(ax)
        ax.plot(pose[:,0],pose[:,1],color='#1462ad',linewidth=2,label='Executed')
        points=np.vstack([req['robot_pose_observation'][:2],xy])
        ax.plot(points[:,0],points[:,1],'o--',color='#97429b',markersize=4,label='Predicted cumulative rows')
        ax.scatter(*xy[4],color='#e2831a',s=60,zorder=5,label='Selected row 4')
        ax.quiver(xy[4,0],xy[4,1],heading[4,0],heading[4,1],angles='xy',scale_units='xy',scale=4,color='#e2831a')
        ax.set_title(f"Request {req['request_id']} | observation t={req['observation_time_sim']-100/60:.2f} s")
        comparison.append(dict(request_id=req['request_id'],observation_pose=req['robot_pose_observation'],
                               world_xy=xy.tolist(),world_heading_vectors=heading.tolist(),
                               native_action_sha256=sha256(req['action_path'])))
    axes[0,0].legend(loc='upper left',bbox_to_anchor=(0,-.24),fontsize=8,ncol=2,frameon=False)
    fig.suptitle('Predicted spatial chunks at observation poses vs executed path\n'
                 'First / middle / last request; rows have no assumed intrinsic timestamps',fontsize=14)
    for extension in ('png','svg'): fig.savefig(output/f'predictions-vs-executed.{extension}',dpi=170)
    plt.close(fig)
    write_json(output/'selected-predictions-world.json',dict(selection='first / middle / last request index',
               semantics='Cumulative x-forward/y-left in 0.1 m units; heading [cos,sin]; rotate at observation pose; no intrinsic row times',
               predictions=comparison))
    write_json(output/'metrics.json',metrics)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/static-obstacle/scientific-001')
    parser.add_argument('--config',type=Path,default=ROOT/'research/frozen-edge-static-obstacle.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.run,args.config,args.output)
    print(json.dumps(result,indent=2))
