"""Additive Isaac viewer for saved fixed-pose predictions, never robot execution."""
import json
import math
import time
import numpy as np
from research.isaac_final_common import ROOT,sha256,write_json,append_json
from research.offline_obstacle_protocol import verify_protocol
from research.offline_obstacle_contracts import compare_pair

LABEL='OFFLINE PREDICTION — NO ROBOT EXECUTION'


def validate_display_values(actual,saved):
    """Viewer-only cross-NumPy check; never replace pinned scientific metrics.

    NumPy 2 scalar promotion differs from the service's NumPy 1.26.4 in the
    unchanged controller AST. Allow 5e-7 absolute display-validation rounding;
    identity, booleans, classifications and container structure remain exact.
    """
    if isinstance(actual,dict):
        assert actual.keys()==saved.keys()
        return max((validate_display_values(actual[k],saved[k]) for k in actual),default=0.)
    if isinstance(actual,list):
        assert len(actual)==len(saved)
        return max((validate_display_values(a,b) for a,b in zip(actual,saved)),default=0.)
    if isinstance(actual,float):
        delta=abs(actual-saved);assert math.isfinite(delta) and delta<=5e-7
        return delta
    assert actual==saved
    return 0.


def load_offline(protocol,config):
    frozen,plan,manifest,expected=verify_protocol(protocol)
    assert config==expected
    run=ROOT/'outputs/edge-obstacle-offline/paired-001'
    data=json.loads((run/'derived/comparisons.json').read_text())
    assert data['protocol_sha256']==sha256(protocol)
    native={}
    for case in data['cases']:
        path=ROOT/case['action_path'];assert sha256(path)==case['action_sha256']
        native[case['case_id']]=np.load(path,allow_pickle=False)
    maximum=0.
    for pair,p,a in zip(data['pairs'],manifest['cases'][::2],manifest['cases'][1::2]):
        actual=compare_pair(native[p['case_id']],native[a['case_id']],p['pose'],config)
        maximum=max(maximum,validate_display_values(actual,{key:pair[key] for key in actual}))
    return dict(data=data,manifest=manifest,protocol_sha256=sha256(protocol),
        viewer_numpy=np.__version__,cross_runtime_maximum_absolute_rounding_difference=maximum)


def run_offline_gui(app,recording,config,output):
    import omni.ui as ui
    from pxr import Gf,UsdGeom
    from omni.kit.viewport.utility import get_active_viewport,capture_viewport_to_file
    from isaacsim.core.utils.viewports import set_camera_view
    from research.isaac_scene import build,box,set_pose
    from research.static_obstacle_scene import add_obstacle
    world,camera=build(config);stage=world.stage
    colors={'present':(.98,.18,.12),'absent':(.10,.55,1.)}
    cases=recording['manifest']['cases'];pairs=recording['data']['pairs']
    assert len(cases)==8 and all(p['status']=='COMPLETE' for p in pairs)
    root='/World/ReplayOverlays'
    UsdGeom.Xform.Define(stage,root)

    def curve(name,points,color,width=.025):
        c=UsdGeom.BasisCurves.Define(stage,root+'/'+name)
        c.CreateTypeAttr('linear');c.CreateCurveVertexCountsAttr([len(points)])
        c.CreatePointsAttr([Gf.Vec3f(*p) for p in points]);c.CreateWidthsAttr([width])
        c.SetWidthsInterpolation('constant');c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return c

    def ring(name,x,y,r,color,z=.06):
        return curve(name,[(x+r*math.cos(t),y+r*math.sin(t),z) for t in np.linspace(0,2*math.pi,65)],color)

    ring('Goal',5,0,.5,(.25,1.,.3))
    box(stage,root+'/GoalCenter',[5,0,.10],[.12,.12,.20],[.25,1.,.3])
    curve('ReferenceFootprint',[(2.2,-.5,.07),(2.8,-.5,.07),(2.8,.5,.07),(2.2,.5,.07),(2.2,-.5,.07)],(1.,.65,.1))
    envelope=[]
    for cx,cy,lo in [(2.8,.5,0),(2.2,.5,90),(2.2,-.5,180),(2.8,-.5,270)]:
        envelope += [(cx+.3*math.cos(t),cy+.3*math.sin(t),.05) for t in np.linspace(math.radians(lo),math.radians(lo+90),17)]
    curve('SafetyEnvelope',envelope+[envelope[0]],(1.,.65,.1),.018)
    body=UsdGeom.Xform.Define(stage,root+'/FixedPose');translation=body.AddTranslateOp()
    box(stage,root+'/FixedPose/Body',[0.,0.,.16],[.32,.24,.20],[.94,.96,1.])
    curve('FixedPose/Heading',[(0,0,.3),(.35,0,.3)],(1,1,1),.04)
    ring('FixedPose/Radius',0,0,.2,(1,1,1))
    paths={key:curve('Prediction_'+key,[(0,0,1.12),(1,0,1.12)],color,.032 if key=='present' else .017) for key,color in colors.items()}
    markers={key:ring('Selected4_'+key,0,0,.055,color,1.14) for key,color in colors.items()}
    viewport=get_active_viewport();assert viewport is not None;viewport.fill_frame=True

    def overview():
        viewport.camera_path='/OmniverseKit_Persp'
        set_camera_view(np.array([2.6,-5.0,7.2]),np.array([2.6,0.,0.]),viewport_api=viewport)

    overview();state=dict(index=0,changed=True,selected_at=time.monotonic(),cycles=0)
    def select(index):
        state.update(index=index,changed=True,selected_at=time.monotonic())
        append_json(output/'case-events.jsonl',dict(case_id=cases[index]['case_id'],event='SELECT_FIXED_CASE',
            pose=cases[index]['pose'],robot_execution=False,wall_time=time.time()))

    window=ui.Window('OmniVLA | Offline obstacle diagnostic',width=550,height=900)
    with window.frame:
        with ui.VStack(spacing=7,height=0):
            ui.Label('OmniVLA-edge | Paired diagnostic',height=32,style={'font_size':22})
            ui.Label(LABEL,height=45,word_wrap=True,style={'font_size':18,'color':0xFF66CCFF})
            with ui.HStack(height=32,spacing=4):
                for i,pair in enumerate(pairs):
                    ui.Button(f"{pair['distance_to_front_m']:.1f} m",clicked_fn=lambda n=i:select(n*2+state['index']%2))
            with ui.HStack(height=32,spacing=4):
                ui.Button('OBSTACLE PRESENT',clicked_fn=lambda:select(state['index']//2*2))
                ui.Button('OBSTACLE ABSENT',clicked_fn=lambda:select(state['index']//2*2+1))
            with ui.HStack(height=25,spacing=5):
                auto=ui.SimpleBoolModel(True);ui.CheckBox(model=auto,width=22)
                ui.Label('Cycle fixed cases (7 s each)');ui.Button('Overview',clicked_fn=overview)
            title=ui.Label('',height=27,style={'font_size':19})
            stats=ui.Label('',height=160,word_wrap=True)
            ui.Label('Red = present prediction; blue = absent prediction.\nSmall rings = selected row 4; white = fixed observation pose.\nOrange = counterfactual footprint + radius/margin boundary.',height=60,word_wrap=True)
            ui.Separator(height=2)
            ui.Label('Original RGB input — sixth independent stationary capture',height=32,word_wrap=True)
            saved_image=ui.Image(cases[0]['history'][-1]['path'],height=230,fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT)
            ui.Label('Case changes jump between fixed poses; there is no driven path.\nPrediction lines are elevated to reveal obstacle crossings.\nOverlay markers never appear in the saved model inputs.',height=64,word_wrap=True)
    for _ in range(30):world.render()
    for name in ('Stage','Layer','Property','Content','Console','Render Settings'):
        panel=ui.Workspace.get_window(name)
        if panel:panel.visible=False
    window.dock_in(ui.Workspace.get_window('Viewport'),ui.DockPosition.RIGHT,.39)
    select(0)
    write_json(output/'ready.json',dict(status='OFFLINE_GUI_READY',label=LABEL,robot_execution=False,
        original_rgb=True,cases=[c['case_id'] for c in cases],protocol_sha256=recording['protocol_sha256'],
        viewer_numpy=recording['viewer_numpy'],
        cross_runtime_maximum_absolute_rounding_difference=recording['cross_runtime_maximum_absolute_rounding_difference']))
    print('OFFLINE_GUI_READY',flush=True)
    rendered=0;next_status=0;capture=None
    while app.is_running() and not (output/'stop-requested').exists():
        start=time.monotonic()
        if auto.as_bool and start-state['selected_at']>=7:
            select((state['index']+1)%8);state['cycles']+=1
        case=cases[state['index']];pair=pairs[state['index']//2]
        if state['changed']:
            x,y,yaw=case['pose'];translation.Set(Gf.Vec3d(x,y,0));set_pose(camera,case['pose'],config['camera']['height_m'])
            if stage.GetPrimAtPath(config['static_obstacle']['path']).IsValid():stage.RemovePrim(config['static_obstacle']['path'])
            if case['condition']=='OBSTACLE_PRESENT':add_obstacle(stage,config)
            for key in colors:
                xy=pair[key]['world_xy'];pts=[case['pose'][:2]]+xy
                paths[key].GetCurveVertexCountsAttr().Set([len(pts)])
                paths[key].GetPointsAttr().Set([Gf.Vec3f(p[0],p[1],1.12) for p in pts])
                sx,sy=xy[4]
                markers[key].GetPointsAttr().Set([Gf.Vec3f(sx+.055*math.cos(t),sy+.055*math.sin(t),1.14) for t in np.linspace(0,2*math.pi,65)])
            saved_image.source_url=case['history'][-1]['path']
            title.text=case['case_id']
            stats.text=(f"Fixed pose x={x:.2f}, y={y:.2f}, yaw=0 | Goal (5,0,0)\n"
                f"Present: {pair['present']['polyline']['classification']}\n"
                f"Absent: {pair['absent']['polyline']['classification']} (counterfactual)\n"
                f"Min clearance P / A: {pair['present']['polyline']['minimum_clearance_m']:.3f} / {pair['absent']['polyline']['minimum_clearance_m']:.3f} m\n"
                f"Max |delta Y|: {pair['maximum_absolute_lateral_difference_m']:.4f} m | Screen >= 0.20 m\n"
                f"Material lateral response: {pair['material_lateral_response']}\n"
                f"RMS position difference: {pair['rms_position_difference_m']:.4f} m")
            state['changed']=False
        world.render();rendered+=1
        if rendered==120:capture=capture_viewport_to_file(viewport,str(output/'overview.png'))
        if start>=next_status:
            (output/'case-status.json').write_text(json.dumps(dict(case_id=case['case_id'],pose=case['pose'],
                rendered_frames=rendered,case_changes=state['cycles'],robot_execution=False,wall_time=time.time()),indent=2)+'\n')
            next_status=start+2
        time.sleep(max(0,1/60-(time.monotonic()-start)))
    return dict(rendered_frames=rendered,case_changes=state['cycles'],window_title=window.title,
        robot_execution=False,screenshot_requested=capture is not None)
