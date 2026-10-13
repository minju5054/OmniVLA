"""Model-free guards for frozen stationary paired inputs and one-shot inference."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from research.isaac_final_common import ROOT, sha256, write_json
from research.isaac_final_contracts import History, local_goal, pixel_hash
from research.static_obstacle_protocol import verify_static

PLAN = ROOT/'research/edge-obstacle-offline-plan.json'
STATIC = ROOT/'research/frozen-edge-static-obstacle.json'
PART_A = ROOT/'outputs/edge-obstacle-offline/existing-chunks/derived-v2'


def red_mask(image):
    a=np.asarray(image).astype(float)
    return (a[:,:,0]>80) & (a[:,:,0]>1.4*np.maximum(a[:,:,1],a[:,:,2]))


def image_difference(a,b,mask=None):
    delta=np.abs(a.astype(float)-b.astype(float))
    selected=delta if mask is None else delta[mask]
    return dict(mean_absolute_channel_difference=float(selected.mean()) if selected.size else None,
        rms_channel_difference=float(np.sqrt(np.mean(selected**2))) if selected.size else None,
        changed_pixel_fraction=float(np.any(delta>0,axis=2)[mask].mean()) if mask is not None
            else float(np.any(delta>0,axis=2).mean()))


def validate_case(case, expected, plan, read_images=True):
    for key in ('case_id','pose_id','pose','condition','request_id','distance_to_front_m'):
        assert case[key]==expected[key], key
    assert case['world_goal']==plan['world_goal']
    assert (case['local_goal'],case['normalized_pose'])==local_goal(case['pose'],case['world_goal'])
    assert case['no_gui_markers']
    present=case['condition']=='OBSTACLE_PRESENT'
    assert case['obstacle_prim_present']==present
    assert bool(case['obstacle_geometry'])==present
    history=case['history']; assert len(history)==6
    assert [f['case_step'] for f in history]==plan['history']['frame_steps']
    assert [f['sim_time'] for f in history]==[i/3 for i in range(6)]
    assert all(a['wall_time']<b['capture_start_wall_time'] for a,b in zip(history,history[1:]))
    assert all(b['absolute_physics_step']-a['absolute_physics_step']==20 for a,b in zip(history,history[1:]))
    assert np.allclose(np.diff([f['absolute_world_sim_time'] for f in history]),1/3,atol=1e-6,rtol=0)
    h=History(); images=[]
    for frame in history:
        h.append(frame)
        assert frame['pose']==case['pose']
        assert frame['capture_start_wall_time']<=frame['wall_time']
        assert np.allclose(frame['camera_position'],[*case['pose'][:2],case['camera_config']['height_m']],
                           atol=plan['input_validity']['stationary_pose_max_abs_error'],rtol=0)
        assert np.allclose(frame['camera_orientation_wxyz'],[1.,0,0,0],atol=1e-6,rtol=0)
        if read_images:
            path=Path(frame['path']); assert path.resolve().is_relative_to(ROOT/'outputs')
            assert sha256(path)==frame['png_sha256']
            array=np.array(Image.open(path).convert('RGB'))
            assert array.shape==(480,640,3) and array.dtype==np.uint8
            assert pixel_hash(array)==frame['rgb_sha256']
            count=int(red_mask(array).sum())
            assert count>=plan['input_validity']['present_red_pixels_min'] if present else count<=plan['input_validity']['absent_red_pixels_max']
            images.append(array)
    if not read_images: return None
    return dict(red_pixel_counts=[int(red_mask(a).sum()) for a in images],
        unique_rgb_hashes=len(set(f['rgb_sha256'] for f in history)),
        within_condition_vs_first=[image_difference(images[0],a) for a in images[1:]],
        within_condition_adjacent=[image_difference(a,b) for a,b in zip(images,images[1:])])


def validate_pair(present, absent):
    assert present['condition']=='OBSTACLE_PRESENT' and absent['condition']=='OBSTACLE_ABSENT'
    for key in ('pose_id','pose','world_goal','local_goal','normalized_pose','camera_intrinsics',
                'camera_config','renderer','scene','distance_to_front_m'):
        assert present[key]==absent[key], key
    assert set(present['stage_paths'])-set(absent['stage_paths'])=={'/World/StaticObstacle'}
    assert not set(absent['stage_paths'])-set(present['stage_paths'])
    for p,a in zip(present['history'],absent['history']):
        for key in ('sim_time','case_step','camera_position','camera_orientation_wxyz','pose','rgb_shape'):
            assert p[key]==a[key],key


def validate_capture(capture, output):
    plan=json.loads(PLAN.read_text()); result=json.loads((capture/'result.json').read_text())
    assert result['status']=='PRE_CLOSE_PASS' and result['requested_exit_code']==0
    assert result['plan_sha256']==sha256(PLAN)
    assert result['capture_source_sha256']==sha256(ROOT/'research/offline_obstacle_capture.py')
    assert len(result['cases'])==len(plan['cases'])==8
    all_ids=[f['frame_id'] for c in result['cases'] for f in c['history']]
    assert len(set(all_ids))==48
    assert len({f['path'] for c in result['cases'] for f in c['history']})==48
    metrics={}
    for case,expected in zip(result['cases'],plan['cases']):
        metrics[case['case_id']]=validate_case(case,expected,plan)
    pairs=[]
    for p,a in zip(result['cases'][::2],result['cases'][1::2]):
        validate_pair(p,a)
        pi=np.array(Image.open(p['history'][-1]['path']).convert('RGB'))
        ai=np.array(Image.open(a['history'][-1]['path']).convert('RGB'))
        obj=red_mask(pi)
        diff=image_difference(pi,ai)
        assert diff['mean_absolute_channel_difference']>=plan['input_validity']['latest_pair_mean_abs_pixel_difference_min']
        # This mask identifies red object pixels only; outside also includes shadow,
        # disocclusion, anti-aliasing and renderer variation (not pure noise).
        pairs.append(dict(pose_id=p['pose_id'],whole_image=diff,
            present_red_object_region=image_difference(pi,ai,obj),
            outside_red_object_region=image_difference(pi,ai,~obj),
            absent_red_pixels_in_present_region=int((red_mask(ai)&obj).sum()),
            red_object_bbox_xyxy=[int(v) for v in [np.where(obj)[1].min(),np.where(obj)[0].min(),np.where(obj)[1].max(),np.where(obj)[0].max()]],
            pairing_pass=True))
    manifest=dict(status='PASS',capture=str(capture.relative_to(ROOT)),plan_sha256=sha256(PLAN),
        capture_result_sha256=sha256(capture/'result.json'),cases=result['cases'],case_metrics=metrics,pair_metrics=pairs,
        robot_execution=False,actual_frame_count=48,frame_copying=False,
        background_caveat='Outside red mask mixes renderer noise, obstacle shadow, disocclusion and edge pixels; no exact equality assumption')
    write_json(output,manifest)
    return manifest


def verify_protocol(path):
    frozen=json.loads(Path(path).read_text())
    assert frozen['status']=='FROZEN_BEFORE_INFERENCE'
    config=json.loads(STATIC.read_text()); verify_static(config,STATIC)
    for key in ('files_sha256','source_sha256'):
        for name,digest in frozen[key].items():
            assert sha256(ROOT/name)==digest, name
    manifest=json.loads((ROOT/frozen['input_manifest']).read_text())
    plan=json.loads(PLAN.read_text())
    assert manifest['status']=='PASS'
    for case,expected in zip(manifest['cases'],plan['cases']): validate_case(case,expected,plan)
    for p,a in zip(manifest['cases'][::2],manifest['cases'][1::2]): validate_pair(p,a)
    return frozen,plan,manifest,config
