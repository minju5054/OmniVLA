"""Pure saved-evidence validity and all-waypoint language-effect measurements."""
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
from research.isaac_final_common import ROOT,sha256
from research.isaac_final_contracts import action,stats
from research.offline_obstacle_contracts import evaluate_chunk,compare_pair
from research.pose_language_contracts import validate_instruction,validate_response


def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def offline_validity(run,plan,outcome=None):
    if outcome is None:outcome=json.loads((run/'outcome.json').read_text())
    assert outcome['status']=='PASS' and outcome['service_exit_code']==0
    assert outcome['service_stopped']==dict(inference_count=32,request_attempts=32,load_count=1,phase='offline')
    requests=rows(run/'raw/requests.jsonl');cases=rows(run/'raw/cases.jsonl');responses=rows(run/'service/responses.jsonl')
    assert len(requests)==len(cases)==len(responses)==32
    manifest=json.loads((ROOT/plan['phase_a']['rgb_manifest']).read_text())
    inputs={c['case_id']:c for c in manifest['cases']}
    hashes={key:set() for key in plan['instructions']}
    for spec,req,case,response in zip(plan['offline_order'],requests,cases,responses):
        assert all(req[k]==spec[k] for k in ('episode_id','request_id','instruction_id'))
        validate_instruction(req,plan);validate_response(response,req)
        assert 'controller_error' not in response,'Upstream controller bug: no navigation'
        assert req['history']==inputs[spec['rgb_case_id']]['history']
        assert req['robot_pose_observation']==inputs[spec['rgb_case_id']]['pose']
        assert response==case['response'] and case['request']==req
        raw=action(np.load(response['action_path'],allow_pickle=False))
        assert raw.tolist()==case['native_prediction']
        assert response['actual_goal_float32']==np.asarray([req['normalized_pose']],dtype=np.float32).tolist()
        hashes[req['instruction_id']].add(response['clip_embedding_sha256'])
    assert len(list((run/'raw/actions').glob('*.npy')))==32
    assert len(set.union(*hashes.values()))>=4,'Distinct token identities must reach text encoder'
    return dict(status='MODALITY8_INPUT_VALID',executed_requests=32,planned_requests=32,model_loads=1,
        native_shape=[1,8,4],native_dtype='float32',all_decoder_masks_verified=True,
        pose_and_rgb_preserved=True,no_truncation=True,
        clip_embedding_hashes={key:sorted(value) for key,value in hashes.items()},
        numerical_errors=0,upstream_controller_bugs=0)


def analyze_offline(run,plan,config):
    gate=offline_validity(run,plan);records=rows(run/'raw/cases.jsonl');metrics=[];native={}
    for record in records:
        response=record['response'];raw=np.load(response['action_path'],allow_pickle=False)
        native[(record['rgb_case_id'],record['instruction_id'])]=raw
        m=evaluate_chunk(raw,record['request']['robot_pose_observation'],config)
        for key in ('pre_clip_vw','clipped_vw','final_vw'):
            assert m['hypothetical_commands'][4][key]==response['controller'][key]
        metrics.append(dict(episode_id=record['episode_id'],rgb_case_id=record['rgb_case_id'],instruction_id=record['instruction_id'],
            request_id=record['request_id'],pose=record['request']['robot_pose_observation'],
            native_shape=response['native_shape'],native_dtype=response['native_dtype'],
            action_path=response['action_path'],action_sha256=response['action_sha256'],
            native_prediction=raw.tolist(),clip_embedding_sha256=response['clip_embedding_sha256'],
            latency_ms={k:response[k] for k in ('model_ms','forward_with_clip_ms','pipeline_ms')},**m))
    manifest=json.loads((ROOT/plan['phase_a']['rgb_manifest']).read_text());comparisons=[];directions=[];visual=[]
    threshold=plan['interpretation']['material_lateral_threshold_m'];epsilon=plan['interpretation']['direction_epsilon_m']
    for c in manifest['cases']:
        cid=c['case_id'];pairs={}
        for x,y in [('L','N'),('R','N'),('L','R'),('D','N')]:
            pair=compare_pair(native[cid,x],native[cid,y],c['pose'],config,threshold)
            pair['delta_definition']=x+' minus '+y
            comparisons.append(dict(rgb_case_id=cid,comparison=x+'-'+y,**pair));pairs[x+'-'+y]=pair
        ln=np.array(pairs['L-N']['per_waypoint_delta_local_xy_m']);rn=np.array(pairs['R-N']['per_waypoint_delta_local_xy_m'])
        left_y=float(native[cid,'L'][0,-1,1])*.1;right_y=float(native[cid,'R'][0,-1,1])*.1
        directions.append(dict(rgb_case_id=cid,left_endpoint_y_m=left_y,right_endpoint_y_m=right_y,
            left_minus_neutral_endpoint_y_m=float(ln[-1,1]),right_minus_neutral_endpoint_y_m=float(rn[-1,1]),
            left_absolute_sign=left_y>epsilon,right_absolute_sign=right_y<-epsilon,
            left_relative_sign=bool(ln[-1,1]>epsilon),right_relative_sign=bool(rn[-1,1]<-epsilon),
            direction_consistency_observed=bool(left_y>epsilon and right_y<-epsilon and ln[-1,1]>epsilon and rn[-1,1]<-epsilon),
            left_material_signed_response=bool(ln[:,1].max()>=threshold),
            right_material_signed_response=bool((-rn[:,1]).max()>=threshold)))
    for present,absent in zip(manifest['cases'][::2],manifest['cases'][1::2]):
        for key in plan['instructions']:
            pair=compare_pair(native[present['case_id'],key],native[absent['case_id'],key],present['pose'],config,threshold)
            visual.append(dict(pose_id=present['pose_id'],instruction_id=key,**pair))
    by_instruction={key:dict(Counter(m['polyline']['classification'] for m in metrics if m['instruction_id']==key)) for key in plan['instructions']}
    summary=dict(technical_validity=gate,status_observations=dict(MODALITY8_INPUT_VALID=True,
        LANGUAGE_RESPONSE_OBSERVED=any(np.max(np.abs(p['per_waypoint_delta_local_xy_m']))>epsilon for p in comparisons if p['comparison'].endswith('-N')),
        DIRECTION_CONSISTENCY_OBSERVED=any(d['direction_consistency_observed'] for d in directions),
        SAFE_PASSAGE_PREDICTED=any(m['polyline']['classification']=='PREDICTED_SAFE_PASSAGE' for m in metrics),
        HORIZON_INSUFFICIENT=any(m['polyline']['classification']=='HORIZON_INSUFFICIENT' for m in metrics),
        PREDICTED_UNSAFE=any(m['polyline']['classification'] in ('PREDICTED_COLLISION','PREDICTED_LOW_CLEARANCE') for m in metrics),
        NO_MATERIAL_DIRECTIONAL_RESPONSE=not any(d['left_material_signed_response'] or d['right_material_signed_response'] for d in directions)),
        classifications_by_instruction=by_instruction,direction_consistency_count=sum(d['direction_consistency_observed'] for d in directions),
        material_left_count=sum(d['left_material_signed_response'] for d in directions),material_right_count=sum(d['right_material_signed_response'] for d in directions),
        cold_latency_ms=metrics[0]['latency_ms'],subsequent_latency_ms={key:stats([m['latency_ms'][key] for m in metrics[1:]]) for key in ('model_ms','forward_with_clip_ms','pipeline_ms')})
    return dict(summary=summary,predictions=metrics,prompt_comparisons=comparisons,directions=directions,visual_pairs=visual)


def analyze_navigation(run,plan,config):
    from research.isaac_final_contracts import integrate,local_goal
    from research.static_obstacle_geometry import Rectangle
    responses=rows(run/'service/responses.jsonl');results=[]
    rectangle=Rectangle.from_config(config['static_obstacle'])
    radius=config['evaluation']['agent_radius_m'];margin=config['evaluation']['safety_margin_m']
    for trial in plan['navigation_order']:
        folder=run/'trials'/trial['instruction_id']/trial['episode_id']
        if not (folder/'summary.json').exists():
            results.append(dict(**trial,status='NOT_RUN',reason=json.loads((run/'outcome.json').read_text()).get('failure')));continue
        summary=json.loads((folder/'summary.json').read_text());states=rows(folder/'raw/robot_state.jsonl')
        requests=rows(folder/'raw/requests.jsonl');applied=rows(folder/'raw/controller.jsonl');history=rows(folder/'raw/history.jsonl')
        episode_responses=[r for r in responses if r.get('request_episode_id')==trial['episode_id']]
        by_id={r['request_id']:r for r in requests};native_metrics=[]
        assert len(episode_responses)==len(requests)
        for response in episode_responses:
            req=by_id[response['request_id']];validate_instruction(req,plan);validate_response(response,req)
            assert req['instruction_id']==trial['instruction_id']
            assert req['robot_pose_observation']==req['history'][-1]['pose']
            observation_step=round((req['observation_time_sim']-100/60)*60)
            assert req['robot_pose_observation']==states[observation_step]['pose']
            assert (req['local_goal'],req['normalized_pose'])==local_goal(req['robot_pose_observation'],req['world_goal'])
            raw=np.load(response['action_path'],allow_pickle=False)
            m=evaluate_chunk(raw,req['robot_pose_observation'],config)
            if 'controller' in response:
                for key in ('pre_clip_vw','clipped_vw','final_vw'):
                    assert m['hypothetical_commands'][4][key]==response['controller'][key]
            native_metrics.append(dict(request_id=req['request_id'],observation_pose=req['robot_pose_observation'],
                observation_time_sim=req['observation_time_sim'],action_sha256=response['action_sha256'],
                clip_embedding_sha256=response['clip_embedding_sha256'],**m))
        min_clearance=rectangle.clearance(states[0]['pose'],radius);first_margin=None;first_collision=None
        for prev,current in zip(states,states[1:]):
            np.testing.assert_allclose(current['pose'],integrate(prev['pose'],current['applied_vw']),atol=1e-12,rtol=0)
            sweep=rectangle.swept(prev['pose'],current['pose'],radius)
            assert abs(sweep['minimum_clearance']-current['swept_minimum_clearance'])<1e-12
            min_clearance=min(min_clearance,sweep['minimum_clearance'])
            if first_margin is None and sweep['minimum_clearance']<margin:
                expanded=rectangle.swept(prev['pose'],current['pose'],radius+margin)
                first_margin=(prev['step']+expanded['first_contact_fraction'])/60
            if first_collision is None and sweep['collision']:
                first_collision=(prev['step']+sweep['first_contact_fraction'])/60
        assert abs(min_clearance-summary['minimum_signed_clearance_m'])<1e-12
        assert first_collision==summary['first_collision_time_sim_s']
        for row in applied:
            assert row['applied_vw']==row['controller']['final_vw']
            step=row['application_physics_step']
            assert row['application_pose_before_step']==states[step-1]['pose']
            assert row['application_pose_after_step']==states[step]['pose']
            assert row['robot_pose_observation']==by_id[row['request_id']]['robot_pose_observation']
        ys=[s['pose'][1] for s in states]
        turn=next((s for s in states if abs(s['applied_vw'][1])>plan['interpretation']['direction_epsilon_m']),None)
        controls=[r['controller'] for r in applied]
        n=len(controls)
        extra=dict(first_safety_margin_violation_sim_s=first_margin,
            minimum_executed_y_m=min(ys),maximum_executed_y_m=max(ys),maximum_absolute_executed_y_m=max(map(abs,ys)),
            first_turn_direction='LEFT' if turn and turn['applied_vw'][1]>0 else ('RIGHT' if turn else 'NONE'),
            first_turn_time_sim_s=turn['elapsed_sim'] if turn else None,
            native_predictions=len(episode_responses),actual_rgb_captures=len(history),
            native_classifications=dict(Counter(m['polyline']['classification'] for m in native_metrics)),
            selected_heading_used_count=sum(m['hypothetical_commands'][4]['heading_used'] for m in native_metrics),
            selected_pre_v_range=[min(c['pre_clip_vw'][0] for c in controls),max(c['pre_clip_vw'][0] for c in controls)] if n else None,
            applied_v_range=[min(c['final_vw'][0] for c in controls),max(c['final_vw'][0] for c in controls)] if n else None,
            applied_w_range=[min(c['final_vw'][1] for c in controls),max(c['final_vw'][1] for c in controls)] if n else None,
            initial_clip_fraction=sum(c['pre_clip_vw']!=c['clipped_vw'] for c in controls)/n if n else None,
            ratio_limit_fraction=sum(c['clipped_vw']!=c['final_vw'] for c in controls)/n if n else None,
            frame_record_to_applied_step_wall_ms=stats([(r['application_step_completed_wall_time']-r['observation_wall_time'])*1000 for r in applied]),
            all_response_model_ms=stats([r['model_ms'] for r in episode_responses]),
            all_response_pipeline_ms=stats([r['pipeline_ms'] for r in episode_responses]),
            controller_parity_count=len(native_metrics),executed_integration_parity_steps=len(states)-1)
        results.append(dict(**summary,instruction_id=trial['instruction_id'],instruction_text=plan['instructions'][trial['instruction_id']]['text'],
            extra=extra,native_predictions=native_metrics))
    return dict(episodes=results)
