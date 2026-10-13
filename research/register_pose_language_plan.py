"""Freeze literal prompts, official CLIP tokens and BOTH phases before inference."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.pose_language_contracts import PLAN,PROMPTS,FLAGS,MASK,utf8_hash,tensor_hash
from research.isaac_final_common import write_json,sha256


def main():
    import clip
    instructions={}
    for key,text in PROMPTS.items():
        tokens=clip.tokenize(text,truncate=False)
        assert (tokens==clip.tokenize(text,truncate=True)).all()
        a=tokens.numpy();assert a.dtype==np.int32 and a.shape==(1,77)
        count=int(np.argmax(a[0]))+1
        assert count<=77 and int(a[0,count-1])==49407
        instructions[key]=dict(id=key,text=text,utf8_sha256=utf8_hash(text),token_ids=a.tolist(),
            token_sha256=tensor_hash(a),token_dtype='int32',token_hash_encoding='C-contiguous little-endian int32 bytes, shape [1,77]',
            token_count_including_sot_eot=count,context_length=77,truncated=False,modality_id=8)
    old=json.loads((ROOT/'research/edge-obstacle-offline-plan.json').read_text())
    offline=[]
    for case in old['cases']:
        for instruction in instructions:
            offline.append(dict(request_id=len(offline),episode_id=case['case_id']+'_'+instruction,
                rgb_case_id=case['case_id'],instruction_id=instruction,pose=case['pose']))
    names=['NEUTRAL','DESCRIPTION','LEFT','RIGHT']
    plan=dict(protocol='POSE_LANGUAGE_OBSTACLE_V1',starting_sha='eeb75f85cbcf316f15856388d0c3831eae4ef87d',
        flags=FLAGS,modality_id=8,decoder_mask=MASK,world_goal=[5.,0.,0.],instructions=instructions,
        offline_order=offline,navigation_order=[dict(episode_id='POSE_LANGUAGE_'+name,instruction_id=key)
            for key,name in zip(instructions,names)],
        phase_a=dict(requests=32,model_loads=1,rgb_manifest='artifacts-local/edge-obstacle-offline/input-manifest.json',
            rgb_manifest_sha256=sha256(ROOT/'artifacts-local/edge-obstacle-offline/input-manifest.json'),
            order='RGB cases in prior D200/D150/D100/D070 PRESENT/ABSENT order, then N,D,L,R within each case'),
        phase_b=dict(primary_trials=4,trial_order=['N','D','L','R'],model_loads=1,
            service_policy='One fresh Phase-B service shared by four independent Isaac processes; no per-condition reseed',
            fresh_world_bootstrap_history_controller_per_trial=True,model_forward_warmup=False,
            start_pose=[0.,0.,0.],scientific_timeout_sim_s=40.,physics_hz=60,observation_hz=3,
            waypoint_index=4,unchanged_controller=True,artificial_latency=False,
            continue_after_valid_collision=True,stop_after_implementation_or_runtime_failure=True),
        seed_policy='Torch and NumPy seed 0 once at each phase service load; eval; no reseed per condition; retain first cold forward in each phase',
        static_config='research/frozen-edge-static-obstacle.json',static_config_sha256=sha256(ROOT/'research/frozen-edge-static-obstacle.json'),
        interpretation=dict(material_lateral_threshold_m=.2,statistical_significance=False,
            direction_epsilon_m=1e-6,language_response='Any metric XY delta >1e-6 m from neutral; descriptive numerical observation',
            direction_consistency='L endpoint local Y >1e-6, R endpoint local Y <-1e-6, L-N endpoint delta Y >1e-6 and R-N <-1e-6; report each separately',
            material_direction='Signed intended-direction row delta reaches 0.20 m; also report max absolute delta independent of sign',
            first_executed_turn='First applied angular command with abs(w)>1e-6 rad/s',
            safety='Previous full polyline criterion, observation included; 0.20-m radius and 0.10-m margin; downstream expanded x=3.1',
            waypoint_timestamps=None,externally_supplied_avoidance_side=True),
        failure_policy='No scientific retries or retuning; Phase B follows technically valid A regardless of geometric outcome; OOM/runtime/routing/data error stops science and records not-run cases',
        stop_rules=['swept collision','goal-region entry (0.50 m)','40 s after bootstrap','upstream controller bug','implementation/runtime failure'])
    write_json(PLAN,plan)
    print({key:(spec['token_count_including_sot_eot'],spec['token_sha256']) for key,spec in instructions.items()})


if __name__=='__main__':main()
