"""Additive ID-8 metadata contracts; unchanged source supplies numerical rules."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from research.isaac_final_common import ROOT,sha256
from research.isaac_final_contracts import action,local_goal
from research.edge_contracts import modality
from research.contracts import _method,_exec_nodes

PLAN=ROOT/'research/pose-language-obstacle-plan.json'
PROTOCOL=ROOT/'research/frozen-pose-language-obstacle.json'
FLAGS=dict(pose_goal=True,satellite=False,image_goal=False,lan_prompt=True)
MASK=[False]*7+[True,True,False]
PROMPTS={
    'N':'Go to the goal.',
    'D':'A large solid red rectangular box is blocking the path directly ahead. Reach the goal while avoiding the obstacle.',
    'L':'A large solid red rectangular box blocks the path directly ahead. Steer to your left, pass around the box with safe clearance, then continue to the goal.',
    'R':'A large solid red rectangular box blocks the path directly ahead. Steer to your right, pass around the box with safe clearance, then continue to the goal.'}


def utf8_hash(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def tensor_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def source_masks(torch):
    nodes=_method(ROOT/'inference/model_omnivla_edge.py','OmniVLA_edge','__init__').body
    start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self.goal_mask_0')
    obj=SimpleNamespace(context_size=5)
    _exec_nodes(nodes[start:],dict(torch=torch,self=obj),ROOT/'inference/model_omnivla_edge.py')
    assert modality(**FLAGS)==8
    assert obj.all_masks[8].tolist()==MASK
    return obj


def instruction_fields(spec):
    return dict(instruction_id=spec['id'],instruction_text=spec['text'],instruction_hash=spec['utf8_sha256'],
        clip_token_ids=spec['token_ids'],clip_token_sha256=spec['token_sha256'],modality_id=8)


def validate_instruction(request,plan):
    spec=plan['instructions'][request['instruction_id']]
    assert spec['text']==PROMPTS[spec['id']]
    for key,value in instruction_fields(spec).items():assert request[key]==value,key
    assert request['instruction_hash']==utf8_hash(request['instruction_text'])
    local,normalized=local_goal(request['robot_pose_observation'],request['world_goal'])
    assert request['local_goal']==local and request['normalized_pose']==normalized
    assert request['world_goal']==plan['world_goal']
    return spec


def validate_response(response,request):
    assert response['modality_id']==8, 'Never fall back to ID 4'
    assert response['request_episode_id']==request['episode_id']
    for key in ('request_id','observation_id','instruction_id','instruction_text','instruction_hash','clip_token_ids','clip_token_sha256'):
        assert response[key]==request[key],key
    assert response['rgb_hashes']==[f['rgb_sha256'] for f in request['history']]
    if request['op']=='dummy':
        assert response['dummy'] and response['lossless_roundtrip'];return
    assert response['selected_index']==4 and response['decoder_mask']==MASK
    assert response['decoder_pool_weights']==[0. if m else 1.25 for m in MASK]
    assert response['clip_embedding_shape']==[1,512] and response['clip_embedding_dtype']=='float32'
    assert len(response['clip_embedding_sha256'])==64
    assert sha256(response['action_path'])==response['action_sha256']
    raw=action(np.load(response['action_path'],allow_pickle=False))
    np.testing.assert_array_equal(raw[0,4],np.asarray(response['selected_native'],dtype=np.float32))
    assert np.allclose(np.asarray(response['selected_metric'])[:2],raw[0,4,:2]*.1,atol=1e-7)


def verify_protocol(path=PROTOCOL,inputs=True):
    from research.static_obstacle_protocol import verify_static
    from research.offline_obstacle_protocol import verify_protocol as verify_previous
    frozen=json.loads(Path(path).read_text());assert frozen['status']=='FROZEN_BEFORE_ANY_INFERENCE'
    for collection in ('source_sha256','evidence_sha256'):
        for p,digest in frozen[collection].items():assert sha256(ROOT/p)==digest,p
    config_path=ROOT/'research/frozen-edge-static-obstacle.json'
    config=json.loads(config_path.read_text());verify_static(config,config_path)
    plan=json.loads(PLAN.read_text())
    assert list(plan['instructions'])==['N','D','L','R']
    assert len(plan['offline_order'])==32 and len(plan['navigation_order'])==4
    for key,spec in plan['instructions'].items():
        assert spec['text']==PROMPTS[key] and spec['utf8_sha256']==utf8_hash(PROMPTS[key])
    if inputs:verify_previous(ROOT/'research/frozen-edge-obstacle-offline.json')
    return frozen,plan,config
