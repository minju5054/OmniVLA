"""Official ID-8/token/preprocessing and one-shot guards, without model inference."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from research.isaac_final_common import ROOT,sha256,write_json
from research.isaac_final_contracts import local_goal,History
from research.edge_contracts import modality,initialize_sample_globals,prepare_function
from research.pose_language_contracts import PLAN,FLAGS,MASK,PROMPTS,source_masks,tensor_hash,utf8_hash,instruction_fields,validate_instruction,validate_response
from research.pose_language_adapter import bind_episode
from research.pose_language_service import base,prepare
from research.run_pose_language import register_once


def extract(path,name,scope):
    node=next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),scope)
    return scope[name]


class PoseLanguageTests(unittest.TestCase):
    def setUp(self):self.plan=json.loads(PLAN.read_text())

    def test_official_routing_and_both_mask_stages(self):
        self.assertEqual(modality(**FLAGS),8)
        obj=source_masks(torch)
        self.assertEqual(obj.all_masks[8].tolist(),[False]*7+[True,True,False])
        self.assertEqual(obj.avg_pool_mask[8].tolist(),[1.25]*7+[0.,0.,1.25])
        self.assertEqual(obj.all_masks[4].tolist(),[False]*7+[True]*3)
        self.assertEqual(obj.all_masks[7].tolist(),[False]*6+[True]*3+[False])

    def test_exact_prompt_tokens_untruncated_distinct(self):
        identities=[]
        for key,spec in self.plan['instructions'].items():
            self.assertEqual(spec['text'],PROMPTS[key]);self.assertEqual(spec['utf8_sha256'],utf8_hash(PROMPTS[key]))
            tokens=base.clip.tokenize(PROMPTS[key],truncate=False)
            self.assertTrue(torch.equal(tokens,base.clip.tokenize(PROMPTS[key],truncate=True)))
            self.assertEqual(tokens.numpy().tolist(),spec['token_ids'])
            self.assertEqual(tensor_hash(tokens.numpy()),spec['token_sha256'])
            self.assertLessEqual(spec['token_count_including_sot_eot'],77);identities.append(spec['token_sha256'])
        self.assertEqual(len(set(identities)),4)

    def test_official_preprocessing_parity_all_four_prompts(self):
        official=base.official;initialize_sample_globals(official);official.device=torch.device('cpu')
        for name,value in FLAGS.items():setattr(official,name,value)
        official.goal_image_PIL=Image.new('RGB',(96,96))
        pixels=np.asarray(Image.open(ROOT/'inference/current_img.jpg').convert('RGB'))
        for spec in self.plan['instructions'].values():
            instance=official.Inference('.',spec['text'],official.goal_utm,official.goal_compass,official.goal_image_PIL)
            expected=prepare_function(official)(instance)
            actual=prepare([pixels]*6,expected['goal_pose_torch'][0].tolist(),spec)
            for key in expected:self.assertTrue(torch.equal(actual[key],expected[key]),key)

    def test_real_history_order_and_only_language_field_changes(self):
        official=base.official;initialize_sample_globals(official);official.device=torch.device('cpu')
        frames=[np.full((480,640,3),30*i,dtype=np.uint8) for i in range(6)]
        pose=[2.,1.,.7];goal=[5.,0.,0.];_,encoding=local_goal(pose,goal)
        neutral=base.prepare(frames,encoding)
        actual=prepare(frames,encoding,self.plan['instructions']['L'])
        for key in actual:
            if key!='obj_inst_lan':self.assertTrue(torch.equal(actual[key],neutral[key]),key)
        self.assertEqual(actual['goal_pose_torch'].dtype,torch.float32)
        self.assertEqual(actual['goal_pose_torch'].tolist(),np.array([encoding],np.float32).tolist())

    def test_episode_code_identity_and_request_metadata_before_logging(self):
        scope=dict(local_goal=local_goal,time=__import__('time'))
        original=extract('research/isaac_final_navigation.py','request',scope)
        episode=extract('research/static_obstacle_navigation.py','episode',scope)
        baseline=SimpleNamespace(request=original,episode=episode)
        bound=bind_episode(baseline,self.plan['instructions']['D'])
        self.assertIs(bound.__code__,episode.__code__)
        self.assertIs(scope['request'],original)
        history=History()
        for i in range(6):history.append(dict(frame_id=i,sim_time=i/3))
        arguments=(Path('/tmp/unit-episode'),'POSE_LANGUAGE_DESCRIPTION',0,history,5/3,[0.,0.,0.],[5.,0.,0.])
        raw=original(*arguments);request=bound.__globals__['request'](*arguments)
        for key,value in raw.items():
            if key!='request_send_wall_time':self.assertEqual(request[key],value)
        self.assertEqual(request['modality_id'],8);validate_instruction(request,self.plan)
        self.assertNotIn('instruction_text',raw)

    def test_instruction_identity_and_observation_goal_mismatch_rejected(self):
        req=dict(robot_pose_observation=[0.,0.,0.],world_goal=[5.,0.,0.],local_goal=[5.,0.,0.],
            normalized_pose=[50.,0.,1.,0.],**instruction_fields(self.plan['instructions']['L']))
        validate_instruction(req,self.plan)
        for key,value in [('instruction_text','Steer right.'),('instruction_id','R'),('modality_id',4),
            ('clip_token_sha256','0'*64),('robot_pose_observation',[.1,0.,0.])]:
            changed=deepcopy(req);changed[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_instruction(changed,self.plan)

    def test_response_never_accepts_modality_four(self):
        with self.assertRaises(AssertionError):validate_response(dict(modality_id=4),{})

    def test_registration_and_writer_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as d,patch('research.run_pose_language.sha256',return_value='unit-protocol'):
            p=Path(d)/'registration.json';register_once(p,'offline',self.plan['offline_order'])
            before=p.read_bytes()
            with self.assertRaises(FileExistsError):register_once(p,'offline',[])
            self.assertEqual(before,p.read_bytes())

    def test_fixed_matrix_and_navigation_order(self):
        self.assertEqual(len(self.plan['offline_order']),32)
        self.assertEqual([r['request_id'] for r in self.plan['offline_order']],list(range(32)))
        for i in range(0,32,4):self.assertEqual([r['instruction_id'] for r in self.plan['offline_order'][i:i+4]],list('NDLR'))
        self.assertEqual([r['instruction_id'] for r in self.plan['navigation_order']],list('NDLR'))
        self.assertTrue(self.plan['phase_b']['continue_after_valid_collision'])


if __name__=='__main__':unittest.main()
