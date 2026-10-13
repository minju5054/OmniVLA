"""Negative tests for paired metadata and evidence guards (no model imports)."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from research.isaac_final_common import sha256
from research.isaac_final_contracts import local_goal
from research.offline_obstacle_protocol import PLAN, validate_case, validate_pair, red_mask, image_difference, verify_protocol


class PairedInputTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads(PLAN.read_text()); self.expected=self.plan['cases'][0]
        self.case=dict(**deepcopy(self.expected),world_goal=self.plan['world_goal'],no_gui_markers=True,
            obstacle_prim_present=True,obstacle_geometry={'present':True},camera_config={'height_m':.5},
            camera_intrinsics={'f':.02},scene={},renderer='RaytracedLighting',stage_paths=['/World','/World/StaticObstacle'])
        self.case['local_goal'],self.case['normalized_pose']=local_goal(self.case['pose'],self.case['world_goal'])
        self.case['history']=[dict(frame_id=i,case_step=i*20,sim_time=i/3,wall_time=100+i+.01,
            capture_start_wall_time=100+i,absolute_physics_step=120+i*20,absolute_world_sim_time=2+i/3,
            pose=[.2,0.,0.],camera_position=[.2,0.,.5],camera_orientation_wxyz=[1.,0.,0.,0.],rgb_shape=[480,640,3]) for i in range(6)]

    def test_good_metadata_and_pair(self):
        validate_case(self.case,self.expected,self.plan,False)
        absent=deepcopy(self.case);absent.update(condition='OBSTACLE_ABSENT',obstacle_prim_present=False,
            obstacle_geometry=None,stage_paths=['/World'])
        validate_pair(self.case,absent)
        absent['normalized_pose'][0]+=.01
        with self.assertRaises(AssertionError): validate_pair(self.case,absent)

    def test_pose_goal_condition_history_rejections(self):
        for field,value in [('pose',[.3,0,0]),('world_goal',[6,0,0]),('condition','OBSTACLE_ABSENT'),
                            ('no_gui_markers',False),('obstacle_prim_present',False)]:
            case=deepcopy(self.case);case[field]=value
            with self.subTest(field=field),self.assertRaises(AssertionError): validate_case(case,self.expected,self.plan,False)
        for field,value in [('case_step',19),('sim_time',.2),('frame_id',0),('capture_start_wall_time',99),
                            ('absolute_physics_step',139),('camera_position',[.3,0,.5])]:
            case=deepcopy(self.case);case['history'][1][field]=value
            with self.subTest(field=field),self.assertRaises((AssertionError,ValueError)): validate_case(case,self.expected,self.plan,False)

    def test_short_history_rejected(self):
        self.case['history'].pop()
        with self.assertRaises(AssertionError): validate_case(self.case,self.expected,self.plan,False)

    def test_image_noise_and_red_mask(self):
        a=np.zeros((4,4,3),np.uint8);b=a.copy();b[0,0]=[180,10,10]
        self.assertEqual(red_mask(b).sum(),1)
        self.assertEqual(image_difference(a,b)['changed_pixel_fraction'],1/16)
        self.assertEqual(image_difference(a,b,~red_mask(b))['mean_absolute_channel_difference'],0)

    def test_frozen_file_tamper_rejected_before_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            evidence=Path(d)/'source';evidence.write_text('before')
            frozen=Path(d)/'frozen.json'
            frozen.write_text(json.dumps(dict(status='FROZEN_BEFORE_INFERENCE',files_sha256={},
                source_sha256={str(evidence):sha256(evidence)})))
            evidence.write_text('after')
            with patch('research.offline_obstacle_protocol.verify_static'),self.assertRaises(AssertionError):
                verify_protocol(frozen)


if __name__=='__main__': unittest.main()
