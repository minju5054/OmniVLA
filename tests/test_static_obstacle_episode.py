"""Execute the real episode function against a deterministic model-free peer.

Only simulator rendering/clock, image capture and the model peer are replaced.
Real scheduler, integration, request formatting, raw response validation and the
upstream controller exercise collision termination without a scientific trial.
"""
import ast
import json
import math
from pathlib import Path
import tempfile
import time
import traceback
import unittest
import numpy as np
from research.edge_contracts import controller
from research.isaac_final_common import ROOT, append_json, sha256, write_json
from research.isaac_final_contracts import History, Scheduler, action, integrate, local_goal, stats, wrap
from research.static_obstacle_geometry import Rectangle, classify


def function(path, name, scope):
    tree=ast.parse((ROOT/path).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]


class EpisodeTests(unittest.TestCase):
    def run_episode(self, goal, obstacle, failure=None):
        class World:
            current_time=0.
            def step(self,render=True): self.current_time+=1/60
        def capture(camera,root,history,frame_id,sim_time,pose):
            history.append(dict(frame_id=frame_id,sim_time=sim_time,wall_time=time.time(),
                                rgb_sha256=f'unit-frame-{frame_id}',pose=list(pose)))
        def bootstrap(world,camera,root):
            history=History()
            for i in range(6): capture(camera,root,history,i,i/3,[0.,0.,0.])
            return history,[0.,0.,0.]
        scope=dict(np=np,math=math,time=time,traceback=traceback,Path=Path,
                   History=History,Scheduler=Scheduler,action=action,integrate=integrate,
                   local_goal=local_goal,stats=stats,wrap=wrap,Rectangle=Rectangle,
                   classify=classify,append_json=append_json,write_json=write_json,
                   sha256=sha256,capture=capture,bootstrap=bootstrap,set_pose=lambda *a:None)
        function('research/isaac_final_navigation.py','request',scope)
        function('research/isaac_final_navigation.py','validate_response',scope)
        episode=function('research/static_obstacle_navigation.py','episode',scope)
        class Peer:
            pending=None
            def send(self,request): self.pending=request
            def poll(self):
                if failure=='implementation': raise ConnectionError('Synthetic unit-test IPC failure')
                if self.pending is None: return None
                request=self.pending;self.pending=None
                raw=np.tile(np.array([10.,0.,1.,0.],np.float32),(1,8,1))
                path=Path(request['action_path'])
                with path.open('xb') as stream: np.save(stream,raw,allow_pickle=False)
                metric=raw[0,4].copy();metric[:2]*=.1
                response=dict(request_id=request['request_id'],observation_id=request['observation_id'],
                              action_path=str(path),action_sha256=sha256(path),modality_id=4,selected_index=4,
                              selected_native=raw[0,4].tolist(),selected_metric=metric.tolist(),
                              rgb_hashes=[f['rgb_sha256'] for f in request['history']],
                              controller=controller(metric),model_ms=0.,forward_with_clip_ms=0.,
                              pipeline_ms=0.,ready_wall_time=time.time())
                if failure=='controller': response['controller_error']='UPSTREAM_CONTROLLER_BUG: clip_angle'
                return response
        config=dict(static_obstacle=obstacle,evaluation=dict(agent_radius_m=.2,safety_margin_m=.1),
                    goal_radius_m=.5,timeout_after_bootstrap_sim_s=40.)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'episode'
            result=episode(World(),None,config,Peer(),root,dict(name='UNIT_ONLY',pose=goal),True)
            states=[json.loads(line) for line in (root/'raw/robot_state.jsonl').read_text().splitlines()]
            self.assertEqual(len(states),result['completed_steps']+1)
            self.assertTrue(all(row['applied_vw'] in ([0.,0.],[.3,0.]) for row in states))
            for path in (root/'raw/actions').glob('*.npy'):
                np.testing.assert_array_equal(np.load(path),np.tile(np.array([10.,0.,1.,0.],np.float32),(1,8,1)))
            return result,states

    def test_collision_stops_before_any_additional_step(self):
        result,states=self.run_episode([5.,0.,0.],dict(center=[2.5,0.,.5],dimensions=[.6,1.,1.]))
        self.assertEqual(result['status'],'COLLISION')
        self.assertFalse(result['goal_reached'])
        self.assertTrue(states[-1]['collision'])
        self.assertTrue(all(not row['collision'] for row in states[:-1]))
        self.assertAlmostEqual(result['first_collision_time_sim_s'],1/60+2/.3)
        self.assertLessEqual(result['elapsed_sim_s']-result['first_collision_time_sim_s'],1/60)

    def test_safe_and_low_clearance_goals(self):
        for center, expected in [([2.5,0.,.5],'SAFE_GOAL_REACHED'),([.45,.75,.5],'GOAL_REACHED_LOW_CLEARANCE')]:
            result,_=self.run_episode([.6,0.,0.],dict(center=center,dimensions=[.6,1.,1.]))
            self.assertEqual(result['status'],expected)
            self.assertTrue(result['goal_reached'])

    def test_implementation_and_upstream_failures(self):
        for failure, expected in [('implementation','IMPLEMENTATION_FAILURE'),('controller','UPSTREAM_CONTROLLER_BUG')]:
            result,_=self.run_episode([5.,0.,0.],dict(center=[2.5,0.,.5],dimensions=[.6,1.,1.]),failure)
            self.assertEqual(result['status'],expected)
