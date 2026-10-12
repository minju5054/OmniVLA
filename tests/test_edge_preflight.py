"""CPU source/unit contracts and separately labelled recorded-runtime evidence."""
import ast
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from research.contracts import ROOT, _method, _exec_nodes, evidence_directory, immutable_action, metric_action, sha256, timed_cuda, write_json
from research.edge_contracts import controller, goal, modality, FULL_REPORT_SHA
from research.fetch_edge_checkpoint import REVISION, SHA256, SIZE


class EdgeSourceUnitTests(unittest.TestCase):
    def test_checkpoint_identity(self):
        m = json.loads((ROOT/'results/edge-checkpoint-manifest.json').read_text())
        self.assertEqual((m['repo'],m['revision'],m['sha256'],m['bytes']),
                         ('NHirose/omnivla-edge',REVISION,SHA256,SIZE))

    def test_output_contract(self):
        self.assertEqual(immutable_action(np.zeros((1,8,4))).shape, (1,8,4))
        with self.assertRaises(ValueError):
            immutable_action(np.zeros((8,4)))

    def test_finite(self):
        for value in (np.nan,np.inf,-np.inf):
            with self.assertRaises(ValueError):
                immutable_action(np.full((1,8,4),value))

    def test_native_immutable(self):
        original = np.zeros((1,8,4),dtype=np.float32)
        frozen = immutable_action(original)
        original[:] = 42
        self.assertTrue((frozen==0).all())
        with self.assertRaises(ValueError):
            frozen.setflags(write=True)

    def test_source_cumulative_and_heading(self):
        path = ROOT/'inference/model_omnivla_edge.py'
        method = _method(path,'OmniVLA_edge','forward')
        start = next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign)
                     and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='torch.cumsum')
        a = torch.tensor([[[1.,2.,3.,4.]]*8])
        scope = dict(torch=torch,F=torch.nn.functional,action_pred=a)
        _exec_nodes(method.body[start:-1],scope,path)
        np.testing.assert_allclose(a[0,:,0],np.arange(1,9))
        np.testing.assert_allclose(a[0,:,1],2*np.arange(1,9))
        np.testing.assert_allclose(a[0,:,2:],np.tile([.6,.8],(8,1)),rtol=1e-6)

    def test_frozen_pose_transforms(self):
        frozen=json.loads((ROOT/'research/frozen-edge-goals.json').read_text())
        for s in frozen['tests']:
            np.testing.assert_allclose(goal(s['delta_east_m'],s['delta_north_m'],0,s['goal_compass_deg']),s['expected_pose'],atol=1e-12)

    def test_goal_radius_clamp(self):
        self.assertAlmostEqual(np.linalg.norm(goal(100,100)[:2]),300)

    def test_pose_only_id(self):
        self.assertEqual(modality(pose_goal=True),4)

    def test_language_id(self):
        self.assertEqual(modality(lan_prompt=True),7)

    def test_masks(self):
        path=ROOT/'inference/model_omnivla_edge.py'
        nodes=_method(path,'OmniVLA_edge','__init__').body
        start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self.goal_mask_0')
        obj=SimpleNamespace(context_size=5)
        _exec_nodes(nodes[start:],dict(torch=torch,self=obj),path)
        self.assertEqual(obj.all_masks[4].tolist(),[False]*7+[True,True,True])
        self.assertEqual(obj.all_masks[7].tolist(),[False]*6+[True,True,True,False])
        self.assertEqual(obj.avg_pool_mask[4,8].item(),0)

    def test_controller_straight(self):
        self.assertEqual(controller([1,0,1,0])['final_vw'],[.3,0])

    def test_controller_left_right(self):
        for sign in [-1,1]:
            c=controller([1,sign,1,0])
            np.testing.assert_allclose(c['pre_clip_vw'],[3,sign*3*np.pi/4])
            np.testing.assert_allclose(c['clipped_vw'],[.5,sign])
            np.testing.assert_allclose(c['final_vw'],[.15,sign*.3])

    def test_controller_zero_x(self):
        np.testing.assert_allclose(controller([0,1,1,0])['final_vw'],[0,.3])

    def test_controller_zero_xy_upstream_bug(self):
        with self.assertRaisesRegex(NameError,'clip_angle'):
            controller([0,0,0,1])

    def test_history_six(self):
        n=next(n for n in _method(ROOT/'inference/run_omnivla_edge.py','Inference','run_omnivla').body
               if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='context_queue')
        self.assertEqual(len(n.value.elts),6)
        self.assertTrue(all(ast.unparse(v)=='current_image_PIL_96' for v in n.value.elts))

    def test_metric_scaling(self):
        a=immutable_action(np.ones((1,8,4)))
        scaled=metric_action(a)
        np.testing.assert_allclose(scaled[...,:2],.1)
        np.testing.assert_allclose(scaled[...,2:],1)
        np.testing.assert_allclose(a,1)

    def test_cuda_sync_boundaries(self):
        order=[]
        cuda=SimpleNamespace(synchronize=lambda:order.append('sync'))
        clock=iter([10.,11.]).__next__
        value,t=timed_cuda(lambda:order.append('forward') or 42,cuda,clock)
        self.assertEqual(order,['sync','forward','sync'])
        self.assertEqual((value,t),(42,1))

    def test_evidence_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'run'
            evidence_directory(p)
            with self.assertRaises(FileExistsError):
                evidence_directory(p)
            write_json(p/'a.json',{'a':1})
            with self.assertRaises(FileExistsError):
                write_json(p/'a.json',{'a':2})

    def test_artifacts_ignored_and_not_tracked(self):
        paths=['models-local/omnivla-edge/x/omnivla-edge.pth','models-local/clip/ViT-B-32.pt',
               'artifacts-local/envs/edge-blackwell/a','outputs/edge-preflight-001/raw/official-native.npy',
               'models-local/omnivla-original/a']
        for p in paths:
            self.assertEqual(subprocess.run(['git','check-ignore','-q',p],cwd=ROOT).returncode,0)
        self.assertEqual(subprocess.check_output(['git','ls-files','models-local','artifacts-local','outputs'],cwd=ROOT),b'')

    def test_full_report_unchanged(self):
        self.assertEqual(sha256(ROOT/'docs/FULL_OMNIVLA_PREFLIGHT.md'),FULL_REPORT_SHA)
        prior=subprocess.check_output(['git','show','3fddf03e89e4c7145ceb241532bf907444b6a87d:docs/FULL_OMNIVLA_PREFLIGHT.md'],cwd=ROOT)
        self.assertEqual((ROOT/'docs/FULL_OMNIVLA_PREFLIGHT.md').read_bytes(),prior)


class EdgeRecordedRuntimeTests(unittest.TestCase):
    """Validate committed measurements; these tests themselves never run a model."""
    def test_strict_load_metadata(self):
        r=json.loads((ROOT/'results/edge-runtime.json').read_text())['strict_load']
        self.assertEqual(r['status'],'EDGE_CHECKPOINT_STRICT_LOAD_PASS')
        self.assertTrue(r['strict'] and r['finite_parameters'] and r['finite_clip_parameters'])
        self.assertEqual(r['missing_keys'],[])
        self.assertEqual(r['unexpected_keys'],[])
        self.assertGreater(r['parameter_count'],0)

    def test_sample_evidence(self):
        s=json.loads((ROOT/'results/edge-sample-summary.json').read_text())
        self.assertEqual(s['status'],'EDGE_OFFICIAL_SAMPLE_PASS')
        a=np.array(s['native'])[None]
        immutable_action(a)
        np.testing.assert_allclose(np.linalg.norm(a[...,2:],axis=-1),1,atol=1e-6)
        self.assertEqual(s['batch']['obs_images']['shape'],[1,18,96,96])
        self.assertEqual(s['modality_id'],7)

    def test_frozen_spec_provenance(self):
        c=json.loads((ROOT/'results/edge-coordinate-summary.json').read_text())
        self.assertEqual(c['frozen_spec_sha256'],sha256(ROOT/'research/frozen-edge-goals.json'))

    def test_pose_model_evidence(self):
        c=json.loads((ROOT/'results/edge-coordinate-summary.json').read_text())
        self.assertEqual(c['status'],'EDGE_POSE_GOAL_SANITY_PASS')
        for name, sign in [('FRONT',0),('LEFT',1),('RIGHT',-1)]:
            s=c['tests'][name]
            self.assertTrue(s['passed'])
            self.assertEqual(s['modality_id'],4)
            self.assertGreater(s['selected_metric'][0],0)
            if sign:
                self.assertGreater(sign*s['selected_metric'][1],0)
                self.assertGreater(sign*s['final_vw'][1],0)
            else:
                self.assertLessEqual(abs(np.arctan2(*s['selected_metric'][1::-1])),np.pi/4)

    def test_masking_model_evidence(self):
        m=json.loads((ROOT/'results/edge-coordinate-summary.json').read_text())['masking']
        self.assertEqual(m['status'],'POSE_ONLY_MASKING_RUNTIME_PASS')
        self.assertEqual((m['max_abs'],m['mean_abs']),(0,0))
        self.assertEqual(m['final_vw_diff'],[0,0])

    def test_latency_and_memory_evidence(self):
        l=json.loads((ROOT/'results/edge-latency-summary.json').read_text())
        self.assertGreaterEqual(l['forward']['count'],50)
        self.assertGreaterEqual(l['pipeline']['count'],20)
        self.assertTrue(l['memory_stable'])
        self.assertEqual(l['repeat_output_max_abs_diff'],0)
        s=json.loads((ROOT/'results/edge-sample-summary.json').read_text())
        self.assertEqual(s['actual_official_vw'],s['final_vw'])


if __name__=='__main__':
    unittest.main()
