"""CPU-only frozen-spec and failure-evidence checks; no Isaac/model execution.

Navigation adapter tests are deferred: Stage A failed before it was authorized
to proceed to model integration, history capture or closed-loop execution.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class IsaacStartupEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.result = json.loads((ROOT/'results/edge-isaac-no-obstacle-summary.json').read_text())
        self.frozen = json.loads((ROOT/'research/frozen-edge-isaac-goals.json').read_text())

    def test_frozen_before_runtime_hash(self):
        digest=hashlib.sha256((ROOT/'research/frozen-edge-isaac-goals.json').read_bytes()).hexdigest()
        self.assertEqual(digest, 'd1f37af3f136839ac835a3403025d5480e4ef86b7ef55e5d76fc9fa6ebe2684f')
        self.assertEqual(digest,self.result['frozen_config_sha256'])

    def test_requested_goals_and_bounds_preserved(self):
        self.assertEqual(self.frozen['start'],[0,0,0])
        self.assertEqual([g['pose'][:2] for g in self.frozen['goals']],[[5,0],[5,2],[5,-2]])
        self.assertEqual(self.frozen['goal_radius_m'],.5)
        self.assertEqual(self.frozen['timeout_after_bootstrap_sim_s'],40)
        self.assertEqual((self.frozen['physics_hz'],self.frozen['observation_hz']),(60,3))

    def test_controller_and_modality_not_retuned(self):
        old=json.loads((ROOT/'results/edge-controller-summary.json').read_text())['constants']
        c=self.frozen['controller']
        for new,prior in [('waypoint_index','waypoint_select'),('metric_scale','metric_waypoint_spacing'),
                          ('DT','DT'),('EPS','EPS'),('maxv','maxv'),('maxw','maxw')]:
            self.assertEqual(c[new],old[prior])
        self.assertEqual(self.frozen['modality']['id'],4)

    def test_failure_cannot_be_navigation_success(self):
        self.assertEqual(self.result['decision'],'ISAAC_STARTUP_BLOCKED')
        self.assertEqual(self.result['startup_classification'],'ISAAC_STARTUP_FAILED')
        self.assertFalse(self.result['constructor_returned'])
        self.assertFalse(self.result['clean_shutdown'])
        self.assertEqual(self.result['task_simulation_steps'],0)

    def test_stop_rule_no_later_execution(self):
        self.assertEqual(self.result['startup_attempts'],1)
        self.assertFalse(self.result['edge_service_started'])
        self.assertEqual(self.result['combined_inference_count'],0)
        self.assertEqual(self.result['navigation_inference_count'],0)
        for run in self.result['runs'].values():
            self.assertEqual(run['status'],'NOT_RUN_BLOCKED_BY_STAGE_A')
            self.assertIsNone(run['goal_reached'])
            self.assertIsNone(run['final_distance_m'])
        self.assertIsNone(self.result['inference_latency_ms'])
        self.assertIsNone(self.result['gpu']['combined_peak_MiB'])

    def test_historical_reports_unchanged(self):
        for name, expected in self.result['historical_report_sha256'].items():
            p=ROOT/name
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),expected)
            self.assertEqual(p.read_bytes(),subprocess.check_output(
                ['git','show',f'9ebb503a5452498dbf90c95a89ef4aa26a5bfec4:{name}'],cwd=ROOT))

    def test_scene_is_declaration_not_measured_stage(self):
        s=json.loads((ROOT/'results/edge-isaac-scene-declaration.json').read_text())
        self.assertEqual(s['status'],'DECLARED_NOT_CONSTRUCTED')
        self.assertFalse(self.result['task_stage_constructed'])
        self.assertFalse(self.result['task_camera_rendered'])

    def test_raw_log_hash_if_local_evidence_available(self):
        p=ROOT/self.result['raw_log_path']
        if not p.exists():
            self.skipTest('Ignored local raw log not present in this checkout')
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),self.result['raw_log_sha256'])
        log=p.read_text()
        self.assertIn('undefined symbol: __nvJitLinkCreate_12_8',log)
        self.assertIn('Fatal Python error: Segmentation fault',log)


if __name__=='__main__':
    unittest.main()
