"""Offline checks of frozen provenance, spatial prediction semantics and evidence."""
import json
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research.isaac_final_common import ROOT, sha256
from research.static_obstacle_protocol import verify_static


class StaticEvidenceTests(unittest.TestCase):
    def test_compact_result_and_frozen_scenario(self):
        result=json.loads((ROOT/'results/edge-static-obstacle-summary.json').read_text())
        config=ROOT/result['frozen_config_path']
        self.assertEqual(result['frozen_config_sha256'],sha256(config))
        self.assertEqual(result['primary_trial_count'],1)
        self.assertEqual(result['retries'],0)
        self.assertEqual(result['infrastructure_status'],'PASS')
        self.assertEqual(result['scientific_outcome'],'COLLISION')
        self.assertFalse(result['metrics']['goal_reached'])
        self.assertLessEqual(result['metrics']['minimum_signed_clearance_m'],0)
        self.assertLess(result['metrics']['first_collision_time_sim_s'],result['metrics']['elapsed_sim_s'])
        self.assertEqual(result['scientific_process']['isaac_exit_code'],0)
        self.assertEqual(result['scientific_process']['service_exit_code'],0)

    def test_observation_anchored_world_prediction_no_second_cumsum(self):
        from research.analyze_static_obstacle import world_prediction
        native=np.tile(np.array([10.,20.,0.,1.],np.float32),(1,8,1))
        original=native.copy()
        xy,heading=world_prediction(native,[3,4,math.pi/2])
        np.testing.assert_allclose(xy,np.tile([1.,5.],(8,1)),atol=1e-12)
        np.testing.assert_allclose(heading,np.tile([-1.,0.],(8,1)),atol=1e-12)
        np.testing.assert_array_equal(native,original)

    def test_recorded_evidence_and_old_artifacts_when_available(self):
        from research.validate_static_obstacle_evidence import validate
        run=ROOT/'outputs/static-obstacle/scientific-001'
        if not run.exists(): self.skipTest('Ignored raw runtime evidence unavailable')
        result=validate(run,ROOT/'research/frozen-edge-static-obstacle.json')
        self.assertEqual(result['validation'],'PASS')
        self.assertEqual(result['baseline_artifacts_verified'],330)
        self.assertEqual(result['episode']['requests'],21)
        self.assertEqual(result['episode']['states'],440)

    def test_config_hash_enforcement_when_gate_files_available(self):
        if not (ROOT/'artifacts-local/static-obstacle/protocol-freeze.json').exists():
            self.skipTest('Ignored gate evidence unavailable')
        config=json.loads((ROOT/'research/frozen-edge-static-obstacle.json').read_text())
        verify_static(config,ROOT/'research/frozen-edge-static-obstacle.json')
        with tempfile.TemporaryDirectory() as temporary:
            other=Path(temporary)/'config.json'
            other.write_text(json.dumps(dict(config,unauthorized_post_freeze_change=True)))
            with self.assertRaisesRegex(ValueError,'configuration hash changed'):
                verify_static(config,other)
