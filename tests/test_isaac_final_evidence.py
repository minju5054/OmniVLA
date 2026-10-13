"""Frozen scientific evidence checks, separated from runtime execution."""
import json
from pathlib import Path
import unittest
from research.isaac_final_common import ROOT, sha256, verify_frozen


class FinalEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'research/frozen-edge-isaac-no-obstacle-final.json').read_text())

    def test_protocol_is_frozen_and_gate_complete(self):
        verify_frozen(self.config)
        self.assertEqual(sha256(ROOT/'research/frozen-edge-isaac-no-obstacle-final.json'),
                         'fb40d57f7999da622d99f33ed019132d1562dde40d14f8c506e40efde63ccbf2')
        self.assertEqual(self.config['scientific_retry_policy']['primary_runs_per_goal'],1)
        self.assertFalse(self.config['scientific_retry_policy']['retries'])
        self.assertFalse(self.config['scientific_retry_policy']['post_outcome_tuning'])

    def test_scene_camera_and_scientific_constants_preserved(self):
        previous=json.loads((ROOT/'research/frozen-edge-isaac-goals.json').read_text())
        for field in ('scene','camera','start','goals','physics_hz','observation_hz',
                      'history','goal_radius_m','timeout_after_bootstrap_sim_s','controller','modality'):
            self.assertEqual(self.config[field],previous[field],field)

    def test_compact_decision_requires_all_three_goals_and_valid_evidence(self):
        result=json.loads((ROOT/'results/edge-isaac-no-obstacle-final.json').read_text())
        self.assertEqual(result['frozen_config_sha256'],sha256(ROOT/'research/frozen-edge-isaac-no-obstacle-final.json'))
        self.assertEqual(result['scientific_attempt_count'],1)
        self.assertEqual(result['scientific_validation'],'PASS')
        for name in ('STRAIGHT','LEFT','RIGHT'):
            episode=result['episodes'][name]
            self.assertEqual(episode['status'],'GOAL_REACHED')
            self.assertLessEqual(episode['final_distance_m'],.5)
            self.assertLessEqual(episode['time_to_goal_sim_s'],40)
            self.assertEqual(episode['zero_xy_bug_occurrences'],0)
        self.assertEqual(result['decision'],'NO_OBSTACLE_GOAL_REACHING_PASS')

    def test_all_raw_evidence_when_available(self):
        from research.validate_isaac_final_evidence import validate
        run=ROOT/'outputs/edge-final/scientific-001'
        if not run.exists():
            self.skipTest('Ignored raw runtime evidence not in this checkout')
        result=validate(run,ROOT/'research/frozen-edge-isaac-no-obstacle-final.json')
        self.assertEqual(result['decision'],'NO_OBSTACLE_GOAL_REACHING_PASS')


if __name__=='__main__':
    unittest.main()
