"""Analytic and randomized model-free evaluation/protocol regression tests."""
import copy
import json
import math
import random
import unittest
from research.isaac_final_common import ROOT
from research.static_obstacle_geometry import Rectangle, classify
from research.static_obstacle_protocol import validate_scenario, verify_static


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rect = Rectangle(2.2,2.8,-.5,.5)

    def test_geometry_from_dimensions(self):
        self.assertEqual(self.rect, Rectangle.from_config(dict(center=[2.5,0,.5], dimensions=[.6,1.,1.])))

    def test_face_corner_inside_and_signed_clearance(self):
        self.assertAlmostEqual(self.rect.clearance([0,0],.2), 2.)
        self.assertAlmostEqual(self.rect.clearance([1.9,0],.2), .1)
        self.assertAlmostEqual(self.rect.clearance([2.2-.3,.5+.4],.2), .3)
        self.assertEqual(self.rect.clearance([2.5,0],.2), -.2)

    def test_tunneling_and_first_contact(self):
        self.assertGreater(self.rect.clearance([0,0],.2),0)
        self.assertGreater(self.rect.clearance([5,0],.2),0)
        sweep=self.rect.swept([0,0],[5,0],.2)
        self.assertTrue(sweep['collision'])
        self.assertAlmostEqual(sweep['minimum_clearance'],-.2)
        self.assertAlmostEqual(sweep['first_contact_fraction'],.4)

    def test_stationary_and_starting_contact(self):
        for point, collision in [([0,0],False),([2.5,0],True)]:
            result=self.rect.swept(point,point,.2)
            self.assertEqual(result['collision'],collision)
            self.assertEqual(result['first_contact_fraction'],0. if collision else None)

    def test_rounded_corner_not_inflated_rectangle(self):
        r=Rectangle(0,1,0,1)
        # In the square-expanded AABB but outside the circular corner offset.
        result=r.swept([-.19,-.19],[-.18,-.18],.2)
        self.assertFalse(result['collision'])
        self.assertAlmostEqual(result['minimum_clearance'],math.hypot(.18,.18)-.2)

    def test_exact_tangent_and_safe_parallel_segment(self):
        r=Rectangle(0,1,0,1)
        tangent=r.swept([-1,1.25],[2,1.25],.25)
        self.assertTrue(tangent['collision'])
        self.assertEqual(tangent['minimum_clearance'],0.)
        self.assertAlmostEqual(tangent['first_contact_fraction'],1/3,places=7)
        self.assertAlmostEqual(r.swept([-1,1.5],[2,1.5],.25)['minimum_clearance'],.25)

    def test_corner_crossing_with_safe_endpoints(self):
        r=Rectangle(0,1,0,1)
        result=r.swept([-.5,.2],[.2,-.5],.25)
        self.assertTrue(result['collision'])
        self.assertGreater(r.clearance([-.5,.2],.25),0)
        self.assertGreater(r.clearance([.2,-.5],.25),0)

    def test_sweep_matches_dense_independent_distance_oracle(self):
        rng=random.Random(77)
        for _ in range(100):
            a=[rng.uniform(-1,5),rng.uniform(-2,2)]
            b=[rng.uniform(-1,5),rng.uniform(-2,2)]
            exact=self.rect.swept(a,b,.2)
            def oracle(t):
                x,y=[a[i]+t*(b[i]-a[i]) for i in range(2)]
                closest=[min(max(x,2.2),2.8),min(max(y,-.5),.5)]
                return math.dist([x,y],closest)-.2
            sampled=min(oracle(i/1000) for i in range(1001))
            self.assertLessEqual(exact['minimum_clearance'],sampled+1e-12)
            self.assertLessEqual(sampled-exact['minimum_clearance'],math.dist(a,b)/1000+1e-12)
            self.assertAlmostEqual(exact['minimum_clearance'],self.rect.swept(b,a,.2)['minimum_clearance'])
            if exact['first_contact_fraction'] not in (None,0.):
                self.assertAlmostEqual(oracle(exact['first_contact_fraction']),0.,places=10)

    def test_invalid_geometry_and_nonfinite(self):
        with self.assertRaises(ValueError): Rectangle(1,0,0,1)
        with self.assertRaises(ValueError): self.rect.swept([math.nan,0],[0,0],.2)
        with self.assertRaises(ValueError): self.rect.clearance([0,0],-.2)


class ClassificationTests(unittest.TestCase):
    config=dict(timeout_after_bootstrap_sim_s=40.,goal_radius_m=.5,evaluation=dict(safety_margin_m=.1))

    def test_outcomes_and_boundary_priority(self):
        for distance, elapsed, clearance, expected in [
            (.5,40.,.1,'SAFE_GOAL_REACHED'),(.5,39.,.099,'GOAL_REACHED_LOW_CLEARANCE'),
            (.2,10.,0.,'COLLISION'),(3.,10.,-.001,'COLLISION'),
            (3.,40.,.2,'MODEL_NAVIGATION_FAILURE'),(3.,10.,.2,None),
            (3.,40.,.05,'MODEL_NAVIGATION_FAILURE')]:
            self.assertEqual(classify(distance,elapsed,clearance,self.config),expected)

    def test_execution_failures_are_distinct(self):
        for failure in ('IMPLEMENTATION_FAILURE','UPSTREAM_CONTROLLER_BUG'):
            self.assertEqual(classify(2,10,.2,self.config,failure),failure)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        path=ROOT/'research/frozen-edge-static-obstacle.json'
        if not path.exists(): path=ROOT/'artifacts-local/static-obstacle/proposed-config.json'
        if not path.exists(): self.skipTest('Static protocol not available')
        self.config=json.loads(path.read_text())

    def test_baseline_preserved(self):
        validate_scenario(self.config)

    def test_reject_changed_camera_controller_geometry(self):
        for field, key in [('camera','width'),('controller','maxv'),('static_obstacle','color')]:
            changed=copy.deepcopy(self.config)
            changed[field][key]=42
            with self.assertRaises(ValueError): validate_scenario(changed)

    def test_unfrozen_or_tampered_protocol_rejected(self):
        changed=copy.deepcopy(self.config)
        changed.pop('static_gates',None)
        with self.assertRaises(ValueError): verify_static(changed)
        changed=copy.deepcopy(self.config)
        changed['implementation_sha256']['research/isaac_scene.py']='wrong'
        with self.assertRaises(ValueError): verify_static(changed)


if __name__ == '__main__':
    unittest.main()
