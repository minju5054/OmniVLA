"""Model-free tests of the all-waypoint, horizon-aware diagnostic contract."""
import json
import math
import unittest
import numpy as np
from research.isaac_final_common import ROOT
from research.static_obstacle_geometry import Rectangle
from research.offline_obstacle_contracts import metric_world, spatial_polyline, evaluate_chunk, compare_pair, command_record
from research.isaac_final_contracts import action


class OfflineSpatialTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'research/frozen-edge-static-obstacle.json').read_text())
        self.rect=Rectangle.from_config(self.config['static_obstacle'])

    def test_metric_frame_no_second_cumsum_and_immutability(self):
        native=np.tile(np.array([10.,20.,0.,1.],np.float32),(1,8,1)); before=native.copy()
        local,world,heading=metric_world(native,[3,4,math.pi/2])
        np.testing.assert_allclose(local,np.tile([1,2],(8,1)))
        np.testing.assert_allclose(world,np.tile([1,5],(8,1)))
        np.testing.assert_allclose(heading,np.tile([-1,0],(8,1)),atol=1e-12)
        np.testing.assert_array_equal(native,before)

    def test_short_safe_prefix_is_insufficient(self):
        self.assertEqual(spatial_polyline([[0,0],[1,0]],self.rect)['classification'],'HORIZON_INSUFFICIENT')

    def test_safe_passage_must_span_expanded_range(self):
        route=[[0,0],[1.5,1.],[3.2,1.]]
        self.assertEqual(spatial_polyline(route,self.rect)['classification'],'PREDICTED_SAFE_PASSAGE')
        route[-1]=[3.,1.]
        self.assertEqual(spatial_polyline(route,self.rect)['classification'],'HORIZON_INSUFFICIENT')

    def test_colliding_segment_with_clear_endpoints(self):
        result=spatial_polyline([[0,0],[4,0]],self.rect)
        self.assertEqual(result['classification'],'PREDICTED_COLLISION')
        self.assertTrue(result['centerline_intersects_rectangle'])

    def test_low_clearance_even_without_full_passage(self):
        self.assertEqual(spatial_polyline([[0,0],[1.95,0]],self.rect)['classification'],'PREDICTED_LOW_CLEARANCE')

    def test_observation_to_first_waypoint_is_included(self):
        raw=np.tile(np.array([40,0,1,0],np.float32),(1,8,1))
        result=evaluate_chunk(raw,[0,0,0],self.config)
        self.assertTrue(all(v>0 for v in result['waypoint_clearance_m']))
        self.assertEqual(result['polyline']['classification'],'PREDICTED_COLLISION')

    def test_all_eight_waypoints_and_late_collision(self):
        raw=np.zeros((1,8,4),np.float32);raw[0,:,0]=np.arange(1,9);raw[:,:,2]=1
        raw[0,7,0]=25
        result=evaluate_chunk(raw,[0,0,0],self.config)
        self.assertEqual(len(result['world_xy']),8)
        self.assertGreater(result['selected_index4_clearance_m'],.1)
        self.assertEqual(result['prefix_classifications'][4],'HORIZON_INSUFFICIENT')
        self.assertEqual(result['polyline']['classification'],'PREDICTED_COLLISION')

    def test_controller_heading_ignored_for_nonzero_xy(self):
        a=command_record(np.array([10,1,1,0],np.float32))
        b=command_record(np.array([10,1,-1,0],np.float32))
        self.assertFalse(a['heading_used'])
        self.assertEqual(a['final_vw'],b['final_vw'])
        self.assertEqual(command_record([0,0,0,1])['error'],'UPSTREAM_CONTROLLER_BUG')

    def test_limiting_delta_and_curvature(self):
        c=command_record(np.array([10,1,1,0],np.float32))
        np.testing.assert_allclose(np.array(c['pre_clip_vw'])+c['total_limit_delta_vw'],c['final_vw'])
        self.assertAlmostEqual(c['curvature_w_over_v']['clipped'],c['curvature_w_over_v']['final'])

    def test_paired_threshold_rms_and_heading_wrap(self):
        absent=np.tile(np.array([10,0,1,0],np.float32),(1,8,1)); present=absent.copy()
        present[0,7,1]=2.
        pair=compare_pair(present,absent,[.2,0,0],self.config)
        self.assertTrue(pair['material_lateral_response'])
        self.assertAlmostEqual(pair['maximum_absolute_lateral_difference_m'],.2)
        self.assertAlmostEqual(pair['rms_position_difference_m'],.2/math.sqrt(8))
        present[:,:,1]=0
        a,b=math.radians(179),math.radians(-179)
        present[:,:,2:]=[math.cos(a),math.sin(a)];absent[:,:,2:]=[math.cos(b),math.sin(b)]
        pair=compare_pair(present,absent,[.2,0,0],self.config)
        self.assertAlmostEqual(pair['heading_angular_difference_rad'][0],math.radians(-2),places=6)

    def test_native_is_immutable_and_diagnostics_repeat_exactly(self):
        native=np.tile(np.array([6.299478054046631,-.1343299150466919,.9999861121177673,.0052617499604821205],np.float32),(1,8,1))
        frozen=action(native)
        with self.assertRaises(ValueError): frozen.setflags(write=True)
        first=evaluate_chunk(frozen,[0,0,0],self.config)
        self.assertEqual(first,evaluate_chunk(frozen,[0,0,0],self.config))
        # Exact original request-0 reference, including float32 in-place XY scale.
        self.assertEqual(first['hypothetical_commands'][4]['final_vw'],[.3,-.03837733827531337])


if __name__=='__main__': unittest.main()
