"""Prevent the NumPy-2 viewer from silently changing NumPy-1 scientific results."""
import unittest
from research.offline_obstacle_gui import validate_display_values


class OfflineViewerTests(unittest.TestCase):
    def test_display_rounding_preserves_saved_data(self):
        saved={'command':[.3,.12],'classification':'HORIZON_INSUFFICIENT','executed':False}
        actual={'command':[.30000001192092896,.12],'classification':'HORIZON_INSUFFICIENT','executed':False}
        self.assertLess(validate_display_values(actual,saved),5e-7)
        self.assertEqual(saved['command'][0],.3)

    def test_real_change_and_identity_mismatch_rejected(self):
        for a,b in [([.3001],[.3]),([.3,.2],[.3]),(True,False),('PREDICTED_SAFE_PASSAGE','HORIZON_INSUFFICIENT'),
                    ({'x':.3},{'y':.3}),(float('nan'),.3)]:
            with self.subTest(actual=a),self.assertRaises(AssertionError):validate_display_values(a,b)


if __name__=='__main__':unittest.main()
