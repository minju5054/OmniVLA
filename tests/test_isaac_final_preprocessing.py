"""CPU exact preprocessing parity; no checkpoint/model construction."""
import unittest
import numpy as np
import torch
from PIL import Image
from research.edge_isaac_service import prepare, official
from research.edge_contracts import initialize_sample_globals, prepare_function


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        initialize_sample_globals(official)
        official.device = torch.device('cpu')
        official.pose_goal, official.satellite, official.image_goal, official.lan_prompt = True, False, False, False
        official.goal_image_PIL = Image.new('RGB', (96,96))

    def test_exact_official_tensor_parity_with_equivalent_inputs(self):
        instance = official.Inference('.', 'xxxx', official.goal_utm,
                                      official.goal_compass, official.goal_image_PIL)
        expected = prepare_function(official)(instance)
        pixels = np.asarray(Image.open('inference/current_img.jpg').convert('RGB'))
        actual = prepare([pixels]*6, expected['goal_pose_torch'][0].tolist())
        for key in expected:
            with self.subTest(key=key):
                self.assertTrue(torch.equal(actual[key],expected[key]), key)

    def test_six_distinct_frames_keep_order_and_latest_large_image(self):
        frames = [np.full((480,640,3),i*40,dtype=np.uint8) for i in range(6)]
        actual = prepare(frames, [50,20,1,0])
        for i, frame in enumerate(frames):
            expected = official.transform_images_PIL_mask(Image.fromarray(frame).resize((96,96)),
                                                          official.mask_360_pil_96)
            self.assertTrue(torch.equal(actual['obs_images'][:,i*3:i*3+3],expected))
        large = official.transform_images_PIL_mask(Image.fromarray(frames[-1]).resize((224,224)),
                                                   official.mask_360_pil_224)
        self.assertTrue(torch.equal(actual['cur_large_img'],large))
        self.assertTrue(np.all(official.mask_360_pil_96==1))
        self.assertTrue(np.all(official.mask_360_pil_224==1))


if __name__ == '__main__':
    unittest.main()
