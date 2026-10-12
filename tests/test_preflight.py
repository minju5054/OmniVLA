"""Source/evidence contracts; these tests do not claim navigation performance."""
import ast
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from research.contracts import (ROOT, SOURCE, canonical_json, checkpoint_identity,
    evidence_directory, immutable_action, metric_action, official_attention_mask,
    official_controller, official_goal, official_modality, parse_remote, sha256, timed_cuda, write_json)


class ProvenanceTests(unittest.TestCase):
    def test_audited_source_is_unchanged(self):
        manifest = json.loads((ROOT / 'results/source-provenance.json').read_text())
        self.assertEqual(manifest['upstream_sha'], '5182600cb4a9ee07684e17cdd2a6cbafc56b8a68')
        for name, expected in manifest['sha256'].items():
            self.assertEqual(sha256(ROOT / name), expected, name)

    def test_checkpoint_manifest_is_full_and_complete(self):
        manifest = json.loads((ROOT / 'results/checkpoint-manifest.json').read_text())
        self.assertEqual(manifest['repo'], 'NHirose/omnivla-original')
        self.assertEqual(manifest['revision'], 'e36a84d4923c041149d441f93f3bdb7092bb5f07')
        shards = [f for f in manifest['files'] if f['name'].endswith('.safetensors')]
        self.assertEqual([f['name'] for f in shards], [f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5)])
        self.assertEqual(sum(f['bytes'] for f in shards), 15082600832)
        for item in manifest['files']:
            self.assertRegex(item['sha256'], r'^[0-9a-f]{64}$')

    def test_remote_parser(self):
        for value in ('https://github.com/NHirose/OmniVLA.git', 'git@github.com:NHirose/OmniVLA.git'):
            self.assertEqual(parse_remote(value), 'NHirose/OmniVLA')
        for bad in ('https://example.org/NHirose/OmniVLA.git', 'https://github.com/a/b/extra'):
            with self.assertRaises(ValueError):
                parse_remote(bad)

    def test_checkpoint_identity_rejects_other_model(self):
        config = {'model_type': 'openvla', 'hf_llm_id': 'meta-llama/Llama-2-7b-hf',
                  'vision_backbone_id': 'dinosiglip-vit-so-224px'}
        tokenizer = {'tokenizer_class': 'LlamaTokenizer'}
        self.assertEqual(checkpoint_identity(config, tokenizer), 'NHirose/omnivla-original')
        config['hf_llm_id'] = 'other-model'
        with self.assertRaises(ValueError):
            checkpoint_identity(config, tokenizer)

    def test_no_model_or_runtime_files_tracked(self):
        files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        forbidden = ('models-local/', 'artifacts-local/', 'outputs/', '.venv')
        for name in filter(None, files):
            self.assertFalse(name.startswith(forbidden), name)
            self.assertNotIn(Path(name).suffix, {'.safetensors', '.pt', '.pth'}, name)
            self.assertLess((ROOT / name).stat().st_size, 20 * 1024**2, name)

    def test_local_storage_ignored(self):
        for name in ('models-local/example.safetensors', 'artifacts-local/envs/x', 'outputs/run/raw/actions.pt'):
            result = subprocess.run(['git', 'check-ignore', name], cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode, 0, name)


class EvidenceTests(unittest.TestCase):
    def test_contract_and_nonfinite_rejection(self):
        self.assertEqual(immutable_action(np.zeros((1, 8, 4))).shape, (1, 8, 4))
        for value in (np.zeros((8, 4)), np.zeros((1, 7, 4)), np.full((1, 8, 4), np.nan),
                      np.full((1, 8, 4), np.inf), np.zeros((1, 8, 4), dtype=int)):
            with self.assertRaises(ValueError):
                immutable_action(value)

    def test_raw_immutability_and_metric_copy(self):
        original = np.ones((1, 8, 4), dtype=np.float32)
        raw = immutable_action(original)
        original[:] = 9
        with self.assertRaises(ValueError):
            raw[0, 0, 0] = 5
        with self.assertRaises(ValueError):
            raw.setflags(write=True)
        derived = metric_action(raw)
        np.testing.assert_array_equal(raw, np.ones_like(raw))
        np.testing.assert_allclose(derived[..., :2], 0.1)
        np.testing.assert_array_equal(derived[..., 2:], 1.0)

    def test_refuse_evidence_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = evidence_directory(Path(tmp) / 'run')
            write_json(path / 'raw/a.json', {'a': 1})
            with self.assertRaises(FileExistsError):
                evidence_directory(path)
            with self.assertRaises(FileExistsError):
                write_json(path / 'raw/a.json', {'a': 2})
            self.assertEqual(json.loads((path / 'raw/a.json').read_text()), {'a': 1})

    def test_deterministic_json(self):
        self.assertEqual(canonical_json({'b': 1, 'a': 2}), canonical_json({'a': 2, 'b': 1}))
        with self.assertRaises(ValueError):
            canonical_json({'a': math.nan})

    def test_cuda_sync_brackets_timing(self):
        events = []
        class CUDA:
            def synchronize(self):
                events.append('sync')
        times = iter((1.0, 1.25))
        def clock():
            events.append('clock')
            return next(times)
        def call():
            events.append('forward')
            return 'output'
        self.assertEqual(timed_cuda(call, CUDA(), clock), ('output', 0.25))
        self.assertEqual(events, ['sync', 'clock', 'forward', 'sync', 'clock'])


class SourceContracts(unittest.TestCase):
    def test_constants(self):
        tree = ast.parse((ROOT / 'prismatic/vla/constants.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'OMNIVLA_CONSTANTS')
        values = {ast.literal_eval(k): ast.literal_eval(v) for k, v in zip(node.value.keys, node.value.values)
                  if isinstance(v, ast.Constant)}
        self.assertEqual(values, {'NUM_ACTIONS_CHUNK': 8, 'ACTION_DIM': 4, 'POSE_DIM': 4})

    def test_front_left_right_at_compass_north(self):
        goals = json.loads((ROOT / 'research/frozen-goals.json').read_text())
        for case in goals['cases']:
            x, y = case['local_metric_goal']
            np.testing.assert_allclose(official_goal(-y, x), case['normalized_pose'], atol=1e-12)

    def test_compass_east_west_and_radius_clamp(self):
        np.testing.assert_allclose(official_goal(3, 2, 90, 90), [30, 20, 1, 0], atol=1e-12)
        np.testing.assert_allclose(official_goal(-3, -2, 270, 270), [30, 20, 1, 0], atol=1e-12)
        np.testing.assert_allclose(official_goal(0, 60), [300, 0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(official_goal(0, 3, 0, 90)[2:], [0, -1], atol=1e-12)

    def test_pose_and_default_modality(self):
        self.assertEqual(official_modality(pose_goal=True), 4)
        self.assertEqual(official_modality(image_goal=True), 6)
        self.assertEqual(official_modality(lan_prompt=True), 7)
        with self.assertRaises(ValueError):
            official_modality()

    def test_actual_source_mask_on_cpu(self):
        import torch
        mask = official_attention_mask(torch, 4)[0]
        self.assertTrue(mask[:257].all())  # BOS + current image
        self.assertFalse(mask[257:513].any())  # goal image
        self.assertTrue(mask[513])  # pose
        self.assertTrue(mask[514:].all())  # placeholder text/action tokens
        image_mask = official_attention_mask(torch, 6)[0]
        self.assertTrue(image_mask[257:513].all())
        self.assertFalse(image_mask[513])

    def test_controller_straight_left_right(self):
        self.assertEqual(official_controller([0.1, 0, 1, 0])['final_vw'], [0.3, 0.0])
        for dy in (0.1, -0.1):
            values = official_controller([0.1, dy, 1, 0])
            np.testing.assert_allclose(values['pre_clip_vw'], [0.3, math.copysign(3 * math.pi / 4, dy)], atol=1e-15)
            np.testing.assert_allclose(values['clipped_vw'], [0.3, math.copysign(1.0, dy)], atol=1e-15)
            np.testing.assert_allclose(values['final_vw'], [0.09, math.copysign(0.3, dy)], atol=1e-15)

    def test_controller_zero_x_and_limits(self):
        for dy in (0.1, -0.1):
            self.assertEqual(official_controller([0, dy, 1, 0])['final_vw'], [0, math.copysign(0.3, dy)])
        self.assertEqual(official_controller([1, 0, 1, 0])['final_vw'], [0.3, 0.0])
        self.assertEqual(official_controller([-1, 0, 1, 0])['final_vw'], [0.0, 0.0])
        v, w = official_controller([0.5, 0.01, 1, 0])['final_vw']
        self.assertLessEqual(v, 0.3)
        self.assertLessEqual(abs(w), 0.3)

    def test_zero_xy_heading_preserves_upstream_failure(self):
        with self.assertRaisesRegex(NameError, 'clip_angle'):
            official_controller([0, 0, 0, 1])


if __name__ == '__main__':
    unittest.main()
