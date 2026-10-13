"""CPU-only launcher/evidence checks; never starts Isaac or loads a model."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'scripts/isaac6_python.sh'
START = '8ddf5c60d9107f8b5f114c68f0fe9777a6f697df'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LauncherTests(unittest.TestCase):
    def probe(self, version='6.0.1-test', args=()):
        with tempfile.TemporaryDirectory(prefix='omnivla-launcher-test-') as temp:
            root = Path(temp)
            if version is not None:
                (root / 'VERSION').write_text(version + '\n')
            launcher = root / 'python.sh'
            probe_code = (
                'import json,os,sys; print(json.dumps({"environment": '
                '{k: os.environ.get(k) for k in ["LD_LIBRARY_PATH", "PYTHONPATH", '
                '"PYTHONEXE", "PYTHONNOUSERSITE", "PYTHONUNBUFFERED"]}, '
                '"arguments":sys.argv[1:]}))')
            launcher.write_text('#!/bin/bash\nexec /usr/bin/python3 -c '
                                + shlex.quote(probe_code) + ' "$@"\n')
            launcher.chmod(0o755)
            env = dict(os.environ, OMNIVLA_ISAAC6_ROOT=temp,
                       LD_LIBRARY_PATH='/injected/cuda',
                       PYTHONPATH='/injected/python',
                       PYTHONEXE='/injected/python-executable',
                       PYTHONNOUSERSITE='0', PYTHONUNBUFFERED='0')
            return subprocess.run([str(WRAPPER), *args], env=env,
                                  capture_output=True, text=True)

    def test_shell_syntax(self):
        subprocess.run(['bash', '-n', str(WRAPPER)], check=True)

    def test_rejects_missing_or_wrong_version(self):
        for version in (None, '5.0.0', '6.0.0', '6.0.10', '6.0.1unexpected'):
            with self.subTest(version=version):
                p = self.probe(version)
                self.assertEqual(p.returncode, 2)
                self.assertEqual(p.stdout, '')  # fake launcher was never executed
                self.assertIn('Isaac', p.stderr)

    def test_accepts_exact_or_suffixed_version_and_forwards_arguments(self):
        for version in ('6.0.1', '6.0.1-rc.7+release.42383.32955d8d.gl'):
            with self.subTest(version=version):
                p = self.probe(version, ('--output', 'a path with spaces', ''))
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(json.loads(p.stdout)['arguments'],
                                 ['--output', 'a path with spaces', ''])

    def test_removes_ld_library_path(self):
        self.assertIsNone(json.loads(self.probe().stdout)['environment']['LD_LIBRARY_PATH'])

    def test_removes_pythonpath(self):
        self.assertIsNone(json.loads(self.probe().stdout)['environment']['PYTHONPATH'])

    def test_removes_pythonexe(self):
        self.assertIsNone(json.loads(self.probe().stdout)['environment']['PYTHONEXE'])

    def test_sets_no_user_site_and_unbuffered(self):
        env = json.loads(self.probe().stdout)['environment']
        self.assertEqual(env['PYTHONNOUSERSITE'], '1')
        self.assertEqual(env['PYTHONUNBUFFERED'], '1')


class RecoveryEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / 'results/edge-isaac-runtime-recovery.json').read_text())

    def test_starting_source(self):
        self.assertEqual(self.result['starting_sha'], START)

    def test_historical_reports_unchanged(self):
        for path, sha in self.result['historical_report_sha256'].items():
            with self.subTest(path=path):
                self.assertEqual(digest(ROOT / path), sha)
                original = subprocess.check_output(['git', 'show', f'{START}:{path}'], cwd=ROOT)
                self.assertEqual((ROOT / path).read_bytes(), original)

    def test_work_log_is_append_only(self):
        original = subprocess.check_output(['git', 'show', f'{START}:docs/WORK_LOG.md'], cwd=ROOT)
        self.assertTrue((ROOT / 'docs/WORK_LOG.md').read_bytes().startswith(original))

    def test_frozen_retry_semantics(self):
        for path, sha in self.result['frozen_sha256'].items():
            with self.subTest(path=path):
                self.assertEqual(digest(ROOT / path), sha)
                if path.startswith('research/'):
                    original = subprocess.check_output(['git', 'show', f'{START}:{path}'], cwd=ROOT)
                    self.assertEqual((ROOT / path).read_bytes(), original)

    def test_exactly_one_new_startup_result_is_accepted(self):
        self.assertEqual(self.result['retry_count'], 1)
        accepted = self.result['accepted_startup_results']
        self.assertEqual(len(accepted), 1)
        expected = 'outputs/edge-isaac-startup-isolated-001/result.json'
        self.assertEqual(accepted[0]['path'], expected)
        self.assertEqual(self.result['launch']['retry_count'], 1)
        # If ignored evidence is present, reject additional isolated run results.
        local = list((ROOT / 'outputs').glob('edge-isaac-startup-isolated-*/result.json'))
        if local:
            self.assertEqual([p.relative_to(ROOT).as_posix() for p in local], [expected])
            self.assertEqual(digest(local[0]), accepted[0]['sha256'])
            self.assertEqual(json.loads(local[0].read_text()), self.result['in_process_result'])

    def test_no_model_or_navigation_outputs_created(self):
        audit = self.result['output_audit']
        prefix = 'outputs/edge-isaac-startup-isolated-001/'
        self.assertEqual(set(audit['new_files']),
                         {prefix + x for x in ('result.json', 'startup-rgb.png', 'raw/startup.log')})
        self.assertFalse(audit['model_or_navigation_outputs_created'])
        self.assertEqual(audit['removed_files'], [])
        before = ROOT / 'artifacts-local/isaac-runtime-recovery/output-inventory-before.json'
        if before.exists():
            self.assertEqual(digest(before), audit['before_inventory_sha256'])
            previous = set(json.loads(before.read_text()))
            current = {p.relative_to(ROOT).as_posix()
                       for p in (ROOT / 'outputs').rglob('*') if p.is_file()}
            # Later authorized tasks may add outputs. Validate only this historical
            # recovery run against its recorded inventory and immutable hashes.
            recovery_files = {p for p in current - previous if p.startswith(prefix)}
            self.assertEqual(recovery_files, set(audit['new_files']))
            for path, sha in audit['new_files'].items():
                self.assertEqual(digest(ROOT / path), sha)
            self.assertEqual(previous - current, set())

    def test_diagnostic_classification_supported(self):
        r = self.result
        self.assertEqual(r['diagnostic_classification'], 'CONFIRMED_LIBRARY_PATH_COLLISION')
        inherited, clean = r['resolution']['inherited'], r['resolution']['clean']
        self.assertNotEqual(inherited['canonical_path'], clean['canonical_path'])
        self.assertEqual(inherited['ctypes_exit_code'], 1)
        self.assertEqual(clean['ctypes_exit_code'], 0)
        by_path = {item['canonical_path']: item for item in r['libraries']}
        self.assertFalse(by_path[inherited['canonical_path']]['exports_required_symbol'])
        self.assertTrue(by_path[clean['canonical_path']]['exports_required_symbol'])
        self.assertIn('undefined symbol: __nvJitLinkCreate_12_8', inherited['error'])
        self.assertTrue(clean['mapped_library_verified_in_proc_self_maps'])

    def test_overall_failure_preserves_in_process_success(self):
        r = self.result
        self.assertEqual(r['in_process_result']['status'], 'ISAAC_STARTUP_PASS')
        self.assertTrue(r['constructor_returned'])
        self.assertTrue(r['stage_opened'])
        self.assertTrue(r['camera_rendered'])
        self.assertTrue(r['camera']['finite'])
        self.assertTrue(r['camera']['nonempty'])
        self.assertEqual(r['camera']['shape'], [480, 640, 3])
        self.assertEqual(r['steps'], 60)
        self.assertTrue(r['clean_shutdown'])  # close returned before native exit
        self.assertFalse(r['process_clean_exit'])
        self.assertFalse(r['timeout_triggered'])
        self.assertFalse(r['cuda_symbol_error'])
        self.assertTrue(r['segmentation_fault'])
        self.assertEqual(r['launcher_exit_code'], 1)
        self.assertEqual(r['decision'], 'ISAAC_STARTUP_RECOVERY_FAIL')
        self.assertIn('Segmentation fault', r['earliest_material_error'])

    def test_retained_log_supports_exit_failure(self):
        path = ROOT / self.result['raw_log']['path']
        if not path.exists():
            self.skipTest('Ignored raw log unavailable in this checkout')
        self.assertEqual(digest(path), self.result['raw_log']['sha256'])
        log = path.read_text()
        self.assertNotIn('undefined symbol:', log)
        self.assertIn(self.result['earliest_material_error'], log)
        self.assertLess(log.index('"clean_shutdown": true'), log.index('Segmentation fault'))


if __name__ == '__main__':
    unittest.main()
