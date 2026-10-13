"""Freeze validated Part A, actual captures, CPU tests and source before inference."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import write_json,sha256
from research.offline_obstacle_protocol import PLAN,STATIC,PART_A,verify_protocol,validate_capture


def main():
    local=ROOT/'artifacts-local/edge-obstacle-offline'
    assert not (local/'scientific-batch-registration.json').exists()
    tests=local/'pre-freeze-tests.log'
    assert '\nOK\n' in tests.read_text() and 'Ran 124 tests' in tests.read_text()
    assert json.loads((local/'capture-001/exit.json').read_text())['returncode']==0
    manifest=local/'input-manifest.json'
    # Revalidate all PNG bytes; compare a fresh manifest without replacing evidence.
    check=validate_capture(ROOT/'outputs/edge-obstacle-offline/capture-001',local/'input-manifest-revalidation.json')
    assert check==json.loads(manifest.read_text())
    prior=json.loads((local/'start-audit/prior-artifact-hashes.json').read_text())
    assert all(sha256(ROOT/p)==digest for p,digest in prior.items())
    write_json(local/'pre-freeze-preservation.json',dict(status='PASS',files=len(prior),wall_time=time.time()))
    source=[
        'research/offline_obstacle_contracts.py','research/offline_obstacle_protocol.py',
        'research/offline_obstacle_capture.py','research/run_offline_obstacle_inference.py',
        'research/analyze_offline_obstacle.py','research/analyze_offline_pairs.py',
        'research/freeze_offline_obstacle.py','tests/test_offline_obstacle_contracts.py',
        'tests/test_offline_obstacle_protocol.py']
    config=json.loads(STATIC.read_text())
    files=[PLAN,STATIC,manifest,PART_A/'metrics.json',PART_A/'complete.json',tests,
        local/'capture-001/exit.json',local/'pre-freeze-preservation.json',
        ROOT/'outputs/edge-obstacle-offline/capture-001/result.json',
        ROOT/'results/edge-checkpoint-manifest.json']
    frozen=dict(status='FROZEN_BEFORE_INFERENCE',protocol='EDGE_OBSTACLE_OFFLINE_V1',
        freeze_wall_time=time.time(),starting_sha=json.loads(PLAN.read_text())['starting_sha'],
        input_manifest=str(manifest.relative_to(ROOT)),plan=str(PLAN.relative_to(ROOT)),
        files_sha256={str(p.relative_to(ROOT)):sha256(p) for p in files},
        source_sha256={**config['implementation_sha256'],**{p:sha256(ROOT/p) for p in source}},
        planned_inferences=8,model_loads=1,robot_execution=False,post_outcome_tuning=False,
        waypoint_timestamps=None,static_protocol_preserved=True,part_a_completed_before_part_b=True,
        cpu_tests=dict(count=124,status='PASS'),input_validation='PASS')
    path=ROOT/'research/frozen-edge-obstacle-offline.json'
    write_json(path,frozen)
    verify_protocol(path)
    write_json(local/'protocol-freeze.json',dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),wall_time=time.time()))
    for p in (ROOT/'outputs/edge-obstacle-offline/capture-001').rglob('*'):
        if p.is_file(): p.chmod(0o444)
    print('FROZEN '+sha256(path))


if __name__=='__main__':main()
