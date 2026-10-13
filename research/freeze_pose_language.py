"""Freeze both scientific phases only after model-free and dummy-IPC gates."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import sha256,write_json
from research.pose_language_contracts import PLAN,PROTOCOL,FLAGS,MASK,source_masks,verify_protocol


def main():
    import torch
    from research.offline_obstacle_protocol import verify_protocol as verify_old
    from research.edge_contracts import modality
    local=ROOT/'artifacts-local/pose-language-obstacle'
    assert not (ROOT/'outputs/pose-language-obstacle/offline-registration.json').exists()
    assert not (ROOT/'outputs/pose-language-obstacle/navigation-registration.json').exists()
    prior=json.loads((local/'start-audit/prior-artifact-hashes.json').read_text())
    assert all(sha256(ROOT/p)==digest for p,digest in prior.items())
    _,_,manifest,config=verify_old(ROOT/'research/frozen-edge-obstacle-offline.json')
    tests=local/'pre-freeze-tests-001.log'
    assert '\nOK\n' in tests.read_text() and 'Ran 135 tests' in tests.read_text()
    infra_path=ROOT/'outputs/pose-language-obstacle/infrastructure-001/outcome.json'
    infra=json.loads(infra_path.read_text())
    assert infra['status']=='PASS' and infra['service_stopped']['load_count']==0
    assert infra['service_stopped']['inference_count']==0
    masks=source_masks(torch)
    gate=dict(status='PASS',modality_selection=modality(**FLAGS),flags=FLAGS,
        decoder_mask=masks.all_masks[8].tolist(),pool_weights=masks.avg_pool_mask[8].tolist(),
        token_order=['RGB0','RGB1','RGB2','RGB3','RGB4','RGB5','pose','map','goal_image','language'],
        previous_rgb_captures=len({f['path'] for c in manifest['cases'] for f in c['history']}),
        preserved_prior_files=len(prior),model_free_tests=135,infrastructure_model_loads=0,
        infrastructure_inferences=0,checkpoint_sha256=config['checkpoint']['sha256'])
    write_json(local/'pre-freeze-gates.json',gate)
    source_paths=[*ROOT.glob('research/pose_language*.py'),ROOT/'research/run_pose_language.py',
        ROOT/'research/register_pose_language_plan.py',Path(__file__),ROOT/'tests/test_pose_language.py']
    previous=json.loads((ROOT/'research/frozen-edge-obstacle-offline.json').read_text())
    evidence=[PLAN,ROOT/'research/frozen-edge-static-obstacle.json',ROOT/'research/frozen-edge-obstacle-offline.json',
        ROOT/'artifacts-local/edge-obstacle-offline/input-manifest.json',tests,infra_path,
        local/'pre-freeze-gates.json',local/'start-audit/prior-artifact-hashes.json']
    frozen=dict(status='FROZEN_BEFORE_ANY_INFERENCE',protocol='POSE_LANGUAGE_OBSTACLE_V1',
        starting_sha='eeb75f85cbcf316f15856388d0c3831eae4ef87d',freeze_wall_time=time.time(),
        plan=str(PLAN.relative_to(ROOT)),offline_requests=32,navigation_primary_trials=4,
        source_sha256={**previous['source_sha256'],**{str(p.relative_to(ROOT)):sha256(p) for p in source_paths}},
        evidence_sha256={str(p.relative_to(ROOT)):sha256(p) for p in evidence},
        gates=gate,scientific_retries=False,post_outcome_tuning=False,
        instrumentation='Official forward unchanged; hooks observe actual text embedding/model modality/decoder masks. Measured forward/model timings include their relevant validation overhead; no artificial delay.')
    write_json(PROTOCOL,frozen);verify_protocol(PROTOCOL)
    write_json(local/'protocol-freeze.json',dict(path=str(PROTOCOL.relative_to(ROOT)),sha256=sha256(PROTOCOL),wall_time=time.time()))
    PLAN.chmod(0o444);PROTOCOL.chmod(0o444)
    print('FROZEN '+sha256(PROTOCOL))


if __name__=='__main__':main()
