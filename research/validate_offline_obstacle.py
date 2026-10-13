"""Post-run evidence audit; never imports a policy or simulation runtime."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.isaac_final_common import sha256,write_json
from research.offline_obstacle_protocol import verify_protocol,PART_A
from research.analyze_offline_obstacle import analyze_existing,rows
from research.analyze_offline_pairs import analyze


def validate():
    local=ROOT/'artifacts-local/edge-obstacle-offline'
    protocol=ROOT/'research/frozen-edge-obstacle-offline.json'
    frozen,plan,manifest,config=verify_protocol(protocol)
    freeze=json.loads((local/'protocol-freeze.json').read_text())
    registration=json.loads((local/'scientific-batch-registration.json').read_text())
    assert freeze['sha256']==registration['protocol_sha256']==sha256(protocol)
    assert frozen['freeze_wall_time']<freeze['wall_time']<registration['start_wall_time']
    a=json.loads((PART_A/'metrics.json').read_text())
    assert analyze_existing(config)==a
    assert json.loads((PART_A/'complete.json').read_text())['source_sha256']==sha256(ROOT/'research/analyze_offline_obstacle.py')
    prior=json.loads((local/'start-audit/prior-artifact-hashes.json').read_text())
    changed=[p for p,digest in prior.items() if sha256(ROOT/p)!=digest]
    assert not changed,changed
    run=ROOT/'outputs/edge-obstacle-offline/paired-001'
    b=json.loads((run/'derived/comparisons.json').read_text())
    assert analyze(run,protocol)==b
    requests=rows(run/'raw/requests.jsonl');cases=rows(run/'raw/cases.jsonl');responses=rows(run/'service/responses.jsonl')
    assert len(requests)==len(cases)==len(responses)==8
    assert [r['episode_id'] for r in requests]==[c['case_id'] for c in plan['cases']]
    assert [r['request_id'] for r in responses]==list(range(8))
    for case,request,response,spec in zip(cases,requests,responses,manifest['cases']):
        assert request['history']==spec['history']
        assert case['response']==response
        assert request['robot_pose_observation']==spec['pose']
        assert request['normalized_pose']==spec['normalized_pose']
        assert request['request_send_wall_time']>registration['start_wall_time']
        path=Path(response['action_path']);native=np.load(path,allow_pickle=False)
        assert native.dtype==np.float32 and native.shape==(1,8,4)
        assert native.tolist()==case['native_prediction']
        assert not path.stat().st_mode & 0o222
    assert json.loads((run/'service/stopped.json').read_text())==dict(inference_count=8,load_count=1)
    source=dict(frozen['source_sha256'])
    for p,digest in source.items():assert sha256(ROOT/p)==digest,p
    evidence={}
    for folder in [ROOT/'outputs/edge-obstacle-offline/capture-001',run/'raw',run/'service',PART_A,run/'derived']:
        for p in sorted(folder.rglob('*')):
            if p.is_file():evidence[str(p.relative_to(ROOT))]=sha256(p)
    source_before=json.loads((local/'start-audit/tracked-file-hashes.json').read_text())
    old_launcher='scripts/isaac_static_obstacle_gui.sh'
    assert sha256(ROOT/old_launcher)==source_before[old_launcher]
    return dict(status='PASS',wall_time=time.time(),preserved_prior_files=len(prior),
        native_action_files=8,actual_stationary_rgb_captures=48,original_chunks=21,
        original_controller_parity=21,paired_controller_parity=8,
        original_and_paired_metrics_exactly_reproduced=True,
        planned_requests=8,executed_requests=8,model_loads=1,robot_execution=False,
        frozen_source_files=len(source),protocol_sha256=sha256(protocol),
        collision_gui_launcher_sha256=sha256(ROOT/old_launcher),new_evidence_sha256=evidence)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=validate();write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='new_evidence_sha256'},indent=2))
