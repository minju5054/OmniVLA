"""Protocol invariants and freeze checks, with no model/simulator imports."""
import json
from research.isaac_final_common import ROOT, sha256, verify_frozen

BASELINE = ROOT/'research/frozen-edge-isaac-no-obstacle-final.json'
PRESERVED = ('camera', 'scene', 'start', 'physics_hz', 'observation_hz', 'history',
             'goal_radius_m', 'timeout_after_bootstrap_sim_s', 'controller',
             'modality', 'scheduling', 'checkpoint', 'seed', 'seed_policy')
STATIC_GATES = {'scene_camera_history', 'regression'}


def validate_scenario(config):
    baseline = json.loads(BASELINE.read_text())
    verify_frozen(baseline)
    for field in PRESERVED:
        if config[field] != baseline[field]:
            raise ValueError(f'Baseline field changed: {field}')
    if config['goals'] != [dict(name='STATIC_CENTER', pose=[5.,0.,0.])]:
        raise ValueError('Expected one central-obstacle straight goal')
    obstacle = config['static_obstacle']
    if obstacle != dict(path='/World/StaticObstacle', center=[2.5,0.,0.5],
                        dimensions=[0.6,1.,1.], color=[0.8,0.08,0.08], opacity=1., yaw_rad=0.):
        raise ValueError('Unexpected obstacle geometry/appearance')
    if config['evaluation']['agent_radius_m'] != .2 or config['evaluation']['safety_margin_m'] != .1:
        raise ValueError('Unexpected footprint or safety margin')


def verify_static(config, config_path=None):
    validate_scenario(config)
    verify_frozen(config)
    if config.get('protocol') != 'STATIC_OBSTACLE_SINGLE_PRIMARY_V1':
        raise ValueError('Static protocol not frozen')
    if set(config.get('static_gates', {})) != STATIC_GATES:
        raise ValueError('Static validation gates missing')
    for group in ('infrastructure_gates', 'static_gates'):
        for gate in config[group].values():
            if gate['status'] != 'PASS' or sha256(ROOT/gate['evidence']) != gate['sha256']:
                raise ValueError('Validation evidence changed or gate failed')
    if config_path is not None:
        freeze = json.loads((ROOT/'artifacts-local/static-obstacle/protocol-freeze.json').read_text())
        if sha256(config_path) != freeze['sha256']:
            raise ValueError('Static configuration hash changed')
