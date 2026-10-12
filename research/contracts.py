"""Small evidence helpers; official formulas are executed from unchanged source.

No model replacements, navigation framework, or controller fixes live here.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = '5182600cb4a9ee07684e17cdd2a6cbafc56b8a68'
HF_REVISION = 'e36a84d4923c041149d441f93f3bdb7092bb5f07'
SOURCE = ROOT / 'inference/run_omnivla.py'


def canonical_json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(canonical_json(value))


def evidence_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    (path / 'raw').mkdir()
    (path / 'derived').mkdir()
    return path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def parse_remote(url):
    match = re.fullmatch(r'(?:https://github\.com/|git@github\.com:)([\w.-]+/[\w.-]+?)(?:\.git)?/?', url.strip())
    if not match:
        raise ValueError('Not an unambiguous GitHub remote')
    return match.group(1)


def checkpoint_identity(config, tokenizer):
    if (config.get('model_type') != 'openvla'
            or config.get('hf_llm_id') != 'meta-llama/Llama-2-7b-hf'
            or config.get('vision_backbone_id') != 'dinosiglip-vit-so-224px'
            or tokenizer.get('tokenizer_class') != 'LlamaTokenizer'):
        raise ValueError('Unexpected FULL checkpoint/config identity')
    return 'NHirose/omnivla-original'


def immutable_action(value):
    value = np.asarray(value)
    if value.shape != (1, 8, 4):
        raise ValueError(f'Expected (1, 8, 4), received {value.shape}')
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError('Actions must be finite floating point values')
    # A bytes-backed array cannot be made writeable again with setflags.
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def metric_action(raw):
    result = immutable_action(raw).copy()
    result[..., :2] *= 0.1
    return result


def _method(path, cls, name):
    tree = ast.parse(Path(path).read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _exec_nodes(nodes, scope, filename=SOURCE):
    code = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(code, str(filename), 'exec'), scope)


def official_controller(waypoint_metric):
    """Execute unchanged controller AST; the upstream missing clip_angle still fails."""
    method = _method(SOURCE, 'Inference', 'run_omnivla')
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'EPS')
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == 'save_robot_behavior')
    nodes = method.body[start:end]
    first_clip = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                      and n.value.func.attr == 'clip')
    scope = dict(np=np, zip=zip)
    scope.update(zip(('dx', 'dy', 'hx', 'hy'), map(float, waypoint_metric)))
    _exec_nodes(nodes[:first_clip], scope)
    pre = [float(scope['linear_vel_value']), float(scope['angular_vel_value'])]
    _exec_nodes(nodes[first_clip:], scope)
    return {'pre_clip_vw': pre,
            'clipped_vw': [float(scope['linear_vel_value']), float(scope['angular_vel_value'])],
            'final_vw': [float(scope['linear_vel_value_limit']), float(scope['angular_vel_value_limit'])]}


def official_goal(delta_east, delta_north, current_compass_deg=0.0, goal_compass_deg=0.0):
    """Execute the source GPS-delta -> normalized local-pose block (30 m clamp)."""
    scope = {'math': math, 'np': np}
    for name in ('calculate_relative_position', 'rotate_to_local_frame'):
        node = _method(SOURCE, 'Inference', name)
        node.decorator_list = []
        _exec_nodes([node], scope)
    scope.update(self=SimpleNamespace(goal_utm=(delta_east, delta_north),
                                     goal_compass=-math.radians(goal_compass_deg),
                                     calculate_relative_position=scope['calculate_relative_position'],
                                     rotate_to_local_frame=scope['rotate_to_local_frame']),
                 cur_utm=(0.0, 0.0), cur_compass=-math.radians(current_compass_deg),
                 thres_dist=30.0, metric_waypoint_spacing=0.1)
    nodes = _method(SOURCE, 'Inference', 'run_omnivla').body
    start = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Tuple) and ast.unparse(n.targets[0]) == '(delta_x, delta_y)')
    end = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'goal_pose_loc_norm')
    _exec_nodes(nodes[start:end + 1], scope)
    return scope['goal_pose_loc_norm']


def official_modality(*, pose_goal=False, satellite=False, image_goal=False, lan_prompt=False):
    node = next(n for n in _method(SOURCE, 'Inference', 'run_forward_pass').body if isinstance(n, ast.If))
    scope = dict(pose_goal=pose_goal, satellite=satellite, image_goal=image_goal, lan_prompt=lan_prompt,
                 torch=SimpleNamespace(as_tensor=np.asarray, float32=np.float32))
    _exec_nodes([node], scope)
    if 'modality_id' not in scope:
        raise ValueError('Unsupported official modality combination')
    return int(scope['modality_id'][0])


def official_attention_mask(torch, modality_id):
    """Exercise the actual MMNv1 mask function with small CPU tensors, no weights."""
    path = ROOT / 'prismatic/extern/hf/modeling_prismatic.py'
    node = _method(path, 'PrismaticForConditionalGeneration_MMNv1', '_build_multimodal_attention_MMN')
    scope = {'torch': torch}
    _exec_nodes([node], scope, path)
    _, mask = scope[node.name](None, torch.zeros(1, 4, 2), torch.zeros(1, 513, 2),
                              torch.ones(1, 4, dtype=torch.bool), None, torch.tensor([modality_id]))
    return mask


def timed_cuda(fn, cuda, clock=time.perf_counter):
    cuda.synchronize()
    start = clock()
    result = fn()
    cuda.synchronize()
    return result, clock() - start
