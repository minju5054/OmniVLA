"""Source-derived Edge contracts; no model or controller repairs."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from research.contracts import ROOT, _method, _exec_nodes

SOURCE = ROOT / 'inference/run_omnivla_edge.py'
FULL_REPORT_SHA = 'bb9f0440281f2672104fdcebb08366cf16654b3ff3bafe4972b5ad6d8c747d03'


def controller(waypoint):
    nodes = _method(SOURCE, 'Inference', 'run_omnivla').body
    start = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'EPS')
    end = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call)
               and ast.unparse(n.value.func) == 'self.save_robot_behavior')
    nodes = nodes[start:end]
    clip = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'np.clip')
    # Preserve NumPy scalar dtype, including float32 division/atan rounding.
    scope = dict(np=np, **dict(zip(('dx', 'dy', 'hx', 'hy'), np.asarray(waypoint))))
    _exec_nodes(nodes[:clip], scope, SOURCE)
    pre = [float(scope['linear_vel_value']), float(scope['angular_vel_value'])]
    _exec_nodes(nodes[clip:], scope, SOURCE)
    return dict(pre_clip_vw=pre,
                clipped_vw=[float(scope['linear_vel_value']), float(scope['angular_vel_value'])],
                final_vw=[float(scope['linear_vel_value_limit']), float(scope['angular_vel_value_limit'])])


def modality(**flags):
    scope = dict(pose_goal=False, satellite=False, image_goal=False, lan_prompt=False)
    scope.update(flags)
    node = next(n for n in _method(SOURCE, 'Inference', 'run_forward_pass').body if isinstance(n, ast.If))
    _exec_nodes([node], scope, SOURCE)
    return scope['modality_id']


def goal(delta_east, delta_north, current_compass=0., goal_compass=0.):
    """Execute Edge's pre-overwrite GPS transform, returning its normalized pose."""
    scope = dict(math=math, np=np)
    for name in ('calculate_relative_position', 'rotate_to_local_frame'):
        n = _method(SOURCE, 'Inference', name)
        n.decorator_list = []
        _exec_nodes([n], scope, SOURCE)
    scope.update(self=SimpleNamespace(goal_utm=(delta_east, delta_north),
                 goal_compass=-math.radians(goal_compass),
                 calculate_relative_position=scope['calculate_relative_position'],
                 rotate_to_local_frame=scope['rotate_to_local_frame']),
                 cur_utm=(0., 0.), cur_compass=-math.radians(current_compass),
                 thres_dist=30., metric_waypoint_spacing=.1)
    nodes = _method(SOURCE, 'Inference', 'run_omnivla').body
    start = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == '(delta_x, delta_y)')
    end = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == 'goal_pose_torch')
    _exec_nodes(nodes[start:end], scope, SOURCE)
    # Evaluate the exact source np.array expression, before tensor/device wrapping.
    expr = next(n for n in ast.walk(nodes[end].value) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == 'np.array')
    return eval(compile(ast.Expression(expr), str(SOURCE), 'eval'), scope)


def initialize_sample_globals(module):
    """Execute official main configuration up to the checkpoint path selection."""
    main = ast.parse(SOURCE.read_text()).body[-1]
    end = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == 'MODEL_WEIGHTS_PATH')
    _exec_nodes(main.body[:end], vars(module), SOURCE)
    masks = [n for n in main.body if isinstance(n, ast.Assign)
             and ast.unparse(n.targets[0]) in ('mask_360_pil_96', 'mask_360_pil_224')]
    _exec_nodes(masks, vars(module), SOURCE)


def prepare_function(module):
    """Compile unchanged official preprocessing, stopping before forward."""
    node = copy.deepcopy(_method(SOURCE, 'Inference', 'run_omnivla'))
    end = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == '(actions, modality_id)')
    node.name = 'prepare_official_batch'
    node.body = node.body[:end] + [ast.Return(value=ast.Name(id='batch', ctx=ast.Load()))]
    _exec_nodes([node], vars(module), SOURCE)
    return module.prepare_official_batch
