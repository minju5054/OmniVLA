"""Pure offline spatial diagnostics; never assign time to waypoint rows."""
import math
import numpy as np
from research.edge_contracts import controller
from research.isaac_final_contracts import action
from research.static_obstacle_geometry import Rectangle

LABELS = ('PREDICTED_COLLISION', 'PREDICTED_LOW_CLEARANCE',
          'PREDICTED_SAFE_PASSAGE', 'HORIZON_INSUFFICIENT')


def metric_world(native, pose):
    raw = action(native)
    local = raw[0, :, :2].astype(np.float64) * .1
    yaw = pose[2]
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    world = local @ rotation.T + np.asarray(pose[:2])
    heading = raw[0, :, 2:].astype(np.float64) @ rotation.T
    return local, world, heading


def spatial_polyline(points, rectangle, radius=.2, margin=.1):
    """Screen the complete prefix including observation center -> row zero.

    Safe passage requires the polyline to span the near and far world-X planes
    of the radius+margin-expanded obstacle (1.9 and 3.1 m for this scenario).
    Positive clearance on a shorter horizon is explicitly insufficient evidence.
    Interpolation is evaluation-only, not an executed/timestamped robot path.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('Need at least two finite XY vertices')
    segments = [rectangle.swept(a, b, radius) for a, b in zip(points, points[1:])]
    minimum = min(s['minimum_clearance'] for s in segments)
    near, far = rectangle.xmin-radius-margin, rectangle.xmax+radius+margin
    covers = bool(points[0, 0] <= near and points[:, 0].max() >= far)
    if minimum <= 0:
        label = LABELS[0]
    elif minimum < margin:
        label = LABELS[1]
    elif covers:
        label = LABELS[2]
    else:
        label = LABELS[3]
    return dict(classification=label, minimum_clearance_m=minimum,
                centerline_intersects_rectangle=minimum <= -radius,
                footprint_collision=minimum <= 0, safety_margin_violation=minimum < margin,
                spans_expanded_obstacle_x_range=covers, near_plane_x_m=near, far_plane_x_m=far,
                forward_world_x_max_m=float(points[:, 0].max()), segments=segments)


def command_record(native_row):
    metric = np.asarray(native_row, dtype=np.float32).copy()
    metric[:2] *= .1  # preserve official float32 scaling/rounding for controller parity
    record = dict(native_waypoint=np.asarray(native_row).tolist(), selected_metric=metric.tolist(),
                  heading_used=bool(abs(metric[0]) < 1e-8 and abs(metric[1]) < 1e-8))
    try:
        value = controller(metric)
    except NameError as exc:
        if 'clip_angle' not in str(exc): raise
        return dict(**record, error='UPSTREAM_CONTROLLER_BUG', detail=str(exc))
    pre, clipped, final = [np.asarray(value[k]) for k in ('pre_clip_vw','clipped_vw','final_vw')]
    record.update(**value, initial_clip_delta_vw=(clipped-pre).tolist(),
                  ratio_limit_delta_vw=(final-clipped).tolist(),
                  total_limit_delta_vw=(final-pre).tolist(),
                  curvature_w_over_v={name: float(vw[1]/vw[0]) if abs(vw[0]) > 1e-12 else None
                    for name, vw in [('pre_clip',pre),('clipped',clipped),('final',final)]})
    return record


def evaluate_chunk(native, pose, config):
    raw = action(native)
    local, world, heading = metric_world(raw, pose)
    rectangle = Rectangle.from_config(config['static_obstacle'])
    radius = config['evaluation']['agent_radius_m']
    margin = config['evaluation']['safety_margin_m']
    points = np.vstack([np.asarray(pose[:2]), world])
    polyline = spatial_polyline(points, rectangle, radius, margin)
    norms = np.linalg.norm(raw[0,:,2:], axis=1)
    local_angles = [float(math.atan2(y,x)) if n > 1e-12 else None
                    for (x,y),n in zip(raw[0,:,2:],norms)]
    world_angles = [float(math.atan2(y,x)) if n > 1e-12 else None
                    for (x,y),n in zip(heading,norms)]
    clearance = [rectangle.clearance(row, radius) for row in world]
    return dict(local_metric_xy=local.tolist(), world_xy=world.tolist(),
                local_heading_vectors=raw[0,:,2:].tolist(),world_heading_vectors=heading.tolist(),
                local_heading_rad=local_angles,world_heading_rad=world_angles,
                heading_norms=norms.tolist(), waypoint_clearance_m=clearance,
                polyline=polyline, prefix_classifications=[
                    spatial_polyline(points[:i+2],rectangle,radius,margin)['classification'] for i in range(8)],
                local_lateral_min_m=float(local[:,1].min()),local_lateral_max_m=float(local[:,1].max()),
                maximum_absolute_local_lateral_m=float(np.max(np.abs(local[:,1]))),
                world_lateral_min_m=float(world[:,1].min()),world_lateral_max_m=float(world[:,1].max()),
                endpoint_local_displacement=local[-1].tolist(),
                endpoint_world_displacement=(world[-1]-pose[:2]).tolist(),
                endpoint_world_xy=world[-1].tolist(),
                selected_index4_clearance_m=clearance[4],
                selected_index4_local_xy=local[4].tolist(),
                hypothetical_commands=[command_record(row) for row in raw[0]])


def compare_pair(present, absent, pose, config, threshold=.2):
    pa = evaluate_chunk(present,pose,config)
    ab = evaluate_chunk(absent,pose,config)
    delta = np.asarray(pa['local_metric_xy']) - np.asarray(ab['local_metric_xy'])
    delta_world = np.asarray(pa['world_xy']) - np.asarray(ab['world_xy'])
    angle = []
    for a,b in zip(pa['local_heading_rad'],ab['local_heading_rad']):
        angle.append(math.atan2(math.sin(a-b),math.cos(a-b)) if a is not None and b is not None else None)
    max_lateral = float(np.abs(delta[:,1]).max())
    cp,ca=pa['hypothetical_commands'][4],ab['hypothetical_commands'][4]
    command_delta = None if 'error' in cp or 'error' in ca else {
        key:(np.asarray(cp[key])-np.asarray(ca[key])).tolist()
        for key in ('pre_clip_vw','clipped_vw','final_vw')}
    return dict(delta_definition='PRESENT minus ABSENT',present=pa,absent=ab,
                per_waypoint_delta_local_xy_m=delta.tolist(),
                per_waypoint_delta_world_xy_m=delta_world.tolist(),
                per_waypoint_world_displacement_difference_m=np.linalg.norm(delta_world,axis=1).tolist(),
                selected_index4_delta_local_xy_m=delta[4].tolist(),
                selected_index4_displacement_difference_m=float(np.linalg.norm(delta[4])),
                maximum_absolute_lateral_difference_m=max_lateral,
                rms_position_difference_m=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
                heading_angular_difference_rad=angle,
                selected_index4_command_delta=command_delta,
                material_lateral_response=max_lateral>=threshold,screening_threshold_m=threshold)
