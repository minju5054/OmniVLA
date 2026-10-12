"""Frozen standard-primitive scene. Import only after SimulationApp starts."""
import math
import numpy as np
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera


def box(stage, path, position, dimensions, color, collision=False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


def build(config):
    world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.)
    stage = world.stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    s = config['scene']
    box(stage, '/World/Floor', s['floor_center'], s['floor_dimensions'], s['floor_color'], True)
    for i,x in enumerate(s['boundary_x']):
        box(stage, f'/World/BoundaryX{i}', [x,0,1.5], [.1,28,3], [.65,.60,.52])
    for i,y in enumerate(s['boundary_y']):
        box(stage, f'/World/BoundaryY{i}', [5,y,1.5], [28,.1,3], [.48,.58,.65])
    for i,x in enumerate(range(-8,20,2)):
        box(stage, f'/World/GridX{i}', [x,0,.001], [.02,28,.001], [.40,.42,.44])
    for i,y in enumerate(range(-14,15,2)):
        box(stage, f'/World/GridY{i}', [5,y,.001], [28,.02,.001], [.40,.42,.44])
    dome = UsdLux.DomeLight.Define(stage, '/World/Dome')
    dome.CreateIntensityAttr(s['dome_intensity'])
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr(s['distant_intensity'])
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25,-35,0))
    c = config['camera']
    camera = Camera('/World/AgentCamera', frequency=60, resolution=(c['width'],c['height']))
    camera.set_world_pose(np.array([0.,0.,c['height_m']]), np.array([1.,0.,0.,0.]), camera_axes='world')
    camera.set_focal_length(c['focal_length_m'])
    camera.set_horizontal_aperture(c['horizontal_aperture_m'])
    camera.set_clipping_range(.01,1000.)
    world.reset()
    camera.initialize()
    return world, camera


def set_pose(camera, pose, height=.5):
    x,y,yaw = pose
    camera.set_world_pose(np.array([x,y,height]), np.array([math.cos(yaw/2),0.,0.,math.sin(yaw/2)]), camera_axes='world')


def rgb(camera):
    value = np.asarray(camera.get_rgba())
    if value.shape != (480,640,4) or value.dtype != np.uint8:
        raise ValueError(f'Invalid camera RGBA: {value.shape}, {value.dtype}')
    return value[:,:,:3].copy()
