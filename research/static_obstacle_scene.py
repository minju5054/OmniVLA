"""One fixed unannotated opaque cuboid; no spectator markers in science."""
from pxr import Gf, Usd, UsdGeom
from research.isaac_scene import box


def add_obstacle(stage, config):
    spec = config['static_obstacle']
    prim = box(stage, spec['path'], spec['center'], spec['dimensions'], spec['color'], True)
    UsdGeom.Gprim(prim).CreateDisplayOpacityAttr([spec['opacity']])
    return prim


def geometry_record(stage, config):
    prim = stage.GetPrimAtPath(config['static_obstacle']['path'])
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    lower, upper = bounds.GetMin(), bounds.GetMax()
    record = dict(center=list((lower+upper)/2), dimensions=list(upper-lower),
                  color=[list(v) for v in UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()],
                  opacity=list(UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Get()))
    assert all(abs(a-b)<1e-6 for a,b in zip(record['center'],config['static_obstacle']['center']))
    assert all(abs(a-b)<1e-6 for a,b in zip(record['dimensions'],config['static_obstacle']['dimensions']))
    assert record['opacity'] == [1.]
    return record
