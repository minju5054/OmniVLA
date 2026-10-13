"""Evaluation-only circle/AABB clearance for straight SE(2) Euler segments.

The center-to-rectangle distance is zero inside the rectangle. Subtracting the
virtual radius follows the declared convention (not a penetration-depth SDF).
No method in this module changes a command or pose.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Rectangle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.xmin, self.xmax, self.ymin, self.ymax)):
            raise ValueError('Nonfinite rectangle')
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError('Rectangle must have positive area')

    @classmethod
    def from_config(cls, obstacle):
        x, y, _ = obstacle['center']
        lx, ly, _ = obstacle['dimensions']
        return cls(x-lx/2, x+lx/2, y-ly/2, y+ly/2)

    def distance(self, point):
        x, y = point[:2]
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Nonfinite center')
        return math.hypot(max(self.xmin-x, 0., x-self.xmax),
                          max(self.ymin-y, 0., y-self.ymax))

    def clearance(self, point, radius):
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('Positive finite radius required')
        return self.distance(point)-radius

    def swept(self, start, end, radius):
        """Exact segment/AABB minimum, then earliest circle contact by bisection.

        A minimum lies at an endpoint, a rectangle-corner projection, or an
        intersection. Slab clipping handles crossing segments whose endpoints
        are both outside (tunneling). Convexity permits bisection between the
        initial point and a minimum-distance point for first radius contact.
        """
        self.clearance(start, radius)
        self.clearance(end, radius)
        a = start[:2]
        delta = [end[i]-a[i] for i in range(2)]
        norm2 = sum(v*v for v in delta)
        point = lambda t: [a[i]+t*delta[i] for i in range(2)]
        candidates = [0., 1.]
        if norm2:
            for x in (self.xmin, self.xmax):
                for y in (self.ymin, self.ymax):
                    t = ((x-a[0])*delta[0]+(y-a[1])*delta[1])/norm2
                    candidates.append(min(1., max(0., t)))
        low, high = 0., 1.
        for i, (lower, upper) in enumerate(((self.xmin, self.xmax), (self.ymin, self.ymax))):
            if delta[i] == 0:
                if not lower <= a[i] <= upper:
                    low, high = 1., 0.
                    break
            else:
                t1, t2 = (lower-a[i])/delta[i], (upper-a[i])/delta[i]
                low, high = max(low, min(t1,t2)), min(high, max(t1,t2))
        if low <= high:
            candidates.append(low)
        fraction = min(candidates, key=lambda t: self.distance(point(t)))
        minimum = self.clearance(point(fraction), radius)
        first = None
        if minimum <= 0:
            if self.clearance(start, radius) <= 0:
                first = 0.
            else:
                left, right = 0., fraction
                for _ in range(64):
                    mid = (left+right)/2
                    if self.clearance(point(mid), radius) <= 0:
                        right = mid
                    else:
                        left = mid
                first = right
        return dict(minimum_clearance=minimum, minimum_fraction=fraction,
                    collision=minimum <= 0, first_contact_fraction=first)


def classify(goal_distance, elapsed, minimum_clearance, config, failure=None):
    """Failure first; collision outranks goal entry; clearance violations persist."""
    if failure:
        if failure not in ('IMPLEMENTATION_FAILURE', 'UPSTREAM_CONTROLLER_BUG'):
            raise ValueError('Unknown execution failure')
        return failure
    if minimum_clearance <= 0:
        return 'COLLISION'
    if elapsed <= config['timeout_after_bootstrap_sim_s'] and goal_distance <= config['goal_radius_m']:
        return ('SAFE_GOAL_REACHED' if minimum_clearance >= config['evaluation']['safety_margin_m']
                else 'GOAL_REACHED_LOW_CLEARANCE')
    if elapsed >= config['timeout_after_bootstrap_sim_s']:
        return 'MODEL_NAVIGATION_FAILURE'
    return None
