"""Deterministic SE(2), actual-history and single-outstanding IPC contracts."""
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import select
import socket
import time
import numpy as np


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def local_goal(pose, goal):
    dx, dy = goal[0]-pose[0], goal[1]-pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    x, y = c*dx+s*dy, -s*dx+c*dy
    yaw = wrap(goal[2]-pose[2])
    radius = math.hypot(x, y)
    scale = min(1., 30./radius) if radius else 1.
    return [x, y, yaw], [x*scale/.1, y*scale/.1, math.cos(yaw), math.sin(yaw)]


def integrate(pose, command, dt=1/60):
    x, y, yaw = pose
    v, w = command
    return [x+v*math.cos(yaw)*dt, y+v*math.sin(yaw)*dt, wrap(yaw+w*dt)]


def action(value):
    a = np.asarray(value)
    if a.shape != (1, 8, 4) or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError(f'Invalid native action: {a.shape}, {a.dtype}')
    # Back with immutable bytes, so writeability cannot be re-enabled.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def pixel_hash(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


class History:
    def __init__(self):
        self.frames = deque(maxlen=6)

    def append(self, frame):
        if self.frames:
            previous = self.frames[-1]
            if frame['frame_id'] <= previous['frame_id']:
                raise ValueError('Frame IDs must increase')
            if not math.isclose(frame['sim_time']-previous['sim_time'], 1/3, abs_tol=1e-6):
                raise ValueError('History must use independent 1/3-s capture slots')
        self.frames.append(dict(frame))

    def snapshot(self, obs_time):
        if len(self.frames) != 6 or self.frames[-1]['sim_time'] != obs_time:
            raise ValueError('A request needs six actual frames ending at observation time')
        return [dict(f) for f in self.frames]


class Scheduler:
    """Poll at a boundary, apply at its next step; skip busy 20-step slots."""
    def __init__(self):
        self.outstanding = None
        self.command = [0., 0.]
        self.missed = 0

    def slot(self, step):
        if step % 20:
            return 'NONE'
        if self.outstanding is not None:
            self.missed += 1
            return 'MISSED_INFERENCE_SLOT'
        return 'SEND'

    def sent(self, request_id):
        if self.outstanding is not None:
            raise RuntimeError('No queue: a request is already outstanding')
        self.outstanding = request_id

    def ready(self, request_id, command):
        if request_id != self.outstanding:
            raise ValueError('Mismatched response ID')
        self.outstanding = None
        self.command = list(command)


class Client:
    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.pending = None
        self.buffer = b''

    def send(self, request):
        if self.pending is not None:
            raise RuntimeError('No queue: only one outstanding IPC request')
        self.pending = (request['request_id'], request['observation_id'])
        self.sock.sendall((json.dumps(request, allow_nan=False)+'\n').encode())

    def poll(self):
        if self.pending is None or not select.select([self.sock], [], [], 0)[0]:
            return None
        part = self.sock.recv(65536)
        if not part:
            raise ConnectionError('Service disconnected')
        self.buffer += part
        if b'\n' not in self.buffer:
            return None
        line, self.buffer = self.buffer.split(b'\n', 1)
        response = json.loads(line)
        if (response['request_id'], response['observation_id']) != self.pending:
            raise ValueError('IPC request/observation ID mismatch')
        self.pending = None
        if response.get('error'):
            raise RuntimeError(response['error'])
        return response

    def close(self):
        self.sock.close()


def stats(values):
    if not values:
        return dict(count=0, median=None, p95=None, max=None)
    a = np.asarray(values, dtype=float)
    return dict(count=len(values), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))
