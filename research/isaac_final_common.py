"""Durable, exclusive evidence writing shared by the separate runtimes."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GATES = ('lifecycle', 'camera', 'history', 'pose', 'ipc', 'strict_load',
         'combined_memory', 'scheduler', 'controller_parity')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def append_json(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()


def sync_tree(root):
    for path in Path(root).rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    for path in [Path(root), *[p for p in Path(root).rglob('*') if p.is_dir()]]:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def verify_frozen(config):
    if set(config.get('infrastructure_gates', {})) != set(GATES):
        raise ValueError('All infrastructure gates must be recorded before science')
    if any(config['infrastructure_gates'][key]['status'] != 'PASS' for key in GATES):
        raise ValueError('Infrastructure gates have not all passed')
    if not config.get('implementation_sha256'):
        raise ValueError('Missing frozen implementation hashes')
    for path, expected in config['implementation_sha256'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f'Frozen implementation changed: {path}')
