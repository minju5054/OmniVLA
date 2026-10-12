"""Download and verify the pinned official Edge checkpoint without overwrites."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

REPO = 'NHirose/omnivla-edge'
REVISION = 'b1361b7e24f101edea795a98b00a826b61a97394'
NAME = 'omnivla-edge.pth'
SIZE = 434276465
SHA256 = 'f93b056f124380899fce3e087a5c2b41ce4d8c07e211daad52b6ebf02e80d04d'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    target = Path('models-local') / 'omnivla-edge' / REVISION / NAME
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{NAME}'
    if not target.exists():
        assert shutil.disk_usage('.').free > 2 * SIZE
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix('.partial')
        with urllib.request.urlopen(url, timeout=120) as src, partial.open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8 << 20)
        assert partial.stat().st_size == SIZE and digest(partial) == SHA256
        partial.rename(target)
        target.chmod(0o444)
    assert target.stat().st_size == SIZE and digest(target) == SHA256
    result = dict(repo=REPO, revision=REVISION, filename=NAME, bytes=SIZE,
                  sha256=SHA256, path=str(target), url=url,
                  command='python3 research/fetch_edge_checkpoint.py')
    manifest = Path('results/edge-checkpoint-manifest.json')
    if manifest.exists():
        assert json.loads(manifest.read_text()) == result
    else:
        with manifest.open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
