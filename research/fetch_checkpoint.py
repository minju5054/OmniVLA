"""Fetch the pinned FULL checkpoint, verify bytes, and preserve existing files."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

REPO = 'NHirose/omnivla-original'
REVISION = 'e36a84d4923c041149d441f93f3bdb7092bb5f07'
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models-local' / 'omnivla-original' / REVISION


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true') as response:
        meta = json.load(response)
    assert meta['sha'] == REVISION and meta['id'] == REPO
    required = [f for f in meta['siblings'] if '/' not in f['rfilename']
                and f['rfilename'] not in {'.gitattributes', 'dist_head--120000_checkpoint.pt',
                                          'modeling_prismatic_____.py'}]
    DEST.mkdir(parents=True, exist_ok=True)
    missing_bytes = sum(f['size'] for f in required if not (DEST / f['rfilename']).exists())
    free = shutil.disk_usage(DEST).free
    if free < missing_bytes + 5 * 1024**3:
        raise RuntimeError(f'Insufficient disk: {free} free, {missing_bytes} download bytes')
    print(f'Free bytes: {free}; required download bytes: {missing_bytes}', flush=True)

    def fetch(item):
        name = item['rfilename']
        path = DEST / name
        if not path.exists():
            part = path.with_name(path.name + '.part')
            offset = part.stat().st_size if part.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            req = urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true', headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                if offset and response.status != 206:
                    raise RuntimeError(f'{name}: server did not honor resume; preserved partial file')
                with part.open('ab' if offset else 'xb') as stream:
                    shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            if part.stat().st_size != item['size']:
                raise RuntimeError(f'{name}: downloaded size mismatch')
            actual = digest(part)
            expected = item.get('lfs', {}).get('sha256')
            if expected and actual != expected:
                raise RuntimeError(f'{name}: SHA256 mismatch')
            # A hard link refuses replacement if another process created the target.
            path.hardlink_to(part)
            part.unlink()
        actual = digest(path)
        if path.stat().st_size != item['size'] or (item.get('lfs') and actual != item['lfs']['sha256']):
            raise RuntimeError(f'{name}: existing file does not match pinned source; left unchanged')
        print(f'Verified {name}: {item["size"]} bytes, {actual}', flush=True)
        return {'name': name, 'bytes': item['size'], 'sha256': actual}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        files = sorted(pool.map(fetch, required), key=lambda f: f['name'])
    manifest = {'repo': REPO, 'revision': REVISION, 'local_path': str(DEST),
                'files': files, 'total_downloaded_bytes': sum(f['bytes'] for f in files),
                'model_shard_bytes': sum(f['bytes'] for f in files if f['name'].endswith('.safetensors')),
                'omitted_unused': ['lora_adapter/', 'dist_head--120000_checkpoint.pt', 'modeling_prismatic_____.py']}
    target = ROOT / 'results' / 'checkpoint-manifest.json'
    content = json.dumps(manifest, indent=2, sort_keys=True) + '\n'
    if target.exists():
        if target.read_text() != content:
            raise RuntimeError('Refusing to replace a different manifest')
    else:
        with target.open('x') as stream:
            stream.write(content)


if __name__ == '__main__':
    main()
