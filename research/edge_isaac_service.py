"""Persistent Edge runtime. Lossless PNG requests, immutable native responses."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'inference')]
import numpy as np
import torch
from PIL import Image
import clip
import run_omnivla_edge as official
from model_omnivla_edge import OmniVLA_edge
from research.edge_contracts import controller, initialize_sample_globals
from research.isaac_final_common import append_json, sha256, write_json
from research.isaac_final_contracts import action, pixel_hash


def prepare(frames, normalized_goal):
    """Official transforms/resize order, with actual history and supplied pose."""
    images = [Image.fromarray(frame) for frame in frames]
    obs = official.transform_images_PIL_mask([im.resize(official.imgsize) for im in images],
                                             official.mask_360_pil_96).to(official.device)
    parts = torch.split(obs, 3, dim=1)
    current = official.transform_images_PIL_mask(images[-1].resize(official.imgsize_clip),
                                                official.mask_360_pil_224).to(official.device)
    blank_map = Image.new('RGB', (352,352), (0,0,0))
    maps = torch.cat((official.transform_images_map(blank_map).to(official.device),
                      official.transform_images_map(blank_map).to(official.device), parts[-1]), axis=1)
    return dict(obs_images=torch.cat(parts, dim=1),
                goal_pose_torch=torch.tensor([normalized_goal], dtype=torch.float32, device=official.device),
                cur_large_img=current, map_images=maps,
                goal_image=official.transform_images_PIL_mask(Image.new('RGB', official.imgsize),
                            official.mask_360_pil_96).to(official.device),
                obj_inst_lan=clip.tokenize('xxxx', truncate=True).to(official.device))


def verify_frames(request):
    records = request['history']
    assert len(records) == 6 and len({x['frame_id'] for x in records}) == 6
    times = [x['sim_time'] for x in records]
    assert all(abs(b-a-1/3) < 1e-6 for a,b in zip(times,times[1:]))
    assert times[-1] == request['observation_time_sim']
    frames = []
    for frame in records:
        path = Path(frame['path']).resolve()
        assert path.is_relative_to(ROOT/'outputs')
        assert sha256(path) == frame['png_sha256']
        rgb = np.asarray(Image.open(path).convert('RGB'))
        assert rgb.shape == (480,640,3) and rgb.dtype == np.uint8
        assert pixel_hash(rgb) == frame['rgb_sha256']
        frames.append(rgb)
    return frames


def load(out):
    assert torch.cuda.is_available(), 'GPU runtime is required; no CPU fallback'
    torch.manual_seed(0)
    np.random.seed(0)
    initialize_sample_globals(official)
    official.pose_goal, official.satellite, official.image_goal, official.lan_prompt = True, False, False, False
    manifest = json.loads((ROOT/'results/edge-checkpoint-manifest.json').read_text())
    checkpoint = ROOT/manifest['path']
    assert sha256(checkpoint) == manifest['sha256'] and checkpoint.stat().st_size == manifest['bytes']
    clip_path = ROOT/'models-local/clip/ViT-B-32.pt'
    assert sha256(clip_path) == '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
    original = OmniVLA_edge.load_state_dict
    original_clip = clip.load
    def checked(self, state, strict=True, **kw):
        outcome = original(self, state, strict=strict, **kw)
        assert strict and not outcome.missing_keys and not outcome.unexpected_keys
        load_record.update(strict=strict, missing_keys=list(outcome.missing_keys),
                           unexpected_keys=list(outcome.unexpected_keys))
        return outcome
    load_record = {}
    def cached_clip(*a, **kw):
        return original_clip(*a, download_root=str(clip_path.parent), **kw)
    with patch.object(OmniVLA_edge, 'load_state_dict', checked), patch.object(clip, 'load', cached_clip):
        model, encoder, _ = official.load_model(str(checkpoint), official.model_params, official.device)
    official.model = model.to(official.device).eval()
    official.text_encoder = encoder.to(official.device).eval()
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert {p.dtype for p in encoder.parameters()} == {torch.float32}
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    assert all(torch.isfinite(p).all().item() for p in encoder.parameters())
    torch.cuda.synchronize()
    load_record.update(status='STRICT_LOAD_PASS', checkpoint=manifest,
                       device=str(official.device), dtype='float32', torch=torch.__version__,
                       cuda=torch.version.cuda, numpy=np.__version__, load_count=1,
                       clip_sha256=sha256(clip_path), allocated_bytes=torch.cuda.memory_allocated(),
                       reserved_bytes=torch.cuda.memory_reserved())
    write_json(out/'strict-load.json', load_record)
    return official.Inference.run_forward_pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--socket', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    forward = load(args.output)
    hook = {}
    def before(*_):
        torch.cuda.synchronize()
        hook['start'] = time.perf_counter()
    def after(*_):
        torch.cuda.synchronize()
        hook['model_seconds'] = time.perf_counter()-hook['start']
    official.model.register_forward_pre_hook(before)
    official.model.register_forward_hook(after)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    os.chmod(args.socket, 0o600)
    server.listen(1)
    server.settimeout(.2)
    write_json(args.output/'ready.json', dict(status='READY', pid=os.getpid(), socket=args.socket))
    seen = set()
    calls = 0
    try:
        while not (args.output.parent/'service-stop').exists():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn, conn.makefile('rb') as stream:
                for line in stream:
                    request = json.loads(line)
                    received = time.time()
                    response = dict(request_id=request['request_id'], observation_id=request['observation_id'])
                    try:
                        key = (request['episode_id'], request['request_id'])
                        if key in seen:
                            raise ValueError('Duplicate request; no replay')
                        seen.add(key)
                        start = time.perf_counter()
                        frames = verify_frames(request)
                        response['rgb_hashes'] = [pixel_hash(f) for f in frames]
                        if request['op'] == 'dummy':
                            response.update(dummy=True, lossless_roundtrip=True)
                        elif request['op'] == 'infer':
                            batch = prepare(frames, request['normalized_pose'])
                            torch.cuda.synchronize()
                            forward_start = time.perf_counter()
                            tensor, modality = forward(None, model=official.model, batch=batch,
                                                       device_id=official.device, mode='vali', idrun=calls)
                            torch.cuda.synchronize()
                            forward_s = time.perf_counter()-forward_start
                            raw = action(tensor.detach().cpu().numpy())
                            assert int(modality.item()) == 4
                            path = Path(request['action_path']).resolve()
                            assert path.is_relative_to(ROOT/'outputs')
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with path.open('xb') as dest:
                                np.save(dest, raw, allow_pickle=False)
                                dest.flush()
                                os.fsync(dest.fileno())
                            path.chmod(0o444)
                            metric = raw[0,4].copy()
                            metric[:2] *= .1  # exact official float32 in-place scaling
                            try:
                                control = controller(metric)
                                response['controller'] = control
                            except NameError as exc:
                                if 'clip_angle' not in str(exc):
                                    raise
                                response['controller_error'] = 'UPSTREAM_CONTROLLER_BUG: '+str(exc)
                            response.update(action_path=str(path), action_sha256=sha256(path),
                                            native_shape=list(raw.shape), native_dtype=str(raw.dtype),
                                            selected_index=4, selected_native=raw[0,4].tolist(),
                                            selected_metric=metric.tolist(), modality_id=4,
                                            model_ms=hook['model_seconds']*1000,
                                            forward_with_clip_ms=forward_s*1000)
                            calls += 1
                        else:
                            raise ValueError('Unknown IPC operation')
                        response.update(pipeline_ms=(time.perf_counter()-start)*1000,
                                        received_wall_time=received, ready_wall_time=time.time())
                    except Exception:
                        response['error'] = traceback.format_exc()
                    append_json(args.output/'responses.jsonl', response)
                    conn.sendall((json.dumps(response, allow_nan=False)+'\n').encode())
    finally:
        server.close()
        Path(args.socket).unlink(missing_ok=True)
        write_json(args.output/'stopped.json', dict(inference_count=calls, load_count=1))


if __name__ == '__main__':
    main()
