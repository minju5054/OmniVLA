"""Additive Edge qualification. Original inference/model/controller stay unchanged.

Run from repository root using the isolated Edge interpreter. Every run requires
a new output directory. --sample-only executes exactly the official sample.
"""
import argparse
import contextlib
import functools
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts-local/edge-matplotlib'))
import numpy as np
import torch
from PIL import Image
import clip
import run_omnivla_edge as official
from model_omnivla_edge import OmniVLA_edge
from research.contracts import evidence_directory, immutable_action, metric_action, sha256, write_json, timed_cuda
from research.edge_contracts import controller, goal, initialize_sample_globals, prepare_function, FULL_REPORT_SHA
from research.fetch_edge_checkpoint import REVISION, SHA256, SIZE


def save_array(path, value):
    with path.open('xb') as f:
        np.save(f, value, allow_pickle=False)
    path.chmod(0o444)


def device_used():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                    '--format=csv,noheader,nounits'], text=True)
        return int(result.splitlines()[0])
    except (OSError, subprocess.CalledProcessError):
        return None


def memory():
    rss = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                   if line.startswith('VmRSS:'))) * 1024
    return dict(rss_bytes=rss, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                device_used_MiB=device_used(), allocated=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                reserved=torch.cuda.memory_reserved() if torch.cuda.is_available() else 0)


class MemorySampler:
    def __init__(self):
        self.values = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.is_set():
            v = device_used()
            if v is not None:
                self.values.append(v)
            self.stop.wait(.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    return timed_cuda(fn, type('Sync', (), {'synchronize': staticmethod(sync)}))


def stats(times):
    a = np.asarray(times) * 1000
    return dict(count=len(a), min_ms=float(a.min()), median_ms=float(np.median(a)), mean_ms=float(a.mean()),
                p90_ms=float(np.percentile(a, 90)), p95_ms=float(np.percentile(a, 95)), max_ms=float(a.max()))


def action_summary(raw):
    a = immutable_action(raw)
    m = metric_action(a)
    return dict(shape=list(a.shape), finite=bool(np.isfinite(a).all()), dtype=str(a.dtype),
                native=a[0].tolist(), metric_xy=m[0, :, :2].tolist(), headings=a[0, :, 2:].tolist(),
                heading_norms=np.linalg.norm(a[0, :, 2:], axis=-1).tolist(),
                selected_native=a[0, 4].tolist(), selected_metric=m[0, 4].tolist(), **controller(m[0, 4]))


def difference(a, b):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    aa, bb = action_summary(a), action_summary(b)
    return dict(max_abs=float(np.abs(delta).max()), mean_abs=float(np.abs(delta).mean()),
                l2=float(np.linalg.norm(delta)), selected_native_diff=delta[0, 4].tolist(),
                selected_metric_diff=(np.asarray(aa['selected_metric']) - bb['selected_metric']).tolist(),
                final_vw_diff=(np.asarray(aa['final_vw']) - bb['final_vw']).tolist())


def run(args, out):
    assert Path.cwd() == ROOT, 'Run from repository root'
    assert sha256(ROOT / 'docs/FULL_OMNIVLA_PREFLIGHT.md') == FULL_REPORT_SHA
    checkpoint = ROOT / 'models-local/omnivla-edge' / REVISION / 'omnivla-edge.pth'
    assert checkpoint.stat().st_size == SIZE and sha256(checkpoint) == SHA256
    frozen_path = ROOT / 'research/frozen-edge-goals.json'
    frozen = json.loads(frozen_path.read_text())
    frozen_hash = sha256(frozen_path)
    write_json(out / 'raw/frozen-spec.json', dict(sha256=frozen_hash, spec=frozen))
    torch.manual_seed(0)
    np.random.seed(0)
    initialize_sample_globals(official)
    runtime = dict(label='BLACKWELL_COMPATIBLE_EDGE_RUNTIME' if torch.cuda.is_available() else 'OFFICIAL_TORCH_CPU_REFERENCE',
                   python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                   device=str(official.device), threads=torch.get_num_threads(),
                   gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                   capability=list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
                   tf32_matmul=torch.backends.cuda.matmul.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                   git_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   frozen_spec_sha256=frozen_hash, full_report_sha256=FULL_REPORT_SHA,
                   source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in
                                 (ROOT/'inference/run_omnivla_edge.py', ROOT/'inference/model_omnivla_edge.py', ROOT/'inference/utils_policy.py')})
    write_json(out / 'runtime.json', runtime)
    before = memory()
    strict = {}
    original_load_state = OmniVLA_edge.load_state_dict
    original_clip_load = clip.load
    cache = ROOT / 'models-local/clip'

    def checked_load(self, state_dict, strict=True, **kwargs):
        result = original_load_state(self, state_dict, strict=strict, **kwargs)
        checked.update(strict=strict, missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys))
        return result

    checked = {}
    def clip_load(*a, **kw):
        result = original_clip_load(*a, download_root=str(cache), **kw)
        strict['after_clip_load'] = memory()
        return result

    load_start = time.perf_counter()
    try:
        with patch.object(OmniVLA_edge, 'load_state_dict', checked_load), patch.object(clip, 'load', clip_load):
            model, text_encoder, _ = official.load_model(str(checkpoint), official.model_params, official.device)
        official.model = model.to(official.device).eval()
        official.text_encoder = text_encoder.to(official.device).eval()
        sync()
        strict['load_seconds'] = time.perf_counter() - load_start
        strict.update(checked)
        strict.update(parameter_count=sum(p.numel() for p in model.parameters()),
                      trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      clip_parameter_count=sum(p.numel() for p in text_encoder.parameters()),
                      dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                      clip_dtypes=sorted({str(p.dtype) for p in text_encoder.parameters()}),
                      devices=sorted({str(p.device) for p in model.parameters()}),
                      finite_parameters=all(torch.isfinite(p).all().item() for p in model.parameters()),
                      finite_clip_parameters=all(torch.isfinite(p).all().item() for p in text_encoder.parameters()),
                      before=before, after_load=memory())
        assert strict['strict'] and not strict['missing_keys'] and not strict['unexpected_keys']
        assert strict['finite_parameters'] and strict['finite_clip_parameters']
        strict['status'] = 'EDGE_CHECKPOINT_STRICT_LOAD_PASS'
    except Exception:
        strict.update(status='EDGE_CHECKPOINT_LOAD_FAILED', traceback=traceback.format_exc())
        write_json(out / 'strict-load.json', strict)
        write_json(out / 'decision.json', dict(decision='BLOCKED_BY_CHECKPOINT'))
        raise
    write_json(out / 'strict-load.json', strict)
    clip_path = cache / 'ViT-B-32.pt'
    clip_url = clip.clip._MODELS['ViT-B/32']
    clip_hash = sha256(clip_path)
    assert clip_hash == clip_url.split('/')[-2]
    write_json(out / 'clip-manifest.json', dict(model='ViT-B/32', package='openai-clip==1.0.1',
               url=clip_url, path=str(clip_path.relative_to(ROOT)), bytes=clip_path.stat().st_size, sha256=clip_hash,
               command='Official load_model -> clip.load("ViT-B/32", download_root="models-local/clip")'))
    inference = official.Inference(str(out / 'derived'), official.lan_inst_prompt, official.goal_utm,
                                   official.goal_compass, official.goal_image_PIL)
    captured = {}
    original_forward = inference.run_forward_pass
    def record_forward(*a, **kw):
        captured['batch'] = kw['batch']
        (actions, mod), elapsed = timed(lambda: original_forward(*a, **kw))
        captured.update(raw=immutable_action(actions.detach().cpu().numpy()), modality=int(mod.item()),
                        forward_with_clip_seconds=elapsed)
        return actions, mod

    # Hooks time the unchanged Edge model, independently of CLIP encoding.
    hook_data = {}
    def pre_hook(*_):
        sync()
        hook_data['start'] = time.perf_counter()
    def post_hook(*_):
        sync()
        hook_data['seconds'] = time.perf_counter() - hook_data['start']
    pre = model.register_forward_pre_hook(pre_hook)
    post = model.register_forward_hook(post_hook)
    inference.run_forward_pass = record_forward
    original_save = inference.save_robot_behavior
    def record_save(*a, **kw):
        sync()
        captured['pipeline_seconds'] = time.perf_counter() - pipeline_start
        return original_save(*a, **kw)
    inference.save_robot_behavior = record_save
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        sync()
        pipeline_start = time.perf_counter()
        actual_vw = inference.run_omnivla()
        raw = captured['raw']
        summary = action_summary(raw)
        np.testing.assert_array_equal(summary['final_vw'], actual_vw)
    except Exception:
        write_json(out / 'sample.json', dict(status='EDGE_OFFICIAL_SAMPLE_FAILED', traceback=traceback.format_exc()))
        write_json(out / 'decision.json', dict(decision='BLOCKED_BY_SAMPLE_INFERENCE'))
        raise
    finally:
        pre.remove()
        post.remove()
    inference.run_forward_pass = original_forward
    inference.save_robot_behavior = original_save
    save_array(out / 'raw/official-native.npy', raw)
    batch = captured['batch']
    batch_meta = {}
    for k, v in batch.items():
        arr = v.detach().cpu().numpy()
        save_array(out / f'raw/input-{k}.npy', arr)
        batch_meta[k] = dict(shape=list(arr.shape), dtype=str(arr.dtype), sha256=hashlib.sha256(arr.tobytes()).hexdigest())
    masks = {}
    for k in ('mask_360_pil_96', 'mask_360_pil_224'):
        arr = getattr(official, k)
        save_array(out / f'raw/{k}.npy', arr)
        masks[k] = dict(shape=list(arr.shape), dtype=str(arr.dtype), unique=np.unique(arr).tolist(), sha256=hashlib.sha256(arr.tobytes()).hexdigest())
    mod = captured['modality']
    summary.update(status='EDGE_OFFICIAL_SAMPLE_PASS', modality_id=mod, language=official.lan_inst_prompt,
                   supplied_goal_pose=batch['goal_pose_torch'].cpu().tolist(), history='Six references to current image resized to 96x96; stopped-robot assumption',
                   current_image_sha256=sha256(ROOT/'inference/current_img.jpg'), goal_image_sha256=sha256(ROOT/'inference/goal_img.jpg'),
                   batch=batch_meta, pixel_masks=masks, attention_mask=model.all_masks[mod].tolist(),
                   pooling_mask=model.avg_pool_mask[mod].tolist(),
                   forward_with_clip_seconds=captured['forward_with_clip_seconds'], cold_model_seconds=hook_data['seconds'],
                   pipeline_seconds=captured['pipeline_seconds'], load_seconds=strict['load_seconds'], memory=memory(),
                   actual_official_vw=list(map(float, actual_vw)),
                   native_path=str(out/'raw/official-native.npy'), figure_path=str(out/'derived/1_ex_omnivla_edge.jpg'))
    write_json(out / 'sample.json', summary)
    if args.sample_only:
        write_json(out / 'decision.json', dict(decision='SAMPLE_ONLY_NO_FULL_PREFLIGHT_DECISION'))
        return
    official.pose_goal, official.satellite, official.image_goal, official.lan_prompt = True, False, False, False
    prepare = prepare_function(official)
    pose_batch = prepare(inference)
    pose_results = {}
    pose_raw = {}
    for spec in frozen['tests']:
        name = spec['name']
        normalized = goal(spec['delta_east_m'], spec['delta_north_m'], 0., spec['goal_compass_deg'])
        np.testing.assert_allclose(normalized, spec['expected_pose'], atol=1e-12)
        b = dict(pose_batch, goal_pose_torch=torch.tensor(normalized, dtype=torch.float32, device=official.device).unsqueeze(0))
        actions, mod = original_forward(model=model, batch=b, device_id=official.device)
        a = immutable_action(actions.cpu().numpy())
        save_array(out / f'raw/pose-{name}.npy', a)
        s = action_summary(a)
        x, y = s['selected_metric'][:2]
        w = s['final_vw'][1]
        if name == 'FRONT':
            passed = x > 0 and abs(np.arctan2(y, x)) <= np.pi/4
        else:
            sign = 1 if name == 'LEFT' else -1
            passed = x > 0 and sign*y > 0 and sign*np.mean(a[0,:,1]) > 0 and sign*w > 0
        s.update(passed=bool(passed), normalized_pose=normalized.tolist(), modality_id=int(mod.item()))
        pose_results[name], pose_raw[name] = s, a
    passed = all(s['passed'] for s in pose_results.values())
    write_json(out / 'coordinates.json', dict(status='EDGE_POSE_GOAL_SANITY_PASS' if passed else 'EDGE_POSE_GOAL_SANITY_FAILED',
               frozen_spec_sha256=frozen_hash, tests=pose_results))
    if not passed:
        write_json(out / 'decision.json', dict(decision='BLOCKED_BY_POSE_MODALITY', stopped_before=['masking', 'latency', 'cadence']))
        return
    front = frozen['tests'][0]['expected_pose']
    b = dict(pose_batch, goal_pose_torch=torch.tensor([front], dtype=torch.float32, device=official.device))
    b['goal_image'] = official.transform_images_PIL_mask(Image.new('RGB', (96,96)), official.mask_360_pil_96).to(official.device)
    masked, _ = original_forward(model=model, batch=b, device_id=official.device)
    masked = immutable_action(masked.cpu().numpy())
    save_array(out/'raw/pose-FRONT-black-goal.npy', masked)
    diff = difference(pose_raw['FRONT'], masked)
    mask_pass = diff['max_abs'] <= 1e-6
    write_json(out/'masking.json', dict(status='POSE_ONLY_MASKING_RUNTIME_PASS' if mask_pass else 'POSE_ONLY_MASKING_RUNTIME_MISMATCH',
               attention_mask=model.all_masks[4].tolist(), pooling_mask=model.avg_pool_mask[4].tolist(), **diff))
    if not mask_pass:
        write_json(out/'decision.json', dict(decision='BLOCKED_BY_POSE_MODALITY', stopped_before=['latency', 'cadence']))
        return
    # Freeze official language sample for warm measurements. No pose/input tuning.
    official.pose_goal, official.satellite, official.image_goal, official.lan_prompt = False, False, False, True
    with torch.no_grad():
        feat = official.text_encoder.encode_text(batch['obj_inst_lan'])
    model_args = (batch['obs_images'], batch['goal_pose_torch'], batch['map_images'], batch['goal_image'],
                  torch.tensor([7], device=official.device), feat, batch['cur_large_img'])
    def model_forward():
        with torch.no_grad():
            return model(*model_args)[0]
    def pipeline():
        fresh = prepare(inference)
        with contextlib.redirect_stdout(io.StringIO()):
            acts, _ = original_forward(model=model, batch=fresh, device_id=official.device)
        arr = acts.float().cpu().numpy()
        controller(metric_action(arr)[0,4])
        return arr
    for _ in range(5):
        model_forward()
        pipeline()
    sync()
    allocations, reservations, forwards, pipelines = [], [], [], []
    max_diff = 0.
    for _ in range(50):
        a, elapsed = timed(model_forward)
        forwards.append(elapsed)
        max_diff = max(max_diff, float(np.abs(a.cpu().numpy()-raw).max()))
        allocations.append(torch.cuda.memory_allocated())
        reservations.append(torch.cuda.memory_reserved())
    # Fail early on memory growth before later pipeline/cadence qualification.
    stable = np.ptp(allocations[-10:]) <= 16*2**20 and np.ptp(reservations[-10:]) <= 16*2**20
    if not stable:
        write_json(out/'decision.json', dict(decision='BLOCKED_BY_MEMORY'))
        return
    for _ in range(20):
        a, elapsed = timed(pipeline)
        pipelines.append(elapsed)
        max_diff = max(max_diff, float(np.abs(a-raw).max()))
        allocations.append(torch.cuda.memory_allocated())
        reservations.append(torch.cuda.memory_reserved())
    stable = bool(stable and np.ptp(allocations[-10:]) <= 16*2**20 and np.ptp(reservations[-10:]) <= 16*2**20)
    if not stable:
        write_json(out/'decision.json', dict(decision='BLOCKED_BY_MEMORY'))
        return
    st = stats(pipelines)
    fraction = float(np.mean(np.asarray(pipelines) > 1/3))
    classification = ('EDGE_3HZ_RUNTIME_FEASIBLE' if st['p95_ms'] <= 1000/3 and fraction <= .05 else
                      'EDGE_3HZ_RUNTIME_MARGINAL' if st['median_ms'] <= 1000/3 else 'EDGE_3HZ_RUNTIME_INFEASIBLE')
    write_json(out/'latency.json', dict(forward=stats(forwards), pipeline=st, forward_seconds=forwards,
               pipeline_seconds=pipelines, warmup=5, repeat_output_max_abs_diff=max_diff, memory_stable=bool(stable),
               warm_allocated_bytes=allocations, warm_reserved_bytes=reservations, fraction_exceeding_333ms=fraction,
               classification=classification, target_interval_ms=1000/3, model_excludes_clip=True,
               pipeline_includes='Disk image read, PIL resize, official masking/normalization, pose/maps/tokenization, device transfers, CLIP text encoding, Edge, CPU action transfer, unchanged source controller; excludes weights/figure/log I/O except in-memory print formatting; synchronized CUDA boundaries'))
    decision = 'READY_FOR_ISAAC_ADAPTER' if classification == 'EDGE_3HZ_RUNTIME_FEASIBLE' else 'BLOCKED_BY_CADENCE'
    write_json(out/'decision.json', dict(decision=decision, classification=classification))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    evidence_directory(out)
    error = None
    with MemorySampler() as sampler:
        try:
            run(args, out)
        except Exception:
            error = traceback.format_exc()
            write_json(out/'error.json', dict(traceback=error))
    measured = memory()
    measured.update(peak_observed_device_MiB=max(sampler.values, default=None),
                    device_sampling_interval_seconds=.1, device_samples=len(sampler.values),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0)
    write_json(out/'memory.json', measured)
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
