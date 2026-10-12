"""Run unchanged full OmniVLA functions with additive, non-overwriting evidence.

No precision/model fallback, CPU offload, or remote inference is implemented.
An OOM terminates model execution immediately, including subsequent probes.
"""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.contracts import (HF_REVISION, canonical_json, checkpoint_identity,
    evidence_directory, immutable_action, metric_action, official_controller,
    official_goal, sha256, timed_cuda, write_json)


def utc():
    return datetime.now(timezone.utc).isoformat()


def memory(torch):
    status = {'time_utc': utc()}
    for name, method in [('allocated', torch.cuda.memory_allocated),
                         ('reserved', torch.cuda.memory_reserved),
                         ('max_allocated', torch.cuda.max_memory_allocated),
                         ('max_reserved', torch.cuda.max_memory_reserved)]:
        try:
            status[name + '_bytes'] = method()
        except Exception as exc:
            status[name + '_error'] = str(exc)
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    status['nvidia_smi_mib_used_free'] = result.stdout.strip()
    for file, keys in [('/proc/meminfo', ('MemTotal:', 'MemAvailable:')),
                       ('/proc/self/status', ('VmRSS:', 'VmHWM:'))]:
        for line in Path(file).read_text().splitlines():
            if line.startswith(keys):
                key, value = line.split(':', 1)
                status[key + '_kib'] = int(value.split()[0])
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'models-local/omnivla-original' / HF_REVISION)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-probe', type=Path, default=ROOT / 'artifacts-local/audit/blackwell-cuda.json')
    parser.add_argument('--warm-calls', type=int, default=20)
    args = parser.parse_args()
    if args.warm_calls < 20:
        parser.error('qualification requires at least 20 warm calls')
    os.chdir(ROOT)
    out = evidence_directory(args.output)
    os.environ.setdefault('HF_HOME', str(ROOT / 'artifacts-local/huggingface'))
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'artifacts-local/matplotlib'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    # The verified local snapshot is the only source for execution.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    result = {'started_utc': utc(), 'command': sys.argv, 'checkpoint_revision': HF_REVISION,
              'checkpoint_path': str(args.checkpoint.resolve()),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'git_status_at_start': subprocess.check_output(['git', 'status', '--short'], text=True),
              'classification': 'FULL_GPU_RUNTIME_BLOCKED', 'sample': 'OFFICIAL_SAMPLE_BLOCKED',
              'coordinate': 'COORDINATE_CONVENTION_UNRESOLVED',
              'pose_only': 'POSE_ONLY_MODALITY_UNRESOLVED', 'latency': None,
              'raw_action_shape': None, 'stage': 'runtime_precondition'}
    result['source_hashes'] = {p: sha256(ROOT / p) for p in (
        'inference/run_omnivla.py', 'prismatic/models/action_heads.py',
        'prismatic/extern/hf/modeling_prismatic.py', 'research/run_preflight.py',
        'research/contracts.py', 'research/frozen-goals.json')}
    result['runtime_environment'] = {k: os.environ.get(k) for k in (
        'CUDA_VISIBLE_DEVICES', 'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_MODULE_LOADING',
        'LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    stop = threading.Event()
    sampler = None
    torch = None
    load_started = None
    try:
        runtime = json.loads(args.runtime_probe.read_text())
        if runtime['status'] != 'GPU_RUNTIME_PASS':
            raise RuntimeError('CUDA probe did not pass; no model execution allowed')
        result['runtime_probe'] = runtime
        result['stage'] = 'import'
        import numpy as np
        import torch
        from PIL import Image
        official = importlib.import_module('inference.run_omnivla')
        result['packages'] = {p: importlib.metadata.version(p) for p in (
            'torch', 'torchvision', 'torchaudio', 'transformers', 'tokenizers', 'timm',
            'huggingface-hub', 'accelerate', 'numpy', 'tensorflow', 'tensorflow-datasets',
            'tensorflow-graphics', 'tensorflow-probability', 'tensorflow-metadata', 'dlimp')}
        result['omitted_packages'] = ['peft', 'flash_attn']
        result['tensorflow_visible_gpus'] = [str(x) for x in sys.modules['tensorflow'].config.get_visible_devices('GPU')]
        result['python'] = sys.version
        result['cuda_runtime'] = torch.version.cuda
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        result['before_load'] = memory(torch)
        result['stage'] = 'checkpoint_verification'
        manifest = json.loads((ROOT / 'results/checkpoint-manifest.json').read_text())
        if manifest['repo'] != 'NHirose/omnivla-original' or manifest['revision'] != HF_REVISION:
            raise ValueError('Wrong checkpoint manifest')
        for file in manifest['files']:
            target = args.checkpoint / file['name']
            if target.stat().st_size != file['bytes'] or sha256(target) != file['sha256']:
                raise ValueError(f'Checkpoint integrity mismatch: {file["name"]}')
        config = json.loads((args.checkpoint / 'config.json').read_text())
        tokenizer = json.loads((args.checkpoint / 'tokenizer_config.json').read_text())
        checkpoint_identity(config, tokenizer)
        result['normalization'] = {'constant': 'BOUNDS_Q99',
                                   'sample_calls_unnormalize_actions': False,
                                   'metric_xy_scale': 0.1,
                                   'config_sha256': sha256(args.checkpoint / 'config.json')}

        def monitor():
            with (out / 'raw/memory.jsonl').open('x') as stream:
                while not stop.is_set():
                    stream.write(json.dumps(memory(torch), sort_keys=True) + '\n')
                    stream.flush()
                    stop.wait(0.5)
        sampler = threading.Thread(target=monitor, daemon=True)
        sampler.start()
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        cfg = official.InferenceConfig()
        cfg.vla_path = str(args.checkpoint.resolve())
        result['stage'] = 'model_load'
        load_started = time.perf_counter()
        (official.vla, official.action_head, official.pose_projector, official.device_id,
         official.NUM_PATCHES, tokenizer_action, processor) = official.define_model(cfg)
        torch.cuda.synchronize()
        result['model_load_seconds'] = time.perf_counter() - load_started
        result['after_load'] = memory(torch)
        for name in ('vla', 'action_head', 'pose_projector'):
            module = getattr(official, name).eval()
            result[name + '_parameter_dtypes'] = sorted({str(p.dtype) for p in module.parameters()})
            if any(p.device.type != 'cuda' for p in module.parameters()):
                raise RuntimeError(f'{name} has non-CUDA parameters; offload is prohibited')
        if result['vla_parameter_dtypes'] != ['torch.bfloat16']:
            raise RuntimeError('Full VLA was not loaded in BF16')
        result['attention_implementation'] = official.vla.language_model.config._attn_implementation
        current = Image.open(ROOT / 'inference/current_img.jpg').convert('RGB')
        goal = Image.open(ROOT / 'inference/goal_img.jpg').convert('RGB')
        for name in ('current_img.jpg', 'goal_img.jpg'):
            source = ROOT / 'inference' / name
            with (out / 'raw' / name).open('xb') as stream:
                stream.write(source.read_bytes())
            result[name + '_sha256'] = sha256(source)
        goal_utm = official.utm.from_latlon(37.8738930785863, -122.26746181032362)
        cur_utm = official.utm.from_latlon(37.87371258374039, -122.26729417226024)
        goal_pose = official_goal(goal_utm[0] - cur_utm[0], goal_utm[1] - cur_utm[1], 270, 0)
        engine = official.Inference(str(out / 'derived'), 'move toward blue trash bin',
                                    goal_utm, 0.0, goal, tokenizer_action, processor)

        def batch_for(image, pose):
            # Fixed seed stabilizes the official dummy action token construction.
            np.random.seed(0)
            return engine.data_transformer_omnivla(current, 'xxxx', image, np.asarray(pose),
                       official.PurePromptBuilder, tokenizer_action, processor)

        def forward(batch, pose_only=False):
            official.pose_goal, official.satellite = pose_only, False
            official.image_goal, official.lan_prompt = not pose_only, False
            return engine.run_forward_pass(
                vla=official.vla, action_head=official.action_head, noisy_action_projector=None,
                pose_projector=official.pose_projector, batch=batch, action_tokenizer=tokenizer_action,
                device_id=official.device_id, use_l1_regression=True, use_diffusion=False,
                use_film=False, num_patches=official.NUM_PATCHES, mode='train')

        def record(name, batch, pose, pose_only=False):
            start = utc()
            (actions, modality), seconds = timed_cuda(lambda: forward(batch, pose_only), torch.cuda)
            end = utc()
            raw = immutable_action(actions.detach().float().cpu().numpy())
            # Preserve exact BF16 tensor plus a lossless readable float32 sidecar.
            with (out / 'raw' / (name + '.pt')).open('xb') as stream:
                torch.save(actions.detach().cpu(), stream)
            with (out / 'raw' / (name + '.npy')).open('xb') as stream:
                np.save(stream, raw, allow_pickle=False)
            metric = metric_action(raw)
            entry = {'start_utc': start, 'end_utc': end, 'seconds': seconds,
                     'goal_pose_normalized': list(map(float, pose)),
                     'modality_id': int(modality.item()), 'raw_action': raw.tolist(),
                     'raw_dtype': str(actions.dtype), 'shape': list(raw.shape),
                     'metric_xy_scale': 0.1, 'selected_waypoint_index': 4,
                     'selected_waypoint_metric': metric[0, 4].tolist(), 'memory': memory(torch)}
            try:
                entry['controller'] = official_controller(metric[0, 4])
            except NameError as exc:
                entry['controller_error'] = str(exc)
            write_json(out / 'raw' / (name + '.json'), entry)
            write_json(out / 'derived' / (name + '.json'), {'metric_action': metric.tolist()})
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(metric[0, :, 0], metric[0, :, 1], 'o-')
            ax.scatter([0], [0], marker='x', color='black')
            ax.set(xlabel='Robot x forward (m)', ylabel='Robot y left (m)', title=name, aspect='equal')
            ax.grid(True)
            fig.savefig(out / 'derived' / (name + '.png'), dpi=120)
            plt.close(fig)
            return raw, entry

        result['stage'] = 'first_forward'
        sample_batch = batch_for(goal, goal_pose)
        sample, entry = record('official_sample', sample_batch, goal_pose)
        result['classification'] = 'FULL_BF16_GPU_PASS'
        result['sample'] = 'OFFICIAL_SAMPLE_EXECUTES'
        result['raw_action_shape'] = list(sample.shape)
        result['cold_latency_seconds'] = entry['seconds']
        result['stage'] = 'coordinate_and_modality'
        frozen = json.loads((ROOT / 'research/frozen-goals.json').read_text())
        write_json(out / 'raw/frozen-goals.json', frozen)
        checks = []
        for case in frozen['cases']:
            pose = case['normalized_pose']
            raw, item = record(case['name'], batch_for(goal, pose), pose, True)
            if case['name'] == 'FRONT':
                front = raw
            selected = item['selected_waypoint_metric']
            w = item.get('controller', {}).get('final_vw', [None, None])[1]
            if w is None:
                passed = False
            elif case['name'] == 'FRONT':
                passed = selected[0] > 0 and abs(w) <= 0.15
            elif case['name'] == 'LEFT':
                passed = selected[1] > 0 and w > 0
            else:
                passed = selected[1] < 0 and w < 0
            checks.append({'case': case['name'], 'passed': passed})
        result['coordinate_checks'] = checks
        result['coordinate'] = 'COORDINATE_CONVENTION_PASS' if all(x['passed'] for x in checks) else 'COORDINATE_CONVENTION_UNRESOLVED'
        pose = frozen['cases'][0]['normalized_pose']
        black = Image.new('RGB', goal.size, (0, 0, 0))
        black_raw, _ = record('pose_black_goal', batch_for(black, pose), pose, True)
        difference = float(np.max(np.abs(front - black_raw)))
        result['pose_only_max_abs_difference'] = difference
        result['pose_only'] = 'POSE_ONLY_MODALITY_PASS' if difference == 0.0 else 'POSE_ONLY_MODALITY_UNRESOLVED'
        result['stage'] = 'latency'
        latencies, differences, memories = [], [], []
        for _ in range(args.warm_calls):
            (actions, _), seconds = timed_cuda(lambda: forward(sample_batch), torch.cuda)
            raw = immutable_action(actions.detach().float().cpu().numpy())
            latencies.append(seconds)
            differences.append(float(np.max(np.abs(raw - sample))))
            memories.append(memory(torch))
        result['latency'] = {'cold': entry['seconds'], 'warm_calls': args.warm_calls,
                            **{key: float(value) for key, value in {
                                'min': np.min(latencies), 'median': np.median(latencies), 'mean': np.mean(latencies),
                                'p90': np.percentile(latencies, 90), 'p95': np.percentile(latencies, 95),
                                'max': np.max(latencies)}.items()},
                            'repeat_max_abs_difference': max(differences),
                            'deterministic_in_this_run': max(differences) == 0.0}
        write_json(out / 'raw/latency.json', {'seconds': latencies, 'max_abs_differences': differences, 'memory': memories})
        result['stage'] = 'complete'
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        result['error'] = str(exc)
        result['traceback'] = traceback.format_exc()
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            result['classification'] = 'FULL_BF16_GPU_OOM'
        elif result['stage'] in ('model_load', 'checkpoint_verification'):
            result['classification'] = 'FULL_MODEL_LOAD_FAILED'
        elif result['stage'] not in ('runtime_precondition', 'import'):
            result['classification'] = 'OTHER_WITH_EVIDENCE'
        if load_started is not None and 'model_load_seconds' not in result:
            result['model_load_attempt_seconds'] = time.perf_counter() - load_started
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=10)
        if torch is not None:
            result['final_memory'] = memory(torch)
        result['finished_utc'] = utc()
        write_json(out / 'result.json', result)
        print(canonical_json(result))
    return int(result['classification'] != 'FULL_BF16_GPU_PASS')


if __name__ == '__main__':
    raise SystemExit(main())
