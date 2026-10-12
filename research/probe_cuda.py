"""Measure architecture recognition and real CUDA kernels; no model fallback."""
import argparse
import json
import platform
import traceback
import warnings
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve output first so a rerun cannot replace prior evidence.
    with args.output.open('x') as stream:
        result = {'python': platform.python_version(), 'status': 'GPU_RUNTIME_FAILED'}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                import torch
                result.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                              cuda_available=torch.cuda.is_available(),
                              compiled_arches=torch.cuda.get_arch_list())
                result['gpu'] = torch.cuda.get_device_name(0)
                result['capability'] = list(torch.cuda.get_device_capability(0))
                result['trivial_result'] = (torch.ones(4, device='cuda') + 1).cpu().tolist()
                torch.cuda.synchronize()
                a = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
                result['bf16_matmul_value'] = float((a @ a)[0, 0])
                torch.cuda.synchronize()
                result['status'] = 'GPU_RUNTIME_PASS'
            except Exception as exc:
                result['error_type'] = type(exc).__name__
                result['error'] = str(exc)
                result['traceback'] = traceback.format_exc()
            result['warnings'] = [str(w.message) for w in caught]
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result['status'] != 'GPU_RUNTIME_PASS')


if __name__ == '__main__':
    raise SystemExit(main())
