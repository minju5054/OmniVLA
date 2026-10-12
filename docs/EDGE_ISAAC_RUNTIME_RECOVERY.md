# OmniVLA-edge Isaac Runtime Recovery

## Question

Was the previous Isaac 6.0.1 startup failure caused by inherited CUDA/Python
library-path contamination, and does process-local environment isolation allow
the exact same simulator-only startup gate to pass?

**Decision: ISAAC_STARTUP_RECOVERY_FAIL.** The library collision is confirmed,
and isolation removes the observed CUDA symbol error. The unchanged gate reaches
its constructor, scene, RGB and 60-step milestones, and `app.close()` returns.
However, Python then segfaults and the launcher exits 1. Its earlier in-process
`ISAAC_STARTUP_PASS` does not satisfy the required no-segfault criterion.

Measured on 2026-10-12. Starting SHA: `8ddf5c60d9107f8b5f114c68f0fe9777a6f697df`.
Branch: `research/omnivla-edge-isaac-runtime-recovery`. Upstream SHA:
`5182600cb4a9ee07684e17cdd2a6cbafc56b8a68`. Compact evidence:
[`results/edge-isaac-runtime-recovery.json`](../results/edge-isaac-runtime-recovery.json).
Exactly one isolated SimulationApp retry was executed; runtime work stopped there.

## Previous measured failure

The preserved [startup report](EDGE_ISAAC_NO_OBSTACLE.md) records one inherited
environment launch on `/home/gpuadmin/isaacsim`, version
`6.0.1-rc.7+release.42383.32955d8d.gl`. Its first material error was:

```text
libcusparse.so.12: undefined symbol: __nvJitLinkCreate_12_8, version libnvJitLink.so.12
```

The constructor never returned; no task stage, RGB, simulation steps, model
loading or navigation occurred. Extension import failure and SIGSEGV followed;
the stock launcher returned 1. Original raw log SHA256:
`5118da8253219d1ad53467c28a0949839428b1448ee218e27b11d1cf955ea6d9`.
That report and both FULL/Edge preflight reports remain byte-identical.

## Historical TIC-VLA reference

Read-only reference files:
`/home/gpuadmin/Workspace/TIC-VLA/scripts/isaac6_python.sh` and
`/home/gpuadmin/Workspace/TIC-VLA/docs/ISAAC6_COMPATIBILITY.md`.
The 2026-10-09 report described inherited ROS/CUDA paths selecting CUDA 12.6
nvJitLink without the required 12.8 symbol, and a clean-process bare constructor
succeeding on this machine. That run used a different driver, 595.84, and did
not establish the present scene/camera or normal shutdown behavior.
Those results are historical reference, not OmniVLA measurements. Neither that
repository nor its environment was modified. All findings below were rechecked.

## Current inherited environment

Working directory: `/home/gpuadmin/Workspace/OmniVLA`.
GPU: NVIDIA GeForce RTX 5060 Ti, 16311 MiB; driver 595.99.02.
Isaac: `/home/gpuadmin/isaacsim`, `6.0.1-rc.7+release.42383.32955d8d.gl`.
`/usr/local/cuda` resolves to `/usr/local/cuda-12.6`; `nvcc --version` reports
release 12.6, V12.6.20. The driver's displayed CUDA support version is not the
version of the library selected by the dynamic loader.

Inherited `LD_LIBRARY_PATH`:

```text
/opt/ros/jazzy/opt/rviz_ogre_vendor/lib:/opt/ros/jazzy/lib/x86_64-linux-gnu:/opt/ros/jazzy/opt/gz_math_vendor/lib:/opt/ros/jazzy/opt/gz_utils_vendor/lib:/opt/ros/jazzy/opt/gz_cmake_vendor/lib:/opt/ros/jazzy/lib:/usr/local/cuda/lib64:
```

Inherited `PYTHONPATH`: `/opt/ros/jazzy/lib/python3.12/site-packages`.
`PYTHONEXE` and `LD_PRELOAD` were absent. The library path contains ROS Jazzy and
`/usr/local/cuda/lib64`, including a trailing empty entry; no other explicit
CUDA/NVIDIA directories appeared in that variable.

The requested `env | sort` capture is retained only in ignored
`artifacts-local/isaac-runtime-recovery/environment-private.txt`, mode 0600.
It is not committed or reproduced. The selected non-secret environment,
`pwd`, CUDA link, nvcc, nvidia-smi and VERSION evidence is in `host.json` there.
Shell startup files and installed environments were unchanged.

## Library candidates

The prior error names:

```text
/home/gpuadmin/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12
```

Canonical cuSPARSE: `/home/gpuadmin/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/cusparse/lib/libcusparse.so.12`.
Targeted `rg --files --hidden -L` searches covered `/usr/local/cuda*` and the
Isaac installation, including the bundled torch/NVIDIA directories. Aliases
resolve to these three relevant files:

| Canonical file | Bytes | SHA256 |
| --- | ---: | --- |
| `/home/gpuadmin/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/cusparse/lib/libcusparse.so.12` | 387990224 | `437651d1ec5d0e5b1b4d07b8c492f153cb4825222fe2d763232a407f667d9bb2` |
| `/usr/local/cuda-12.6/targets/x86_64-linux/lib/libnvJitLink.so.12.6.20` | 50354760 | `b70f0f325cbd43f37028c559612b49d9c208649ed9184cc24b6e55abf04e341b` |
| `/home/gpuadmin/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib/libnvJitLink.so.12` | 94101392 | `0369e6867d44b800437de4e146d72c65afc6c75adf677a15c2ecd8e6a7ac135f` |

System `libnvJitLink.so.12` is a symlink to `libnvJitLink.so.12.6.20`;
`/usr/local/cuda/lib64` and CUDA target-directory aliases resolve to the system
file above. The bundled cuSPARSE and nvJitLink files are regular files, with no
direct symlink target. Full alias and direct-target records are in the JSON.
No library was replaced, installed, or modified.

## Symbol audit

Read-only `readelf --dyn-syms --wide` established:

| File | Actual symbol evidence |
| --- | --- |
| Bundled cuSPARSE | Undefined reference `__nvJitLinkCreate_12_8@libnvJitLink.so.12` |
| System nvJitLink 12.6.20 | Required symbol absent |
| Bundled nvJitLink | Export `__nvJitLinkCreate_12_8@@libnvJitLink.so.12` present |

This finding comes from ELF symbols, not filenames. `readelf -d` on cuSPARSE
shows `NEEDED libnvJitLink.so.12` and
`RUNPATH [$ORIGIN/../../nvjitlink/lib:$ORIGIN]`.

## Inherited resolution

`ldd` selected `/usr/local/cuda/lib64/libnvJitLink.so.12`, canonically:

```text
/usr/local/cuda-12.6/targets/x86_64-linux/lib/libnvJitLink.so.12.6.20
```

A minimal `/home/gpuadmin/isaacsim/kit/python/bin/python3` process with
`LD_DEBUG=libs` and `ctypes.CDLL` on the exact failing cuSPARSE path reproduced
the missing `__nvJitLinkCreate_12_8` error and exited 1. No SimulationApp,
Torch import, model, or CUDA computation was used in this probe. The external
LD_LIBRARY_PATH entry takes priority over this cuSPARSE RUNPATH.
Raw evidence: `ldd-inherited.txt`, `loader-inherited.stdout`,
`loader-inherited.stderr`, and `loader-inherited.json` in the ignored audit folder.

## Clean-process resolution

The same probes removed `LD_LIBRARY_PATH`, `PYTHONPATH`, and `PYTHONEXE`, and
set `PYTHONNOUSERSITE=1` in the child environment only. `ldd` then resolved the
RUNPATH-relative bundled library, canonically:

```text
/home/gpuadmin/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib/libnvJitLink.so.12
```

The minimal CDLL load succeeded (exit 0). `/proc/self/maps` confirmed both
bundled cuSPARSE and bundled nvJitLink mapped in that process. No replacement
CUDA path or LD_PRELOAD was injected into this probe. Raw evidence:
`ldd-clean.txt`, `loader-clean.stdout`, `loader-clean.stderr`, `loader-clean.json`.

## Diagnostic classification

**CONFIRMED_LIBRARY_PATH_COLLISION**, recorded before the retry:

1. Inherited loader resolution selects the external CUDA 12.6 nvJitLink and
   reproduces the exact previous symbol error.
2. Its ELF exports lack the required 12.8 symbol.
3. The Isaac-bundled nvJitLink exports that symbol.
4. Process-local isolation selects that bundled file and the same CDLL load passes.

This directly supports the CUDA search-path collision. It does not independently
establish whether removing PYTHONPATH or PYTHONEXE is necessary: the wrapper
removes all three together. The subsequent exit-time SIGSEGV has a different
observed location in the lifecycle; its native cause is not established here.

## Process-local launcher

Added [`scripts/isaac6_python.sh`](../scripts/isaac6_python.sh). It uses
`set -euo pipefail`, selects `${OMNIVLA_ISAAC6_ROOT:-${HOME}/isaacsim}`, checks
readable VERSION for `6.0.1` or `6.0.1-*`, and executes:

```bash
exec env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONEXE \
  PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${root}/python.sh" "$@"
```

Before the retry, `bash -n` passed. A fake installation/environment probe
confirmed all three inherited variables were absent at the launcher boundary,
and both required flags were `1`. A harmless real Python probe confirmed the
Isaac executable/version and absence of inherited ROS/system CUDA entries.
These probes did not construct SimulationApp.

The stock Isaac launcher subsequently populates its own LD_LIBRARY_PATH and
PYTHONPATH; those final variables are therefore not literally absent. It also
already sets its own `kit/libcarb.so` preload. No preload override was added
and no stock launcher code was changed. Probe JSON is retained locally and the
compact result records both boundary and real-launch environments.

## Single isolated startup retry

Before launch the following SHA256 values were recorded and rechecked:

| File | SHA256 |
| --- | --- |
| `research/isaac_startup_gate.py` | `46ddb1ae0043dae05c2e30d929ad264172e604e757f20f3e3efb37977bbd5055` |
| `research/isaac_scene.py` | `41c3fb6519bf8724f68857eb2320202678c04634bed39e5bbc5f67a07b49ac88` |
| `research/frozen-edge-isaac-goals.json` | `d1f37af3f136839ac835a3403025d5480e4ef86b7ef55e5d76fc9fa6ebe2684f` |
| `scripts/isaac6_python.sh` | `d71e06f693ecf9bd92051fa350af0e88b4c5d7b7f8fcfefbc76a1fb2e69a49ad` |

The startup script, scene helper and frozen configuration are byte-identical
to the starting commit. The same installation, headless mode,
RaytracedLighting, GPU 0, 640×480 image and 60 startup steps were used.
The exact executed command, from the repository root, was:

```bash
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME="$PWD/artifacts-local/isaac-cache-isolated" \
XDG_DATA_HOME="$PWD/artifacts-local/isaac-data-isolated" \
timeout --signal=TERM --kill-after=15s 300s \
scripts/isaac6_python.sh \
  research/isaac_startup_gate.py \
  --output outputs/edge-isaac-startup-isolated-001 \
  > artifacts-local/isaac-runtime-recovery/startup-isolated-001.log 2>&1
```

Start UTC: `2026-10-12T09:07:21.801055+00:00`.
Finish UTC: `2026-10-12T09:08:11.332696+00:00`.
Host elapsed: 49.531551 s. Timeout did not trigger.
The one-use registration file was created with exclusive creation before launch;
the existing gate also refuses an existing output directory. No second launch,
alternative renderer, Isaac 5 fallback, or other runtime workaround was tried.

| Observation | Measured value |
| --- | --- |
| In-process status | `ISAAC_STARTUP_PASS` |
| Constructor returned | true, 12.112168 s |
| Task stage created | true |
| Camera RGB | 480×640×3, uint8, finite and non-empty |
| RGB standard deviation | 8.959227583606568 |
| Completed steps | 60 |
| Reported simulation time | 1.0333333872258663 s |
| `app.close()` returned / in-process `clean_shutdown` | true |
| Clean process termination | false |
| Launcher exit code | 1 |
| Timeout | false |
| CUDA symbol error | absent |
| Segmentation fault | present, after result emission |

The earliest material failure in this retry is the shell's terminal message:

```text
/home/gpuadmin/isaacsim/python.sh: line 69: 99833 Segmentation fault      (core dumped) $python_exe "${filtered_args[@]}" $args
```

The stock launcher's `error_exit` maps the failed child to exit 1. The raw result
and log are preserved without reinterpretation: the in-process PASS and close
flag were emitted before the native crash. No crash backtrace was collected or
second diagnostic startup attempted after the stop rule.

Raw log: `outputs/edge-isaac-startup-isolated-001/raw/startup.log`,
41126 bytes, SHA256 `03ef9c73a63a2a7bf1973bef6cb7dbdd0fadeea425040a6b3331029d586df8d0`.
In-process result SHA256: `98ffc8cacff56ea40edcb1bfd3906e4379b649337282d6855badceb92664ae23`.
PNG SHA256: `9d83f710678f5f7a473fb46b4dd7e4ae64f672fd40318026acefbd638ec70a5b`.
RGB pixel SHA256: `8f0c2938b92b90a316c0e0c547c44177daaf5fe3962b721a37e7715988082725`.
A separate CPU PNG read verified dimensions, dtype, finite pixels, variation
and the pixel hash. Only the result, PNG and preserved raw log were added under
`outputs/`; the before/after inventory is recorded in the compact result.

## Positive evidence

The current workspace independently reproduces and removes the original CUDA
symbol-resolution error by process-local isolation. The unchanged simulator-only
gate progresses through constructor, stage, camera and all 60 steps. The close
call returns and writes its result. This narrows the remaining failure to process
termination after those successful milestones.

## Negative evidence

The retry ends in SIGSEGV, exit 1. Therefore the required startup recovery PASS
is not established. The in-process PASS alone is insufficient.
No Edge loading/inference, combined memory test, marker calibration, RGB history,
pose-goal inference, STRAIGHT/LEFT/RIGHT navigation, obstacles or reconciliation
was performed. Their outcomes remain unmeasured, not zero or successful.

## Limitations

One isolated retry provides no reliability distribution. The native exit crash
has not been diagnosed. Camera aperture adjustment, DLSS resolution, PCIe,
powersave and other warnings appear in the retained log; none was modified or
promoted to a proven crash cause. A finite varied RGB frame is not camera
calibration or navigation evidence. Peak startup and combined GPU memory were
not measured. Isolation was tested as the specified bundle of environment
changes, not as independent PYTHONPATH and LD_LIBRARY_PATH ablations.

Focused CPU tests cover wrapper behavior, historical preservation, unchanged
experiment hashes, single-result accounting, output scope and classification.
They are software/evidence checks, not additional simulator measurements.
All 24 focused checks passed (16 new plus 8 existing startup evidence tests):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_isaac*.py' -v
```

Validation log: `artifacts-local/isaac-runtime-recovery/validation.log`.
`git diff --check` also passed.

## Decision

**ISAAC_STARTUP_RECOVERY_FAIL**

The library-resolution hypothesis is confirmed, but the overall recovery gate
fails its no-segmentation-fault requirement. Runtime work stops after the one
authorized retry. No experiment parameters, drivers, CUDA installation, system
packages, shell startup files, or other repositories were changed.

## Exact next uncertainty

Can the unchanged simulator-only gate exit without a post-close SIGSEGV under the clean launcher, while retaining the successful constructor, stage, RGB, and 60-step evidence?
