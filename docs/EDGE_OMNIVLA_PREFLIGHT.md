# OmniVLA-edge Preflight

## Research question

Can geometric / temporal inconsistency between successive online navigation VLA
trajectory chunks be resolved while preserving already executed OLD motion and
the new navigation intent of FRESH? Advisor feedback requires valid upstream
obstacle avoidance before reconciliation research. This report qualifies an
upstream generator only. It contains no navigation or obstacle-avoidance result.

Measured decision: **READY_FOR_ISAAC_ADAPTER**. This task stops before Isaac.
Primary evidence is `outputs/edge-preflight-002/`; compact committed summaries
are under `results/edge-*.json`. Source claims, measurements, interpretations,
and future proposals are identified separately below.

## Why Edge replaced Full OmniVLA

Historical FULL preflight at `3fddf03e89e4c7145ceb241532bf907444b6a87d` loaded
BF16 weights but its first forward OOMed: `FULL_BF16_GPU_OOM` /
`BLOCKED_BY_VRAM`. No successful FULL action exists. Edge is a separately
qualified model, not evidence that FULL inference works. The FULL branch,
weights, environments and evidence are preserved. The historical report is
byte-identical, SHA256
`bb9f0440281f2672104fdcebb08366cf16654b3ff3bafe4972b5ad6d8c747d03`.

## Repository-confirmed facts

Sources inspected: [README](../README.md), [SETUP](../SETUP.md),
[pyproject](../pyproject.toml), [Edge sample](../inference/run_omnivla_edge.py),
[Edge model](../inference/model_omnivla_edge.py),
[policy helpers](../inference/utils_policy.py), training script, configuration,
and GNM, CAST, Frodobots, BDD and LeLaN dataset implementations.
All upstream files remain unchanged. `results/edge-import-audit.json` records
the inference import closure and source hashes.

Architecture, from actual sample configuration and model implementation:

| Component | Source-confirmed configuration |
| --- | --- |
| Observation encoder | EfficientNet-B0, 3 channels per frame; shared over six frames; 1280 features compressed to 1024 |
| Map encoder | Separate EfficientNet-B0, 9 channels: current map, goal map, current RGB; compressed to 1024 |
| Image goal encoder | Separate EfficientNet-B0, 6 channels: current RGB + goal RGB; late fusion false; compressed to 1024 |
| Pose encoder | Linear 4 → 1024 |
| Language | Separate CLIP ViT-B/32 text encoder, 512 features; image-conditioned FiLM network with 8 residual blocks/128 channels; flattened 4096 features → 1024 |
| Decoder | 10 tokens (6 observations, pose, map, image, language), sinusoidal positions; TransformerEncoder 4 layers, 4 heads, width 1024, FF 4096, GELU, norm-first |
| Pool/output | Masked average pooling; append modality ID; 1025 → 1024 → 256 → 128 → 64 → 32, ReLUs; linear action head 32 → 32; separate unused distance output |
| Prediction | `len_traj_pred=8`, four parameters per row; `context_size=5` means six RGB observations |

Input shapes measured: observations `(1,18,96,96)`, goal `(1,3,96,96)`, map
`(1,9,96,96)`, current large image `(1,3,224,224)`, pose `(1,4)`, CLIP tokens
`(1,77)`. Two black 352×352 satellite placeholders are resized to 96×96.
Pixel masks are all-one float32 arrays at 96×96×3 and 224×224×3. Official
normalization is retained exactly, including float image × mask then `/255`.
`encoding_size=256` and `goal_encoding_size=1024` exist in the sample config;
the loader uses `obs_encoding_size=1024` for both actual token widths.

Source modality mapping: 0 satellite, 1 pose+satellite, 2 satellite+image,
3 pose+satellite+image, 4 pose, 5 pose+image, 6 image, 7 language,
8 language+pose. The sample selector also offers 9 language+image, but the
model's `all_masks` has only nine rows (indices 0–8); that path would exceed
the mask table and was not exercised. Qualification covers IDs 7 and 4 only.

## Git provenance

| Item | Verified value |
| --- | --- |
| Repository | `/home/gpuadmin/Workspace/OmniVLA` |
| origin | `https://github.com/minju5054/OmniVLA.git` |
| upstream | `https://github.com/NHirose/OmniVLA.git` |
| Expected and fetched upstream/main | `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68` |
| FULL branch and branch-point commit | `research/full-omnivla-preflight`, `3fddf03e89e4c7145ceb241532bf907444b6a87d` |
| Active branch | `research/omnivla-edge-preflight` |
| Research commit | The commit containing this report; resolve with `git rev-parse HEAD` |

`git fetch --all --prune`, remote checks and an FF-only FULL pull preceded
branch creation. No reset, force push or upstream source edits. The pre-existing
untracked `.vscode/` directory was left untouched and is not staged. Execution
metadata records the pre-commit FULL base plus source hashes; the final
research SHA is recorded by Git and reported after normal push verification.

## Edge checkpoint provenance

Official public [Hugging Face repository](https://huggingface.co/NHirose/omnivla-edge),
resolved through its API before download. It was accessible, not gated or
private; free disk was 1,634,425,389,056 bytes. API listing: `.gitattributes`
(1519 bytes), `README.md` (24), `omnivla-edge.pth` (434,276,465).

| Artifact | Identity |
| --- | --- |
| HF revision | `b1361b7e24f101edea795a98b00a826b61a97394` |
| Checkpoint | `omnivla-edge.pth`, 434,276,465 bytes |
| SHA256 | `f93b056f124380899fce3e087a5c2b41ce4d8c07e211daad52b6ebf02e80d04d` |
| Ignored local path | `models-local/omnivla-edge/b1361b7e24f101edea795a98b00a826b61a97394/omnivla-edge.pth` |
| Download command | `python3 research/fetch_edge_checkpoint.py` |

The downloader pins the revision in its resolve URL, verifies published LFS
size/hash, makes the finished file read-only and verifies an existing artifact
without replacing it. Compact manifest: `results/edge-checkpoint-manifest.json`.
No numerical golden tensor was found in the public repository file list or
source; the existing illustrative JPEG is not numerical parity evidence.

CLIP is a separate model dependency: OpenAI `ViT-B/32`, package
`openai-clip==1.0.1`, 353,976,522 bytes, SHA256
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
Its exact [artifact URL](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt)
is retained in `results/edge-runtime.json`. Cache location:
`models-local/clip/ViT-B-32.pt`. The wrapper supplies only `download_root` to
unchanged `clip.load`, preventing an invisible home-directory cache. The URL's
embedded hash and measured file hash agree. Checkpoint and cache are ignored.

## Minimal runtime / dependency audit

An independent Python 3.10 environment was created at
`artifacts-local/envs/edge-blackwell/`. Source-derived direct requirements:
NumPy 1.26.4, Pillow 11.1.0, Matplotlib 3.10.1, utm 0.8.1, PyYAML 6.0.2,
efficientnet-pytorch 0.7.1, openai-clip 1.0.1, setuptools 78.1.0, Torch and
Torchvision. CLIP imports `pkg_resources.packaging`; setuptools is explicitly
pinned to retain that API. GPU Torch already supplied it; the CPU environment
initially lacked it, and the dependency was supplied before model execution.

`research/requirements-edge.txt` is the minimal manifest;
`scripts/edge_environment.sh` performs dry-run resolution before each install,
then `uv pip check` and freeze. Exact 45-package GPU and 48-package CPU freezes
are committed in `results/edge-*-freeze.txt` (line count is authoritative).
Resolution/install logs remain in `artifacts-local/edge-audit/`.

Exact deviations from official environment: Torch **2.7.1+cu128** replaces
2.2.0, Torchvision **0.22.1+cu128** replaces 0.17.0; Python is uv-managed 3.10.21
rather than a Conda environment. Torchaudio 2.2.0 is omitted because Edge does
not import it. No editable package install: upstream TOML still lacks the comma
between `av` and `openai-clip` (lines 66–67). Unused FULL/training dependencies
are omitted: custom Transformers, TensorFlow/TFDS/TF Graphics, dlimp, PEFT,
FlashAttention and the FULL checkpoint stack. Previously unpinned Edge
libraries are pinned above and transitive versions are frozen. The unreachable
`define_model` function contains leftover FULL names but is not called by the
Edge entry point and does not justify installing its stack. No precision,
architecture, controller, waypoint, modality or checkpoint change was made.

## Blackwell runtime

Measured `BLACKWELL_COMPATIBLE_EDGE_RUNTIME`, not official dependency parity.
Python 3.10.21, Torch 2.7.1+cu128, CUDA runtime 12.8, Torchvision 0.22.1+cu128.
RTX 5060 Ti, 16,311 MiB, compute capability 12.0, driver 595.99.02. The CUDA
probe includes `sm_120` and passed a CUDA arithmetic test and BF16 matrix
multiply. Edge itself executes float32. CUDA matmul TF32 is false; cuDNN
benchmark is false. No custom autocast, compilation, quantization or offload.
The earlier official Torch sm_120 failure was not rerun.

## Optional official-Torch CPU reference

**OFFICIAL_CPU_REFERENCE_UNAVAILABLE**. A separate environment at
`artifacts-local/envs/edge-official-cpu/` used Torch 2.2.0+cu121 and Torchvision
0.17.0+cu121 from the existing download cache, with `CUDA_VISIBLE_DEVICES=''`.
These are CUDA-capable wheels executing only on CPU, not CPU-only wheel builds.
No prior FULL environment was modified. The same Edge and CLIP files were used.

After resolving CLIP's missing setuptools import, exactly one official model
sample was attempted. Strict checkpoint loading and finite-parameter checks
passed. The official forward fails at `model_omnivla_edge.py:272`:
`obs_img.get_device()` is `-1` on CPU, so `self.all_masks.to(device)` raises
`RuntimeError: Device index must not be negative`. This is an upstream CPU
portability issue, not evidence of a GPU semantic failure. It was not patched.
No CPU action exists; max/mean absolute, L2, selected waypoint and command
comparisons are all unavailable, not zero. Raw traceback is preserved at
`outputs/edge-official-cpu-001/error.json`. GPU qualification proceeds under
the explicitly optional reference provision.

## Strict checkpoint load

Measured **EDGE_CHECKPOINT_STRICT_LOAD_PASS** through the unchanged
`utils_policy.load_model`. Its actual `load_state_dict(strict=True)` result was
observed: no missing or unexpected keys. All Edge and CLIP parameters finite.
Edge has **108,305,527 parameters**, all float32 on `cuda:0`; the separately
loaded CLIP model has **151,277,313 parameters**, also float32 after the official
loader conversion. CLIP's unused vision encoder is included, not removed.

Primary run load time with cached files: **1.734094 s**. Initial run including
the CLIP download: 25.225973 s. Load time excludes the subsequent full finite
parameter scan. Before load RSS 811,540,480 bytes; after load 1,740,619,776;
peak process RSS across the primary run 2,678,702,080. Full allocation snapshots
are retained in `results/edge-runtime.json`.

## Official sample inference

Measured **EDGE_OFFICIAL_SAMPLE_PASS** using unchanged
`Inference.run_omnivla`, `run_forward_pass`, loader, transforms and plotter.
The wrapper changes storage paths and adds observation/timing only. Main
configuration is extracted from the official AST. Default flags are
`False, False, False, True` (pose, satellite, image, language), **language ID 7**,
actual instruction **`blue trash bin`**. History is the stopped-robot duplicate
sample, not fabricated motion. Supplied pose after the official overwrite:
`[10, -100, 0.000046326793381, -1]`, float32 (the source uses `3.1415`).
The computed GPS pose is intentionally overwritten; it does not affect this
language-only sample because pose is masked.

Current image SHA256:
`efe5e0c4be8902da94eddae06ebde0bf0fc92c28ad9ff5688fe108cfcba1cc18`.
Goal image SHA256:
`113713bbb970041ffa40e2e88b885f9c1ecbbe54aeaf3b98d6facf971cc15d37`.
Raw input arrays, full pixel masks, hashes and context construction are saved;
compact metadata is in `results/edge-sample-summary.json`. Token order is six
observations, pose, map, image, language. ID 7 attention mask is
`[F,F,F,F,F,F,T,T,T,F]`; corresponding masked average pooling is recorded too.

Native output is finite float32 `(1,8,4)`. Exact raw array:
`outputs/edge-preflight-002/raw/official-native.npy`; a readable complete 8×4
array is committed in `results/edge-sample-summary.json` under `native`.
Selected index 4 native waypoint:
`[4.803703785, -0.324020684, 0.987104356, -0.160078451]`.
Metric waypoint: `[0.480370373, -0.032402068, 0.987104356, -0.160078451]`.
Before clipping `(v,w)=[1.441111118,-0.202050716]`; first clipping
`[0.5,-0.202050716]`; final **`[0.3,-0.12123042941093444]`**.
The wrapper's source-extracted controller exactly equals the actual official
sample return, including NumPy float32 scalar arithmetic.

Official figure:
`outputs/edge-preflight-002/derived/1_ex_omnivla_edge.jpg`, visually inspected.
Current and goal panels, language label and forward/right curve agree with
numeric evidence. Upstream fixes the vertical axis to 10 normalized units,
cropping the last native x=10.5335 point; all eight numeric rows are preserved.
No plot repair or numerical/paper parity claim was made.

From repository root, rerun the sample with a new output name:

```bash
artifacts-local/envs/edge-blackwell/bin/python research/run_edge_preflight.py --sample-only --output outputs/edge-sample-rerun-001
```

## Native action semantics

Source plus runtime: `F_j=[x_j,y_j,h_xj,h_yj]`. **x is front, y is left**, in the
current robot frame. The source plot explicitly documents those axes. The
source GPS transform negates compass, rotates UTM deltas and produces
`[relative_y/0.1,-relative_x/0.1,cos(delta_heading),sin(delta_heading)]`, with a
30 m radius clamp before normalization. Synthetic north-facing FRONT/LEFT/RIGHT
transforms were checked independently of model predictions.

**Returned XY is cumulative**, not increments: Edge's own `forward` applies
`torch.cumsum(action_pred[:,:,:2], dim=1)` before returning. This returned
tensor is the native/raw action. Internal pre-cumsum logits are not exported
as native actions. External metric conversion multiplies XY by **0.1 m** only;
heading is untouched. Heading components represent cosine/sine of local yaw;
`F.normalize(...,dim=-1)` is performed internally. Measured heading norms are
unit within float32 tolerance. Normalization alone cannot promise a unit vector
for a hypothetical exact zero input (PyTorch's epsilon behavior).

Current-origin interpretation is supported by the inference pose convention
and published local-coordinate dataset targets. Precise training provenance
of this Edge checkpoint is not published in the available files, so the
published FULL training path is not claimed to prove its complete training
recipe. **INTRINSIC_WAYPOINT_TIMING_UNRESOLVED**. No row timestamps are added;
eight rows and a 3 Hz controller do not establish a 2.67 s intrinsic horizon.

## Observation-history semantics

Source sample: `context_size=5`, six identical 96×96 current frames, explicitly
assuming the robot stopped. Observed concatenated shape `(1,18,96,96)` and
all six frame contents agree. The current 224×224 image feeds the language
FiLM branch. No history queue is implemented here.

Published training/dataset evidence (not a recovered Edge checkpoint recipe):

| Dataset/path | History/action evidence and limits |
| --- | --- |
| GNM | `gnm_dataset.py:281` selects current−5×spacing through current in `waypoint_spacing` steps; action positions/yaws use the same index stride (`:229–258`), transformed relative to current; dataset-specific metric normalization; no universal seconds-per-index |
| Frodobots | Defaults 10 fps (`:233`); delta timestamps at context/action spacing × 0.1 s (`:281–282`); published caller sets both spacings to 3 (`train_omnivla_dataset.py:1069–1070`), hence 0.3 s for that path; IL poses and 0.25 normalization are dataset-specific |
| BDD | History uses `img_id−ih` clamped at zero (`:141–146`), then current plus five entries (`:205–207`), duplicating current; timestamps are not established; trajectories come from stored MBRA annotations |
| LeLaN | Current plus `iv−ih` frames with boundary clamping (`:308–315`); reverses before concatenation (`:378`); includes duplicate current; NoMaD annotation trajectories and 0.25 pose normalization; no universal seconds-per-index |
| CAST | Published adapter reads current image/state sequence, nine states for eight local actions and archive normalization; it does not establish Edge's six-frame temporal contract |
| MBRA conversion | `robot_pos_model` integrates velocity at 0.333 s (`train_omnivla_dataset.py:322`) with 0.125 spatial normalization; that local conversion is not a universal Edge output clock |

No Edge training entry point is present (search for `OmniVLA_edge` references).
**CONTEXT_TIMING_UNRESOLVED**. A future Isaac adapter must use actual past RGB
observations in the source order and explicitly document its chosen sampling
policy. The stopped-sample duplication does not validate moving-history use.

## Pose-goal coordinate tests

Before pose output was viewed, `research/frozen-edge-goals.json` was created
and SHA256 recorded:
`0c48da8acaa670b0e4a627d11646ddb66727753feecb04c8a7152660e87da88a`.
Each run checks that frozen specification and saves a copy. North-facing test
UTM deltas (east,north) are FRONT `(0,5)`, LEFT `(-5,5)`, RIGHT `(5,5)` meters.
Goal compasses are 0°, −45°, +45°. Native poses are respectively
`[50,0,1,0]`, `[50,50,cos45,sin45]`, `[50,-50,cos45,-sin45]`.

Predeclared FRONT criterion: selected x positive and absolute selected bearing
≤45°. LEFT/RIGHT: selected x positive, selected and mean trajectory lateral
sign matching left/right, and command angular sign matching. Current/history,
checkpoint, masks, language placeholder and goal-image placeholder were fixed.
No signs, thresholds, waypoint index or modality were changed after output.

Measured **EDGE_POSE_GOAL_SANITY_PASS**:

| Test | Selected metric (x,y), m | Final (v,w) | Frozen criterion |
| --- | --- | --- | --- |
| FRONT | (0.480154, −0.002604) | (0.300000, −0.009763) | PASS |
| LEFT | (0.508349, 0.115730) | (0.223371, 0.300000) | PASS |
| RIGHT | (0.514540, −0.104154) | (0.250347, −0.300000) | PASS |

All native 8×4 arrays, metric XY, headings, poses and commands are in
`results/edge-coordinate-summary.json` and separate immutable local raw files.
This is pose-direction sanity on one visual scene, not goal-reaching evidence.

## Pose-only modality

Source-derived pose-only flags: `True,False,False,False`, **ID 4**. Actual
inference returned that ID for all three tests. The pose remains unmasked;
map, image-goal and language tokens are masked. Language is tokenized as `xxxx`
in this mode; those computations still execute upstream. There is no claim
that masked encoders are skipped or that their weights can be removed.

## Masking runtime test

Measured **POSE_ONLY_MASKING_RUNTIME_PASS**. Holding the FRONT pose, current
image/history, model, runtime and all other inputs fixed, the goal-image tensor
was changed from the official image to a black image through the same transform.
Native max absolute difference **0.0**, mean **0.0**, L2 **0.0**; selected
waypoint difference `[0,0,0,0]`, final command difference `[0,0]`.

Source path: all encoders run, token padding mask blocks excluded keys/values,
then masked pooling excludes their query outputs. ID 4 mask is
`[F,F,F,F,F,F,F,T,T,T]`, pooling weights are 10/7 on seven retained tokens and
zero on the last three. Runtime independence was measured for this pair of
placeholders; source intention alone was not treated as proof.

## Official controller

Source-extracted unchanged AST, synthetic CPU tests and exact official sample
return comparison pass on supported branches. Constants: nominal tick **3 Hz**,
zero-based waypoint **4**, scale **0.1**, **DT=1/3**, **EPS=1e−8**, initial clips
**v∈[0,0.5], w∈[−1,1]**, final **maxv=maxw=0.3**. For nonzero dx,
`v=dx/DT`, `w=atan(dy/dx)/DT`, followed by first clipping and ratio-preserving
final limits; the ratio is preserved after the first clip. If linear exceeds
maxv and `abs(w)<=0.001`, the controller emits `[maxv,0]`.

STRAIGHT `[1,0,1,0]` → `[0.3,0]`; LEFT `[1,1,1,0]` → `[0.15,0.3]`;
RIGHT `[1,-1,1,0]` → `[0.15,-0.3]`; zero-x `[0,1,1,0]` → `[0,0.3]`.
**Upstream negative evidence:** zero-XY with heading `[0,0,0,1]` raises
`NameError: clip_angle is not defined`. That branch was not repaired. Synthetic
parity is recorded as preservation of the failure, not successful control.
The successful sample and pose tests do not enter it. A future integration
must explicitly handle this upstream limitation in a separate authorized task.

## GPU memory

Measured **EDGE_GPU_MEMORY_PASS**. No OOM or monotonic allocation growth.
Primary run device-wide usage includes the desktop/other processes, so it is
not Edge-exclusive memory; polling is an observed bound, not an exact GPU peak.

| Stage / metric | Bytes unless marked MiB |
| --- | ---: |
| Before load: allocated / reserved | 0 / 0 |
| Before load: device-wide MiB | 1214 |
| After initial CLIP load: allocated / reserved | 359466496 / 385875968 |
| After initial CLIP load: device-wide MiB | 1685 |
| After checkpoint + CLIP float32 load: allocated / reserved | 1049351680 / 1335885824 |
| After complete load: device-wide MiB | 2666 |
| Forward peak allocated / reserved | 1108572160 / 1344274432 |
| Peak observed device-wide MiB | **2694 (2.631 GiB)** |
| Final process RSS / peak RSS | 2110173184 / 2678702080 |

Device polling requests every 0.1 s plus `nvidia-smi` execution time; 30 samples
in the primary run. Torch peaks reset just before inference, so they describe
forward execution rather than transient loading. Load snapshots are separate.
Warm forward and pipeline allocations/reservations remain stable within the
frozen 16 MiB bound (actual last-ten spans zero). Checkpoint is 434,276,465 bytes.

## Natural inference latency

Measured only after strict load, sample, all three pose tests and masking passed.
No artificial sleep, latency injection or concurrent model job. Shared desktop
activity was not disabled. CUDA synchronized before/after measured boundaries.
Five warmups each precede 50 model-only forwards and 20 complete pipelines.
The model-only test caches CLIP features; the pipeline recomputes them.

| Timing | Cold | Min | Median | Mean | p90 | p95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Edge model, ms | 112.212 | 8.647 | **9.167** | 9.335 | 9.767 | **10.427** | **14.169** |
| Full pipeline, ms | 199.934 | 16.778 | **17.328** | 21.051 | 20.772 | **23.916** | **81.789** |

Cold CLIP + Edge forward: 192.601 ms. Cold pipeline also contains observation
hooks, memory-copy evidence capture and original console formatting, so it is
not an uninstrumented production latency. Warm model excludes CLIP, input
preparation, controller and device-to-host result transfer; CUDA synchronization
is included. Warm pipeline includes current-image disk read, resize/masks,
normalization, pose/maps, tokenization, device transfers, CLIP, unchanged Edge,
CPU action copy, source controller and in-memory official print formatting.
The goal image is already loaded, as in the official initialization. Weights,
plot saving, physical robot I/O, loop wait and filesystem log I/O are excluded.
All durations and scope are in `results/edge-latency-summary.json`.

Repeated output max absolute difference: **0.0** against the frozen official
sample for both warm series. Warm memory stable. The 81.789 ms pipeline maximum
is retained; it was not removed as an outlier. The first run is retained too;
its 17.354 ms median was not selected in preference to the finalized run.

## 3 Hz feasibility

Measured **EDGE_3HZ_RUNTIME_FEASIBLE**. Target interval is 333.333333 ms.
Median 17.328 ms, p95 23.916 ms, **0/20 (0%)** pipelines exceed the target.
Sequential non-overlapping inference can theoretically meet 3 Hz under this
fixed-sample workstation condition, leaving approximately 309 ms after p95
for work not measured here. This is computational preflight only, not measured
closed-loop or Isaac cadence. The official `run()` waits before its tick and
resets its clock after inference; that implementation itself does not guarantee
exact 3 Hz wall-clock updates. No scheduler was changed or tested here.

## Positive evidence

The pinned Edge checkpoint strictly loads; native sample output is finite with
the expected cumulative XY and normalized headings. Frozen pose directions and
pose-only masking pass. The unchanged controller matches on exercised branches.
Memory is stable and measured computation fits the 3 Hz budget. Independent
CPU source/unit contracts and validation of recorded model evidence are named
separately in `tests/test_edge_preflight.py`. Together with preserved FULL tests,
**45 tests passed**. Their passing is not a replacement for recorded GPU runs.

## Negative evidence

Official Torch 2.2 CPU forward cannot complete due to the upstream device-index
assumption, so numerical cross-runtime parity is unavailable. The zero-XY
controller branch is undefined. The sample overwrites GPS pose and duplicates
current history. The official plot clips its last trajectory point. The
published TOML is still invalid. No universal row or history clock and no
complete Edge training recipe were established. These are retained limitations,
not repaired away to obtain a pass.

## Implementation limitations

Additive wrappers observe unchanged source and execute selected AST blocks for
preprocessing/controller/pose transforms; tests and source hashes guard those
boundaries. A narrowly scoped `clip.load` wrapper redirects only its cache.
The initial controller helper promoted NumPy scalars to Python float, producing
an approximately 7e−9 angular discrepancy. It was corrected to preserve scalar
dtype, then the complete qualification was rerun in a new output directory;
`edge-preflight-001` remains immutable. No upstream model/controller changed.

Device-wide memory polling may miss sub-poll transients; Torch counters cover
PyTorch allocations and not every driver allocation. Fifty forwards and twenty
pipelines cover one frozen scene, not long-duration thermal or multi-process
stress. Timing/memory gates remain local to this configuration. CPU reference
absence prevents official-version numerical comparison. Test guards prove normal
Git ignore behavior and no tracked model/cache artifacts; a human `git add -f`
can intentionally bypass Git ignores and is not claimed impossible.

Exact full rerun command, from repository root with a new output directory:

```bash
artifacts-local/envs/edge-blackwell/bin/python research/run_edge_preflight.py --output outputs/edge-preflight-rerun-001
```

Focused verification:

```bash
artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -p 'test_*preflight.py' -v
```

Evidence directories and raw array writes refuse overwrites. Upstream source,
FULL report, frozen specification, native files and input/figure hashes are
recorded in the compact summaries and `results/edge-evidence-manifest.json`.

## Insufficient evidence

There is no Isaac result, real-robot result, goal-reaching test, obstacle
avoidance evidence, moving-history validation, intrinsic 2.67 s horizon,
OLD/FRESH pair, correspondence, alignment, reconciliation, smoothing, training,
quantization, remote inference or paper-level numerical parity. A successful
sample is not navigation; pose sanity is not goal reaching; goal reaching is
not obstacle avoidance; computational cadence is not closed-loop cadence.

## Exact next uncertainty

Can an Isaac adapter preserve the measured pose/frame, real RGB history, and
official command semantics during bounded no-obstacle integration?

## Decision

**READY_FOR_ISAAC_ADAPTER**

Interpretation: the required GPU preflight gates pass with the explicit CPU,
timing and controller limitations above. The next separately authorized task
may qualify a bounded no-obstacle adapter. Isaac was not launched or implemented.
