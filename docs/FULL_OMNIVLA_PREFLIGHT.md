# Full OmniVLA Preflight

## Research question

Can the released FULL OmniVLA model and official inference path be reproduced
on this RTX 5060 Ti machine well enough to justify a later Isaac integration?

Evidence labels used below: **Source** = repository-confirmed fact;
**Measured** = local observation; **Interpretation** = conclusion from evidence;
**Proposal** = untested future work. No navigation, obstacle-avoidance, Isaac,
or successive-chunk experiment is part of this preflight.

## Repository-confirmed facts

**Source.** Full OmniVLA builds on OpenVLA/OpenVLA-OFT. The released checkpoint
configuration names `meta-llama/Llama-2-7b-hf` and the fused DINOv2/SigLIP backbone
`dinosiglip-vit-so-224px` (224×224 inputs). Its model type is `openvla`.
The official FULL loader registers the repository's
`OpenVLAForActionPrediction_MMNv1`, not the edge model.

Authoritative source for all local line references is
[NHirose/OmniVLA at 5182600](https://github.com/NHirose/OmniVLA/tree/5182600cb4a9ee07684e17cdd2a6cbafc56b8a68).
The checkpoint source is
[NHirose/omnivla-original at e36a84d](https://huggingface.co/NHirose/omnivla-original/tree/e36a84d4923c041149d441f93f3bdb7092bb5f07).
No golden numeric sample tensor was found in the audited repository; successful
execution would not mean numerical reproduction of the paper.

## Fork / Git provenance

**Measured.** The real target folder was `EMPTY_DIRECTORY`. Sandbox-created
read-only placeholder directories initially obscured that state; an authorized
host read verified emptiness before cloning into the existing folder.

| Item | Value |
| --- | --- |
| Local workspace | `/home/gpuadmin/Workspace/OmniVLA` |
| Official upstream | `https://github.com/NHirose/OmniVLA.git` |
| Prompt's expected SHA | `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68` |
| Fetched upstream/main SHA | `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68` |
| Upstream commit | `Add publication` |
| Authenticated account | `minju5054` (existing HTTPS credentials; unchanged) |
| Fork / origin | `https://github.com/minju5054/OmniVLA.git` |
| Verified fork parent | `NHirose/OmniVLA` |
| Visibility / default branch | Public / `main` |
| Initial origin/main SHA | Same as fetched upstream/main |
| Research branch | `research/full-omnivla-preflight` |

Research work is additive except appended ignore rules. Official source files
are unchanged. The final research commit is the commit containing this report;
the pushed SHA is verified separately after commit (a file cannot contain its
own final Git commit SHA). Exact commands are in [WORK_LOG.md](WORK_LOG.md).

## Official source audit

All of `README.md`, `SETUP.md`, `pyproject.toml`, `inference/run_omnivla.py`,
`prismatic/vla/constants.py`, and `prismatic/models/action_heads.py` were read.
Training target inspection covered `vla-scripts/train_omnivla_dataset.py`, the
sample trainer/dummy dataset, and GNM, LeLaN, Frodobots, BDD, CAST loaders.

| Contract | Source-confirmed result |
| --- | --- |
| Chunk / action / pose dimensions | `NUM_ACTIONS_CHUNK=8`, `ACTION_DIM=4`, `POSE_DIM=4` |
| Expected batch-one output | `(1,8,4)`; head reshapes action hidden states to eight rows and regresses four values |
| Row layout | `[x,y,hx,hy]`, with heading represented as cos/sin in targets |
| Robot axes | x forward, y left; explicit comment in `run_omnivla.py:260`, reinforced by target construction and flips |
| Inference XY metric scale | `0.1` metres per normalized XY unit, applied outside the model |
| Default actual script modality | image-only goal, ID 6; README incorrectly says language |
| Pose-only modality | pose=True, satellite=False, image=False, language=False, ID 4 |
| Image processing | both current and goal PIL RGB images processed, then concatenated along channels |
| Language disabled | `xxxx` sentinel becomes `No language instruction` in the prompt |
| Goal pose | four-vector appended through a learned pose projector |
| Output normalization | no call to generic action unnormalization in this inference path |
| Hard-coded sample state | current GPS, compass, goal GPS/compass, language prompt and image paths |

`data_transformer_omnivla` creates random dummy actions for label/token layout;
MMNv1 zeros action-token embeddings. The wrapper seeds this construction and
freezes the resulting batch for latency. These dummy labels are not ground truth
for the navigation output.

**Source.** The full loader uses `torch_dtype=torch.bfloat16`,
`low_cpu_mem_usage=True`, then `.to(cuda)`; this is ordinary CPU checkpoint
loading followed by full GPU residency, not CPU inference offload. The VLA and
action head are BF16. The official pose projector is left FP32 and used under
BF16 autocast; the wrapper preserves that behavior. It loads step 120000 action
and pose/proprio projector files. `use_lora=True` exists in config but the loader
does not attach the separate adapter. It also does not load the distance head.

Checkpoint storage is ignored under
`models-local/omnivla-original/e36a84d4923c041149d441f93f3bdb7092bb5f07/`.
[Checkpoint manifest](../results/checkpoint-manifest.json) records downloaded
files, individual bytes, SHA256 verification, total bytes, and omitted unused
artifacts. HF publishes four model shards (roughly 4.93, 4.95, 4.95, 0.263 GB),
an action head, FP32 proprio projector, unused distance head, and LoRA adapter.
The four-shard index reports **15,082,474,368 tensor bytes**; file bytes include
headers. Tokenizer identity is `LlamaTokenizer`, with `<PAD>` ID 32000 and maximum
length 2048. The config names Transformers 4.40.1. HF provenance is pinned to
`e36a84d4923c041149d441f93f3bdb7092bb5f07`, never moving `main` at execution.

## Runtime and dependency audit

**Measured.** Ubuntu 24.04.4 LTS, kernel `7.0.0-38-generic`, AMD Ryzen 7 9800X3D
(8 cores / 16 threads), about 30 GiB RAM and 39 GiB swap; roughly 1.6 TiB disk
free before downloads. GPU is NVIDIA GeForce RTX 5060 Ti, 16,311 MiB reported
VRAM, compute capability 12.0. Driver is 595.99.02, reporting CUDA capability
13.2; installed `nvcc` is 12.6.20. Neither determines the CUDA runtime bundled
with a Torch wheel.

Host evidence is in ignored `artifacts-local/audit/host.json` and `gpu.csv`.
Sandbox GPU visibility failed; GPU measurements therefore run on the host with
the same isolated workspace interpreter. No driver/toolkit/system Python or
other research environment was changed.

Only Python 3.11.16 and 3.12.3 were locally available initially. uv downloaded
Python 3.10.21 to `artifacts-local/python/`. Two distinct environments are used:

- `artifacts-local/envs/official`: official pinned Torch/vision/audio probe set.
- `artifacts-local/envs/blackwell`: explicitly experimental CUDA 12.8 runtime.

**Source.** `pyproject.toml:66–67` lacks a comma between `av` and `openai-clip`;
it is invalid TOML, so unmodified `pip install -e .` is not a reproducible setup.
The inference import closure reaches model/backbone modules and training
utilities through package `__init__` files. It requires `wandb` and `jsonlines`
via metrics even though this is not training. The initial static closure
missed `prismatic.vla.__init__`: an actual import failed on dlimp through
`vla.materialize -> datasets -> rlds.dataset`. That chain imports TensorFlow
and TFDS, and the eagerly imported DROID utilities require TensorFlow Graphics.
These packages are therefore required even for the unchanged inference import.
The Diffusers import in the
action-head file is commented out; FlashAttention is mentioned for training
and is not installed speculatively.

The selected minimal closure is pinned in
[requirements-inference.txt](../research/requirements-inference.txt). The custom
Transformers fork is preserved at
`bc339d9ad707454c0c115970db43c260067c61ab`, with tokenizers 0.19.1,
timm 0.9.10, HF Hub 0.29.1, sentencepiece 0.1.99, draccus 0.8.0 and NumPy 1.26.4.
Unconstrained upstream packages are pinned for repeatability. There is no
blanket editable install and no training dataset download. Following the actual
import failure, the upstream versions TensorFlow 2.15.0, TFDS 4.9.3, and TensorFlow
Graphics 2021.12.3 are added with dlimp commit
`040105d256bd28866cc6620621a3d5f7b6b91b46`. TensorFlow Metadata 1.15.0,
TensorFlow Probability 0.23.0, and protobuf 3.20.3 constrain old-stack
compatibility. [TFP 0.23 release notes](https://github.com/tensorflow/probability/releases/tag/v0.23.0)
confirm compatibility with TensorFlow 2.15. The official RLDS module itself
calls `tf.config.set_visible_devices([], "GPU")`; preserving this import behavior
is not model CPU offload. The VLA still must reside entirely on CUDA.
PEFT is pinned to 0.11.1 upstream but is absent from the inference import closure
and is not installed. Its training compatibility is not claimed. Unused
`json-numpy` is omitted. `einops` enters transitively through the required
TensorFlow-related dependency set; the final freeze records its version.

## Blackwell compatibility

**Measured.** The official `torch==2.2.0+cu121` build detects the GPU but supports
compiled architectures only through sm_90. It warns that RTX 5060 Ti sm_120 is
incompatible. Even `torch.ones(4, device="cuda") + 1` fails with
`CUDA error: no kernel image is available for execution on the device`.
The BF16 matmul check is never reached. Exact warning/traceback is preserved in
[official-cuda.json](../results/official-cuda.json). The wrapper's official
runtime gate consequently records `FULL_GPU_RUNTIME_BLOCKED` without loading
any checkpoint (`outputs/official-runtime-blocked/`).

**Source.** [PyTorch's 2.7 release documentation](https://pytorch.org/blog/pytorch-2-7/)
introduces Blackwell support with CUDA 12.8 wheels. A separate experimental
2.7.1+cu128 environment is therefore measured separately; this does not restore
parity with the official 2.2.0 dependency pin.

**Measured.** `torch==2.7.1+cu128`, torchvision 0.22.1+cu128, torchaudio
2.7.1+cu128 on Python 3.10.21 report CUDA runtime 12.8 and include sm_120.
CUDA arithmetic returns four 2.0 values; BF16 32×32 ones matmul returns 32.0.
No warnings occurred. See [blackwell-cuda.json](../results/blackwell-cuda.json).
Only this environment proceeds to full-model execution. The official-runtime
negative result remains valid and separate.

## Full BF16 VRAM feasibility

**Measured.** **FULL_BF16_GPU_OOM**, at the **first forward pass**, after a
successful full GPU load. The unmodified `define_model` loaded in **2.848262 s**.
VLA and action-head parameters were all BF16 on CUDA; the official pose
projector remained FP32 on CUDA. SDPA was selected by the unchanged loader.
No CPU model parameters, quantization, edge model, remote inference, allocator
retuning, or reduction of the input/model was used.

| Measurement | Before load | After load | At failure / peak |
| --- | ---: | ---: | ---: |
| Torch allocated MiB | 0 | 14,684.934 | 14,693.059 current; **14,842.554 peak** |
| Torch reserved MiB | 0 | 14,728 | **14,926 peak** |
| Device-wide used MiB | 703 | 15,554 | **15,772 observed** |
| Device-wide free MiB | 15,139 | 288 | 70 |
| Process RSS MiB | 1,320.730 | 1,537.848 | 2,443.109 |
| Process peak RSS MiB | 1,320.730 | 6,301.363 | 6,301.363 |
| Host available RAM MiB | 22,701.492 | 22,499.449 | 21,875.484 |

Peak Torch allocation is **14.494682 GiB**, reserved peak **14.576172 GiB**.
The device-wide figure is the maximum observed across periodic and terminal
snapshots, not a guarantee that sub-sample peaks were captured. The OOM reports
15.47 GiB usable device capacity, 69.56 MiB free, and a failed 12 MiB allocation
inside `LlamaMLP.forward` at custom Transformers `modeling_llama.py:240`.
The nominal nvidia-smi total (16,311 MiB) includes memory not all available to
the CUDA allocation pool; current desktop usage is part of this measured
machine condition. This result is not a proof about every possible 16 GB setup.

Model execution stopped immediately on OOM. The exception's allocator advice
was not acted on. [Compact measurement](../results/full-bf16-summary.json)
preserves before/after/peak bytes, versions, source hashes, sample hashes, exact
exception, and stop stage. Full local evidence is `outputs/full-bf16-001/` and
`artifacts-local/audit/full-bf16-001.log`.

## Official sample inference

**Measured.** **OFFICIAL_SAMPLE_BLOCKED**. The official current/goal image
pair was processed by the original transformer/collator, and the first
`run_forward_pass` entered the Llama MLP before OOM. No raw action tensor,
selected waypoint, or final `(v,w)` was returned. Expected shape is `(1,8,4)`;
measured output shape is **unavailable**, not a validated shape pass.

The attempted modality was the actual script default, image-only ID 6. The
sample GPS/compass values are unchanged; reconstructed normalized goal pose is
`[149.00875555,-199.10510136,-1.8369702e-16,-1]`. This reconstructed input is
explicitly distinguished from runtime output in the local derived sidecar.
The NumPy dummy-label seed is 0. Original sample images were not overwritten;
ignored raw copies and their SHA256 hashes were preserved. No trajectory
figure exists because no trajectory was produced. No golden-output or paper
numerical reproduction is claimed.

## Raw action semantics

**Source.** These are normalized robot-local positions relative to the current
observation, rather than per-row motion increments. GNM `_compute_actions`
(`gnm_dataset.py:229–260`) transforms all future positions relative to the same
current position/yaw, takes `waypoints[1:]`, and divides XY by dataset metric
spacing times frame stride. CAST (`cast_dataset.py:48–63`) does the analogous
local-position conversion. Frodobots (`frodobots_dataset.py:585–621`) derives
each future relative transform against the same current origin and divides XY
by 0.25. LeLaN consumes saved `nomad_traj_norm`; BDD consumes saved MBRA targets.
Their cached-generation provenance cannot be established by loading absent
datasets in this task.

The training path (`train_omnivla_dataset.py:309–359`) accumulates rigid motion
transforms `Tacc = Tacc @ Todom` from velocity commands for synthetic MBRA
targets, then emits cumulative local positions and cos/sin heading. It does
not produce a delta sequence that the inference caller should cumulatively
sum. Target selection at lines 476–483 chooses dataset or MBRA target tensors
before regression loss. The action head and inference caller contain no
post-head XY `cumsum`; `cumsum` in training utilities is for token masks.

The heading is a regressed `[cos(theta),sin(theta)]` target representation, not
a scalar yaw. The head does not enforce unit norm. Raw values must be retained.
The `BOUNDS_Q99` constant and inherited manipulation `norm_stats` in config
exist, but the FULL sample calls the regression head directly and performs no
quantile-based unnormalization. Only XY is multiplied by sample scale 0.1
outside the model; heading components remain unchanged. That sample scale is
not a universal training-dataset spacing or a timestamp.

**Source/interpretation.** Some training targets have dataset-specific temporal
sampling (Frodobots default 10 fps × training action stride 3 gives 0.3 s per
target interval; MBRA integration uses 0.333 s).
GNM spacing is dataset-specific, and precomputed LeLaN/BDD target provenance
is only partially exposed. The returned `(8,4)` rows carry no timestamps.
There is insufficient evidence for a universal 8/3-second intrinsic horizon:
**INTRINSIC_WAYPOINT_TIMING_UNRESOLVED**.

**Proposal only.** A later `OmniTrajectoryChunk` may carry `raw_action`,
`poses_local_xy`, `heading_vector`, `observation_time`, `ready_time`,
`application_time`, `frame`, `source`, and `request_id`. The three timestamps
belong to a future Isaac stage, not invented row times. No such framework is
implemented here.

## Pose-goal coordinate convention

**Source.** With UTM delta `(E,N)`, `c=-compass_degrees*pi/180`, the code computes
`rx=E*cos(c)+N*sin(c)`, `ry=-E*sin(c)+N*cos(c)`, clamps radius to 30 m, then uses
`[ry/0.1,-rx/0.1,cos(goal_c-c),sin(goal_c-c)]`. At compass 0° (north), the
robot-local axes are `(N,-E)`. At compass 90° (east), they are `(E,N)`.
Ignoring the final swap/negation would create a sign/frame error.

The frozen cases in [frozen-goals.json](../research/frozen-goals.json) were
declared before model output inspection: FRONT `(3,0)` → `[30,0,1,0]`, LEFT
`(3,2)` → `[30,20,1,0]`, RIGHT `(3,-2)` → `[30,-20,1,0]`. All use pose-only ID 4.
FRONT requires selected x positive and `abs(w)<=0.15`; LEFT requires selected y
and w positive; RIGHT requires both negative. These are limited qualitative
sanity criteria, not navigation success criteria. No post-hoc sign repair is
permitted. Source-transform unit tests alone cannot qualify model behavior.

**Measured limitation.** The FRONT/LEFT/RIGHT source-transform CPU tests pass,
including compass 0/90/270 degrees and the 30 m clamp. Full-model cases were
**not run**, because execution stopped on the earlier OOM.
**COORDINATE_CONVENTION_UNRESOLVED** for model behavior. The compact
[unit results](../results/controller-coordinate-unit.json) explicitly leave
model actions null. No contradiction or agreement of model turn signs was measured.

## Pose-only modality

**Source.** `run_forward_pass` selects ID 4 for pose-only settings.
`PrismaticForConditionalGeneration_MMNv1._build_multimodal_attention_MMN`
(`modeling_prismatic.py:919–1002`) leaves 256 current-image tokens visible,
masks 256 goal-image tokens, and exposes the one pose token. Both images are
still encoded before the attention mask, so a placeholder tensor is required.
The disabled-language sentinel text remains in the attention mask; this is
not a physically absent text sequence. A CPU mask test checks the source
function, but cannot prove full-model masked-input independence.

**Measured limitation.** The actual mask function passes the CPU unit test,
but the full-model black-goal versus sample-goal comparison was **not run**.
Raw-action difference is **unavailable**; masked-input independence is not
asserted. **POSE_ONLY_MODALITY_UNRESOLVED**.

## Controller constants

**Source.** `inference/run_omnivla.py:86–240` defines:

| Item | Official value / rule |
| --- | --- |
| `tick_rate` | 3 Hz loop threshold; sample breaks after one tick |
| `waypoint_select` | 4 (zero-based; fifth of eight) |
| `metric_waypoint_spacing` | 0.1 m per normalized XY unit |
| `DT` | 1/3 s controller formula constant |
| `EPS` | 1e-8 |
| General pre-limit command | `v=dx/DT`, `w=atan(dy/dx)/DT` (not atan2) |
| dx≈0 with dy nonzero | `v=0`, `w=sign(dy)*pi/(2*DT)` |
| Both XY≈0 | intended `v=0`, `w=clip_angle(atan2(hy,hx))/DT`; undefined function |
| First clip | v to [0,0.5], w to [-1,1] |
| Final limits | `maxv=0.3`, `maxw=0.3`, preserve clipped v/w ratio when limited |
| Near-straight high-v branch | if `abs(w)<=0.001`, clamp v and set w=0 |

**Source/negative evidence.** No definition/import for `clip_angle` exists in
the FULL script. ZERO_XY_WITH_HEADING raises `NameError` in unchanged source;
there is no valid numerical result for this branch. The helper extracts and
executes original AST statements instead of substituting a repaired formula.
STRAIGHT `(0.1,0,1,0)` gives `(0.3,0)`; LEFT `(0.1,0.1,1,0)` gives `(0.09,0.3)`;
RIGHT mirrors w to -0.3, after intermediate clipping. The model's heading is
otherwise unused by the selected-waypoint controller.

These are controller-unit checks, not motion or navigation evidence. The 3 Hz
threshold does not establish model latency, row timestamps, or a 2.67 s horizon.

## Natural inference latency

**Not measured.** The first forward pass failed; there is no completed cold
latency, and zero warm calls were executed. Min/median/mean/p90/p95/max and
repeated-output differences are all unavailable. The 2.848262 s checkpoint-load
time is not inference latency. Neither deterministic output nor memory
stability over repeated calls has been established.

## Positive evidence

- Verified official Git source, exact parented fork, remotes, and research branch.
- Pinned full checkpoint downloaded with all required file sizes/hashes verified.
- Experimental Blackwell CUDA arithmetic/BF16 matmul pass; original FULL module
  imports after installing its measured eager dependency closure.
- Full BF16 GPU residency and checkpoint/head loading succeeded, with official
  FP32 pose-projector behavior preserved.
- All 19 focused CPU source/evidence tests pass in both Torch environments.
- Source confirms frame/sign, cumulative local-position targets, heading-vector
  representation, default/pose-only IDs, and controller formulas.

## Negative evidence

- Official Torch 2.2.0+cu121 cannot execute even a trivial kernel on sm_120.
- Full BF16 first forward OOMs under the measured local GPU conditions.
- No sample output, coordinate behavior, modality independence, or latency result.
- Upstream packaging TOML is invalid; eager imports unexpectedly require the
  TensorFlow/RLDS stack. Initial incomplete closure and failed dependency
  resolution are retained in the work log, with corrections.
- README's modality default disagrees with executable source.
- Upstream zero-XY heading controller branch raises `NameError: clip_angle`.
- Universal intrinsic waypoint timing is not established.

## Implementation limitations

- Upstream TOML cannot be parsed; the task uses an additive inference dependency
  list and direct repository imports. This is explicitly not full environment parity.
- The controller's zero-XY heading branch is broken upstream and remains broken.
- Training data and external target generators are not downloaded. Source audit
  does not validate every cached target or dataset coordinate annotation.
- Coordinate and modality CPU tests exercise formulas/masks, not a navigation model.
- A sample execution, if available, would still provide no obstacle-avoidance
  evidence and no golden-tensor or paper-result reproduction.
- The wrapper calls unchanged official `define_model`, image transformation,
  collator, and `run_forward_pass`; it saves separate evidence rather than running
  the script's busy-wait loop or overwriting its sample visualization.
- The wrapper's later model coordinate/modality/latency paths remain unexecuted
  after the OOM; their presence is not a runtime qualification.
- Evidence is local and ignored; compact verified provenance/results are committed.

## Insufficient evidence

No runtime action tensor exists, so action semantics are source-supported
but not qualified through a successful full-model output. There is no measured
FRONT/LEFT/RIGHT behavior, pose-only dummy-image independence, output
determinism, successful forward latency, navigation, obstacle avoidance, or
Isaac compatibility. A model-load pass is not a forward-pass pass.

Exact environment deviations are explicit: Torch 2.2.0 → 2.7.1+cu128,
torchvision 0.17.0 → 0.22.1+cu128, torchaudio 2.2.0 → 2.7.1+cu128, and wheel
CUDA 12.1 → 12.8; uv-managed Python 3.10.21 venvs replace the README's Conda
workflow. Invalid editable packaging is bypassed with explicit repository
imports and requirements. Upstream unpinned dependencies are fixed to concrete
versions; TFP 0.23.0/TF Metadata 1.15.0/protobuf 3.20.3 are compatibility choices.
The custom Transformers and dlimp forks are pinned to recorded commits. Unused
PEFT, FlashAttention, Diffusers, datasets, and other application/training extras
are omitted. The [final experimental freeze](../results/runtime-blackwell-freeze.txt)
and [official Torch probe freeze](../results/runtime-official-freeze.txt) record
all installed versions. `uv pip check` reports all 130 installed distributions
compatible. The model source, checkpoint, precision, and controller are unchanged.

## Exact next uncertainty

How much additional usable GPU memory is required for the unchanged FULL
BF16 sample forward to complete? That must be resolved before the still-untested
coordinate and pose-only gates, and before any Isaac adapter decision.

## Decision

**Interpretation.** This machine has not reproduced the official FULL sample
well enough to justify an Isaac adapter. Loading succeeded in the explicitly
experimental runtime, but first forward failed. All model execution stopped at
the OOM gate; remaining work was documentation, source tests, and Git finalization.
No Isaac integration or obstacle experiment was started.

To inspect/reproduce the completed non-model tests:

```bash
cd /home/gpuadmin/Workspace/OmniVLA
artifacts-local/envs/blackwell/bin/python -m unittest discover -s tests -v
```

The exact failed BF16 invocation was:

```bash
artifacts-local/envs/blackwell/bin/python research/run_preflight.py --output outputs/full-bf16-001
```

That evidence path refuses overwrite. For an explicitly authorized future rerun,
use a **new** output directory (the tested configuration is expected to OOM):

```bash
artifacts-local/envs/blackwell/bin/python research/run_preflight.py --output outputs/full-bf16-002
```

The wrapper uses the original FULL loader and forward functions plus official
images; it never substitutes reduced precision or a smaller model. Fresh setup
commands are in `scripts/preflight_environment.sh` and the append-only work log.
A runtime probe can be repeated with `research/probe_cuda.py --output <new.json>`;
checkpoint re-verification uses `python3 research/fetch_checkpoint.py` and reuses
matching existing files. GPU commands need host device access, as this session's
sandbox hides the driver.

BLOCKED_BY_VRAM
