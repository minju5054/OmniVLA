# Work log (append-only)

## 2026-10-12 — initialization and provenance

- Request: full OmniVLA local BF16 preflight; fork, commit, and normal push are
  explicitly authorized. No previous research repository was modified.
- `pwd; ls -la` inside the sandbox showed protected empty placeholder directories
  (`.git`, `.agents`, `.codex`, `.aws`); Git reported no repository. Repeating the
  read-only inspection outside the sandbox showed the actual folder was empty.
  Classification: `EMPTY_DIRECTORY`. No files moved, removed, or overwritten.
- Before Git mutation: `/home/gpuadmin/Workspace/OmniVLA`, empty actual folder;
  Git status, remotes, HEAD, and log not applicable.
- `git clone https://github.com/NHirose/OmniVLA.git .`
- `git status --short; git remote -v; git rev-parse HEAD; git log -1 --oneline`
- `git fetch origin main; git rev-parse origin/main`
- Expected and fetched upstream SHA both
  `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68` (Add publication).
- `gh auth status; gh api user --jq .login`: active user `minju5054`, HTTPS;
  credentials unchanged. No credential contents saved.
- `gh repo view minju5054/OmniVLA --json nameWithOwner,parent,defaultBranchRef,isPrivate,url`
  initially reported the repository absent.
- `gh repo fork NHirose/OmniVLA --clone=false --remote=false`
- Repeated `gh repo view` verified public `minju5054/OmniVLA`, parent
  `NHirose/OmniVLA`, default `main`, URL https://github.com/minju5054/OmniVLA.
- `gh api repos/minju5054/OmniVLA/commits/main --jq .sha`: same upstream SHA.
- `git remote rename origin upstream`
- `git remote add origin https://github.com/minju5054/OmniVLA.git`
- `git remote -v; git fetch --all --prune; git rev-parse origin/main upstream/main`
  verified both remote main revisions equal the expected SHA.
- `git switch -c research/full-omnivla-preflight upstream/main`
- Initial sandbox `nvidia-smi` could not access the driver; host inspection
  succeeded. This is a sandbox visibility issue, not GPU incompatibility.
- Host GPU: RTX 5060 Ti, 16311 MiB, driver 595.99.02, driver CUDA capability 13.2.
  Toolkit `nvcc` 12.6.20. No system software changed.
- `uv python list --only-installed --no-config` found Python 3.11.16 and 3.12.3,
  no Python 3.10. Available isolated option: uv-managed Python under ignored
  workspace storage. User informed before proceeding.
- Source inspection found invalid upstream TOML (missing comma before
  `openai-clip`) and README/script disagreement on default goal modality.
  Upstream code will remain unchanged; inference requirements will be explicit.
- Primary metadata: HF `NHirose/omnivla-original` revision
  `e36a84d4923c041149d441f93f3bdb7092bb5f07`; custom Transformers upstream HEAD
  `bc339d9ad707454c0c115970db43c260067c61ab`.

## 2026-10-12 — isolated runtime and source audit preparation

- Added research rules/context, ignored local storage, source-derived helpers,
  focused unit tests, a pinned checkpoint fetcher, and an additive FULL wrapper.
  No official Python/model/controller file was changed.
- Source hashes recorded in `results/source-provenance.json`; checkpoint hashes
  and sizes recorded in `results/checkpoint-manifest.json`.
- Host inspection commands: `nvidia-smi`, `nvcc --version`, `uname -a`,
  `cat /etc/os-release`, `lscpu`, `free -h`, `df -h . /tmp`, Python version
  commands, `uv --version`, `uv python list --only-installed --no-config`.
- `UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/artifacts-local/python" uv python install 3.10 --no-config`
  installed isolated CPython 3.10.21. Log: `artifacts-local/audit/python-install.log`.
- `bash scripts/preflight_environment.sh official > artifacts-local/audit/official-install.log 2>&1`
  initially used PyPI, resolved the official 2.2.0/0.17.0/2.2.0 versions, but
  large wheel transfers became very slow. The task-owned uv process was
  identified by `/proc/<pid>/cmdline` and terminated with SIGTERM. The partial
  cache and original log were retained. This was not a CUDA compatibility result.
- A 2 MiB range-read from the official Torch 2.2.0+cu121 wheel host took 3.49 s.
  Retry used the same requested versions with the explicit CUDA 12.1 index:
  `UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" uv pip install --no-config --python artifacts-local/envs/official/bin/python --dry-run numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121`
  and the same command without `--dry-run`. Log:
  `artifacts-local/audit/official-cu121-install.log`. The setup script now uses
  this exact official wheel source and refuses to overwrite existing environments.
- `UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" uv pip install --no-config --python artifacts-local/envs/official/bin/python --dry-run -e .`
  confirmed the upstream TOML parse failure at line 67. A concurrent environment
  lock wait was also logged; no editable install occurred.
- Inference import closure inspected through `prismatic` model/backbone,
  training utility, metrics, and package initializer imports. No TensorFlow,
  TFDS, dlimp, datasets, PEFT, FlashAttention, json-numpy, or einops install was
  justified by this closure; these remain omitted. wandb/jsonlines are required
  by eagerly imported metrics. Requirements are in `research/requirements-inference.txt`.
- Hugging Face metadata read using its public API with `blobs=true`; no weights
  were present in the standard full-model cache. Free disk checked before fetch.
- `python3 research/fetch_checkpoint.py > artifacts-local/audit/checkpoint-download.log 2>&1`
  downloaded and verified 20 files, 15,353,995,680 bytes. Four model shards total
  15,082,600,832 file bytes; tensor metadata reports 15,082,474,368 bytes.
  Individual LFS SHA256 hashes matched. Unused adapter and distance-head weights
  were not downloaded. No model/cache/environment data is tracked.
- The pinned custom Transformers `modeling_llama.py` was fetched read-only to
  `artifacts-local/audit/transformers-modeling-llama.py` for attention review.
  Its SDPA path repeats the final mask row and sets `is_causal=False`; masked
  columns remain masked. Full-model independence is still a measurement gate.
- FRONT/LEFT/RIGHT inputs and pass criteria were frozen in
  `research/frozen-goals.json` before any full-model execution.
- Source issues preserved: script default is image-only ID 6 (README says
  language); zero-XY controller calls undefined `clip_angle`; target timing is
  heterogeneous (Frodobots defaults 10 fps with training action stride 3,
  versus MBRA integration dt 0.333 s). No universal row timing asserted.
- `python3 -m compileall -q research tests`, `bash -n scripts/preflight_environment.sh`,
  and `git diff --check` passed syntax/whitespace checks at preparation time.

## 2026-10-12 — official CUDA negative result and CPU contracts

- `artifacts-local/envs/official/bin/python research/probe_cuda.py --output artifacts-local/audit/official-cuda.json`
  on the host measured Torch `2.2.0+cu121`, CUDA runtime 12.1, Python 3.10.21.
  GPU detection succeeded (capability 12.0), but the compiled architectures
  stop at sm_90. The trivial `torch.ones(4, device='cuda') + 1` failed with
  `CUDA error: no kernel image is available for execution on the device`.
  Warning explicitly says sm_120 is incompatible. Exact output is preserved
  in `results/official-cuda.json`; BF16 matmul was not reached.
- `UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" uv pip freeze --python artifacts-local/envs/official/bin/python > artifacts-local/audit/official-freeze.txt`
  saved the official probe environment. It is a Torch subset, not a successfully
  installed full upstream dependency environment (upstream TOML is invalid).
- `artifacts-local/envs/official/bin/python -m unittest discover -s tests -v`
  passed all 19 CPU tests in 0.509 s. This includes expected `NameError` for
  upstream ZERO_XY_WITH_HEADING, not a claim that this controller branch works.
  No navigation evidence is inferred.
- In accordance with the authorized fallback-runtime gate:
  `bash scripts/preflight_environment.sh blackwell > artifacts-local/audit/blackwell-install.log 2>&1`
  creates a distinct Python 3.10 environment with Torch 2.7.1, torchvision
  0.22.1, torchaudio 2.7.1 from the official cu128 wheel index. This is an
  explicit experimental runtime deviation, with full BF16 unchanged.

## 2026-10-12 — experimental CUDA pass and inference resolution

- Blackwell package installation completed, but the surrounding shell reported
  `hon: command not found` after installation: the setup script had been edited
  while that Bash invocation was still reading it. No package install was
  rolled back or hidden. The current script passes `bash -n`; subsequent
  probes independently verified the actual installed runtime. No running
  setup script will be edited again during this task.
- `artifacts-local/envs/blackwell/bin/python research/probe_cuda.py --output artifacts-local/audit/blackwell-cuda.json`
  measured Torch 2.7.1+cu128, CUDA runtime 12.8, Python 3.10.21. Compiled
  architectures include sm_120. Trivial CUDA arithmetic and BF16 matrix
  multiplication both passed without warnings. Saved in `results/blackwell-cuda.json`.
- `UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" uv pip compile --no-config --python-version 3.10 --torch-backend cu128 --constraint research/constraints-blackwell.txt research/requirements-inference.txt -o artifacts-local/audit/inference-resolution.txt`
  resolved successfully before inference installation. No TensorFlow, TFDS,
  dlimp, PEFT, or flash-attn package appeared in the resolution.
- `bash scripts/preflight_environment.sh inference > artifacts-local/audit/inference-install.log 2>&1`
  performs a second explicit dry run and then installs only that inference
  closure. The Torch family is constrained to the already measured cu128 set.
  NumPy selected transitively during the base Torch install is explicitly
  returned to upstream SETUP's 1.26.4; Pillow is pinned to 11.1.0.
- CPU source results, with null model outputs, saved separately in
  `results/controller-coordinate-unit.json`.
- `artifacts-local/envs/official/bin/python research/run_preflight.py --runtime-probe artifacts-local/audit/official-cuda.json --output outputs/official-runtime-blocked`
  verified that the failed official-runtime gate refuses checkpoint loading.

## 2026-10-12 — measured import correction (supersedes initial closure assumption)

- The first actual unchanged FULL-module import failed with
  `ModuleNotFoundError: No module named 'dlimp'`. The traceback in
  `artifacts-local/audit/inference-import.log` proves this eager chain:
  `prismatic.vla.__init__ -> materialize -> datasets -> rlds.dataset`.
  Static inspection had missed this initializer; the earlier claim that the
  inference closure did not require the TensorFlow stack was incorrect.
- Further source inspection shows `rlds.dataset` imports TensorFlow/TFDS,
  and `rlds.oxe.utils.droid_utils` imports TensorFlow Graphics. This satisfies
  the user's explicit prerequisite for installing otherwise training-related
  dependencies. No import shims or upstream semantic edits were substituted.
- `git ls-remote https://github.com/moojink/dlimp_openvla.git HEAD` returned
  `040105d256bd28866cc6620621a3d5f7b6b91b46`; its setup pins TensorFlow 2.15.0.
- Added official TensorFlow 2.15.0, TFDS 4.9.3, TensorFlow Graphics 2021.12.3,
  and that exact dlimp revision. Constrained TensorFlow Probability to 0.23.0
  (official release documents TF 2.15 compatibility), TensorFlow Metadata to
  1.15.0, and protobuf to 4.25.8. These are explicit transitive choices, not
  claims of full upstream environment parity.
- `bash scripts/preflight_environment.sh inference > artifacts-local/audit/inference-rlds-install.log 2>&1`
  dry-runs the corrected closure, then installs it in the experimental env.
  The previous install and failed-import logs remain intact.
- Upstream `prismatic/vla/datasets/rlds/dataset.py:35` itself calls
  `tf.config.set_visible_devices([], "GPU")`. Preserving that behavior is
  unrelated to VLA CPU offload; the model wrapper rejects non-CUDA parameters.

- Corrected-closure dry run rejected protobuf 4.25.8: TensorFlow Metadata 1.15.0
  requires `protobuf>=3.20.3,<4.21` on Python below 3.11. No install occurred
  from that failed resolution. Changed the explicit protobuf pin to 3.20.3.
- `bash scripts/preflight_environment.sh inference > artifacts-local/audit/inference-rlds-install-v2.log 2>&1`
  retries dry-run/install with that compatible intersection. Earlier failure
  remains recorded; it is dependency-resolution evidence, not a model failure.

## 2026-10-12 — completed full BF16 gate; mandatory model-execution stop

- Corrected dependency installation succeeded. `inference-import-v2.log` proves
  the unchanged FULL module imports, with TensorFlow/TFDS/Graphics/dlimp eagerly
  loaded and TensorFlow visible GPUs empty. TensorFlow emits duplicate CUDA
  factory/missing optional GPU library warnings, but the import succeeds.
  No system libraries were changed to suppress those messages.
- einops entered transitively through the now-required TensorFlow-related stack;
  the previous proposed omission no longer describes the final environment.
- `UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" uv pip freeze --no-config --python artifacts-local/envs/blackwell/bin/python > artifacts-local/audit/blackwell-final-freeze.txt`
- `UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" uv pip check --no-config --python artifacts-local/envs/blackwell/bin/python`
  checked 130 distributions and reported all compatible.
- `artifacts-local/envs/blackwell/bin/python research/run_preflight.py --output outputs/full-bf16-001 > artifacts-local/audit/full-bf16-001.log 2>&1`
  executed on host GPU access. Local time: 2026-10-12 16:21:47–16:22:05 KST
  (raw evidence uses UTC 07:21:47–07:22:05).
- Full VLA and action head loaded BF16, pose projector FP32 as upstream, all
  parameters on CUDA. No offload/quantization/model substitution. Load wall
  time 2.848261953 s. After-load torch allocated/reserved:
  15,398,269,440 / 15,443,427,328 bytes; device-wide used 15,554 MiB.
- First official image-goal sample forward failed inside LlamaMLP with a CUDA
  OOM requesting 12 MiB. Classification: FULL_BF16_GPU_OOM. Peak Torch allocated
  15,563,546,112 bytes (14,842.554 MiB / 14.494682 GiB); peak reserved
  15,651,045,376 bytes (14,926 MiB). Maximum observed device-wide usage,
  including the terminal snapshot, 15,772 MiB. Before load: 703 MiB device-wide,
  Torch allocated/reserved zero. See immutable original result and memory JSONL.
- STOP: no further model forward, pose-goal model test, dummy-image comparison,
  warm benchmark, allocator tuning, precision change, CPU offload, remote
  inference, or Isaac work after OOM. No successful raw action exists.
- OFFICIAL_SAMPLE_BLOCKED; coordinate behavior and pose-only independence remain
  unresolved; all latency statistics and determinism are unavailable.
- `results/full-bf16-summary.json` is a compact copy/summary with a SHA256 pointer
  to the original local result. Unit/source-derived input records are explicitly
  separate from model outputs. No trajectory figure was generated, so no
  trajectory figure visual verification is applicable.
- Final scientific decision: BLOCKED_BY_VRAM under the measured local GPU state.

## 2026-10-12 — final non-model validation and Git preparation

- `git status --short; git diff --check; git diff; git diff --cached` reviewed
  before staging. The only changed pre-existing file is additive `.gitignore`.
  All official Python files and sample images remain unchanged.
- Verified every runtime source hash in the raw result still matches the final
  wrapper/helper and official source files. The run occurred before the first
  research commit, so its `git_head` is upstream; recorded content hashes bind
  the executed additions to the committed implementation.
- `artifacts-local/envs/blackwell/bin/python -m unittest discover -s tests -v`
  passed 19/19 in 0.636 s in the final dependency environment. Log:
  `artifacts-local/audit/unit-tests-final.txt`. These are CPU contract checks;
  no model execution resumed after the OOM.
- All intended code/doc/result additions are below 100 KB each. Checkpoints,
  environments, downloaded caches, and generated evidence remain ignored.
- Read-only post-exit `nvidia-smi` recorded 798 MiB used / 15045 MiB free,
  confirming that the failed model process no longer occupies its prior VRAM.
- Git finalization uses explicit file staging, diff/size checks, the requested
  commit message, then normal `git push -u origin research/full-omnivla-preflight`.
  Remote branch SHA is compared with local HEAD after push. No force push,
  history rewrite, main-branch research commit, or Isaac work is authorized here.

## 2026-10-12 — Git author identity repair

- The first `git commit` was rejected because no Git author identity was
  configured. The following normal push in that shell invocation created the
  remote research branch at the upstream SHA without a research commit; no
  source/history was rewritten. The staged research changes remained intact.
- Read-only `gh api user --jq '{login: .login, id: .id, name: .name, email: .email}'`
  reconfirmed minju5054, account ID 112682808, no public name/email.
- Use repository-local author name `minju5054` and the GitHub account's standard
  ID-based noreply address `112682808+minju5054@users.noreply.github.com`.
  This is commit metadata, not an authentication credential change; global Git
  configuration and all other repositories remain untouched.
- Retry commit and push with fail-fast shell execution. The push is a normal
  fast-forward from the initial remote research branch. Final local/remote
  equality and clean status are checked after the push, outside the self-
  referential committed report.

## 2026-10-12 — OmniVLA-edge preflight (new authorized task)

The user selected Edge after historical FULL_BF16_GPU_OOM / BLOCKED_BY_VRAM.
Scope: source/provenance, isolated inference, native actions, pose/masking,
controller, memory and computational 3 Hz only. No Isaac or reconciliation.
Preserved FULL weights, environments, branch, evidence and report. Initial
status had only untracked `.vscode/`, which remains untouched.

Repository commands (host escalation needed for protected Git and networking):

```bash
pwd
git status --short
git remote -v
git branch --show-current
git rev-parse HEAD
git log -1 --oneline
git fetch --all --prune
git rev-parse upstream/main
git rev-parse origin/research/full-omnivla-preflight
git branch --list research/omnivla-edge-preflight
git switch research/full-omnivla-preflight
git pull --ff-only origin research/full-omnivla-preflight
git switch -c research/omnivla-edge-preflight
sha256sum docs/FULL_OMNIVLA_PREFLIGHT.md
```

Origin/upstream identities and FULL commit matched the request. Fetched upstream
remained 5182600cb4a9ee07684e17cdd2a6cbafc56b8a68. FULL report hash was
bb9f0440281f2672104fdcebb08366cf16654b3ff3bafe4972b5ad6d8c747d03 and is unchanged.
Updated AGENTS and current context additively to preserve FULL as history and
make Edge preflight active. No original inference/training source changed.

Read README, SETUP, pyproject, all three Edge inference files, relevant training
and GNM/CAST/Frodobots/BDD/LeLaN data construction. Verified six duplicates,
language ID 7, test pose overwrite, pose ID 4, internal XY cumsum, heading
normalization and source controller. No Edge trainer found; no universal action
or history clock inferred from FULL training examples. Strictly kept native
outputs before external metric scaling and attached no intrinsic timestamps.

Verified rebooted GPU through host `nvidia-smi`: RTX 5060 Ti, driver 595.99.02,
16311 MiB, later runtime capability 12.0. Did not rerun the old official Torch
GPU failure. Public HF API `https://huggingface.co/api/models/NHirose/omnivla-edge?blobs=true`
was read with Python urllib before downloading; response preserved at
`artifacts-local/edge-audit/hf-api.json`. API file list/size/access and 1.634 TB
free space were inspected. Revision b1361b7e24f101edea795a98b00a826b61a97394;
checkpoint 434276465 bytes, LFS SHA256 f93b056f124380899fce3e087a5c2b41ce4d8c07e211daad52b6ebf02e80d04d.
The first metadata command used absent `python`; it performed no download and
was immediately rerun with `python3`.

Exact setup/execution commands, from repository root:

```bash
python3 research/fetch_edge_checkpoint.py > artifacts-local/edge-audit/download.log 2>&1
bash scripts/edge_environment.sh blackwell > artifacts-local/edge-audit/environment-blackwell.log 2>&1
bash scripts/edge_environment.sh official-cpu > artifacts-local/edge-audit/environment-cpu.log 2>&1
sha256sum research/frozen-edge-goals.json > artifacts-local/edge-audit/frozen-goals-before-runtime.sha256
UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" uv pip freeze --no-config --python artifacts-local/envs/edge-blackwell/bin/python > results/edge-blackwell-freeze.txt
UV_CACHE_DIR="$PWD/artifacts-local/uv-cache" uv pip check --no-config --python artifacts-local/envs/edge-blackwell/bin/python
artifacts-local/envs/edge-blackwell/bin/python research/probe_cuda.py --output artifacts-local/edge-audit/cuda-probe.json
artifacts-local/envs/edge-blackwell/bin/python research/run_edge_preflight.py --output outputs/edge-preflight-001 > artifacts-local/edge-audit/preflight-001.log 2>&1
```

Setup scripts perform dry-runs, install and pip checks. Used cached Torch wheel
families, new envs only. Minimal source closure omitted torchaudio and the FULL
TF/Transformers stack. A standalone initial freeze/check command omitted the
local UV_CACHE_DIR and --no-config; sandbox refused writing the home cache and
uv warned about known invalid upstream TOML. The command was corrected as above;
no environment mutation or source repair resulted. Subsequent checks passed.

Frozen FRONT/LEFT/RIGHT specification hash, recorded before pose execution:
0c48da8acaa670b0e4a627d11646ddb66727753feecb04c8a7152660e87da88a.
No criteria, signs, inputs or modality were retuned after viewing outputs.
CLIP's own downloader fetched ViT-B-32.pt to ignored models-local/clip via a
cache-path-only wrapper. 353976522 bytes and URL-embedded SHA256
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af verified.

First GPU run passed strict load, sample, frozen poses, masking and warm tests.
Review identified that the additive controller helper promoted float32 NumPy
scalars to Python floats, causing ~7e-9 angular deviation from the original
sample return. Removed that promotion, added exact return equality assertion,
and extended stable-memory checks across pipeline iterations. No upstream code
changed. Preserved the entire first run and repeated qualification in a new
output directory:

```bash
artifacts-local/envs/edge-blackwell/bin/python research/run_edge_preflight.py --output outputs/edge-preflight-002 > artifacts-local/edge-audit/preflight-002.log 2>&1
```

This is the primary run. Edge 108305527 parameters, CLIP 151277313, all float32
and finite; strict=True, no missing/unexpected keys. Sample native (1,8,4)
finite. FRONT/LEFT/RIGHT pass, masked goal black-vs-sample difference exactly
zero. Official controller supported branches match; zero-XY case deliberately
retains upstream NameError for missing clip_angle. Memory stable, 2694 MiB peak
observed total device usage; Torch peak allocated 1108572160 and reserved
1344274432 bytes. Fifty warm model forwards median/p95/max 9.167/10.427/14.169
ms; twenty pipelines 17.328/23.916/81.789 ms, 0/20 above 333.333 ms. No outlier
removed. Repeated native output max difference zero. Both official figures
visually inspected; source y-limit clips last point at native x=10.5335, with
all eight rows retained numerically. No generated image staged.

Optional CPU attempt initially stopped at CLIP import because Torch 2.2's
minimal environment lacked pkg_resources. Source audit justified the same
setuptools==78.1.0 already in the GPU env; added it to the shared Edge manifest.
Exact corrective dependency commands:

```bash
export UV_CACHE_DIR="$PWD/artifacts-local/uv-cache"
uv pip install --no-config --python artifacts-local/envs/edge-official-cpu/bin/python --dry-run setuptools==78.1.0
uv pip install --no-config --python artifacts-local/envs/edge-official-cpu/bin/python setuptools==78.1.0
uv pip check --no-config --python artifacts-local/envs/edge-official-cpu/bin/python
uv pip freeze --no-config --python artifacts-local/envs/edge-official-cpu/bin/python > results/edge-official-cpu-freeze.txt
CUDA_VISIBLE_DEVICES='' artifacts-local/envs/edge-official-cpu/bin/python research/run_edge_preflight.py --sample-only --output outputs/edge-official-cpu-001 > artifacts-local/edge-audit/cpu-reference-after-setuptools.log 2>&1
```

Exactly one model sample attempted on CPU, same artifact paths and official
input construction. Strict load passes, but model forward fails when get_device()
returns -1 and mask.to(-1) raises Device index must not be negative. Stopped CPU
work without a model patch: OFFICIAL_CPU_REFERENCE_UNAVAILABLE. No CPU output
or invented GPU/CPU numerical comparison. This optional upstream portability
failure does not override successful required GPU gates.

Promoted compact JSON summaries and hashes from primary immutable local evidence;
no raw image/tensor/weights copied into Git. Added report with required sections,
source/unit tests separately from recorded-runtime validation, minimal setup and
reproducible sample/full wrapper. Initial 42 tests passed; added three additional
recorded pose/masking/latency consistency checks for final verification.

Final verification / publication commands:

```bash
artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -p 'test_*preflight.py' -v
git status --short
git diff --check
git diff
git diff --cached
git diff --cached --check
git diff --cached --stat
git commit -m "Qualify OmniVLA-edge runtime and action semantics"
git push -u origin research/omnivla-edge-preflight
git rev-parse HEAD
git ls-remote origin refs/heads/research/omnivla-edge-preflight
```

Decision READY_FOR_ISAAC_ADAPTER with EDGE_3HZ_RUNTIME_FEASIBLE; no Isaac work
performed. Explicit remaining uncertainty: whether a bounded no-obstacle Isaac
adapter can preserve pose/frame, actual RGB history and official commands.
Unresolved universal row/context timing, upstream zero-XY controller defect,
and unavailable CPU numerical parity remain documented limitations.

## 2026-10-12 — Bounded Edge / Isaac no-obstacle task, stopped at startup

User explicitly authorized a new bounded obstacle-free Isaac task and required
one simulator-only gate before model integration. Refreshed source exactly
matched expected 9ebb503a5452498dbf90c95a89ef4aa26a5bfec4. Commands before edits:

```bash
git status --short
git remote -v
git fetch --all --prune
git rev-parse HEAD
git rev-parse origin/research/omnivla-edge-preflight
git log -1 --oneline origin/research/omnivla-edge-preflight
git rev-parse upstream/main
git branch --list research/omnivla-edge-isaac-no-obstacle
git switch -c research/omnivla-edge-isaac-no-obstacle origin/research/omnivla-edge-preflight
```

Origin/upstream correct; upstream remains 5182600cb4a9ee07684e17cdd2a6cbafc56b8a68.
Existing untracked `.vscode/` remains untouched. Read current instructions,
context/report/work log, official inference/model/helpers, research wrapper and
contracts, frozen goals and prior runtime/latency/coordinate/sample evidence.
Updated current branch instructions/context only to permit the requested
bounded SE(2) obstacle-free task, preserving historical scope/context entries.

Read-only simulator inventory found /home/gpuadmin/isaacsim VERSION
6.0.1-rc.7+release.42383.32955d8d.gl, python.sh, bundled Python 3.12.13 and NumPy
2.3.1. Read alternate /home/gpuadmin/isaacsim-ticvla-5.0.0/VERSION only:
5.0.0-rc.45+release.23960.184afb15.gl; no alternate launch/modification. Inspected
installed SimulationApp, camera APIs and standalone examples. No TIC-VLA repo
source was copied or modified, and its environments were untouched.

Official/public camera audit: local README and inference source, project page
https://omnivla-nav.github.io/ and paper https://arxiv.org/html/2509.19480v1 .
No exact Edge camera calibration recovered. Declared requested adapter geometry
640x480 RGB pinhole, HFOV90, height0.5m, pitch0, all-one masks. Frozen JSON includes
scene primitives/light settings, start/goals, 60Hz/3Hz, history, 0.5m threshold,
40s timeout, controller constants and stop rules. Hash recorded before execution:
d1f37af3f136839ac835a3403025d5480e4ef86b7ef55e5d76fc9fa6ebe2684f.
The new primitive scene and camera helpers were written for the startup gate.

Host GPU read before startup: RTX 5060 Ti, driver595.99.02,16311MiB total,
1221MiB used. Executed exactly ONE bounded simulator-only attempt:

```bash
mkdir -p artifacts-local/isaac-edge-audit artifacts-local/isaac-cache artifacts-local/isaac-data
sha256sum research/frozen-edge-isaac-goals.json > artifacts-local/isaac-edge-audit/frozen-before-runtime.sha256
PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME="$PWD/artifacts-local/isaac-cache" XDG_DATA_HOME="$PWD/artifacts-local/isaac-data" timeout --signal=TERM --kill-after=15s 300s /home/gpuadmin/isaacsim/python.sh research/isaac_startup_gate.py --output outputs/edge-isaac-startup-001 > artifacts-local/isaac-edge-audit/startup-001.log 2>&1
```

Failure: SimulationApp constructor did not return. At Kit offset7.392s:
libcusparse.so.12 could not resolve __nvJitLinkCreate_12_8 (libnvJitLink.so.12),
from Isaac's own ml_archive/pip_prebundle tree. Later core.prims import failed
with KeyError isaacsim.core.utils.types, then SIGSEGV at simulation_app.py:576
_start_app. python.sh converted child failure to exit1; no timeout expiry.
Last Kit offset13.155s is not claimed as total wall duration. No stage/camera,
no task steps, no clean shutdown. Python finally could not execute after native
segfault, so host-created failure summary explicitly identifies its provenance.
Copied raw log losslessly to ignored outputs/.../raw/startup.log, made read-only,
and recorded size/SHA. No model inference or navigation artifacts fabricated.

Strict stop rule enforced immediately: no restart, package installation, library
path workaround, alternate runtime, renderer change, environment/driver/CUDA
change, or model/controller/camera/history/scene/goal/timeout tuning. The log's
suggestion to install Torch was not followed. Only read failure logs after the
stop. Post-failure nvidia-smi:863MiB used, no compute processes. These point
readings are not a startup or combined memory peak. Isaac's normal runtime log
also wrote to its default kit/logs path; installed software was not edited.

Camera/history/pose/combined-memory/cadence gates NOT_RUN. Edge service never
started. STRAIGHT/LEFT/RIGHT NOT_RUN_BLOCKED_BY_STAGE_A, no time-to-goal/final
errors/saturation/slot/latency values. Existing preflight timings are not reused
as simulator measurements. No IPC/navigation adapter development continued after
Stage A failure. Requested navigation unit/runtime/figure work is deferred by
that explicit gate; only frozen-config/failure-evidence tests were added.

Regression command:

```bash
artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -p 'test_*.py' -v > artifacts-local/isaac-edge-audit/tests.log 2>&1
```

53 tests PASS:45 existing preflight regressions and8 new evidence/spec checks.
Both simulator scripts parse syntactically; this is not a runtime PASS. Both
historical reports and official model/controller source remain unchanged.
Created EDGE_ISAAC_NO_OBSTACLE.md with required sections and null later-stage
metrics. Numeric scene sidecar explicitly declares an unconstructed stage.
No figures exist because no camera or motion data exists to visualize.

Finalization uses git status/diff/diff--cached/diff--check, size/ignore checks,
normal commit and normal push of the NEW branch, followed by local/remote SHA
verification. Suggested message retained: Qualify OmniVLA-edge obstacle-free
Isaac navigation; report clearly states ISAAC_STARTUP_BLOCKED, not success.
Exact remaining uncertainty: whether existing Isaac6.0.1 can complete the
unchanged simulator-only gate after its CUDA-library loading incompatibility
is resolved in a separately authorized runtime task. No obstacle work began.


## 2026-10-12 — OmniVLA-edge Isaac runtime recovery

- Starting source: `8ddf5c60d9107f8b5f114c68f0fe9777a6f697df`; new branch
  `research/omnivla-edge-isaac-runtime-recovery`. The TIC-VLA wrapper/report
  were read as historical reference only; its workspace was not modified.
- Independently confirmed `CONFIRMED_LIBRARY_PATH_COLLISION`: inherited
  ROS/CUDA paths select system CUDA 12.6 nvJitLink without
  `__nvJitLinkCreate_12_8`. ELF symbols, ldd and a minimal CDLL loader show
  process-local isolation instead selects the compatible Isaac bundled file.
- Added `scripts/isaac6_python.sh`; shell syntax and harmless boundary/real
  Python probes passed before the retry. No system, CUDA or shell changes.
- Exactly one isolated SimulationApp retry used byte-identical startup script,
  scene and frozen configuration. Constructor returned, stage and finite
  640×480 RGB succeeded, 60 steps completed, and `app.close()` returned.
  The written result says `ISAAC_STARTUP_PASS` / `clean_shutdown=true`, but
  Python subsequently segfaulted and the launcher exited 1, without timeout.
  No CUDA symbol-resolution error recurred. Overall decision:
  **ISAAC_STARTUP_RECOVERY_FAIL**. No second retry or later runtime work.
- Raw result, PNG and log are preserved under
  `outputs/edge-isaac-startup-isolated-001`; diagnostics under
  `artifacts-local/isaac-runtime-recovery`. The full environment stays ignored
  and private. Report: `docs/EDGE_ISAAC_RUNTIME_RECOVERY.md`; compact result:
  `results/edge-isaac-runtime-recovery.json`. Historical reports unchanged.
- Added 16 focused CPU tests covering launcher behavior, historical/frozen
  evidence, exactly one accepted result, output scope and process-level
  failure classification. The final focused suite also includes the 8 existing
  startup evidence checks; see the report for validation outcome.
- No Edge loading/inference, combined memory, calibration/history, pose-goal
  inference, navigation, obstacles, controller changes or reconciliation.
- Next uncertainty: can the unchanged simulator-only gate exit without a
  post-close SIGSEGV under the clean launcher, while retaining its successful
  constructor, stage, RGB and 60-step evidence? Native crash cause remains
  unmeasured and needs a separately authorized task.

Validation outcome: all 24 focused CPU checks passed; `git diff --check` passed.


## 2026-10-13 — Final OmniVLA-edge obstacle-free Isaac qualification

Starting SHA `c5ed3b911e0eb4c59d07058eda4a15f6ec405b25`; branch `research/omnivla-edge-no-obstacle-final`.
Updated AGENTS.md to allow logged infrastructure debugging retries while keeping
frozen scientific trials one-shot, with no post-outcome tuning. Reused the
verified process-local CUDA isolation; installation/library hashes were unchanged.
Inspected installed SimulationApp source and adopted default fast shutdown,
durable pre-close evidence, and final `app.close(exit_code=code)`.

Infrastructure attempts lifecycle-001, camera-history-001 and combined-001 all
passed, with exit 0 and no residual Isaac process. Camera marker x centers were
319.5/159.5/479.5 px (FRONT/LEFT/RIGHT), FOV 90 degrees. Actual six-frame history,
pose/source parity, lossless single-outstanding IPC, strict float32 Edge load,
scheduler and controller parity passed. Combined active-inference memory was
3720 MiB device-wide; scientific peak 3821 MiB, sampled every 0.25 s. A first
socket unit test was blocked by sandbox EPERM; unchanged host-access retry passed.
All attempts and failures remain recorded locally.

After 83 passing pre-freeze checks, froze protocol SHA256
`fb40d57f7999da622d99f33ed019132d1562dde40d14f8c506e40efde63ccbf2`. Exactly one scientific group
scientific-001 ran STRAIGHT/LEFT/RIGHT, each once. All reached the 0.5 m region:
STRAIGHT 15.50 s / 0.495436577 m; LEFT 16.80 s / 0.497001901 m;
RIGHT 16.683333 s / 0.495582257 m. All frozen runtime/config hashes unchanged.
148 finite native outputs, 146 applied; 2 endpoint responses retained unapplied.
One missed slot, 146 initial clips and final ratio limits, zero zero-XY bugs.
Model median/p95/max 14.537/17.488/144.385 ms; forward+CLIP
22.975/30.323/251.577 ms. No artificial latency. Derived frame-record-to-applied-
step wall delay is 95.402/147.025/358.917 ms; raw request-send timing is preserved
and distinguished in the report.

Decision: **NO_OBSTACLE_GOAL_REACHING_PASS**. Offline raw evidence validation
passed; final suite 87 tests passed. Historical reports unchanged; no model,
controller, scene, camera or timing tuning after freeze. No obstacle, pedestrian,
latency-injection, OLD/FRESH or reconciliation work performed.
Report: `docs/EDGE_ISAAC_NO_OBSTACLE_FINAL.md`; compact result:
`results/edge-isaac-no-obstacle-final.json`; ignored evidence: `outputs/edge-final/`.

Next uncertainty: Can the same frozen OmniVLA-edge + Isaac adapter safely detour around a predeclared static obstacle without changing model, camera, history, controller, cadence, or goal semantics?

## 2026-10-13 — Isaac GUI replay of completed navigation

User requested viewing the driving results in the Isaac Sim GUI. Added
`research/isaac_gui_replay.py` on the existing final branch. It reads the saved
scientific-001 SE(2) states and the original RGB images, recreates the frozen
scene, and adds explicitly visual-only pose arrows, colored paths and 0.5 m goal
rings. No model is loaded and no new scientific navigation is performed.
The panel supports goal selection, pause/restart, speed and automatic cycling.
Playback at 1x follows recorded simulation time, with a 3-second endpoint hold.

Commands (both launches used the existing isolated Isaac wrapper):

```bash
python3 -m py_compile research/isaac_gui_replay.py
bash scripts/isaac6_python.sh research/isaac_gui_replay.py --output artifacts-local/gui-replay/attempt-001/viewer
bash scripts/isaac6_python.sh research/isaac_gui_replay.py --output artifacts-local/gui-replay/attempt-002/viewer
```

The launchers ran the GUI in a detached subprocess and recorded the exact command,
PID, display and launch time in each attempt's `launch.json`; stdout/stderr are in
`console.log`. Attempt 001 successfully cycled all three goals; inspection found
a deprecated docking API and an unnecessary Render Settings panel that reduced
the viewing area. Stopped that viewer through its local stop-request file, then
updated only the viewer's docking/layout and launched attempt 002. Attempt 002
reached `GUI_REPLAY_READY`, cycled STRAIGHT/LEFT/RIGHT, and has no Error/Traceback
entries at verification. The GUI was left open for the user. X11 window inventory,
desktop screenshots, viewport captures and advancing playback status verified
the visible application. All 164 original RGB hashes and the frozen source hashes
passed read-only checks. Evidence remains under ignored `artifacts-local/gui-replay/`.
No new Git branch was created and the scientific protocol/results are unchanged.

To reopen later, use the same command with a new, unused `--output` directory;
the viewer refuses to reuse an existing directory or write inside its input run.

## 2026-10-13 — Default GUI presentation and one-line launcher

User requested GUI result presentation after every future experiment and a
one-line launch command. Recorded that preference in `AGENTS.md`, including
explicit selection of the just-completed run, replay/live labeling and preserving
the scientific evidence. Added `scripts/isaac_gui_replay.sh`, which resolves the
repository from its own location and allocates a fresh timestamp/PID diagnostics
path on every invocation. It forwards `--run`, `--config` and other viewer
arguments through the existing isolated Isaac launcher. Defaults remain the
completed edge-final/scientific-001 run; future experiments must select their own
compatible run/config or adapt the viewer to their scene/schema.

One-line command from any working directory:

```bash
bash ~/Workspace/OmniVLA/scripts/isaac_gui_replay.sh
```

## 2026-10-13 — Static obstacle qualification preparation

User explicitly requested one primary central-static-obstacle trial on
`research/omnivla-edge-static-obstacle`, followed by analysis, existing GUI replay,
commit and push, then stop. Start local/remote obstacle-free HEAD both matched
`89b4e39c55af02cb9dc1a87c65f783f273e54ff6`. Read AGENTS, work log, final report,
compact results, frozen runner/service/controller/scheduler/validation and local
GUI additions. `git ls-remote --heads origin` showed no static-obstacle branch;
`git switch -c research/omnivla-edge-static-obstacle` preserved the uncommitted
AGENTS/WORK_LOG/viewer/launcher changes and pre-existing untracked `.vscode/`.
Initial hashes and request copy: `artifacts-local/static-obstacle/start-audit/`.

Added separate geometry, protocol, obstacle scene, episode, validation, runner
and supervisor modules. Baseline frozen runtime files remain byte-identical.
The episode reuses baseline request/response/bootstrap and unchanged scheduler,
integration and model service. New checks only measure clearance and terminate;
they never modify controls or poses. Circle/AABB swept distance uses segment
intersection, endpoint and corner projections, with first contact found by
64 bisections; the completed 60 Hz detection step is retained without clipping
the motion. Footprint is an evaluation convention, not physical chassis geometry.

Proposed before inference: red opaque cuboid center (2.5,0,0.5), dimensions
(0.6,1,1), RGB displayColor (0.8,0.08,0.08), yaw 0; radius 0.20 and margin 0.10 m.
Draft config: `artifacts-local/static-obstacle/proposed-config.json`.
Existing GUI was extended additively to show this obstacle, footprint, measured
clearance and actual recorded outcome; original RGB is shown from saved files.
Added a small launcher that forwards explicit static run/config to the existing
viewer. Scientific execution remains headless with no replay markers.

Model-free commands and preserved logs:

```bash
python3 -m unittest discover -s tests -p test_static_obstacle.py -v
python3 research/run_static_obstacle_attempt.py --phase infrastructure --run-id infrastructure-001 --config artifacts-local/static-obstacle/proposed-config.json --change 'Model-free obstacle geometry, visibility, yaw, history, spectator isolation, dummy IPC and swept collision qualification'
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -v
python3 -m unittest discover -s tests -p 'test_static_obstacle*.py' -v
```

Infrastructure-001 PASS, launcher exit 0, no residual processes or segmentation
fault. No model loaded or inferred. Measured geometry matches within 1e-6 m;
21,051 red pixels at image x=319.108 initially, x=411.127 at +15 degree yaw.
Spectator-only camera addition preserved observation pose/intrinsics, with
0.192786 mean absolute RGB difference after render settling (threshold 1).
Calibration and actual rolling history gates passed; six model-free dummy
IPC requests were applied with zero held motion. Scripted straight motion was
detected at step 401, with the continuous contact at approximately 6.6667 s.
First full suite: 101 passed. Added real-episode model-free peer tests for
collision immediate stop, safe/low-clearance goal, implementation failure and
upstream-controller failure: 17 new evaluation tests pass. Re-running the full
suite to include them before freeze. All 330 obstacle-free raw artifact hashes
and historical report/source hashes remain unchanged.

## 2026-10-13 — Single static-obstacle trial, collision and completion

Full pre-freeze suite passed 104 tests. Froze the new protocol at
`research/frozen-edge-static-obstacle.json`, SHA256
`f8cdbfe22a32a335968976c80a4c86994154866fcf859277a81f4ce911b73728`.
Baseline config and runtime source files remain unchanged. Registration was
written exclusively before the scientific launch, preventing a second primary.

```bash
python3 research/run_static_obstacle_attempt.py --phase scientific --run-id scientific-001 --config research/frozen-edge-static-obstacle.json --timeout 600 --change 'Frozen f8cdbfe22a32a335968976c80a4c86994154866fcf859277a81f4ce911b73728; one primary central static-obstacle trial; no tuning or retries'
PYTHONDONTWRITEBYTECODE=1 artifacts-local/envs/edge-blackwell/bin/python research/validate_static_obstacle_evidence.py --output artifacts-local/static-obstacle/scientific-validation-001.json
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python research/analyze_static_obstacle.py --output outputs/static-obstacle/scientific-001/derived
bash scripts/isaac_static_obstacle_gui.sh --output artifacts-local/static-obstacle/gui-001/viewer
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -v
git diff --check
```

Scientific outcome: **COLLISION**. First swept contact 7.301832592 sim s;
detection/termination at step 439, 7.316666667 sim s. Minimum signed clearance
-0.004438785 m; first 0.10 m safety-boundary crossing 6.967805882 sim s. Goal
not reached; final goal distance 2.995643566 m. Maximum lateral deviation
0.022212310 m and path length 2.005 m. Final pose
(2.004438785,-0.022212310,-0.072277013). The executed endpoint was retained,
not moved back to the contact surface; no additional step was executed.

Exactly 21 native predictions, all applied, 27 actual RGB captures and 440 states.
One missed inference slot; all 21 controls underwent initial clipping and final
ratio limitation. Zero upstream zero-XY exceptions. Cold first inference retained;
no artificial delay. Median model/CLIP-inclusive/pipeline times
15.667/26.349/76.970 ms; recorded-frame-to-completed-applied-step median
96.136 ms, with the same post-PNG timestamp convention as the baseline.
Isaac and Edge service exit 0, no residuals/SIGSEGV. Device-wide sampled peak
3866 MiB. Scientific runtime/config hashes unchanged; no tuning or retry.

Offline validation PASS for native actions, history, controller parity, pose
integration, swept clearance, collision termination and timestamp linkage.
All 330 obstacle-free artifact hashes and historical reports/results match.
Primary raw/run files marked read-only. Derived PNG/SVG plots cover XY and
footprint/safety envelopes, goal distance, clearance, v/w, and first/middle/last
model chunks with verified cumulative XY/local heading semantics. No intrinsic
waypoint timestamps assumed. Plots were visually inspected; prior STRAIGHT is
explicitly an obstacle-free reference, not a controlled repeat.

Existing GUI replay extended additively and launched after science. It reached
GUI_REPLAY_READY, rendered 4400 frames and cycled the recorded collision episode
eight times before normal window closure. No model inference in replay. Viewport
capture inspected; logs/command/PID under `artifacts-local/static-obstacle/gui-001/`.
Repeatable one-line GUI command: `bash ~/Workspace/OmniVLA/scripts/isaac_static_obstacle_gui.sh`.

Final suite passed **108 tests**, including four new offline evidence/transform/
config-hash checks; `git diff --check` passed. Final report
`docs/EDGE_STATIC_OBSTACLE.md`; compact result
`results/edge-static-obstacle-summary.json`. Git publication includes code,
tests, compact results, report and preserved GUI additions, excluding `.vscode/`
and all ignored raw images/tensors/checkpoints/environments/caches.

Interpretation: technical infrastructure passes but this single frozen navigation
trial fails static-obstacle qualification. It does not establish general model
incapability or identify a causal perception/controller failure. Exact next
uncertainty: checkpoint visual/domain sensitivity versus the fixed index-4
closed-loop interface as the source of the absent lateral avoidance response.
No dynamic obstacles, OLD/FRESH analysis, reconciliation, graph optimization or
follow-up experiment started. Normal commit/push to the requested branch only;
remote SHA verification is the final publication step.

Staged whitespace check subsequently detected extra EOF blank lines in the new
episode and offline validator; the previous unstaged check did not inspect new
untracked additions. Trimmed the unfrozen offline validator. Kept the executed
episode byte-identical to its frozen SHA and documented a per-file `blank-at-eof`
exception in `.gitattributes`; all other whitespace checks remain active. This
packaging correction changes no scientific source/config or result. Rechecked
the staged diff and frozen protocol before committing.
