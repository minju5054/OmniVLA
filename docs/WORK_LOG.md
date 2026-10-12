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
