# OmniVLA-edge Isaac No-Obstacle Qualification

## Research question

Using actual RGB history and an observation-time robot-local pose goal, while
preserving OmniVLA-edge's official 3 Hz / waypoint-index-4 / controller
semantics, can OmniVLA-edge achieve obstacle-free goal-region entry in Isaac Sim?

**Decision: ISAAC_STARTUP_BLOCKED.** The one authorized simulator-only attempt
failed inside `SimulationApp(...)`, before the constructor returned. No task
stage, camera, Edge service, combined inference, or navigation was executed.
The navigation question remains unanswered. This is an upstream simulator
runtime failure, not a measured Edge navigation failure.

## Repository-confirmed facts

Refreshed origin source branch `research/omnivla-edge-preflight` and expected
starting commit both resolved to `9ebb503a5452498dbf90c95a89ef4aa26a5bfec4`.
New branch: `research/omnivla-edge-isaac-no-obstacle`. Origin is
`https://github.com/minju5054/OmniVLA.git`; upstream is
`https://github.com/NHirose/OmniVLA.git`; fetched upstream/main remains
`5182600cb4a9ee07684e17cdd2a6cbafc56b8a68`. The final research SHA is the commit
containing this report, reported separately after normal push verification.
The only pre-existing untracked path was `.vscode/`, left untouched.

Reviewed current research instructions/context, both prior work-log history and
Edge report, official Edge sample/model/helpers, the additive Edge wrapper and
contracts, frozen pose goals, and runtime/latency/coordinate/sample summaries.
Their existing findings were reused; no preflight model rerun was performed.
The historical reports and source remain byte-identical:

| Report | SHA256 |
| --- | --- |
| FULL | `bb9f0440281f2672104fdcebb08366cf16654b3ff3bafe4972b5ad6d8c747d03` |
| Edge | `c4097bf147774f04417816a6baa8cccd177f6b4681581735e8b8d18e3d0ed5ac` |

Validated model identity, inherited from preflight rather than loaded in Isaac:
`NHirose/omnivla-edge`, revision
`b1361b7e24f101edea795a98b00a826b61a97394`; checkpoint 434,276,465 bytes, SHA256
`f93b056f124380899fce3e087a5c2b41ce4d8c07e211daad52b6ebf02e80d04d`.
Separate CLIP ViT-B/32 hash:
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
Existing inference runtime remains Python 3.10.21 / Torch 2.7.1+cu128, unchanged.
Its preflight speed and memory figures are not Isaac measurements.

## Adapter conventions

The requested conventions were frozen before startup/model use in
`research/frozen-edge-isaac-goals.json`, SHA256
`d1f37af3f136839ac835a3403025d5480e4ef86b7ef55e5d76fc9fa6ebe2684f`.
No parameter changed after the failed attempt.

| Declared choice | Value; execution status |
| --- | --- |
| Isaac installation | `/home/gpuadmin/isaacsim`, VERSION `6.0.1-rc.7+release.42383.32955d8d.gl`; attempted |
| Entry point | `/home/gpuadmin/isaacsim/python.sh`, bundled Python 3.12.13; NumPy 2.3.1 |
| Rendering | Headless `RaytracedLighting`, one GPU; selected before startup |
| Camera | Pinhole RGB 640×480, horizontal FOV 90°, height 0.5 m, pitch 0°, +X forward; not constructed |
| Physics/state | 60 Hz, simple kinematic SE(2), explicit Euler; not executed |
| History acquisition | Six independent actual RGB captures, 3 Hz simulation time, oldest→newest; not executed |
| Success | XY distance ≤0.5 m; yaw reported separately; not evaluated |
| Timeout | 40.0 simulation seconds after bootstrap; not reached |
| Seed | 0; no randomized scene content |

Camera audit: repository source, [official project page](https://omnivla-nav.github.io/),
and [paper platform/Edge descriptions](https://arxiv.org/html/2509.19480v1)
were inspected. Exact Edge intrinsics/extrinsics were not recovered. Source
mentions optional fisheye masking but supplies all-one masks in the sample.
The frozen pinhole convention is an adapter choice, not a recovered training
calibration. The proposed focal length/aperture are 0.02/0.04 m, giving 90°
horizontal FOV; no camera or FOV tuning occurred.

Declared scene: 30×30 m flat floor, centered at (5,0), top z=0; wide free region
x∈[−5,12], y∈[−8,8]. Visual-only boundary walls at x=−9/19 and y=−14/14,
3 m high, 0.1 m thick; thin 2 m floor grid for structure. Dome intensity 900,
distant light 1500. Floor alone has CollisionAPI; boundaries/grid are visual
only. No blocking geometry inside the free region, robot articulation or
actuator model. This geometry was declared in code but never constructed.

Start (0,0,0), same for all goals: STRAIGHT (5,0,0); LEFT
(5,2,atan2(2,5)); RIGHT (5,−2,−atan2(2,5)). The numeric top-down sidecar
`results/edge-isaac-scene-declaration.json` is explicitly
`DECLARED_NOT_CONSTRUCTED`, not a measured stage export.

## Coordinate frames

Declared world +X forward at start, +Y left, +Z up, yaw positive CCW.
The camera helper requests Isaac's documented `camera_axes='world'`; orientation
would be quaternion `[cos(yaw/2),0,0,sin(yaw/2)]`, translated to agent x/y and
height 0.5 m. The camera API was inspected read-only in the installed source.
That helper was not executed because construction never reached it.

The model-free color gate was planned with FRONT (4,0,0.5), LEFT (4,+2,0.5),
RIGHT (4,−2,0.5), red/green/blue non-colliding panels, then removal before
navigation. LEFT/RIGHT have lateral offsets on a front plane so both fit the
90° field of view. No panels were placed and no centroid result exists:
`CAMERA_FRAME = NOT_RUN`. No frame alignment PASS is claimed.

## RGB history semantics

Required future request history is `[I(k−5),...,I(k)]`, six actual capture
events spaced 1/3 simulation second. Stationary bootstrap may yield identical
pixels, but it must have distinct capture IDs/timestamps and separate images.
The latest image also supplies the current 224×224 input. No future frames,
copied moving history, or synthetic observations are allowed.

3 Hz history acquisition remains an adapter convention; the checkpoint's
universal context timing is unresolved. Here **no RGB was captured** and no
rolling queue was implemented after the startup stop: `HISTORY_INPUT = NOT_RUN`.
The requested PNGs/contact sheets cannot be produced from absent observations.

## Pose-goal computation

Required observation-time convention:
`T_Robs_G = inverse(T_W_R(t_obs)) @ T_W_G`. Rotate the world XY delta by
`−yaw_obs`, clamp its norm to 30 m, divide local forward/left by 0.1, append
`cos(wrap(yaw_goal−yaw_obs))` and `sin(...)`. Do not re-anchor at response or
application time. Record the matrix, observation pose and exact normalized
vector with each request.

No observation-time request was constructed here. Prior Edge FRONT/LEFT/RIGHT
source-contract regression tests still pass, but they are not evidence that
an Isaac pose adapter was implemented or validated: `POSE_TRANSFORM = NOT_RUN`.

## Scheduling / observation-ready-application timing

Frozen requirement: 60 Hz integration, observations every 20 physics steps,
one outstanding request, no queue/catch-up, skipped busy slots explicitly
logged, old command held while inference runs, zero before the first result,
response applied on the next control step. Success is checked before requesting
another prediction. No artificial latency or schedule tuning is permitted.

Isaac and Edge must be separate runtimes, with lossless RGB and request/response
ID/hash checks. The intended Edge service would load once and use pose-only
ID 4, black goal/maps, `xxxx` language and official transforms. The IPC service,
scheduler and navigation adapter were **not implemented or started after Stage A
failed**. Observation cadence, request latency, observation-to-application delay,
missed slots, update rate and sim/wall ratio are **unavailable**, not zero:
`CADENCE = NOT_RUN`. No prior preflight timing is relabeled as Isaac timing.

## Controller parity

Inherited source contract remains waypoint index 4, metric XY×0.1, DT=1/3,
EPS=1e−8, first clips v=[0,0.5], w=[−1,1], final maxv=maxw=0.3 with the
unchanged ratio limiter. Existing source-extracted controller tests pass.
No commands were generated/applied in Isaac; saturation statistics are N/A.
The undefined upstream `clip_angle` zero-XY branch is unchanged. It was not
exercised in this task, so no navigation occurrence count is reported.

Isaac's bundled NumPy is 2.3.1 whereas the qualified Edge helper uses 1.26.4.
Exact numerical controller parity in the simulator process remains untested;
this is a further implementation check for a future adapter, not an explanation
asserted for the constructor failure. No dependency was changed.

## Simulator-only gate

**Measured classification: ISAAC_STARTUP_FAILED. Adapter gate: FAIL.**
Exactly one attempt, run ID `edge-isaac-startup-001`.

Read-only inventory found preferred Isaac 6.0.1 above and a separate
`/home/gpuadmin/isaacsim-ticvla-5.0.0` VERSION
`5.0.0-rc.45+release.23960.184afb15.gl`. The latter's version file was read;
its runtime was not used or modified. NVIDIA RTX 5060 Ti was visible to Vulkan,
16,311 MiB, driver 595.99.02. Device memory before launch was 1,221 MiB.

The bounded command, from `/home/gpuadmin/Workspace/OmniVLA`, was:

```bash
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME="$PWD/artifacts-local/isaac-cache" \
XDG_DATA_HOME="$PWD/artifacts-local/isaac-data" \
timeout --signal=TERM --kill-after=15s 300s \
/home/gpuadmin/isaacsim/python.sh research/isaac_startup_gate.py \
  --output outputs/edge-isaac-startup-001 \
  > artifacts-local/isaac-edge-audit/startup-001.log 2>&1
```

The child failed before the 300 s limit; timeout did not fire. First material
error at Kit log offset 7.392 s:

```text
libcusparse.so.12: undefined symbol: __nvJitLinkCreate_12_8,
version libnvJitLink.so.12
```

The error names the library under Isaac's
`extsDeprecated/omni.isaac.ml_archive/pip_prebundle/torch/lib/../../nvidia/cusparse/lib/`.
Subsequent evidence: `isaacsim.core.prims` failed to import;
`KeyError: 'isaacsim.core.utils.types'`; fatal segmentation fault in
`simulation_app.py:576`, `_start_app`, called from its constructor and this
task's startup script line 22. Last Kit log offset is approximately 13.155 s;
this is a log-relative offset, not a measured end-to-end wall duration.

`python.sh` reports a segmentation fault and converts child failure to launcher
exit code 1. The constructor never returned; task scene construction, camera
initialization and all 60 proposed steps were not reached. Clean shutdown did
not occur. The process died before Python's `finally` could emit `result.json`;
`host-failure-summary.json` is explicitly reconstructed from the raw log and
launcher result, not represented as an in-process measurement.

The observable failure is CUDA-library symbol resolution followed by extension
errors and a crash. Exact loaded-library provenance and the causal chain have
not been established. No installer, library-path workaround, alternate renderer,
second launch, driver/CUDA change, Isaac package change or Edge environment
change was attempted. The log's suggestion to install Torch was not followed.
Standard runtime logging produced an Isaac Kit log at its own default path;
XDG data/cache were directed to ignored local artifacts. No installed source
or package was edited.

## Combined memory result

`ISAAC_EDGE_COMBINED_MEMORY = NOT_RUN`. Edge was never launched with Isaac;
no combined peak, finite model result or responsive simulator result exists.
Before-startup device-wide reading: 1,221 MiB. After crash: 863 MiB, with no
compute processes reported by `nvidia-smi`. These two observations are **not a
peak bound** and are not combined runtime memory. No memory workaround was tried.

## STRAIGHT result

**NOT_RUN_BLOCKED_BY_STAGE_A.** No navigation request or control step. Goal
entry, time-to-goal, final distance/yaw error, saturation, missed inference slots
and upstream controller bug occurrences are all unavailable.

## LEFT result

**NOT_RUN_BLOCKED_BY_STAGE_A.** No navigation request or control step; all
navigation metrics unavailable. No altered goal or retry.

## RIGHT result

**NOT_RUN_BLOCKED_BY_STAGE_A.** No navigation request or control step; all
navigation metrics unavailable. No altered goal or retry.

## Positive evidence

Expected source SHA/remotes were verified after fetch; the task uses a new
branch and preserves both historical branches/reports. Existing Isaac version,
entry point, bundled Python/NumPy and GPU visibility were verified. Configuration
was frozen before model use. The simulator-only gate produced a concrete,
hashed failure log and the stop rule prevented later experiments.

**53 CPU-only tests pass:** 45 existing FULL/Edge preflight regressions and eight
new frozen-spec/failure-evidence checks. New checks cover immutable config hash,
requested bounds/goals, unchanged controller/modality constants, failure status,
no later execution, preserved reports, declared-vs-measured geometry and raw-log
integrity. Syntax parsing passes for startup/scene scripts. None proves simulator
startup, RGB history, IPC, or navigation. The requested full set of navigation
adapter tests is deferred with the adapter after Stage A failure.

## Negative evidence

SimulationApp did not return, no task stage opened, camera did not render,
zero task simulation steps completed, and no clean shutdown. The startup gate
fails all required completion conditions. This invalidates proceeding to camera,
history, combined memory or navigation gates, irrespective of the earlier
successful standalone Edge preflight. The primary requested goal-reaching
qualification therefore does not pass; its failure stage is simulator startup.

## Implementation limitations

Only startup-gate scaffolding, a frozen primitive-scene builder and the immutable
experiment specification were added before the gate. Scene/camera code remains
unexecuted and unvalidated. No claim that it is ready for closed-loop use.
No IPC, history queue, pose-goal request builder or navigation scheduler was
implemented after the stop. No lossless run logs/actions or derived navigation
figures exist because no observations or predictions occurred. Generating them
would invent evidence. The raw subprocess log survives the native crash;
Python's own exception/finally path cannot recover from SIGSEGV.

## Experimental artifacts

| Artifact | Meaning |
| --- | --- |
| `research/frozen-edge-isaac-goals.json` | Frozen planned scene/camera/goals/rates/stop rules |
| `research/isaac_startup_gate.py` | Single startup/stage/camera/60-step/close gate; no Edge imports |
| `research/isaac_scene.py` | Standard primitive scene and camera helpers, not reached |
| `outputs/edge-isaac-startup-001/raw/startup.log` | Immutable full subprocess log |
| `outputs/edge-isaac-startup-001/host-failure-summary.json` | Host-reconstructed crash evidence |
| `results/edge-isaac-no-obstacle-summary.json` | Compact provenance, gate failure and null later metrics |
| `results/edge-isaac-scene-declaration.json` | Numeric top-down planned geometry, explicitly unconstructed |
| `results/edge-isaac-source-audit.json` | Calibration/runtime source audit |
| `artifacts-local/isaac-edge-audit/frozen-before-runtime.sha256` | Pre-execution freeze record |
| `artifacts-local/isaac-edge-audit/tests.log` | Regression/evidence test output |

No trajectory figure, marker RGB, contact sheet, USD export or video exists to
inspect. Only evidence-backed artifacts are included. Raw logs/caches remain
ignored, with compact JSON hashes committed. Existing large weights and all
prior evidence remain untouched.

Replay inspection without launching Isaac:

```bash
cat results/edge-isaac-no-obstacle-summary.json
cat outputs/edge-isaac-startup-001/raw/startup.log
artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

The launch command above is the exact single attempted command. No second launch
is authorized within this stopped qualification. A later, separately authorized
runtime task must use a fresh output directory; existing directories refuse
overwrite. There is no closed-loop replay/run command because that implementation
and its input/output data were not reached.

## Insufficient evidence

No camera-frame PASS, actual-history PASS, Isaac pose-transform PASS, combined
memory PASS, achieved cadence, model latency under Isaac, command application
delay, goal reaching, obstacle avoidance, wheel dynamics, robot-level validation,
OLD/FRESH analysis or reconciliation evidence. No intrinsic waypoint timestamp
or 2.67 s horizon is inferred. No navigation failure is attributed to Edge.

## Decision

**ISAAC_STARTUP_BLOCKED**

Startup classification `ISAAC_STARTUP_FAILED`; primary outcome
`NO_OBSTACLE_GOAL_REACHING_FAIL` because required Stage A did not pass.
STRAIGHT/LEFT/RIGHT were not run, rather than attempted and failed to reach goals.
The exact startup error is recorded without changing system software or tuning
any experiment parameter. Task work ends at the prescribed stop gate.

## Exact next uncertainty

Can the existing Isaac 6.0.1 runtime complete the unchanged simulator-only gate
after its CUDA-library loading incompatibility is resolved in a separately
authorized runtime task?
