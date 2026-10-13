# OmniVLA-edge Obstacle-Free Isaac Qualification

## Research question

Using actual RGB history and an observation-time robot-local pose goal, while
preserving OmniVLA-edge's official 3 Hz / waypoint-index-4 / controller semantics,
can OmniVLA-edge achieve obstacle-free closed-loop goal-region entry in Isaac Sim?

**NO_OBSTACLE_GOAL_REACHING_PASS.** All nine infrastructure gates passed before
protocol freeze. STRAIGHT, LEFT and RIGHT each reached the 0.5 m XY goal region
in its single primary trial. This establishes the declared obstacle-free
kinematic qualification, not obstacle avoidance or real-robot reliability.

Date: 2026-10-13, Asia/Seoul. Starting SHA:
`c5ed3b911e0eb4c59d07058eda4a15f6ec405b25`. Branch: `research/omnivla-edge-no-obstacle-final`.
Upstream source SHA: `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68`.
Compact result: [`results/edge-isaac-no-obstacle-final.json`](../results/edge-isaac-no-obstacle-final.json).

| Goal | Result | Time to region, sim s | Final XY distance, m | Signed yaw error, rad (degrees) |
| --- | --- | ---: | ---: | ---: |
| STRAIGHT | GOAL_REACHED | 15.500000 | 0.495436577 | -0.004691037 (-0.269°) |
| LEFT | GOAL_REACHED | 16.800000 | 0.497001901 | +0.005198377 (+0.298°) |
| RIGHT | GOAL_REACHED | 16.683333 | 0.495582257 | +0.054776966 (+3.138°) |

## Infrastructure vs scientific protocol

`AGENTS.md` now distinguishes iterative infrastructure debugging from frozen
scientific trials. Infrastructure corrections can be retried with every command,
run ID, change, result and failure reason retained on this branch. Scientific
trials start only after the infrastructure gates and protocol hash exist; one
primary trial per goal, no post-outcome tuning or silent implementation retries.

Three infrastructure process attempts were performed in this task; all passed.
The first socket unit-test execution hit sandbox `EPERM` at AF_UNIX bind; the
unchanged test passed with authorized host socket access. Both test attempts are
recorded locally. This was a tooling restriction, not a measured model failure.
No unsuccessful simulator attempt was omitted.

| Attempt | Change / purpose | Result | Host elapsed, s |
| --- | --- | --- | ---: |
| lifecycle-001 | Use installed default fast shutdown; fsync evidence before final app.close(exit_code=code) | PASS; exit 0 | 12.370 |
| camera-history-001 | Add model-free FRONT/LEFT/RIGHT marker and actual stationary/rolling history validation; same scene and camera geometry | PASS; exit 0 | 15.376 |
| combined-001 | Add persistent strict-loaded Edge service, lossless dummy IPC/scheduler and three stationary inference checks with Isaac alive; no model commands applied | PASS; exit 0 | 21.808 |

The one subsequent scientific process, `scientific-001`, contains exactly one
STRAIGHT, LEFT and RIGHT episode. Its registration was created exclusively
after freeze. There was no scientific retry. The predeclared policy would stop
on an escaped implementation error or upstream controller bug; a model timeout
would remain a negative result without parameter changes. Neither occurred.

## Reused known infrastructure fixes

Reused the confirmed library-path fix from the preserved
[recovery report](EDGE_ISAAC_RUNTIME_RECOVERY.md): the workspace wrapper clears
inherited LD_LIBRARY_PATH, PYTHONPATH and PYTHONEXE and sets PYTHONNOUSERSITE=1
and PYTHONUNBUFFERED=1 for its child. No arbitrary CUDA directories or preload
were added. Relevant library and launcher hashes matched the prior audit, so
the full ELF investigation was not repeated.

Read `/home/gpuadmin/Workspace/TIC-VLA/scripts/isaac6_python.sh` and
`docs/ISAAC6_COMPATIBILITY.md` there as historical reference only. The same
installed Isaac source was then inspected directly. Installed
`simulation_app.py` SHA256: `6af5372bb0cdda6e665a30cd285ff724b3e5bd012dbdd4df458e0726173fc903`.
Its default is `fast_shutdown=True`; supported signature is:

```python
close(self, wait_for_replicator: bool = True,
      skip_cleanup: bool = False, exit_code: int = 0) -> None
```

The source states that default fast shutdown calls immediate framework/process
termination. Nonzero `exit_code` flushes stdio and uses `os._exit(code)` before
Kit can replace the code. Arbitrary post-close Python work is not expected.
The additive runner writes/flushes/fsyncs all evidence first, then makes
`app.close(exit_code=code)` its final lifecycle action. It does not force
`fast_shutdown=False`. The historical startup script and prior reports remain
unchanged. No TIC-VLA, installed Isaac, driver, CUDA, package or shell changes.

## Isaac lifecycle result

**ISAAC_LIFECYCLE_PASS.** Isaac version `6.0.1-rc.7+release.42383.32955d8d.gl` at
`/home/gpuadmin/isaacsim`, RTX 5060 Ti, driver 595.99.02.
The dedicated lifecycle attempt returned from the constructor, created the
unchanged stage, rendered finite nonempty 640×480 RGB, and executed 60 steps.
Its pre-close result was durable before close; the host observed exit 0, no
symbol error, no SIGSEGV, and no residual Isaac process.

Camera/history, combined-memory and scientific launches also terminated
successfully. Both scientific Isaac and persistent Edge service exited 0 with
no residual processes. Thus there was one dedicated shutdown-fix attempt and
four successful Isaac process lifecycles in total, including scientific work.
The earlier `fast_shutdown=False` exit crash remains historical negative
evidence and was not rewritten as a success.

## Camera adapter

Same 30×30 m floor and fixed lighting, visual-only boundary walls and thin floor
grid; only the floor has collision geometry. No blocking obstacle occupies the
navigation corridor. The robot is a kinematic SE(2) state driving the camera;
there is no physical robot chassis or dynamic collision controller.

The declared pinhole convention is RGB 640×480, horizontal FOV 90°, height
0.5 m, pitch 0°, camera forward along agent +X, +Y left and positive yaw CCW.
All-one 96×96 and 224×224 masks are preserved. This is an adapter choice, not
an assertion of official OmniVLA camera calibration.

Temporary red FRONT, green LEFT and blue RIGHT panels gave:

| Marker | Pixel center (x,y) | Dominant-color pixels |
| --- | --- | ---: |
| FRONT | (319.5,239.5) | 1600 |
| LEFT | (159.5,239.5) | 1680 |
| RIGHT | (479.5,239.5) | 1680 |

Measured camera aperture/focal ratio gives 90.0° horizontal FOV. All pass the
predeclared center/left/right pixel checks. Markers were removed after this
model-free test and were never added during scientific episodes. Camera
geometry and scene parameter values are unchanged from the prior config.

## RGB history

`context_size=5` means six images. Every scientific request used six actual
independently rendered/captured PNGs, oldest to newest, at 20-step intervals
(1/3 simulation second). Bootstrap is stationary at steps 0,20,40,60,80,100;
its 1.666667 s is excluded from the 40 s scientific timeout. Frame IDs and
simulation timestamps are unique; identical pixels while stationary are allowed.
The model-free moving test verified changing past-frame hashes.

At a busy inference slot a real frame is still captured into history, but no
request is queued. Histories never include future observations. The offline
validator checked every PNG, decoded RGB hash, request history identity and
observation timestamp. Scientific frame counts: STRAIGHT 52, LEFT 56, RIGHT 56.
The 1/3 s history interval is this adapter's declared convention; source alone
does not establish universal Edge training-history timing.

## Pose-goal convention

For observation pose `(x,y,yaw)` and world goal, evaluate
`inverse(T_W_R(t_obs)) @ T_W_G`: forward `c*dx+s*dy`, left `-s*dx+c*dy`,
and wrapped goal yaw minus observation yaw. Encode
`[forward/0.1, left/0.1, cos(delta_yaw), sin(delta_yaw)]` with the official
30 m radial position clamp. Goals here are within that clamp.

Deterministic FRONT/LEFT/RIGHT, translated/rotated-pose and clamp tests passed,
including comparison to the upstream compass-path convention. Every recorded
request was rechecked against the 60 Hz state at its observation time. No
model-ready or command-application pose was substituted as the goal anchor.

## IPC

A persistent Edge process in `artifacts-local/envs/edge-blackwell/` uses a local
Unix socket with newline JSON metadata. RGB transport is read-only lossless PNG
files, verified by file SHA256 and decoded-pixel SHA256. Native responses are
exclusive, read-only `.npy` arrays of finite float32 shape `(1,8,4)`, with file
hash verification. Request and observation IDs must match; episode IDs identify
trials. A client permits at most one outstanding request, with no pending-request
queue or catch-up. A dummy end-to-end exchange passed before science.

Edge and CLIP load once and remain alive across all three scientific episodes.
The existing checkpoint at revision `b1361b7e24f101edea795a98b00a826b61a97394`
was hash-verified, strictly loaded, and kept float32 on GPU. CLIP artifact hash
was verified; no redownload occurred. All-one masks, official resize/normalization
functions, pose-only modality 4, latest 224×224 RGB, black map/goal placeholders
and `xxxx` text were used. CPU tests demonstrated exact tensor equality with
unchanged official preprocessing for equivalent inputs and ordered six-frame
handling for distinct inputs.

The controller executes the unchanged upstream AST in the Edge NumPy runtime,
retaining float32 waypoint scaling and rounding. It selects index 4, multiplies
native XY by 0.1, uses DT=1/3, EPS=1e-8, clips v to [0,0.5], w to [-1,1], and
preserves maxv/maxw=0.3 ratio limiting. The zero-XY `clip_angle` bug is not fixed.

148 native predictions were retained; 146 were applied. At goal entry, two
outstanding endpoint responses were drained and recorded without further motion. All applied controller results exactly matched independent upstream-AST
recomputation. Native arrays were never changed for analysis or control.

## Combined memory

Combined infrastructure gate: three finite model inferences while Isaac remained
responsive (31, 2 and 11 rendered steps while waiting). No model-derived command
was applied during these infrastructure checks. Both processes exited 0.

Measured device-wide usage during the active combined-inference window:
**3,720 MiB / 16,311 MiB**. Device-wide peak during scientific execution: **3,821 MiB**.
Sampling interval is 0.25 wall seconds; values include desktop/other host GPU
work and may miss shorter peaks. These are observed memory measurements, not
tensor-allocation-only totals. No OOM, precision reduction, quantization,
resolution reduction, offload or model substitution was used.

## Scheduler / cadence

Physics/control is 60 Hz; observations target 3 Hz simulation time, every 20
steps. The agent holds zero before its first response and holds the last command
while one request is outstanding. A busy slot logs `MISSED_INFERENCE_SLOT` and
skips sending; there is no queue or catch-up burst. A completed response is
polled at a step boundary and used by the next physics step.

Simulation is stepped with synchronous rendering as fast as the runtime permits,
without wall-time pacing or artificial delay. The declared application simulation
time is the timestamp of the first completed step using that prediction.
The achieved update rate below is applied-count / episode duration; finite-window
endpoints can make it slightly greater than 3 even with a fixed 3 Hz request grid.

All interval columns below show **median / p95 / max**.

| Goal | RGB observation interval, sim s | Sent-request interval, sim s | Applied updates / sim s | Missed slots | Sim/wall ratio |
| --- | --- | --- | ---: | ---: | ---: |
| STRAIGHT | 0.333 / 0.333 / 0.333 | 0.333 / 0.333 / 0.667 | 2.903226 | 1 | 1.404979 |
| LEFT | 0.333 / 0.333 / 0.333 | 0.333 / 0.333 / 0.333 | 3.035714 | 0 | 1.415537 |
| RIGHT | 0.333 / 0.333 / 0.333 | 0.333 / 0.333 / 0.333 | 2.997003 | 0 | 1.483432 |

All model latency columns include every response, even unapplied endpoint
responses, and retain the first cold forward. Values are **median / p95 / max ms**.

| Goal | Inferences | Edge forward | Forward including CLIP | Service pipeline |
| --- | ---: | --- | --- | --- |
| STRAIGHT | 46 | 14.694 / 17.906 / 144.385 | 23.746 / 29.517 / 251.577 | 71.962 / 90.840 / 298.869 |
| LEFT | 51 | 12.575 / 17.424 / 17.708 | 20.072 / 29.310 / 30.393 | 68.016 / 83.413 / 94.941 |
| RIGHT | 51 | 15.047 / 17.436 / 17.923 | 24.803 / 30.551 / 31.031 | 68.725 / 87.237 / 147.331 |

| Goal | Observation → applied step, sim s | Recorded frame → applied step, wall ms |
| --- | --- | --- |
| STRAIGHT | 0.167 / 0.233 / 0.400 | 109.923 / 187.147 / 358.917 |
| LEFT | 0.150 / 0.233 / 0.250 | 97.914 / 132.697 / 135.689 |
| RIGHT | 0.183 / 0.233 / 0.250 | 90.422 / 148.453 / 157.563 |

Aggregate over 148 inferences: Edge forward
**14.537 / 17.488 / 144.385 ms**;
forward including CLIP **22.975 / 30.323 / 251.577 ms**;
service pipeline **70.028 / 85.977 / 298.869 ms**.
For 146 applied predictions, recorded-frame-to-applied-step wall delay is
**95.402 / 147.025 / 358.917 ms**, and simulation
delay is **0.167 / 0.233 / 0.400 s**.

Timing bookkeeping is explicit: the original episode summary's field named
`observation_to_application_wall_ms` measures request-send to the pre-step
response-handling marker. It is preserved unchanged in raw evidence. The
reported wall delay above is derived from the latest frame's recorded wall
timestamp to `application_step_completed_wall_time`, matching the declared
first-completed-step convention. Frame wall timestamps are recorded after PNG
save, so acquisition/PNG-write time before that timestamp is excluded; this is
not a calibrated sensor-exposure-to-actuation wall latency. Simulation timestamps
and the 60 Hz state/observation alignment were independently verified.

## Frozen scientific protocol

The protocol was written only after all nine gates passed: lifecycle, camera,
history, pose, IPC, strict load, combined memory, scheduler and controller parity.
The 83 pre-freeze CPU/regression checks passed. Config:
[`research/frozen-edge-isaac-no-obstacle-final.json`](../research/frozen-edge-isaac-no-obstacle-final.json).
SHA256:

```text
fb40d57f7999da622d99f33ed019132d1562dde40d14f8c506e40efde63ccbf2
```

It records the scene/camera/start/goals, physics and observation rates, history,
checkpoint, modality, unchanged controller, IPC, scheduler timing, success radius,
40 s timeout, seed, retry policy, gate evidence hashes and implementation hashes.
The seed is 0 for Edge Torch/NumPy; renderer bitwise determinism is not claimed.
A runtime guard rejects changed implementation files or incomplete gates.
An exclusive scientific registration prevents a second scientific group here.
All frozen runtime hashes and config hash remain unchanged after execution.

## STRAIGHT result

One primary run, **GOAL_REACHED**. Goal `[5.0, 0.0, 0.0]`; final pose
`[4.528027582172443, -0.15066332738515878, 0.004691037107258922]`. First XY goal-region entry at
15.500000 simulation seconds after bootstrap, distance
0.495436577 m, signed yaw error -0.004691037 rad.
Executed 930 physics steps in 11.032193 wall seconds.
46 requests, 45 applied predictions,
1 response(s) retained without application at episode end.
Missed slots: 1. First clip / final ratio limitation:
45 / 45 applied predictions.
Zero-XY upstream bug occurrences: 0.
No retry or parameter change followed this outcome.

## LEFT result

One primary run, **GOAL_REACHED**. Goal `[5.0, 2.0, 0.3805063771123649]`; final pose
`[4.708965660849915, 1.597122968039312, 0.3753079998027515]`. First XY goal-region entry at
16.800000 simulation seconds after bootstrap, distance
0.497001901 m, signed yaw error +0.005198377 rad.
Executed 1008 physics steps in 11.868291 wall seconds.
51 requests, 51 applied predictions,
0 response(s) retained without application at episode end.
Missed slots: 0. First clip / final ratio limitation:
51 / 51 applied predictions.
Zero-XY upstream bug occurrences: 0.
No retry or parameter change followed this outcome.

## RIGHT result

One primary run, **GOAL_REACHED**. Goal `[5.0, -2.0, -0.3805063771123649]`; final pose
`[4.5251658167691895, -1.8581047135768658, -0.4352833431959091]`. First XY goal-region entry at
16.683333 simulation seconds after bootstrap, distance
0.495582257 m, signed yaw error +0.054776966 rad.
Executed 1001 physics steps in 11.246444 wall seconds.
51 requests, 50 applied predictions,
1 response(s) retained without application at episode end.
Missed slots: 0. First clip / final ratio limitation:
50 / 50 applied predictions.
Zero-XY upstream bug occurrences: 0.
No retry or parameter change followed this outcome.

## Positive evidence

All three primary runs entered the goal region with actual RGB history and an
observation-time pose-only goal. The official native action shape, index 4,
controller, masks, precision and preprocessing are preserved. Every raw frame,
request, applied response and 60 Hz pose transition passed offline validation.
All four current Isaac processes exited successfully. Source/library knowledge
was reused without repeating the solved environment audit.

## Negative evidence

One inference slot was missed in STRAIGHT during the cold first prediction;
no request queue concealed it. All 146 applied predictions required both initial
clipping and final ratio limitation. This is the preserved official controller's
behavior, not a parameter tuned to make the goals pass. Zero upstream zero-XY
bug occurrences were observed; that does not repair or rule out the bug.
Two valid endpoint responses were deliberately left unapplied after success.
The prior startup failure and post-close SIGSEGV reports remain unchanged.

## Implementation limitations

This is a static, obstacle-free primitive scene with kinematic SE(2) integration,
not wheel dynamics, a physical chassis or collision avoidance. Camera calibration
and history timing are explicit adapter choices. PNG/hash/file transport adds
CPU and storage overhead; the service pipeline metric includes RGB reads,
verification, official preprocessing, inference, native-action write and controller
work, but excludes socket response serialization. Application-delay wall timestamp
limitations are stated above. Rendering may vary between runs despite the model
seed; no bitwise run-replay guarantee is made.

The historical recovery test was narrowed to its own recorded output directory
so future authorized runs do not invalidate historical evidence checks; its old
raw hashes and reports are still checked. No experiment code was modified after
freeze. Subsequent additions were offline evidence validation, tests, summaries
and documentation only.

## Experimental artifacts

Ignored local evidence is under `outputs/edge-final/` and
`artifacts-local/edge-final/`. Every process attempt includes command, change,
source hashes, raw log, result, exit status, memory samples and residual-process
check. The scientific group contains one `STRAIGHT`, `LEFT`, `RIGHT` directory,
each with:

```text
raw/rgb/                  # lossless actual rendered PNGs
raw/actions/              # immutable native (1,8,4) .npy
raw/events.jsonl
raw/history.jsonl
raw/requests.jsonl
raw/controller.jsonl
raw/robot_state.jsonl
summary.json
```

Per prediction: episode/request/observation IDs, six history IDs/times/hashes,
observation pose, world/local/normalized goal, send/ready/application wall times,
application simulation time, selected native/metric waypoint, preclip/clipped/
final/applied commands, native path/hash. Robot state is recorded at every 60 Hz
step with pose, command, distance and signed yaw error.

Offline validation manifest:
`artifacts-local/edge-final/scientific-validation-002.json`, SHA256
`9679cebf7712de2c5b19442ebb01d63805e4cdd44397784d7199b4feddd61e5e`. It records artifact hashes and validates all 148 native
outputs, 164 RGB images and 2,942 state records including three initial states.
Derived trajectory/distance figure:
`outputs/edge-final/scientific-001/derived/navigation-summary.png` (also SVG).
Large/raw artifacts are not committed.

Exact commands executed from `/home/gpuadmin/Workspace/OmniVLA` follow. The
supervisor rejects existing run directories; the scientific registration rejects
a second scientific group in this workspace. Reproduction requires a fresh
qualification workspace/evidence root, never overwriting these runs.

```bash
python3 research/run_isaac_final_attempt.py --run-id lifecycle-001 --phase lifecycle --change 'Use installed default fast shutdown; fsync evidence before final app.close(exit_code=code)'

python3 research/run_isaac_final_attempt.py --run-id camera-history-001 --phase camera --change 'Add model-free FRONT/LEFT/RIGHT marker and actual stationary/rolling history validation; same scene and camera geometry'

python3 research/run_isaac_final_attempt.py --run-id combined-001 --phase combined --change 'Add persistent strict-loaded Edge service, lossless dummy IPC/scheduler and three stationary inference checks with Isaac alive; no model commands applied'

python3 research/run_isaac_final_attempt.py --run-id scientific-001 --phase scientific --config research/frozen-edge-isaac-no-obstacle-final.json --timeout 600 --change 'Frozen protocol fb40d57f7999da622d99f33ed019132d1562dde40d14f8c506e40efde63ccbf2; one primary trial per declared goal, no tuning or retries'

PYTHONDONTWRITEBYTECODE=1 artifacts-local/envs/edge-blackwell/bin/python \
  research/validate_isaac_final_evidence.py \
  --run outputs/edge-final/scientific-001 \
  --output artifacts-local/edge-final/scientific-validation-002.json

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" \
  artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -v
```

The supervisor sets PYTHONDONTWRITEBYTECODE=1 and workspace-local XDG caches,
launches Isaac through `scripts/isaac6_python.sh`, and uses the existing Edge
interpreter in a separate process. Exact child commands and environment are in
each `attempt.json`. Final full suite: 87 passed, including existing Edge
regressions, infrastructure contracts and raw scientific evidence checks.
Unit tests are not substituted for simulator/model measurements.

## Insufficient evidence

One primary trial per goal is insufficient to claim general reliability,
robustness across scenes, real-robot performance or obstacle avoidance. No static
obstacle, pedestrian, moving obstacle, injected latency, OLD/FRESH chunk analysis
or reconciliation was performed.

Documented future direction only: (1) static-obstacle positive control;
(2) pedestrian/human avoidance positive control; (3) only after validated Edge
avoidance, collect successive OLD/FRESH chunks; (4) study latency separately.
No part of those stages was implemented in this task.

## Natural latency interpretation

The historical standalone medians were approximately 9 ms model forward and
17 ms full pipeline. In this combined rendered scene, the all-inference model
median is 14.537 ms; including CLIP it is 22.975 ms. The larger 70.028 ms service
pipeline includes file/PNG integrity and preprocessing overhead. The first cold
forward is retained in maxima and explains the early skipped slot.

These observations do not make Edge a naturally high-latency VLA. No artificial
latency was injected. A future explicit delay experiment must be motivated by
remote/cloud VLA deployment or communication delay and analyzed separately from
native model inference time.

## Decision

**NO_OBSTACLE_GOAL_REACHING_PASS**

All frozen infrastructure and scientific criteria were met. Scientific outcomes
are three GOAL_REACHED results; no MODEL_NAVIGATION_FAILURE,
IMPLEMENTATION_FAILURE or UPSTREAM_CONTROLLER_BUG occurred. The reliability and
obstacle claims listed above remain INSUFFICIENT_EVIDENCE. Work stops before any
obstacle or latency experiment.

## Exact next uncertainty

Can the same frozen OmniVLA-edge + Isaac adapter safely detour around a predeclared static obstacle without changing model, camera, history, controller, cadence, or goal semantics?
