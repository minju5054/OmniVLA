# OmniVLA-edge Static Obstacle Avoidance Qualification

**Scientific outcome: `COLLISION`. Static obstacle qualification did not pass.**
Infrastructure and recorded-evidence validation passed. The single primary
trial drove almost straight into the virtual collision envelope and stopped at
the first detected collision step. No scientific retry or post-outcome tuning
was performed.

## 1. Research question and scope

Can the same OmniVLA-edge configuration that reached obstacle-free goals safely
detour around a fixed cuboid blocking the STRAIGHT route and reach the goal
region? This is a prerequisite qualification for future research, not a test of
OLD/FRESH consistency, trajectory reconciliation or SE(2) graph optimization.
None of those methods, dynamic obstacles or subsequent trials were implemented.

## 2. Starting state and execution environment

- Date: 2026-10-13, Asia/Seoul.
- Branch: `research/omnivla-edge-static-obstacle`.
- Starting and scientific execution Git HEAD:
  `89b4e39c55af02cb9dc1a87c65f783f273e54ff6`.
- Upstream: `5182600cb4a9ee07684e17cdd2a6cbafc56b8a68`.
- Local and remote obstacle-free HEADs agreed before branch creation. The new
  branch preserved uncommitted GUI scripts and AGENTS/WORK_LOG changes. Existing
  `.vscode/` remains untracked and untouched. No reset, stash, rebase or force push.
- Research additions were executed before the final research commit; frozen
  content hashes bind those exact runtime sources to the resulting commit.
- Isaac: `6.0.1-rc.7+release.42383.32955d8d.gl` at `/home/gpuadmin/isaacsim`.
- GPU: NVIDIA RTX 5060 Ti, 16,311 MiB, driver 595.99.02.
- Edge service: existing isolated `artifacts-local/envs/edge-blackwell/` runtime,
  Torch 2.7.1+cu128, CUDA runtime 12.8, NumPy 1.26.4. This is the previously
  documented Blackwell runtime deviation, not official dependency parity.
- Same float32 Edge and float32 CLIP checkpoints; strict load, no missing or
  unexpected keys. Edge revision `b1361b7e24f101edea795a98b00a826b61a97394`, SHA256
  `f93b056f124380899fce3e087a5c2b41ce4d8c07e211daad52b6ebf02e80d04d`.
- Reused process-local CUDA isolation and default fast shutdown. No installed
  Isaac, system Python/CUDA, LightNav or TIC-VLA changes.

The original [obstacle-free report](EDGE_ISAAC_NO_OBSTACLE_FINAL.md), compact
result, frozen configuration and all 330 manifest-listed raw artifacts retain
their original hashes. Historical reports and compact results also match the
start audit. Existing model/service/controller/scheduler files were not edited.

## 3. Frozen scenario and evaluation convention

Configuration: [`research/frozen-edge-static-obstacle.json`](../research/frozen-edge-static-obstacle.json).
Protocol SHA256:

```text
f8cdbfe22a32a335968976c80a4c86994154866fcf859277a81f4ce911b73728
```

| Item | Frozen value |
| --- | --- |
| Initial SE(2) | (0, 0, 0) |
| World goal | (5, 0, 0) |
| Cuboid center | (2.5, 0, 0.5) m |
| Cuboid dimensions | (0.6, 1.0, 1.0) m |
| Cuboid yaw / motion | 0 rad / fixed throughout episode |
| Appearance | Opaque solid red, displayColor (0.8, 0.08, 0.08); no directional hint |
| Virtual circular footprint radius | 0.20 m |
| Required clearance | 0.10 m |
| Goal radius / timeout | 0.50 m / 40 simulation seconds after bootstrap |
| Physics / observation target | 60 Hz / 3 Hz simulation time |
| Camera | 640×480, 90° horizontal FOV, height 0.50 m, pitch 0° |

Only the obstacle, evaluation geometry/termination and additional measurements
were added. Checkpoint, official preprocessing and masks, pose-only modality 4,
observation-time goal frame, six actual RGB history frames spaced 1/3 s apart,
index-4 controller, velocity limits, Euler integration, IPC and scheduling remain
the baseline. No controller bug was repaired. No artificial latency or warmup
forward was inserted; the first scientific inference was cold.

The footprint is an explicit evaluation convention, **not a physical robot
chassis**. Clearance is Euclidean distance from the center to the closed XY
rectangle minus 0.20 m. Distance inside the rectangle is zero, so this convention
is not a full penetration-depth signed distance field.

The checker evaluates the complete straight center segment between consecutive
explicit-Euler poses, including when both endpoints are outside. It computes
the segment/AABB minimum using intersections, endpoints and corner projections;
rounded circle/rectangle corners are not replaced by an expanded square AABB.
It finds first contact by 64 bisections along the segment up to a minimum.
Circular footprint orientation does not affect clearance.

Collision (`clearance <= 0`) outranks simultaneous goal entry. Any swept
clearance below 0.10 m persists in whole-episode safety classification. A goal
entry with no collision is `SAFE_GOAL_REACHED` only if the minimum remains at
least 0.10 m; otherwise it is `GOAL_REACHED_LOW_CLEARANCE`. Timeout is
`MODEL_NAVIGATION_FAILURE`; execution faults and the upstream zero-XY controller
bug have separate classifications.

On contact, the current commanded 60 Hz step is recorded as executed and no next
step is taken. The checker does not clip a pose back to the surface, alter a
velocity or choose a turn. Hence the recorded detection endpoint can penetrate
slightly beyond first continuous contact. Both times are reported below.

## 4. Infrastructure validation and freeze

One model-free Isaac attempt, `infrastructure-001`, passed with exit 0, no
residual process, CUDA symbol error or segmentation fault. It made **zero model
inferences**. There were no failed Isaac infrastructure attempts in this task.

| Check | Measured result |
| --- | --- |
| USD world geometry | Center exactly (2.5,0,0.5); x extent 0.60000002384 m, other extents 1 m; within 1e-6 tolerance |
| Start footprint clearance | 2.0 m |
| Whole goal-region minimum footprint clearance | 1.5 m |
| Direct path blocked with radius + margin | PASS, first expanded-envelope contact fraction 0.38 |
| Actual red obstacle pixels | 21,051; image center x=319.108, y=239.907 |
| Positive 15° yaw | Obstacle image moves right to x=411.127 |
| Baseline camera-axis/FOV gate | PASS; temporary calibration panels removed |
| Actual stationary and rolling six-frame history | PASS |
| Spectator isolation | Observation camera pose unchanged; mean RGB difference 0.192786/255 after settling, below predeclared 1/255 threshold |
| Dummy lossless IPC / no-queue scheduler | 6 requests, 6 responses applied with zero commands; no model load |
| Scripted straight collision | Contact at about 6.6667 s, detected at step 401; no steering correction |
| Full pre-freeze suite | 104 tests passed |

The spectator gate adds a separate USD camera with no geometry. Render settling
is not bitwise deterministic; the small RGB difference is reported rather than
claimed to be zero. Scientific execution contains no replay markers or spectator
camera. The GUI later reads original recorded PNGs for its camera inset.

Tests include analytic face/corner distances, tangent contact, zero-length motion,
safe endpoints with a colliding sweep, rounded-corner noncollision, 100 seeded
dense-sampling comparisons, outcome priority, actual episode collision stop and
safe/low-clearance/failure handling using a deterministic model-free peer.
Existing controller parity, pose transform, preprocessing, raw action,
history, IPC and scheduler regressions also passed. Previously measured strict
load/combined-memory gates were reused; no new pretrial model forward occurred.

The new protocol was frozen only after these checks. It binds runtime source
hashes and gate evidence hashes. A separate exclusive registration at
`outputs/static-obstacle/scientific-registration.json` permits one primary
trial and refuses a second registration. The baseline frozen configuration is
unchanged. Scientific runtime/config hashes still match after execution.

## 5. Single primary scientific result

**`COLLISION`**, with no model retry. Isaac and the separate Edge service both
exited 0; no residual processes, segmentation fault or implementation failure.

| Metric | Result |
| --- | ---: |
| Goal reached | No |
| Geometric collision | Yes |
| First continuous contact after bootstrap | 7.301832592 sim s |
| Collision detection / termination | 7.316666667 sim s, step 439 |
| Minimum signed clearance | −0.004438785 m |
| Required-margin violation | Yes |
| First crossing of 0.10 m safety boundary | 6.967805882 sim s |
| First sampled violating state | 6.983333333 sim s |
| Final distance to goal | 2.995643566 m |
| Maximum absolute lateral deviation | 0.022212310 m |
| Executed path length | 2.005000000 m |
| Time to goal | Not reached |
| Inferences / applied predictions | 21 / 21 |
| Unapplied endpoint responses | 0 |
| Missed inference slots | 1 |
| Initial clip / final ratio limitation count | 21 / 21 |
| Upstream zero-XY bug occurrences | 0 |
| Actual images / recorded states | 27 / 440, including initial state |
| Episode wall time / sim-to-wall ratio | 4.830808 s / 1.514584 |
| Peak device-wide memory | 3,866 MiB / 16,311 MiB |

Final SE(2): `(2.004438785, -0.022212310, -0.072277013)`.
After the initial held-zero interval, linear velocity remained 0.30 m/s.
Angular velocity ranged from −0.100640 to +0.031566 rad/s; maximum absolute
heading was 4.141168°. Small turn-sign changes did not produce a detour. There
was no sustained stop, in-place rotation or timeout before collision.

Memory is sampled every 0.25 wall seconds and includes desktop/other GPU work;
it is not a model-only allocation peak. The scientific process group took
22.043077 wall seconds including startup/teardown. GUI replay ran afterward and
did not contribute to scientific scheduler or latency measurements.

### Timing measurements

All 21 predictions are included, including the cold first forward. Median / p95 /
maximum values below are milliseconds unless stated otherwise.

| Measurement | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| Edge forward | 15.667 | 18.694 | 139.971 |
| Forward including CLIP | 26.349 | 30.376 | 263.409 |
| Service pipeline | 76.970 | 85.647 | 315.024 |
| Recorded RGB frame → inference ready | 77.883 | 86.701 | 316.032 |
| Recorded RGB frame → first completed applied step | 96.136 | 143.527 | 328.884 |
| Observation → applied step, **simulation seconds** | 0.200 | 0.233 | 0.650 |

The service pipeline includes PNG reads/hash checks, official preprocessing,
inference, native action write and controller work, but excludes socket response
serialization. Frame wall timestamps are recorded **after PNG save**; the wall
delays above exclude acquisition/save time before that metadata timestamp.
They are not calibrated sensor-exposure-to-actuation delays. The legacy raw
summary field `observation_to_application_wall_ms` is retained and explicitly
means request-send to the pre-step application marker, as in the baseline.

Observations remained at 1/3 simulation second; requests skipped one occupied
slot without queueing or catch-up. Applied count/duration was 2.870159 Hz. The
renderer stepped faster than real time, without artificial delay. GUI playback
at 1x follows recorded **simulation time**, not the original wall-time pace.

## 6. Evidence and visualizations

Compact result: [`results/edge-static-obstacle-summary.json`](../results/edge-static-obstacle-summary.json).
Ignored raw root: `outputs/static-obstacle/scientific-001/`.

```text
attempt.json, outcome.json, isaac.log, service.log, gpu-memory.jsonl
service/strict-load.json, responses.jsonl, stopped.json
isaac/result.json, startup-rgb.png
isaac/STATIC_CENTER/summary.json
isaac/STATIC_CENTER/raw/rgb/              # 27 original lossless images
isaac/STATIC_CENTER/raw/actions/          # 21 original (1,8,4) float32 arrays
isaac/STATIC_CENTER/raw/history.jsonl
isaac/STATIC_CENTER/raw/requests.jsonl
isaac/STATIC_CENTER/raw/controller.jsonl
isaac/STATIC_CENTER/raw/robot_state.jsonl
isaac/STATIC_CENTER/raw/events.jsonl
derived/metrics.json
derived/navigation-summary.png          # XY, goal distance, clearance, v/w
derived/navigation-summary.svg
derived/predictions-vs-executed.png
derived/predictions-vs-executed.svg
derived/selected-predictions-world.json
```

Offline validation passed: request/observation identity and timestamps, actual
six-frame inputs/hashes, observation-time world/local goals, every raw action,
upstream-controller recomputation, held commands, all 60 Hz pose integrations,
swept clearance, first collision and immediate termination agree. The raw
episode manifest contains 54 files. Original scientific files are read-only.
Validation manifest: `artifacts-local/static-obstacle/scientific-validation-001.json`.
All original obstacle-free artifacts and historical reports passed preservation
checks again. Large/raw artifacts are ignored and are not included in Git.

The main figure overlays the prior obstacle-free STRAIGHT path and goal distance
as a reference. That prior run had **no obstacle**; its line crossing the new
obstacle footprint is not a historical collision. These separate one-shot runs
are not controlled repeats or statistical evidence of a performance difference.

Predicted spatial comparison uses fixed first/middle/last request indices
0/10/20, observed at 0/3.6667/7.0 simulation seconds after bootstrap. Verified
source semantics: Edge returns already-cumulative x-forward/y-left coordinates,
metric conversion is XY ×0.1 m, heading is local `[cos(yaw), sin(yaw)]`, and the
world transform uses the observation pose. There is no second cumulative sum
and **no intrinsic timestamp assigned to a waypoint row**. Source tests verify
`torch.cumsum` and heading normalization. Exact checkpoint training provenance
is still incomplete; spatial plots do not resolve intrinsic waypoint timing.

### Existing GUI replay

The existing GUI was extended, not replaced. It shows the actual obstacle,
recorded pose/heading, virtual evaluation footprint, goal, executed path, current
clearance, collision/safety status and original camera images. Visual overlays
exist only in replay; no model is loaded and no command is computed by the GUI.

`gui-001` reached `GUI_REPLAY_READY`, rendered 4,400 frames and cycled the recorded
episode eight times before normal window closure. No Python exception was
recorded. A saved viewport capture was visually checked. Replay evidence and
the launch command/PID are under `artifacts-local/static-obstacle/gui-001/`.
This is playback verification, not another scientific trial.

One-line command from any working directory:

```bash
bash ~/Workspace/OmniVLA/scripts/isaac_static_obstacle_gui.sh
```

## 7. Positive and negative evidence

**Positive technical evidence:** the fixed obstacle was rendered in the actual
input camera, the baseline frame/pose/control contracts held, native outputs
were finite and preserved, and swept collision monitoring terminated correctly.

**No positive avoidance evidence:** this trial neither maintained the required
margin nor reached the goal. It did not demonstrate a safe detour.

**Negative navigation evidence:** the agent continued toward the central object,
deviated laterally by at most 2.22 cm, violated the safety boundary and collided.
The final selected spatial prediction in the fixed comparison also points into
the obstacle footprint. All 21 applied commands underwent official clipping and
ratio limiting. There was no implementation fault or upstream zero-XY exception
to explain away this measured navigation outcome.

## 8. Implementation limitations

- Primitive synthetic scene and adapter camera geometry can differ from the
  checkpoint's training distribution. Rendering the obstacle in RGB does not
  establish how the model represents or uses it.
- Collision is circle/rectangle evaluation of piecewise linear kinematic motion;
  there is no physical chassis, wheel dynamics, contact response or braking
  simulation. USD's float extent differs from the declared double extent by
  roughly 2.4e-8 m; geometry validation allows 1e-6 m.
- Detection occurs at a 60 Hz step boundary. The 4.44 mm terminal clearance below
  zero is retained instead of repairing the executed pose. Continuous first
  contact is reported separately.
- Every command saturated. The outcome evaluates the unchanged upstream
  checkpoint plus this controller/adapter, not an isolated perception network.
- Headless science and GUI replay have different rendering/wall-time behavior.
  Only headless records enter scientific latency measurements.
- Cumulative XY and local heading conventions have source/test support;
  universal intrinsic waypoint timing and full Edge training provenance remain
  unresolved. No OLD/FRESH temporal alignment is inferred.

## 9. Scientific interpretation and stop

Under this one frozen configuration, OmniVLA-edge with the preserved index-4
controller **failed static obstacle avoidance qualification**. This is a valid
negative navigation observation with passing infrastructure, not a general proof
that the model cannot avoid obstacles. One trial cannot establish robustness,
success probability or statistical degradation from the obstacle-free reference.

The result does not identify whether the cause is visual/domain sensitivity,
pose-only conditioning, action generation or closed-loop control. A visually
present obstacle and almost straight output are insufficient to claim the model
did not perceive it. No counterfactual/ablation trial was authorized or run.

No conclusion about dynamic avoidance, OLD/FRESH reconciliation or graph
optimization success/failure follows. Work stops after this qualification,
report, GUI presentation and Git publication; no follow-up is started.

## 10. Exact next uncertainty

What causes the missing lateral avoidance response to this visible obstacle in
pose-only mode: checkpoint visual/domain sensitivity or the fixed
waypoint-index-4 closed-loop interface? This causal distinction remains
unresolved by the single trial and is not investigated automatically.

## Reproducibility, attempts and checks

Exact child commands, environment, source hashes, launch timestamps, lifecycle
results and memory samples are in each attempt directory. Only one infrastructure
attempt and one scientific attempt were run; all completed without execution
failure. Unit/analysis/GUI runs never execute the VLA policy.

```bash
python3 research/run_static_obstacle_attempt.py --phase infrastructure --run-id infrastructure-001 --config artifacts-local/static-obstacle/proposed-config.json --change 'Model-free obstacle geometry, visibility, yaw, history, spectator isolation, dummy IPC and swept collision qualification'

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -v

python3 research/run_static_obstacle_attempt.py --phase scientific --run-id scientific-001 --config research/frozen-edge-static-obstacle.json --timeout 600 --change 'Frozen f8cdbfe22a32a335968976c80a4c86994154866fcf859277a81f4ce911b73728; one primary central static-obstacle trial; no tuning or retries'

PYTHONDONTWRITEBYTECODE=1 artifacts-local/envs/edge-blackwell/bin/python research/validate_static_obstacle_evidence.py --output artifacts-local/static-obstacle/scientific-validation-001.json

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python research/analyze_static_obstacle.py --output outputs/static-obstacle/scientific-001/derived

bash scripts/isaac_static_obstacle_gui.sh --output artifacts-local/static-obstacle/gui-001/viewer
git diff --check
```

These recorded experiment commands are not a request to rerun the primary:
existing output directories and the exclusive registration reject reuse.
The GUI launcher alone is repeatable and allocates a fresh ignored diagnostics
directory. Pre-freeze logs: `artifacts-local/static-obstacle/pre-freeze-tests.log`;
post-analysis suite: `artifacts-local/static-obstacle/final-tests.log`.
Final full suite: **108 tests passed**. Syntax checks and `git diff --check`
passed; all frozen runtime/config hashes remained unchanged after analysis.
Staged whitespace validation initially found trailing blank lines in two new
files (the earlier unstaged check does not examine untracked additions). The
unfrozen offline validator was trimmed. The already-executed frozen episode's
trailing blank is retained under a documented per-file `blank-at-eof` exception
in `.gitattributes`, preserving its exact protocol SHA; other whitespace checks
remain enabled. This is a packaging correction, not a scientific source change.
