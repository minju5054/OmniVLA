# OmniVLA-edge obstacle response: offline diagnostic

Date: 2026-10-13. Branch: `research/omnivla-edge-obstacle-offline-diagnostic`.

**Result:** The paired inputs produce different native predictions, especially
their forward extent at 1.5 and 1.0 m from the obstacle. No pair reaches the
predeclared 0.20 m lateral-response screen, and no prediction demonstrates a
safe passage. The original 21 chunks contain no hidden safe detour in their
eight-waypoint spatial polylines. Controller saturation also removes the
observed differences in the selected waypoint's forward extent from final
linear commands. These findings separate output sensitivity from safe avoidance;
they do not establish the model's internal obstacle recognition or a successful
alternative controller.

No new navigation was executed. Part A used existing immutable evidence only;
Part B captured stationary camera inputs and made exactly eight offline requests
with one strict float32 Edge/CLIP load. No additional scientific experiment,
retry, reconciliation, graph optimization, training, or controller repair ran.

## Research questions and source of truth

- RQ1: Does native prediction change between obstacle-present and obstacle-absent
  RGB at the same pose and goal?
- RQ2: Do the eight waypoints contain a safe detour missed by selection of row 4?

Starting local and remote static-obstacle SHA matched:
`8ac8352a06b5ac05a778cda09fbe9269604c7a5b`. Upstream source SHA:
`5182600cb4a9ee07684e17cdd2a6cbafc56b8a68`. The research revision is the commit
containing this report; exact executed additions were hashed before inference
because they were still uncommitted at execution time.

The [frozen protocol](../research/frozen-edge-obstacle-offline.json) has SHA256
`bef44a47d5679f68581f4a0fe804f7bb8a8ebe0a5b55f80506c54f6ecd8fa47c`.
It records 31 source hashes, the initial plan, complete Part A metrics, input
manifest, prior-evidence audit, capture termination, and 124 passing CPU tests.
Freeze precedes exclusive batch registration and every inference request.
The [compact result](../results/edge-obstacle-offline-diagnostic.json) contains
all pair deltas, case command/latency summaries, and native file hashes.

Confirmed repository facts:

- `inference/model_omnivla_edge.py` returns **already cumulative** XY via
  `torch.cumsum` and normalized local heading vectors. The diagnostic never
  applies another cumulative sum.
- Returned shape is `(1,8,4)`, float32. XY represents local +X forward, +Y left;
  metric scale is 0.1 m. World XY and heading rotate by observation-time yaw,
  then XY translates by observation-time position. Rows have no intrinsic
  timestamps. Straight interpolation is evaluation-only.
- The unchanged controller AST in `research/edge_contracts.py` selects row 4
  after the official float32 in-place XY scaling. For nonzero `dx`, it computes
  `v=dx/(1/3)` and `w=atan(dy/dx)/(1/3)`. It does **not** use the heading vector
  in this branch. The zero-XY `clip_angle` NameError remains unfixed and tested.
- Initial limits are `v:[0,0.5]`, `w:[-1,1]`; subsequent coupled limits are
  `maxv=maxw=0.3`. The near-zero angular branch can set `w=0`.
- Original controller, scheduler, model, precision, preprocessing, scene and
  camera modules remain byte-identical. The scheduler is not invoked for this
  independent offline request batch.

Geometry uses the existing `research/static_obstacle_geometry.py`: rectangle
X `[2.2,2.8]`, Y `[-0.5,0.5]`, virtual radius 0.20 m, required clearance 0.10 m.
Clearance is center-to-closed-rectangle distance minus radius; values inside
the rectangle saturate at -0.20 m, rather than measuring penetration depth.
Every polyline includes observation center → row 0 and all seven later segments.
Centerline intersection and virtual-footprint collision are recorded separately.

The predeclared conservative safe-passage test requires minimum clearance
at least 0.10 m **and** a span from observation X <=1.9 to predicted X >=3.1 m
(radius-plus-margin-expanded obstacle extent). Collision takes precedence,
then low clearance, then safe passage; otherwise the horizon is insufficient.
A short clear prefix is never counted as a successful passage.

## Part A: all 21 original predictions

Source: `outputs/static-obstacle/scientific-001/isaac/STATIC_CENTER/raw/`.
No new simulation or inference occurred during this analysis.

| Classification | Count | Fraction | Request IDs |
|---|---:|---:|---|
| PREDICTED_COLLISION | 7 | 33.33% | 14–20 |
| PREDICTED_LOW_CLEARANCE | 3 | 14.29% | 11–13 |
| PREDICTED_SAFE_PASSAGE | 0 | 0% | none |
| HORIZON_INSUFFICIENT | 11 | 52.38% | 0–10 |

Thus 10/21 (47.62%) have collision or margin risk. None of the 168 waypoint
prefixes satisfies safe passage. A prefix stopping before the obstacle cannot
demonstrate that an alternative waypoint index would solve avoidance.

| Waypoint index | Point collision count /21 | Maximum absolute local Y (m) | Hypothetical final w range (rad/s) |
|---|---:|---:|---:|
| 0 | 1 | 0.00385 | -0.05843 to +0.09809 |
| 1 | 2 | 0.01179 | -0.08634 to -0.01352 |
| 2 | 3 | 0.01854 | -0.09003 to -0.00401 |
| 3 | 4 | 0.02820 | -0.10248 to -0.00254 |
| 4 | 4 | 0.03468 | -0.10064 to +0.03157 |
| 5 | 5 | 0.04389 | -0.10587 to +0.03674 |
| 6 | 6 | 0.05231 | -0.10848 to +0.04667 |
| 7 | 7 | 0.06014 | -0.10998 to +0.04268 |

All-eight-waypoint maximum absolute local lateral displacement is 0.06014 m;
maximum absolute world Y is 0.09334 m. Selected row-4 local Y ranges from
-0.03468 to +0.01005 m. Later rows extend further into the obstacle, rather
than revealing a safe alternate polyline. Commands for all 168 rows are retained
as **hypothetical comparisons**, with no alternative-index navigation run.

All 21 selected-row commands reproduce the original pre-clip, clipped, final,
selected float32 metric values, and actually applied commands **exactly** in
the pinned Edge environment. All 21 use XY without heading. No zero-XY bug
occurred for any of the 168 hypothetical commands.

| Stage | v range (m/s) | w range (rad/s) |
|---|---:|---:|
| Pre-clip | 1.356207–1.928404 | -0.167733 to +0.052610 |
| Initial clip | 0.500000 throughout | -0.167733 to +0.052610 |
| Final / actually applied | 0.300000 throughout | -0.100640 to +0.031566 |

Initial clipping reduces v by 0.856207–1.428404 m/s in all 21 requests and
does not clip w. Ratio limiting removes another 0.20 m/s. It multiplies w by
0.6 in 20 cases; one near-zero value is set to zero. Signed angular change
from pre-clip to final ranges from -0.021044 to +0.067093 rad/s.
Except for that near-zero branch, ratio limiting preserves **w/v** relative
to the initially clipped command. Initial v clipping actually increases
absolute curvature relative to the raw controller command. A reduction in
absolute w alone is therefore insufficient evidence that limiting removed
an otherwise safe turn. No such safe turn is present in these predictions.

All per-chunk native arrays, transformed XY/headings, eight clearances, swept
segments, endpoint displacements, classifications and clipping deltas are in
`outputs/edge-obstacle-offline/existing-chunks/derived-v2/metrics.json`.
The initial `derived/` folder is retained; `derived-v2/` corrects figure aspect
and documents the near-zero curvature exception without changing predictions.

![All 21 chunks](../outputs/edge-obstacle-offline/existing-chunks/derived-v2/all-21-diagnostics.png)

![Final six chunks](../outputs/edge-obstacle-offline/existing-chunks/derived-v2/last-six-chunks.png)

## Part B: frozen paired inputs and inference

The [plan](../research/edge-obstacle-offline-plan.json) was registered before
new captures or model results. Order: 2.0, 1.5, 1.0, 0.7 m from obstacle front;
PRESENT then ABSENT at each distance. Camera poses are respectively
`(0.2,0,0)`, `(0.7,0,0)`, `(1.2,0,0)`, `(1.5,0,0)`; goal is `(5,0,0)`.
Normalized goals are `[48,0,1,0]`, `[43,0,1,0]`, `[38,0,1,0]`, `[35,0,1,0]`.

One scene instance uses the unchanged primitive scene, lighting, camera and
RaytracedLighting settings. Only the original opaque red cuboid is added or
removed between conditions. Each case settles for 120 steps, then independently
captures six actual `camera.get_rgba` frames at steps 0,20,40,60,80,100 at 60 Hz.
There are **48 distinct acquisition records and PNG paths**, not six copies
of one frame. Recorded relative times are 0 through 5/3 s; paired relative
times match while absolute simulation, physics-step and wall timestamps remain
independent and increasing. All six hashes happened to differ within each case.

Input gates passed: actual camera positions/orientations and USD intrinsics,
pose/goal encoding, renderer/scene metadata, paired relative history times,
condition identity, pixel hashes and absence of replay/calibration markers.
The only differing stage prim between each pair is `/World/StaticObstacle`.
All 24 absent captures contain zero red-dominant pixels after settling; saved
images show the removed object and its shadow are absent. The red-pixel gate
is a color/ghost screen, not a semantic segmentation guarantee.

Pixel differences below are channel means on the 0–255 scale, using the latest
capture in each pair. Same-condition variation uses frames 1–5 versus frame 0.

| Front distance | Present red pixels | Whole-image paired MAE | Outside red-mask MAE | Within-present MAE range | Within-absent MAE range |
|---|---:|---:|---:|---:|---:|
| 2.0 m | 25,599 | 7.1222 | 0.7268 | 0.1567–0.2874 | 0.1518–0.4936 |
| 1.5 m | 45,010 | 12.2517 | 1.1231 | 0.1963–0.2399 | 0.1277–0.1772 |
| 1.0 m | 102,393 | 26.4405 | 1.7276 | 0.2995–0.4007 | 0.1704–0.2167 |
| 0.7 m | 208,344 | 51.7787 | 2.1183 | 0.4426–0.5745 | 0.1140–0.1612 |

Outside-object pixels include shadows, disocclusion, indirect illumination,
anti-aliasing and renderer variation; they are not assumed identical or pure
noise. A supplementary **post-inference descriptive** shadow-associated screen
counts lower-half pixels outside the red mask whose Rec.709 luminance is >5
units darker with the obstacle: 6,840 / 10,846 / 21,968 / 7,160 pixels, with
mean darkening 10.344 / 10.595 / 9.022 / 14.985 units. This is a documented
proxy, not semantic shadow attribution or a new acceptance threshold.
It did not modify the frozen input gates, material screen, or scientific run.

![Input variation and shadow proxy](../outputs/edge-obstacle-offline/paired-001/derived/input-noise-and-shadow.png)

Capture process exit code was 0. Inputs and source hashes were frozen only
after capture validation and 124 passing model-free CPU tests. The service
then processed the eight scheduled requests once each, with no warmup or
unrecorded forward. No request was omitted, retried or tuned. Service exit
code 0; stopped record reports `inference_count=8`, `load_count=1`.

Runtime: RTX 5060 Ti, Edge environment Python 3.10, Torch `2.7.1+cu128`, CUDA
12.8, NumPy 1.26.4; Isaac 6.0.1 uses separate Python 3.12 / NumPy 2.3.1.
Model and CLIP are native float32, eval mode, strict checkpoint load with zero
missing/unexpected keys, no substitution, quantization or offload. Seeds are
Torch/NumPy 0 once at load, no per-condition reseeding. Checkpoint revision
`b1361b7e24f101edea795a98b00a826b61a97394`, SHA256
`f93b056f124380899fce3e087a5c2b41ce4d8c07e211daad52b6ebf02e80d04d`;
CLIP SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.

Cold request 0 latency: model 118.606 ms, forward including CLIP 202.218 ms,
pipeline 239.207 ms. Remaining seven median / p95: model 10.367 / 11.610 ms;
forward including CLIP 14.269 / 15.878 ms; pipeline 49.221 / 52.026 ms.
These measurements are offline pipeline latencies, not navigation latency.
Cold and subsequent results remain separate; cold output was not discarded.

## Paired prediction measurements

Every difference is PRESENT minus ABSENT. Both predictions are evaluated
against the same original obstacle geometry. An absent prediction labeled
collision below is a **counterfactual spatial comparison**, not an absent-scene
collision experiment. I = HORIZON_INSUFFICIENT, C = PREDICTED_COLLISION.

| Front distance | Max abs ΔY (m) | RMS XY difference (m) | Row-4 displacement difference (m) | Min clearance P / A (m) | Class P / A |
|---|---:|---:|---:|---:|---|
| 2.0 m | 0.007574 | 0.017150 | 0.016703 | +0.789973 / +0.811869 | I / I |
| 1.5 m | 0.020414 | 0.189332 | 0.186812 | +0.606634 / +0.303947 | I / I |
| 1.0 m | 0.011054 | 0.515523 | 0.507677 | +0.621176 / -0.200000 | I / C |
| 0.7 m | 0.010613 | 0.042871 | 0.043659 | -0.200000 / -0.200000 | C / C |

All four pairs are below the predeclared **max absolute lateral difference
>=0.20 m** material screen. This screen is neither statistical significance
nor a claim that smaller differences are absent. In particular, forward
endpoint X for PRESENT / ABSENT is 1.010027 / 0.988131, 0.693366 / 0.996053,
0.178824 / 1.002719, and 0.942217 / 1.007874 m. The D100 forward contraction
is **0.823896 m**, a large longitudinal change despite a small lateral change.
Maximum absolute heading-angle difference over the rows is 4.316°, 5.842°,
8.780°, and 1.459°, respectively. All per-row local/world deltas and wrapped
heading angles are retained in the compact JSON and full comparisons JSON.

![Paired native trajectories and original inputs](../outputs/edge-obstacle-offline/paired-001/derived/paired-trajectories-and-rgb.png)

![Pair differences](../outputs/edge-obstacle-offline/paired-001/derived/paired-differences.png)

## Model versus controller interpretation

**Measured controller behavior:** each new native row-4 command exactly matches
the service's unchanged controller under the Edge NumPy runtime. All eight
final v values are 0.3 m/s. The selected-row command differences are:

| Distance | Δv pre-clip (m/s) | Δw pre-clip (rad/s) | Δv final (m/s) | Δw final (rad/s) |
|---|---:|---:|---:|---:|
| 2.0 m | +0.045836 | -0.032233 | 0 | -0.019340 |
| 1.5 m | -0.560069 | +0.048400 | 0 | +0.029040 |
| 1.0 m | -1.522710 | +0.158662 | 0 | +0.124441 |
| 0.7 m | -0.130728 | -0.016530 | 0 | -0.009918 |

At D100, pre-clip v is 0.367638 (PRESENT) versus 1.890348 (ABSENT); both become
0.3 m/s. Angular commands remain different: +0.110470 versus -0.013971 rad/s.
Thus forward-extent differences can disappear through this interface even
when raw predictions change. This is evidence of interface compression,
not proof that the model intended a stop: waypoint rows have no known intrinsic
timing, and the conversion to velocity is the controller's convention.

**Research interpretation, RQ1:** condition-associated native response exists,
most visibly in longitudinal extent at D150/D100. Calling the model globally
RGB-insensitive would contradict these measurements. Lateral avoidance
sensitivity is small in these four pairs, and no safe spatial passage is
observed. D070 is non-monotonic relative to D100: close-range present output
again extends into the obstacle. No additional cases were run to explain it.

**Research interpretation, RQ2:** the original complete eight-point chunks
provide no safe bypass that index 4 alone discards. Other rows sometimes
produce different hypothetical angular commands, but their spatial prefixes
are insufficient or unsafe. The observed heading-vector changes are unused
by the normal controller branch; this is an interface fact, not evidence that
heading-based control would succeed. Absolute w attenuation alone does not
demonstrate a causal controller failure because v also changes.

**Positive evidence:** valid obstacle-conditioned RGB changes, native geometric
differences, substantial D100 forward contraction, strict source/controller
parity, complete request execution and reproducible analysis.

**Negative evidence:** zero safe-passage chunks in Part A or Part B, zero
material lateral-response pairs, and a colliding present chunk at D070. The
new stationary inputs do not establish a reliable upstream safe FRESH generator.

**Unverified hypotheses and limits:** semantic recognition of the cuboid,
intent to brake, domain mismatch as the cause, and an alternative index or
controller solving avoidance remain untested. The scene is a synthetic red
cuboid with simple background; camera calibration is the frozen adapter choice,
not recovered training calibration. Each condition has one model request;
renderer noise, shadows and object appearance co-vary, with no same-condition
output repetitions. Seed/eval policy does not establish bitwise GPU determinism.
The old navigation used moving six-frame histories while these histories are
stationary, so Part A and B do not isolate motion-history effects. Native
polylines are neither timestamped nor necessarily dynamically executable.

Edge remains a candidate upstream generator with measured visual-condition
sensitivity. It is not qualified as a safe obstacle-avoiding FRESH generator
by this experiment, and these results do not justify starting reconciliation
under an assumption of upstream safe avoidance.

**Exact next uncertainty:** whether the strong D100 longitudinal contraction
is reproducible beyond within-condition rendering/input variation.
**One recommended next experiment, not performed:** a preregistered repeated
stationary PRESENT/ABSENT offline test at D100 only, using independent actual
six-frame histories per replicate and the unchanged model/camera/interface,
to estimate the longitudinal effect and its within-condition variability.

## GUI, integrity and reproduction

```bash
bash ~/Workspace/OmniVLA/scripts/isaac_obstacle_offline_gui.sh
```

The existing viewer is extended through explicit `--offline-protocol` dispatch;
the collision launcher remains byte-identical. GUI mode uses saved data from
this diagnostic, with fixed case poses, both native predictions, selected row
4, goal, obstacle/reference footprint, safety envelope and original RGB. It
labels `OFFLINE PREDICTION — NO ROBOT EXECUTION`. Case switches jump between
fixed poses; no fake driven path or old successful run is substituted. Prediction
lines are elevated only for visibility; the original input images contain no
viewer overlays. The viewer loads no model and applies no predicted command.

GUI verification `artifacts-local/edge-obstacle-offline/gui-002/verification.json`
records all eight case IDs, 5,091 rendered frames and 12 automatic case changes;
the actual Isaac window and original RGB were visually inspected. Screenshots
are `window.png`, `desktop.png`, and `overview.png` in that directory.
The first viewer attempt stopped before creating a GUI because exact numeric
validation crossed NumPy 1.26.4 → 2.3.1 scalar-promotion behavior. Maximum
difference was 2.48856e-7. Viewer-only validation now allows <=5e-7 absolute
rounding while reading the original saved numbers; identity/classification
checks remain exact. Scientific metrics and all frozen sources were unchanged.
The full scientific re-analysis under the pinned Edge runtime is exactly equal.

Final tests: **126 passed** (124 pre-inference, plus two viewer regression tests).
Validation checks source/input/protocol hashes, shape/dtype/immutability,
pose/goal/history identity, present/absent pixels, all-waypoint geometry,
classification, original row-4 parity, and exact repeated metrics. All **416**
files under the prior obstacle-free and static scientific runs match the
pre-task manifest, including original 21 actions and 27 static RGB files.
New native `.npy` and raw records are read-only; prior evidence is never edited.

Local evidence:

| Evidence | Location |
|---|---|
| Original-chunk metrics / six requested plots | `outputs/edge-obstacle-offline/existing-chunks/derived-v2/` |
| 48 stationary RGB and capture metadata | `outputs/edge-obstacle-offline/capture-001/` |
| Eight native actions, full arrays and request metadata | `outputs/edge-obstacle-offline/paired-001/raw/` |
| Strict load, eight service responses, single-load stop | `outputs/edge-obstacle-offline/paired-001/service/` |
| Full paired metrics / plots / input diagnostics | `outputs/edge-obstacle-offline/paired-001/derived/` |
| Input manifest, freeze, tests, launch/failure/exit logs | `artifacts-local/edge-obstacle-offline/` |
| Preservation and reproduced metrics audit | `artifacts-local/edge-obstacle-offline/final-validation-002.json` |

Generated images, raw tensors, checkpoints, environments and caches are ignored
local evidence, not GitHub payloads. Relative report image links resolve in
this local workspace. The pre-existing untracked `.vscode/` is untouched.

Completed commands (from repository root; exclusive writers refuse overwrite):

```bash
# Part A ran before capture or model inference.
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python research/analyze_offline_obstacle.py --part A --output outputs/edge-obstacle-offline/existing-chunks/derived-v2
# Actual capture was supervised with a 300-s host timeout and exit record.
bash scripts/isaac6_python.sh research/offline_obstacle_capture.py --output outputs/edge-obstacle-offline/capture-001
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 artifacts-local/envs/edge-blackwell/bin/python research/freeze_offline_obstacle.py
# Already completed; exclusive registration intentionally prohibits a second batch.
PYTHONDONTWRITEBYTECODE=1 artifacts-local/envs/edge-blackwell/bin/python research/run_offline_obstacle_inference.py --protocol research/frozen-edge-obstacle-offline.json
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python research/analyze_offline_pairs.py --output outputs/edge-obstacle-offline/paired-001/derived
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$PWD/artifacts-local/edge-matplotlib" artifacts-local/envs/edge-blackwell/bin/python research/validate_offline_obstacle.py --output artifacts-local/edge-obstacle-offline/final-validation-002.json
```

For model-free reproduction, choose a **new** output directory for the Part A,
paired-analysis or validator command; no inference rerun is needed. Supplementary
pixel plots reproduce with `research/analyze_offline_input_pixels.py --output`
and a fresh directory under the same pinned Edge Python environment.
