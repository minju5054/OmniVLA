# OmniVLA-edge Pose + Language obstacle avoidance

Date: 2026-10-13. **Technically valid; not qualified for safe language-assisted obstacle avoidance in this frozen scenario.** All 32 offline requests completed. No safe full bypass was predicted. All four independent primary navigation trials collided; none entered the goal region. These are negative scientific outcomes, not runtime failures.

## Research question and provenance

Does the official Pose + Language modality improve native obstacle-conditioned trajectories and closed-loop avoidance of the same central red box? Separate text response, directional instruction following, visual-condition sensitivity, native geometric safety, and actual goal-reaching.

Source branch `research/omnivla-edge-obstacle-offline-diagnostic` was verified remotely at `eeb75f85cbcf316f15856388d0c3831eae4ef87d`; it had not advanced. This task uses only `research/omnivla-edge-pose-language-obstacle`. No historical branch was rewritten or deleted. Scientific launches record that starting Git HEAD because additive code was frozen before its publication commit. Exact executed code identity is the protocol's source SHA256 map; the publication SHA is the commit containing this report, verified against the remote after push and reported in the final response.

Historical evidence remains unchanged: obstacle-free 3/3 goal-region entries; pose-only static collision at 7.301833 simulation seconds with max absolute lateral deviation 0.022212 m; 21 original chunks with zero safe passages; historical D100 PRESENT/ABSENT forward endpoints 0.178824/1.002719 m with both final controller v=0.3 m/s. These are context, not new Modality-4 controls. Primary prompt comparisons below remain entirely within Modality 8.

References: [Edge preflight](EDGE_OMNIVLA_PREFLIGHT.md), [obstacle-free final](EDGE_ISAAC_NO_OBSTACLE_FINAL.md), [static obstacle](EDGE_STATIC_OBSTACLE.md), [offline diagnostic](EDGE_OBSTACLE_OFFLINE_DIAGNOSTIC.md), [compact new results](../results/edge-pose-language-obstacle.json).

## Source-confirmed interface and preservation

Official selection is `inference/run_omnivla_edge.py:327-328`: `pose_goal=True`, `satellite=False`, `image_goal=False`, `lan_prompt=True` selects ID 8. The unchanged official forward calls CLIP `encode_text` at line 335. `inference/model_omnivla_edge.py:192-195,208` defines ID-8 decoder masking and average-pooling weights. Token order is `[RGB0, RGB1, RGB2, RGB3, RGB4, RGB5, pose, map, goal_image, language]`:

```text
mask (True = masked): F F F F F F F T T F
pool weights:        1.25 1.25 1.25 1.25 1.25 1.25 1.25 0 0 1.25
```

Actual model and decoder hooks validated the ID, masks, pooling weights, finite float32 `(1,512)` CLIP embedding and its hash on every new inference. RGB history, pose and language were active. Map and goal image remained masked. No ID-4 fallback, alternate encoder, neural forward warmup or per-condition reseeding occurred.

The additive service reuses the original loader, image transforms, pose tensor preprocessing and official model forward. The new adapter constructs a function using the **identical original episode code object**, replacing only request metadata injection and response validation in a private globals mapping. Original episode/module globals are unchanged. Observation-time local goal is `(T_world_robot_obs)^-1 T_world_goal`, encoded as `[forward/.1, left/.1, cos(relative_yaw), sin(relative_yaw)]`, with the original clamp and float32 conversion. Every B observation pose was independently matched to saved robot state and the newest of six real captured frames.

Controller is unchanged: selected index 4 (zero-based fifth row), native XY scaled by 0.1 m in float32, `DT=1/3`, `EPS=1e-8`, `atan(dy/dx)`, original clipping and ratio-preserving final maxv=maxw=0.3. Nonzero-XY branches ignore heading-vector fields; the undefined `clip_angle` zero-XY bug is preserved and occurred zero times. No extra slowdown, steering, stopping policy, path repair or controller replacement was used. Collision only terminates an episode after the completed step, retaining its penetration and interpolated first contact time.

All official model/controller files and all frozen historical scientific sources remain byte-identical. The final audit verified 40 frozen source hashes and 504 historical evidence files. The only edited existing runtime file is the additive, non-scientific GUI dispatcher; old static and offline launchers are unchanged.

## Frozen prompts, tokenizer and protocol

**N** (7 tokens including SOT/EOT):

> Go to the goal.

**D** (23 tokens including SOT/EOT):

> A large solid red rectangular box is blocking the path directly ahead. Reach the goal while avoiding the obstacle.

**L** (33 tokens including SOT/EOT):

> A large solid red rectangular box blocks the path directly ahead. Steer to your left, pass around the box with safe clearance, then continue to the goal.

**R** (33 tokens including SOT/EOT):

> A large solid red rectangular box blocks the path directly ahead. Steer to your right, pass around the box with safe clearance, then continue to the goal.

LEFT means observation-frame +Y; RIGHT means observation-frame -Y. Directions are externally supplied task information. No exact obstacle coordinates, oracle path or geometric clearance evaluator were passed to the model. Each prompt is constant throughout its episode.

Official CLIP ViT-B/32 `tokenize(..., truncate=False)` succeeded for all four texts, matched the nontruncating result with `truncate=True`, and produced distinct int32 `(1,77)` padded token identities. Exact punctuation, UTF-8 SHA256, all padded IDs, token count and token-byte SHA256 are in the immutable [plan](../research/pose-language-obstacle-plan.json). Token SHA256 hashes C-contiguous little-endian int32 bytes. Per-request embedding hashes were constant within each instruction across both phases and different across all four instructions.

| ID | UTF-8 SHA256 | Token SHA256 |
| --- | --- | --- |
| N | a0434ecc60ebb1e2b3bd752fddeaf39389f3ea6fe8b407d58103b7c33bf737e7 | c395e4d58231b32dcb9f599d457804e42b1118626488cae23579ec3215b12c28 |
| D | fc40b18704c76f934d1e4493805e5735d09d54cfdf896a02192a3ec76e2f2804 | f3ddf0bd7fe2f7c2bef8980f9b3f3e471b37192fd045a315659d672f659b9bc7 |
| L | 3049f0963531fdddfe738aa1560afc5023a3751c8fcd45d04625141140514bf3 | 699060a675db64d52838ca6b1d8245a632a1ade7bc01d2e7957f9e8bcebd644a |
| R | 855499c3b61fec6e08940f170fdd62cde0417f6b2d4bfe76cc2383e7faaf0c5b | 3baaf9feabf22e51659622d551cdc4195911baecc3f7c7b1d57da0ac7685b309 |

| ID | Float32 CLIP embedding SHA256 |
| --- | --- |
| N | 254cd8534e329d49dfa6ba0ea476aff88421c57f16e916bb7fa77810d7e89fd9 |
| D | 7b1a055b90b5007b13e7a0493b2bc1e0fcb47b62a9a5f5b402924f0f48fe8a49 |
| L | 813d8ea5d57be932a5063c870715387df9a07bb4c19cf1d0b5549fb65d0e2e74 |
| R | 663a7830f91f4ca6803384a5a1dd81d024e6994af46442aef3fe2f8eab386062 |

Plan SHA256: `289dea38528cce277629dd347734e8da9c96235d6e96a6e31de216c6ff4b6c36`. Full [protocol](../research/frozen-pose-language-obstacle.json) SHA256: `6148e88e3e4d417ee78cad03efde6567e91e68379aa828f2a63239608fa4f397`. Both phases, order, thresholds and stop rules were frozen before the first new inference. Exclusive global phase registrations refuse another scientific run even with a new run ID. Freeze/registration/first-request wall times are in the compact result and local final audit.

Phase A reuses the original 48 immutable PNG captures: D200/D150/D100/D070, PRESENT then ABSENT; N,D,L,R within each case. Exactly 32 requests, one strict Edge/CLIP load. No image was duplicated to fabricate history. Both PNG and RGB identities were checked.

Phase B follows the technical validity gate, even though Phase A was negative. Order N,D,L,R, one fresh Phase-B service shared across four fresh independent Isaac processes. Each trial has a fresh world, six-frame bootstrap and zero initial controller state. Seed 0 is set once per phase load; no per-condition reseeding. First cold inference in each phase is retained. A valid collision does not stop the other frozen conditions. No scientific retry occurred.

The model-free infrastructure attempt used six dummy requests, zero model loads, zero model inferences and stationary motion. Its outcome passed before freeze. The pre-freeze suite passed all 135 tests.

## Frozen scene and evaluation

Start `(0,0,0)`, goal `(5,0,0)`, goal XY tolerance 0.50 m, timeout 40 simulation seconds after bootstrap. Red opaque fixed cuboid center `(2.5,0,0.5)`, size `(0.6,1.0,1.0)` m. Kinematic SE(2) camera, virtual circle radius 0.20 m, required clearance 0.10 m. This is swept 2D virtual-footprint collision, not contact dynamics of a physical robot body. Lighting, floor, camera, scheduler, rendering and collision stop semantics come from the original frozen static configuration. Physics is 60 Hz; observation target 3 Hz; one outstanding request, hold prior command while waiting, no catch-up queue. Camera 640×480, horizontal FOV 90°, height 0.50 m, pitch 0°.

Native float32 `(1,8,4)` is preserved without modification. All-eight-row XY is transformed at the **observation pose**. Evaluation includes the segment from observation center to row 0 and every following segment. No intrinsic timestamps are assigned to rows. Safe full passage requires minimum clearance >=0.10 m and a polyline spanning expanded X planes 1.9 to 3.1 m. A short positive-clearance prefix is horizon-insufficient. ABSENT predictions are evaluated counterfactually against the original box, not reported as real collisions in an absent scene.

Frozen direction consistency requires L endpoint Y>1e-6 m, R endpoint Y<-1e-6 m, L-N endpoint Y>1e-6 m and R-N endpoint Y<-1e-6 m together. The 0.20 m intended-sign lateral difference screen is descriptive, not a statistical test and not a substitute for full geometric safety.

## Phase A: all 32 measured predictions

The technical gate passed: 32 planned / 32 executed / 1 load, all actual ID-8 masks and embeddings valid, no numerical or controller error. Each cell below is endpoint `(forward, left)` in meters / minimum full-polyline clearance in meters. Classification is identical across N,D,L,R within each RGB case.

| RGB case | N | D | L | R | Classification |
| --- | --- | --- | --- | --- | --- |
| D200 PRESENT | (0.836596, -0.003442) / 0.963404 | (0.901445, 0.001095) / 0.898555 | (0.905822, -0.005704) / 0.894178 | (0.939317, -0.004140) / 0.860683 | insufficient |
| D200 ABSENT | (0.780814, -0.034769) / 1.019186 | (0.736909, -0.045842) / 1.063091 | (0.756384, -0.042205) / 1.043616 | (0.743555, -0.042949) / 1.056445 | insufficient |
| D150 PRESENT | (0.900845, 0.025220) / 0.399155 | (0.944153, 0.024817) / 0.355847 | (0.953011, 0.015935) / 0.346989 | (0.970020, 0.016926) / 0.329980 | insufficient |
| D150 ABSENT | (0.775867, -0.037165) / 0.524133 | (0.731621, -0.045822) / 0.568379 | (0.750399, -0.042976) / 0.549601 | (0.737999, -0.044033) / 0.562001 | insufficient |
| D100 PRESENT | (0.736487, -0.022562) / 0.063513 | (0.752199, -0.007800) / 0.047801 | (0.792456, -0.014366) / 0.007544 | (0.798353, -0.014457) / 0.001647 | low clearance |
| D100 ABSENT | (0.775274, -0.043249) / 0.024726 | (0.726097, -0.048294) / 0.073903 | (0.746921, -0.046617) / 0.053079 | (0.734554, -0.047633) / 0.065446 | low clearance |
| D070 PRESENT | (0.718339, -0.056426) / -0.200000 | (0.600896, -0.054903) / -0.100896 | (0.635323, -0.054534) / -0.135323 | (0.635607, -0.055109) / -0.135607 | collision |
| D070 ABSENT | (0.770006, -0.045445) / -0.200000 | (0.720576, -0.050486) / -0.200000 | (0.742639, -0.048524) / -0.200000 | (0.730352, -0.049428) / -0.200000 | collision |

Counts: **0 safe / 16 horizon-insufficient / 8 low-clearance / 8 collision**. Each instruction has 0/4/2/2 respectively. Minimum predicted clearance is -0.200000 m; maximum absolute lateral extent across all 32 native chunks is 0.103963 m. The chunks do not provide the roughly ±0.80 m centerline offset needed alongside the box at the declared margin. Full geometry, not this rough side-offset observation, determines classifications.

| Observation | Value |
| --- | --- |
| MODALITY8_INPUT_VALID | True |
| LANGUAGE_RESPONSE_OBSERVED | True |
| DIRECTION_CONSISTENCY_OBSERVED | False |
| SAFE_PASSAGE_PREDICTED | False |
| HORIZON_INSUFFICIENT | True |
| PREDICTED_UNSAFE | True |
| NO_MATERIAL_DIRECTIONAL_RESPONSE | True |

### Fixed RGB/pose prompt comparisons

| Comparison | Largest absolute per-row lateral delta (m) | RMS position difference range (m) | Largest absolute heading-angle delta (rad) |
| --- | --- | --- | --- |
| L-N | 0.016374 | 0.015136–0.050023 | 0.141173 |
| R-N | 0.017513 | 0.021896–0.061452 | 0.140641 |
| L-R | 0.001710 | 0.001123–0.013332 | 0.015640 |
| D-N | 0.022169 | 0.016307–0.049587 | 0.143133 |

| RGB case | L endpoint Y | R endpoint Y | L-N endpoint Y | R-N endpoint Y | Joint direction consistency |
| --- | --- | --- | --- | --- | --- |
| D200 PRESENT | -0.005704 | -0.004140 | -0.002262 | -0.000698 | False |
| D200 ABSENT | -0.042205 | -0.042949 | -0.007435 | -0.008180 | False |
| D150 PRESENT | 0.015935 | 0.016926 | -0.009286 | -0.008294 | False |
| D150 ABSENT | -0.042976 | -0.044033 | -0.005812 | -0.006868 | False |
| D100 PRESENT | -0.014366 | -0.014457 | 0.008197 | 0.008106 | False |
| D100 ABSENT | -0.046617 | -0.047633 | -0.003368 | -0.004384 | False |
| D070 PRESENT | -0.054534 | -0.055109 | 0.001892 | 0.001317 | False |
| D070 ABSENT | -0.048524 | -0.049428 | -0.003079 | -0.003983 | False |

Joint directional consistency: **0/8**. Intended-sign material response: L 0/8 and R 0/8. All 32 fixed-input prompt comparisons have nonzero numerical changes, but L and R are nearly coincident: largest lateral difference only 0.001710 m. D200/D100/D070 L and R endpoints are both negative; D150 PRESENT endpoints are both positive. Directional instruction following is not demonstrated by these changes. Heading vectors also differ, but the unchanged selected-row controller used XY, not those vectors.

### PRESENT versus ABSENT under the same instruction

| Pose | ID | PRESENT endpoint X | ABSENT endpoint X | P-A endpoint X | Max absolute lateral delta | RMS position delta |
| --- | --- | --- | --- | --- | --- | --- |
| D200 | N | 0.836596 | 0.780814 | 0.055782 | 0.069458 | 0.185656 |
| D200 | D | 0.901445 | 0.736909 | 0.164536 | 0.073304 | 0.247458 |
| D200 | L | 0.905822 | 0.756384 | 0.149438 | 0.061509 | 0.243623 |
| D200 | R | 0.939317 | 0.743555 | 0.195762 | 0.062453 | 0.258433 |
| D150 | N | 0.900845 | 0.775867 | 0.124978 | 0.111582 | 0.131407 |
| D150 | D | 0.944153 | 0.731621 | 0.212532 | 0.103194 | 0.159031 |
| D150 | L | 0.953011 | 0.750399 | 0.202612 | 0.093022 | 0.161463 |
| D150 | R | 0.970020 | 0.737999 | 0.232021 | 0.093716 | 0.172033 |
| D100 | N | 0.736487 | 0.775274 | -0.038787 | 0.050666 | 0.132027 |
| D100 | D | 0.752199 | 0.726097 | 0.026102 | 0.067357 | 0.120277 |
| D100 | L | 0.792456 | 0.746921 | 0.045536 | 0.057380 | 0.125296 |
| D100 | R | 0.798353 | 0.734554 | 0.063800 | 0.059036 | 0.123186 |
| D070 | N | 0.718339 | 0.770006 | -0.051667 | 0.010981 | 0.023755 |
| D070 | D | 0.600896 | 0.720576 | -0.119680 | 0.018345 | 0.044295 |
| D070 | L | 0.635323 | 0.742639 | -0.107316 | 0.025071 | 0.044295 |
| D070 | R | 0.635607 | 0.730352 | -0.094745 | 0.023470 | 0.039103 |

Visual-condition sensitivity is observed; maximum lateral difference is 0.111582 m, still below 0.20 m. The sign and size of longitudinal response depend on distance and instruction. PRESENT extends farther than ABSENT at D200 and D150. At D100 only N contracts (-0.038787 m); D/L/R extend by +0.026102/+0.045536/+0.063800 m. At D070 all contract, by 0.051667–0.119680 m. The old ID-4 D100 contraction of 0.824 m is **not reproduced as a within-ID-8 effect**. Comparing it to ID-8 N would change both modality masking and text context, so it is not a clean isolated language-effect comparison.

### Native prediction versus controller conversion

Every D200/D150 PRESENT prompt produces selected pre-clip v from 1.137772 to 1.632336 m/s and final v=0.3 m/s. The official clipping/limiting therefore collapses those forward command differences, although angular commands differ. Conversely, it would be incorrect to reuse the old claim that all new D100 inputs produce v=0.3:

| D100 ID | P/A selected pre-clip v (m/s) | P/A final v (m/s) | P/A final w (rad/s) |
| --- | --- | --- | --- |
| N | 0.292826 / 0.764901 | 0.134369 / 0.214949 | -0.300000 / -0.300000 |
| D | 0.249421 / 0.667499 | 0.245266 / 0.218469 | -0.300000 / -0.300000 |
| L | 0.273418 / 0.714283 | 0.218953 / 0.233555 | -0.300000 / -0.300000 |
| R | 0.271412 / 0.699908 | 0.227636 / 0.230147 | -0.300000 / -0.300000 |

D100 forward differences partly survive into final v, while all selected angular commands saturate at -0.3 rad/s. This is measured conversion, not evidence of intended braking: row timing is undefined and endpoint X is not the selected-row speed input. No controller alternative was tested and no controller rescue is inferred.

## Phase B: four primary closed-loop outcomes

All four conditions completed as scientifically valid **COLLISION**, with goal_reached=false, collision=true, no goal-entry time and no runtime/numerical/upstream-controller errors. Each episode terminated immediately on its first swept collision; no motion correction or retry followed. Signed Y is world left (+), right (-).

| ID | First margin violation (s) | First contact (s) | Recorded stop (s) | Min clearance (m) | Min / max Y (m) | Max abs Y (m) | Path length (m) | Final goal distance (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 7.303651 | 7.727881 | 7.733333 | -0.001251 | -0.292428 / 0.000000 | 0.292428 | 2.064274 | 3.012973 |
| D | 6.973040 | 7.416014 | 7.416667 | -0.000144 | -0.310566 / 0.000000 | 0.310566 | 2.077954 | 3.015889 |
| L | 7.024623 | 7.459656 | 7.466667 | -0.001586 | -0.325371 / 0.000000 | 0.325371 | 2.082179 | 3.016016 |
| R | 7.425517 | 7.873757 | 7.883333 | -0.002113 | -0.338621 / 0.000000 | 0.338621 | 2.092958 | 3.016951 |

| ID | First turn / time (s) | Predicted / applied | RGB captures | Native insufficient / low / collision / safe | Missed slots | Action updates per sim s |
| --- | --- | --- | --- | --- | --- | --- |
| N | RIGHT / 0.400000 | 23 / 22 | 29 | 11 / 1 / 11 / 0 | 1 | 2.844828 |
| D | RIGHT / 0.083333 | 23 / 22 | 28 | 11 / 1 / 11 / 0 | 0 | 2.966292 |
| L | RIGHT / 0.100000 | 23 / 23 | 28 | 11 / 1 / 11 / 0 | 0 | 3.080357 |
| R | RIGHT / 0.233333 | 24 / 24 | 29 | 11 / 1 / 12 / 0 | 0 | 3.044397 |

N and D each retained one final completed prediction that was never applied because the robot had already collided: all 93 service responses and native tensors are analyzed; 91 became controller updates. These are not dropped measurements. Raw event logs retain terminal in-flight status, responses retain inference-ready timestamps, and unused predictions have no application timestamp. Applied-only latency aggregates in original episode summaries must not be mislabeled as all-inference aggregates.

All four first turns were RIGHT. L did not execute the requested left detour. Later angular commands include brief positive turns, but all four executed Y maxima remain 0 at the start and their net paths remain on the right side. R's matching first-turn sign is insufficient evidence of language control because N, D and L do the same. More lateral travel than the historical pose-only run did not provide safe clearance or goal-reaching. Single asynchronous episodes cannot isolate a causal prompt effect on collision timing.

### Commands and saturation

| ID | Applied-update pre-clip v range | Applied-update final v range | Applied-update final w range | Initial clipping | Final ratio limit |
| --- | --- | --- | --- | --- | --- |
| N | [0.326027, 1.571555] | [0.18585, 0.3] | [-0.3, 0.114148] | 17/22 (77.3%) | 22/22 (100%) |
| D | [0.320485, 1.665323] | [0.185147, 0.3] | [-0.3, 0.048547] | 18/22 (81.8%) | 22/22 (100%) |
| L | [0.335107, 1.700243] | [0.181101, 0.3] | [-0.3, 0.100164] | 19/23 (82.6%) | 23/23 (100%) |
| R | [0.303209, 1.724011] | [0.164635, 0.3] | [-0.3, 0.3] | 19/24 (79.2%) | 24/24 (100%) |

Ranges above are applied updates; initial held bootstrap commands are zero. Every applied update underwent final ratio limiting; angular limiting sometimes reduces v below 0.3. The full command time series and selected pre-clip values are plotted in local `analysis-001/navigation-commands-clearance.png`. Heading fields were unused in all 93 selected-row conversions. Exact saved controller parity passed for all 125 predictions across both phases. Reintegrating all 1,830 physics steps reproduces poses to 1e-12 absolute tolerance; swept clearance and first-contact checks match the saved summaries.

### Latency and runtime

Values below are median / p95 / max in milliseconds. Model and forward timings include their relevant hook validation overhead; no artificial latency was introduced. Service pipeline includes preprocessing/controller/evidence work up to ready. Cold starts are retained. All four trials share the fresh Phase-B service; N therefore contains its cold start.

| ID | All-inference model | All-inference CLIP + forward | All-inference service pipeline | Observation to application marker (applied only) | Frame record to completed applied step |
| --- | --- | --- | --- | --- | --- |
| N | 12.515 / 18.379 / 140.324 | 20.134 / 31.600 / 237.216 | 71.460 / 103.924 / 288.011 | 97.701 / 154.233 / 290.997 | 105.618 / 163.330 / 297.215 |
| D | 15.843 / 17.314 / 18.006 | 25.644 / 28.810 / 28.911 | 78.084 / 107.397 / 149.766 | 84.900 / 135.960 / 161.146 | 92.038 / 143.893 / 169.535 |
| L | 15.376 / 17.890 / 18.016 | 23.675 / 29.213 / 29.552 | 76.315 / 86.132 / 86.591 | 94.561 / 143.367 / 145.227 | 101.786 / 150.940 / 152.927 |
| R | 11.541 / 17.693 / 18.925 | 17.086 / 28.694 / 29.256 | 69.291 / 83.419 / 153.530 | 86.836 / 145.275 / 194.751 | 97.181 / 154.297 / 202.705 |

| ID | Observation-to-application simulation seconds (median / p95 / max) | Simulation / wall ratio |
| --- | --- | --- |
| N | 0.150 / 0.266 / 0.400 | 1.407 |
| D | 0.217 / 0.265 / 0.300 | 1.541 |
| L | 0.167 / 0.233 / 0.250 | 1.517 |
| R | 0.125 / 0.233 / 0.233 | 1.368 |

The application marker is recorded before the applied physics step; the completed-step wall measurement includes that step. Observation wall time is the captured frame record, not a hardware exposure timestamp. Thus these are replayable software-pipeline timings, not sensor hardware latency claims.

Phase A cold model/CLIP-inclusive/pipeline: 117.414 / 199.718 / 244.821 ms. Its remaining 31 requests have median/p95/max model 10.011/10.659/11.214 ms, CLIP-inclusive 13.640/14.328/14.827 ms and pipeline 47.977/56.686/119.056 ms. Phase B first request: 140.324 / 237.216 / 288.011 ms. The one N missed slot is retained; others missed zero. Real wall-time scheduling contributes to trajectory differences.

Edge runtime: Python 3.10.21, Torch 2.7.1+cu128, CUDA runtime 12.8, NumPy 1.26.4. Isaac 6.0.1 rc.7 uses its isolated Python 3.12 / NumPy 2.3.1 environment. RTX 5060 Ti (16,311 MiB), driver 595.99.02. This is recorded compatibility evidence, not official dependency parity. No system Python, CUDA, driver or Isaac installation changed. Peak sampled device memory: Phase A 2,436 MiB, Phase B 3,931 MiB, dummy infrastructure 2,364 MiB. Each service strict-loaded once; missing/unexpected keys are empty, original Edge and CLIP precision float32. All scientific services and episode processes exited 0 with no residual processes.

Checkpoint SHA256: `f93b056f124380899fce3e087a5c2b41ce4d8c07e211daad52b6ebf02e80d04d` (released revision `b1361b7e24f101edea795a98b00a826b61a97394`). CLIP SHA256: `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.

## GUI verification and local evidence

The existing Isaac GUI was extended through optional `--language-protocol`; original navigation/offline modes and old launchers remain available. The new launcher uses this exact navigation run, never an older successful run. It displays the physical red obstacle, all four executed paths, start/goal, heading arrow, virtual radius, rounded clearance envelope, collision outcome, exact current instruction, active applied native chunk anchored at its recorded observation pose, selected waypoint 4, and original RGB. Purple prediction XY is elevated to Z=1.08 m solely for visibility, with a vertical observation-anchor marker; its XY is unchanged and row timestamps are not invented. Unapplied terminal predictions remain in saved metrics/plots.

The GUI was labeled `RECORDED REPLAY - NO LIVE INFERENCE`. No policy was loaded. It opened and advanced all four conditions to their recorded terminal times, rendering 3,532 frames. The viewer exited normally (code 0) after its application loop ended; this task did not create a stop-requested file. Exact text and changing active observation anchors are logged per condition. Saved viewport screenshot: `artifacts-local/pose-language-obstacle/gui-001/overview.png`; verification: `gui-verification.json`; readiness and sample logs are under `gui-001/`. The screenshot covers the viewport, not the control panel; control-panel instruction values are evidenced by ready/playback metadata. All 114 navigation PNG and decoded RGB hashes were independently checked. No viewer overlay entered a model observation.

```bash
bash ~/Workspace/OmniVLA/scripts/isaac_pose_language_gui.sh
```

Immutable raw evidence is local and ignored:

- `outputs/pose-language-obstacle/offline-001/`: 32 native tensors, requests, cases, response metadata, strict-load, technical gate and service lifecycle.
- `outputs/pose-language-obstacle/navigation-001/trials/{N,D,L,R}/POSE_LANGUAGE_*/raw/`: actual RGB, actions, events, history, requests, controller and robot states. Service responses include all 93 predictions.
- Each run's `derived/metrics.json` contains full saved-data measurements, every native geometry/heading and controller conversion. Raw files are mode 0444 and never overwritten.
- `artifacts-local/pose-language-obstacle/analysis-001/`: exact reproduction receipt, compact analysis export and four scientific plots (offline matrix, native/executed paths, commands/clearance, visual/text response).
- `new-evidence-hashes.json` inventories all 306 new immutable scientific evidence files; `final-audit.json` verifies their provenance, chronology and historical preservation. Plot/report/viewer products are separate from raw evidence.

## Reproduction and tests

Read-only reanalysis in the **Edge NumPy 1.26.4 runtime**, using a fresh derived directory:

```bash
cd ~/Workspace/OmniVLA
artifacts-local/envs/edge-blackwell/bin/python research/analyze_pose_language.py --output /tmp/pose-language-analysis-NEW
PYTHONDONTWRITEBYTECODE=1 artifacts-local/envs/edge-blackwell/bin/python -m unittest discover -s tests -v
git diff --check
```

Reanalysis exactly matched both stored metrics files and generated the four inspected plots; no model inference or simulation executes in this analysis script. Raw evidence/checkpoints/environments are required locally and are intentionally not in Git. The GUI uses saved scientific metrics and float64 geometry, so it does not recompute the float32 controller under Isaac's different NumPy version.

Historical execution commands, **already consumed** (do not delete registrations or rerun science):

```bash
artifacts-local/envs/edge-blackwell/bin/python research/register_pose_language_plan.py
artifacts-local/envs/edge-blackwell/bin/python research/run_pose_language.py --phase infrastructure --run-id infrastructure-001
artifacts-local/envs/edge-blackwell/bin/python research/freeze_pose_language.py
artifacts-local/envs/edge-blackwell/bin/python research/run_pose_language.py --phase offline --run-id offline-001
artifacts-local/envs/edge-blackwell/bin/python research/run_pose_language.py --phase navigation --run-id navigation-001
```

All 135 model-free tests passed before inference and in final validation, including all 126 historical tests and 9 new tests covering actual source masks, tokenizer parity, six-image official preprocessing/pose parity, adapter code identity, modality/metadata rejection and one-shot overwrite refusal. Final attempt 001 hit the execution sandbox's Unix-socket bind restriction in one existing IPC test (`PermissionError`); unchanged tests passed in the permitted host context as `final-tests-002.log`. This was a test-environment retry, not a scientific retry or code fix. One dummy infrastructure attempt and one GUI attempt succeeded; neither invokes a neural model.

## Assessment, limitations and final decision

**Positive evidence:** official ID-8 routing, both pose/language activation, distinct real CLIP embeddings, small fixed-input prompt effects, and substantial PRESENT/ABSENT output sensitivity were measured. The real unchanged controller and asynchronous Isaac pipeline completed four scientifically valid trials with preserved observations and native actions.

**Negative evidence:** consistent LEFT/RIGHT directional response 0/8, intended-direction material response 0/8 per side, safe full native passages 0/32 offline and 0/93 during navigation, safe goal-reaching 0/4. All four first turns were right and all collided. No validated safe FRESH future is supplied by this experiment for a reconciliation study.

**Insufficient evidence:** small numerical prompt effects are not successful language following. Visual-condition sensitivity does not identify internal obstacle semantics; object appearance, shadows and rendered input variation co-vary. Native safety does not follow from direction signs, and executed safety cannot be inferred from a short clear prefix. The matrix uses one scene, four literal instructions, one checkpoint, one stationary history per RGB condition and one asynchronous trial per instruction. No repeated-history variance estimate, statistical significance, general inability claim, controller-rescue claim or causal explanation of failures is justified. Cold/warm service differences and wall scheduling limit across-trial causal comparisons. Instructed L/R supply avoidance-side information externally and would be language-assisted navigation even if successful; this is not autonomous side selection. The underlying agent remains a kinematic virtual footprint rather than a physical robot.

**Decision:** preserve this negative qualification. Do not begin reconciliation, OLD/FRESH correspondence, SE(2) graph work, artificial latency, dynamic obstacles, prompt search, placement search, training or controller modification. No further experiment was launched.

**Exact next uncertainty:** With independently acquired but otherwise matched six-frame RGB histories at D100, is the near-identical LEFT/RIGHT native lateral response repeatable under the frozen Modality-8 interface? This uncertainty is stated only; testing it requires a separately authorized preregistered experiment.
