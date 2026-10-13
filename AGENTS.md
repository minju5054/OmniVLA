# OmniVLA research workspace

- Official GitHub upstream source is the source of truth. Preserve upstream model,
  inference, and controller semantics; make research additions distinguishable.
- Prefer additive `research/`, `scripts/`, `tests/`, `docs/`, and compact `results/`.
- Record upstream/research Git SHA, checkpoint revision and hashes, runtime
  versions, and exact commands for every qualification run.
- Raw model output is immutable. Store raw and derived evidence separately;
  refuse to overwrite prior runs. `docs/WORK_LOG.md` is append-only.
- User preference: after each experiment, show its results in the Isaac Sim GUI
  and include a one-line GUI launch command in the final response. For navigation,
  replay that experiment's saved poses and original camera images with visible
  goals, paths and outcome; verify the GUI opens and playback advances. Clearly
  label recorded replay versus live inference. Use the actual run/config, never
  silently substitute a previous successful run. If an experiment produced no
  replayable navigation evidence, explain that limitation. Keep visualization
  additive and do not rerun or alter a scientific experiment just to show it.
- Current recorded-navigation launcher: `bash scripts/isaac_gui_replay.sh`.
  Its defaults show edge-final/scientific-001; explicitly pass `--run` and
  `--config` for another compatible run. Each launch uses a fresh ignored output
  directory. Update the viewer as needed for future experiment scene/schema changes.
- Never commit checkpoints, Hugging Face caches, datasets, generated runs,
  large images, virtual environments, Isaac caches, or secrets. Use ignored
  `artifacts-local/`, `models-local/`, or `outputs/` for local evidence.
- Never force push, reset user history, or rebase published work without an
  explicit user request. Do not make research commits on main.
- Do not modify LightNav or TIC-VLA repositories or their environments.
- Distinguish repository-confirmed facts, measured runtime facts, research
  interpretations, and not-yet-implemented proposals.
- Current research state is SE(2), unless explicitly changed later. Active
  candidate is OmniVLA-edge, an upstream generator, not a reconciliation baseline.
- FULL preflight is historical `FULL_BF16_GPU_OOM` / `BLOCKED_BY_VRAM` evidence.
  Preserve its branch, report, checkpoints, environments, and recorded results.
- On `research/omnivla-edge-isaac-no-obstacle`, the authorized scope is bounded
  obstacle-free Isaac integration: simple kinematic SE(2), actual six-frame RGB
  history, observation-time pose-only goals, and unchanged official controller.
  No obstacle experiments, OLD/FRESH analysis, training, fine-tuning, quantization,
  CPU offload, remote inference, model substitution, or reconciliation.
- Preserve each official model's precision and semantics (Edge uses float32).
  An isolated official-Torch CPU reference is allowed for Edge; no CPU offload.
  On OOM stop model execution and finish non-runtime work. Preserve both preflight
  reports unchanged. Do not tune or retry navigation after seeing its outcome;
  enforce the frozen scene/camera/goals and strict stage stop rules.
- Checkpoint load is not sample inference; sample inference is not navigation;
  navigation is not obstacle avoidance. Controller cadence is not intrinsic
  waypoint timing; newer Torch compatibility is not official dependency parity.
- On `research/omnivla-edge-no-obstacle-final`, distinguish infrastructure
  debugging from scientific experiments. Infrastructure retries are allowed to
  establish technical correctness; record every command, run ID, change, result
  and failure reason on this branch. Do not create a branch/report per retry.
- Scientific navigation starts only after all infrastructure gates pass and the
  protocol is hashed. Use one primary run per frozen goal, no post-outcome tuning
  or silent reruns. Classify escaped implementation failures separately from
  model navigation failures. No obstacle or artificial-latency experiments here.
- The Isaac CUDA library-path collision is confirmed and resolved by the local
  `scripts/isaac6_python.sh` isolation. Reuse validated infrastructure knowledge
  across repositories when source/runtime equivalence is demonstrated; do not
  repeat solved environment investigations unless relevant hashes changed.
  Preserve historical reports and original experiment evidence unchanged.
- On `research/omnivla-edge-static-obstacle`, the user authorizes exactly one
  primary static-obstacle trial after model-free validation and a new protocol
  freeze. Preserve the obstacle-free frozen files, evidence and local GUI work.
  Use one fixed opaque cuboid, virtual circle radius 0.20 m and safety margin
  0.10 m. Evaluate swept center motion between consecutive poses. Collision
  checks may terminate the episode but must never repair commands or poses.
  No scientific retry, post-outcome tuning, dynamic obstacles, reconciliation
  or graph optimization. Reuse the existing GUI only for recorded result replay
  after the headless trial; keep replay markers out of model observations.
  Static result GUI: `bash scripts/isaac_static_obstacle_gui.sh`.
- On `research/omnivla-edge-obstacle-offline-diagnostic`, the completed scope is
  immutable analysis of the 21 static predictions, followed by exactly eight
  frozen stationary present/absent requests using one Edge/CLIP load. No robot
  execution, scientific retry, controller repair, or further experiment is
  authorized by that task. Preserve the frozen source/input protocol and raw data.
  Offline result GUI: `bash scripts/isaac_obstacle_offline_gui.sh`; label fixed
  predictions `OFFLINE PREDICTION — NO ROBOT EXECUTION`. Preserve the collision
  launcher. Use the Edge Python/NumPy runtime for exact scientific re-analysis;
  Isaac's NumPy 2 viewer reads saved metrics and permits only documented display
  comparison rounding. A short positive-clearance horizon is not safe passage.
- On `research/omnivla-edge-pose-language-obstacle`, the completed scope is
  exactly 32 frozen ID-8 offline requests and four independent primary N/D/L/R
  navigation trials, with one strict Edge/CLIP service load per phase. Both
  phases were frozen before inference; all four valid collision outcomes are
  retained. No prompt tuning, scientific retry, controller repair, or further
  inference is authorized by this task. Preserve the new protocol, prompt plan,
  all frozen source hashes, historical reports, and all raw observations/actions.
  Saved-data reanalysis uses the Edge environment; recorded GUI replay uses
  `bash scripts/isaac_pose_language_gui.sh`. The optional language GUI mode
  displays exact instructions and native XY at observation poses; overlays are
  display-only and must never enter model inputs. Distinguish small numerical
  text effects, directional instruction following, visual sensitivity, safe
  native geometry, and actual goal-reaching. This matrix established no safe
  passage and no safe goal entry; do not begin reconciliation on this evidence.
