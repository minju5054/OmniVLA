# OmniVLA research workspace

- Official GitHub upstream source is the source of truth. Preserve upstream model,
  inference, and controller semantics; make research additions distinguishable.
- Prefer additive `research/`, `scripts/`, `tests/`, `docs/`, and compact `results/`.
- Record upstream/research Git SHA, checkpoint revision and hashes, runtime
  versions, and exact commands for every qualification run.
- Raw model output is immutable. Store raw and derived evidence separately;
  refuse to overwrite prior runs. `docs/WORK_LOG.md` is append-only.
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
