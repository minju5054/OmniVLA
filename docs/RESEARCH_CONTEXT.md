# Research context

The long-term question is whether successive online navigation-VLA trajectory
chunks can be reconciled while preserving already executed OLD motion and the
new navigation intent of FRESH. Advisor feedback requires valid upstream
obstacle-avoidance evidence before reconciliation or corner cases.

Full OmniVLA is an upstream trajectory generator. It is not the contribution
or a reconciliation baseline. The current scope is repository setup and local
official-model/runtime preflight on RTX 5060 Ti. No Isaac work, navigation or
obstacle tests, chunk analysis, correspondence, Hermite interpolation, rigid
alignment, blending, SE(2) optimization, training, or model reduction belongs
in this task. Earlier LightNav and TIC-VLA workspaces remain untouched.

The research state is SE(2). Future reconciliation documentation must distinguish
source facts, measurements, interpretation, and proposals. A full BF16 failure
is a useful result and must not be hidden by precision or model changes.

Only after all pre-Isaac gates pass may the decision be
`READY_FOR_ISAAC_ADAPTER`; that decision itself authorizes no Isaac work here.

## Active candidate: OmniVLA-edge (2026-10-12)

The completed FULL preflight at `3fddf03e89e4c7145ceb241532bf907444b6a87d`
strictly loaded the full BF16 model but its first forward pass exhausted GPU
memory: `FULL_BF16_GPU_OOM` / `BLOCKED_BY_VRAM`. No FULL action was produced.
Its report, branch, environments, checkpoint and evidence remain preserved.

The active candidate is now OmniVLA-edge. The current task qualifies its
checkpoint, isolated runtime, official sample, native action semantics, pose
modality, controller, memory and computational cadence only. It neither begins
Isaac integration nor tests navigation, obstacles, or reconciliation. Edge is
an upstream generator, not the contribution or a reconciliation baseline.

## Active task: bounded obstacle-free Isaac qualification

The Edge preflight at `9ebb503a5452498dbf90c95a89ef4aa26a5bfec4` concluded
READY_FOR_ISAAC_ADAPTER. The user's next explicit request authorizes, only on
`research/omnivla-edge-isaac-no-obstacle`, simple kinematic SE(2) navigation
using actual six-frame RGB history, observation-time local pose goals, and the
unchanged pose-only Edge model and waypoint-4 controller. It permits simulator,
camera/history/coordinate, combined memory and three frozen no-obstacle runs.
It does not authorize obstacle avoidance or reconciliation. Both historical
preflight reports, branches, environments and evidence remain unchanged.
