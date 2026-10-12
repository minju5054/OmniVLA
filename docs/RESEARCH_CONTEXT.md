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
