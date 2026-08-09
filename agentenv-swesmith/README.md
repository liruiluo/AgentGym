# AgentEnv SWE-smith

Native SWE-smith episodes for AgentMemoryGym reinforcement learning.

Each episode exposes one unchanged issue and one persistent repository through
Codex-style `shell_command` and `apply_patch` actions. The policy receives no
gold patch, hidden test list, verifier command, or dedicated memory API.

The runtime separately attests the Hugging Face dataset revision and the pinned
SWE-smith source revision. A formal launch also requires an OCI image manifest
that records both identities; a dataset snapshot SHA is never reused as a
source-code SHA.

The `/detail` endpoint is an audit-only route. It is disabled unless the server
has `SWESMITH_DETAIL_TOKEN` and the caller supplies the matching
`X-SWESMITH-Detail-Token` header, so hidden grader evidence never enters the
policy-facing HTTP surface.

Formal RL launches set `SWESMITH_AUDIT_ROOT` to a server-private directory. On
episode close, the server atomically persists the exact observations, raw policy
outputs, tool results, workspace diffs, terminal reward, and hidden F2P/P2P grade
before deleting the episode workspace. The sink path and its contents are never
returned by the policy-facing API.

## Interaction budget

AgentMemoryGym training uses at most 75 policy turns per SWE-smith episode.
Every sampled policy output consumes one turn, including `shell_command`,
`apply_patch`, final submission, parser errors, and policy-authored context
compaction. Successful submissions terminate early. This is a bounded training
contract, not the upstream benchmark default.

The pinned Mini-SWE-Agent SWE-bench configuration at commit
`a83fcae82d2a08f0ee0c688f9d137b3566c097f8` uses `step_limit: 250` in
`src/minisweagent/config/benchmarks/swebench.yaml`. Held-out native evaluation
uses that 250-turn reference limit so training-budget exhaustion is not reported
as a coding-capability failure. Runtime metadata reports both values.
