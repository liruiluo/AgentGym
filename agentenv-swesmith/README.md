# AgentEnv SWE-smith

Native SWE-smith episodes for AgentMemoryGym reinforcement learning.

Each episode exposes one unchanged issue and one persistent repository through
Codex-style `shell_command` and `apply_patch` actions. The policy receives no
gold patch, hidden test list, verifier command, or dedicated memory API.

The policy has two complementary memory mechanisms. Near the context limit it
writes a short continuation state that replaces the earlier conversation. For
details that should survive repeated lossy compaction, it may maintain ordinary
workspace files containing debugging hypotheses, attempted commands and tests,
exact evidence, failed approaches, partial results, and next checks. It chooses
the paths, structure, write cadence, and read cadence. On a long debugging path,
the policy updates this ledger at meaningful evidence changes rather than waiting
until the compaction request, which cannot execute a file action. After compaction it must
rediscover and read any needed files with normal shell commands; the harness
does not author, enumerate, summarize, or restore their contents. Neither a
compaction nor a file action receives a separate task reward.

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

## Upstream interaction parity

The pinned interaction reference is Mini-SWE-Agent
`SWE-agent/mini-swe-agent@a83fcae82d2a08f0ee0c688f9d137b3566c097f8`.
SWE-smith `9b74ac08118a85c39c356802f7961893af73e07f` supplies task/image
assets, while Mini-SWE-Agent supplies submission and horizon semantics.

- A submission is recognized only when a successful shell command has
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` as its first stdout line. Plain text,
  including `final`, is a parser error.
- The pinned upstream command is `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  && cat patch.txt`. AMG intentionally emits only the sentinel because its
  persistent no-`.git` workspace is graded directly; this is a workspace-grade
  adapter, not byte-equivalent patch transport.
- The policy prompt preserves the upstream repair order: inspect, reproduce,
  make a localized non-test source edit, rerun the reproduction, run relevant
  existing tests and edge checks, then submit immediately. The action syntax is
  adapted to one Codex-style action per policy turn, and the no-`.git` workspace
  removes the upstream `patch.txt` transport step.
- Upstream `LimitsExceeded` returns an empty submission and does not grade the
  current workspace. The bounded-training contract maps this outcome once to
  reward `-0.01`, keeps `grade=None`, and makes no hidden-grader call.
- Parser or executor rejection also terminates once with reward `-0.01`. A
  recognized submission with no non-generated source change, or a valid but
  unresolved official grade, terminates with reward `0`; a resolved official
  submission receives reward `1`. A valid shell command that exits nonzero
  (including a failing test) remains ordinary observable feedback and does not
  terminate. Grader/backend failures are reward `0` and are separately marked
  `sample_excluded` so infrastructure failures are resampled before PPO instead
  of being attributed to the policy.

## Interaction budget

AgentMemoryGym's endpoint default is 75 policy turns per SWE-smith episode;
the r4 formal lineage used a 30-turn curriculum override.
Every sampled policy output consumes one turn, including `shell_command`,
`apply_patch`, sentinel submission, parser errors, and policy-authored context
compaction. Successful submissions terminate early. This is a bounded training
contract, not the upstream benchmark default. The initial observation states
the exact configured budget and warns that compactions consume the same turns,
so a policy can submit before an otherwise ungraded horizon.

The pinned Mini-SWE-Agent SWE-bench configuration at commit
`a83fcae82d2a08f0ee0c688f9d137b3566c097f8` uses `step_limit: 250` in
`src/minisweagent/config/benchmarks/swebench.yaml`. Held-out native evaluation
uses that 250-turn reference limit so training-budget exhaustion is not reported
as a coding-capability failure. Runtime metadata reports both values.
