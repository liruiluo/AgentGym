# LiteResearcher Stage-1 RL Intake

Status: CPU intake scaffold only. No GPU gate or formal run is authorized by
this document.

## Frozen source

- Upstream LiteResearcher commit: `779e7d5f6a043d4100149ba0992a39507f69a974`
- Dataset: `simplex-ai-inc/LiteResearcher-Data`
- Dataset revision: `fff6b0cfef718859543a16f542ea248d30d1ac34`
- Configuration: `stage1`, `train`
- Upstream Stage-1 row count: `10,398`
- Native upstream contract: one natural episode of `search -> visit -> answer`;
  no hand-authored sessions
- Upstream Stage-1 policy-turn limit: `40`

The initial intake selects train indices `0..63` and a disjoint held-out panel
`64..71`. The manifest is
`agentenv-agentmemory/fixtures/literesearcher_stage1_coverage.json`; its
SHA256 is checked when loaded. Each selected row stores the question, gold
aliases, private source URL, an opaque local URL, and a deterministic page
excerpt.

The committed page excerpts are a plumbing fixture, not the released 32M-page
corpus. They are sufficient to test URL routing, answer scoring, and receipt
privacy. A formal result must not be reported until every selected row is
covered by an independently frozen source excerpt or a task-covering local
corpus with a provenance audit. The full 32M search corpus and 937GB browse
corpus are explicitly out of scope for this intake.

## Policy-facing contract

The model receives only:

```text
<tool_call>{"name":"search","arguments":{"query":["..."]}}</tool_call>
<tool_call>{"name":"visit","arguments":{"url":["https://literesearcher.local/page/00000"],"goal":"..."}}</tool_call>
<answer>...</answer>
```

Search returns deterministic snippets with opaque local URLs. Visit returns
the frozen page excerpt. `mask_url`, target aliases, and any source URL from
the upstream row never appear in search results or service metadata. Unknown
URLs, malformed requests, and backend failures fail closed; there is no live
web fallback and no exact-match judge fallback on backend failure.

The initial reward contract is terminal-answer-only binary reward:

- search: `0`
- visit: `0`
- `shell_command` / `apply_patch`: `0`
- correct terminal answer: `1`
- wrong terminal answer: `0`
- backend error: `0`, `sample_excluded=true`

There is no memory-specific reward shaping, no SFT data, and no demonstration
injection. Workspace tools are the existing Codex-style
`shell_command`/`apply_patch` contract. A workspace is created per episode and
never shared across environment IDs.

## Context compaction

The wrapper receives a tokenizer-derived `context_token_count` from the rollout
adapter. With the Stage-1 contract (`model_context_tokens=32768`,
`max_response_tokens=2048`, `compaction_margin_tokens=256`), the wrapper marks
the next row as requiring compaction when the response budget plus margin no
longer fits.

Compaction is a normal policy action and consumes one environment step. The
policy writes the summary:

```text
<context_compaction>model-authored continuity summary</context_compaction>
```

The wrapper does not author a summary, call search/visit, or write the
workspace. It emits the task-neutral receipt:

```json
{
  "schema": "agentmemory_task_neutral_context_transition_v1",
  "operation": "replace_messages",
  "continuity_id": "stage1:00000",
  "workspace_path": ".agent_memory",
  "messages": [
    {"role": "system", "content": "...tool contract..."},
    {"role": "user", "content": "...question..."},
    {"role": "assistant", "content": "...policy summary..."}
  ],
  "policy_authored": true
}
```

`native_environment_call_count` remains unchanged for a compaction row. A
summary containing a server-private URL is rejected without a backend call.
The shared rollout only transports this opaque receipt; lifecycle timing and
parsing remain wrapper-owned.

## 1-3 update gate prerequisites

Before a GPU gate can be launched, the following evidence is required:

1. Run the focused CPU tests below. They cover manifest split, search/private
   URL isolation, gold/wrong/tampered answers, backend fail-closed behavior,
   per-episode workspace isolation, and policy-authored compaction.
2. Replace or independently certify the deterministic page fixture for all 64
   selected rows. The page body must be source-backed and must not be generated
   from the gold field for a formal result.
3. Add the wrapper to the existing task-neutral server/factory in a separate
   integration change, with no `vllm_rollout.py` diff. The first gate should
   use batch `64`, run only `1-3` optimizer updates, and verify nonzero terminal
   reward, actor/critic parameter changes, exact sampled-token/logprob
   retention, and eight-rank checkpoint/cleanup evidence.

No 1-3 update gate has been started from this intake branch. This separation is
intentional: the present fixture proves plumbing and privacy, not research
performance or source-answer quality.

## Verification

From `AgentGym/agentenv-agentmemory`:

```bash
PYTHONPATH=. python3 -m unittest -v tests/test_literesearcher_intake.py
```

The test suite currently runs eight tests with no external service, GPU, or
full-corpus dependency.
