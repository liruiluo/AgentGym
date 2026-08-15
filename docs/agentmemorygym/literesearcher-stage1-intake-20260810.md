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

The intake contains 64 source-certified train rows and eight disjoint held-out
rows. The first string-match pass was not accepted as a data-quality gate: all
72 rows were reviewed against a question-relevant source passage. An
independent review then removed every weak or rejected row from the stale
manifest (`13` rejected and `8` weak rows, `21` total) and replaced them with
`21` source-supported rows from the same frozen Stage-1 revision. The final
split uses an explicit 64/8 partition chosen from the reviewed set; the held-out
rows stay clear of the title/search-only-easy items.

The manifest is
`agentenv-agentmemory/fixtures/literesearcher_stage1_coverage.json`; its
SHA256 is checked when loaded. The machine-readable review is
`agentenv-agentmemory/fixtures/literesearcher_stage1_semantic_audit.json`.
The loader binds the two files and rejects missing, altered, or non-matching
review evidence. The committed v3 manifest was materialized at
`2026-08-10T10:05:00Z`; manifest SHA256 is
`95c5e336cdd63216d3b8d4fd58471be1ecda094365cc87fed401f26e6e88c58d`
and semantic-audit SHA256 is
`c81d3c36abadcb5ac2578e550b9f5b9c848dd762a6f6ccf7e6a0bfb22be8ea25`.
Each selected row stores the question, verifier-only gold aliases and source
URL, an opaque local URL, frozen source text, retrieval time, extraction
method, license note, evidence anchors, and a content SHA256.

The v1 gold-synthesized placeholder pages have been removed and the loader now
rejects their marker text, missing provenance, missing evidence anchors, and
content-hash mismatches. The resulting 2.7 MB corpus is still an intake-scale
source snapshot, not the released 32M-page search corpus or 937 GB browse
corpus. It is sufficient for the first RL plumbing/learnability gate; it does
not support a claim about full LiteResearcher benchmark performance.

The final corpus is `72/72` source-backed (`100%`). The removed stale-manifest
indices are `66, 353, 362, 411, 584, 875, 899, 902, 989, 1489, 1780, 1878,
2191, 2315, 2705, 2911, 3166, 3838, 3859, 3874, 4558`; their failure
categories and passages are retained in the semantic audit. Replacement
indices are `2666, 4489, 5034, 5046, 5058, 5256, 5327, 5330, 5397, 5400,
5464, 5500, 5717, 5761, 5806, 5815, 5857, 5905, 5909, 6754, 6918`. The audit
also labels `5815, 5905, 5909` as title/search-only-easy items, and the
explicit held-out partition keeps them in train instead of held-out.

The bounded-page audit keeps every reviewed evidence quote reachable. Public
goal BM25 ranks the evidence-bearing window top-1 for `55/72`, top-3 for
`65/72`, and top-5 for `72/72` rows. Pages use an 8,192-character window with
1,024-character overlap; page counts range from 1 to 18 and the largest
serialized visit response is 9,303 characters.

The source snapshot is reproducible from the exact 7,076,368-byte Stage-1
parquet (`SHA256=493f3d0cc87dc5f0f42340d3891d9df0f8b687d496c911847cd479250610371d`)
with:

```bash
uv run --with pyarrow python scripts/materialize_literesearcher_stage1.py \
  --parquet /path/to/stage1/train.parquet \
  --scan-rows 7000 \
  --workers 12 \
  --audit fixtures/literesearcher_stage1_semantic_audit.json \
  --output fixtures/literesearcher_stage1_coverage.json
```

Wikipedia text is frozen from the Jina reader plain-text surface and retains
its CC BY-SA/GFDL attribution note and resolved source URL server-side. Neither
the resolved URL nor the upstream `mask_url` enters policy observations.

## Policy-facing contract

The model receives only:

```text
<tool_call>{"name":"search","arguments":{"query":["..."]}}</tool_call>
<tool_call>{"name":"visit","arguments":{"url":"https://literesearcher.local/page/00000","goal":"...","page":1}}</tool_call>
<answer>...</answer>
```

Search returns deterministic snippets with opaque local URLs. Visit accepts
exactly one URL and returns one bounded page plus `page`, `page_count`, and
`next_page`; the policy follows `next_page` with the same URL and goal when it
needs more evidence. Bulk-URL visits, out-of-range pages, unknown URLs,
malformed requests, and backend failures fail closed. `mask_url`, target
aliases, and upstream source URLs never enter policy observations or service
metadata, and there is no live-web or judge fallback on backend failure.

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

The task-neutral controller computes tokenizer pressure from the active model
capacity and passes it to the LiteResearcher client wrapper. The wrapper alone
decides when compaction is needed; the server and shared rollout contain no
LiteResearcher timing branch. Capacity is derived from the active
`model_context_tokens`, response budget, observation budget, and serialization
envelope rather than a second hard-coded prompt limit.

Compaction is a normal trainable policy output and consumes one wrapper policy
turn. No special tag is required: the sampled model response is preserved
verbatim as the assistant continuity summary.

The client wrapper does not author a summary, call search/visit, or write the
workspace. Native research-call count stays unchanged while policy-step count
advances by one. It emits the task-neutral receipt:

```json
{
  "schema": "agentmemory_task_neutral_context_transition_v1",
  "operation": "replace_messages",
  "messages": [
    {"role": "system", "content": "...tool contract..."},
    {"role": "user", "content": "...question..."},
    {"role": "assistant", "content": "...policy summary..."},
    {"role": "user", "content": "Continue the same task in the unchanged workspace."}
  ]
}
```

`native_environment_call_count` remains unchanged for a compaction row. The
shared rollout only transports this opaque receipt; lifecycle timing and
parsing remain wrapper-owned. Server-private URLs never enter the policy
context, so compaction does not add a second summary parser or filtering path.

## 1-3 update gate prerequisites

Before a GPU gate can be launched, the following evidence is required:

1. Run the focused CPU tests below. They cover manifest split, search/private
   URL isolation, gold/wrong/tampered answers, backend fail-closed behavior,
   per-episode workspace isolation, and policy-authored compaction.
2. Load and hash-check all 64 train plus eight held-out source snapshots. The
   loader rejects gold-synthesized placeholders and absent provenance.
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
PYTHONPATH=. python3 -m unittest -q \
  tests/test_literesearcher_intake.py \
  tests/test_literesearcher_materializer.py \
  tests/test_domain_launch_v3.py \
  tests/test_domain_server_factory_v3.py \
  tests/test_service_identity.py
```

The focused CPU suite currently runs 89 tests with no external service, GPU, or
full-corpus dependency. It includes split-local search/visit, all-row top-1
self-search, gold/wrong/tampered rewards, placeholder-text rejection, content
tampering, and missing-source-provenance controls.
