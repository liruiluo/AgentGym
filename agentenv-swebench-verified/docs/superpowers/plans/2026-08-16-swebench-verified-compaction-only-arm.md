# SWE-bench Verified Compaction-Only Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the frozen SWE-bench Verified wrapper from two arms to exactly `native`, `amg_compaction_only`, and `amg_memory`, with the two AMG arms sharing one compaction implementation while only `amg_memory` declares durable external memory.

**Architecture:** Keep the benchmark-native manager, slot/capability boundary, `/testbed` shell/apply-patch surface, TestSpec binding, runtime identity, and official patch exporter unchanged. Add the third protocol identity, replace the four client-side `amg_memory` compaction gates with one explicit two-arm capability predicate, and leave the existing memory prompt addendum gated exclusively on `amg_memory`. Prove the split at the client, manager, and exporter boundaries with triad tests, then validate the real pinned 500-row/TestSpec assets read-only.

**Tech Stack:** Python 3.10+, `unittest`, AgentGym `BaseEnvClient`, existing SWE-bench Verified HTTP adapter, Git CLI.

---

## Frozen ownership and scope

- Clean recorded base before any write: `6cecfb579e9c38672390602dcc50894e3b2107d0` (`feat: add SWE-bench Verified AMG adapter`).
- Owner: `amg-swebench-compaction-only-0816`; original owner confirmed inactive before work began.
- Allowed implementation paths: `agentenv-swebench-verified/**` and `agentenv/agentenv/envs/swebench_verified.py` only.
- Protected paths: runner/controller/registry/shared rollout, datasets, harness, images, OCI/rootfs, and all runtime/GPU launch surfaces.
- One final scoped follow-up commit is required; do not push or integrate another branch.

## File map

- Modify `agentenv-swebench-verified/agentenv_swebench_verified/protocol.py`: define the exact triad order and the compaction-only model label.
- Modify `agentenv/agentenv/envs/swebench_verified.py`: expose one explicit compaction-enabled arm predicate while retaining the memory addendum only for `amg_memory`.
- Modify `agentenv-swebench-verified/tests/test_client.py`: verify capability absence, trigger/summary/transition/accounting parity, and ordinary action parity across the triad.
- Modify `agentenv-swebench-verified/tests/test_environment.py`: verify metadata and benchmark-native manager dispatch parity across all three arms.
- Modify `agentenv-swebench-verified/tests/test_exporter.py`: verify byte-identical solution patches and reserved-memory exclusion for every arm.
- Modify `agentenv-swebench-verified/README.md`: document the exact three-arm contract and contrasts without describing a memory-only fourth arm.
- Create this plan only under the authorized adapter documentation subtree.

### Task 1: Add failing triad protocol and capability tests

**Files:**
- Modify: `agentenv-swebench-verified/tests/test_client.py`
- Modify: `agentenv-swebench-verified/tests/test_environment.py`
- Modify: `agentenv-swebench-verified/tests/test_exporter.py`

- [ ] **Step 1: Change endpoint fixtures to require the exact triad**

Use this exact arm/model identity in client metadata assertions:

```python
["native", "amg_compaction_only", "amg_memory"]
{
    "native": "qwen35-4b-native",
    "amg_compaction_only": "qwen35-4b-amg-compaction-only",
    "amg_memory": "qwen35-4b-amg-memory",
}
```

- [ ] **Step 2: Add a compaction-only absence-and-transition test**

Construct `amg_compaction_only`, assert its system prompt contains neither `.agent_memory` nor durable-memory guidance, bind its initial context, trigger the frozen pressure threshold, and assert:

```python
candidate == SBV_CONTEXT_COMPACTION_REQUEST
transition["operation"] == "replace_messages"
transition["messages"][-2:] == [
    {"role": "assistant", "content": summary},
    {"role": "user", "content": POLICY_CONTINUATION_MARKER},
]
backend.memory == {}
native_call_count_after == 0
policy_step_after == 1
context_epoch_after == 1
```

- [ ] **Step 3: Add exact compaction parity coverage**

Drive identical pressure and summary through `amg_compaction_only` and `amg_memory`. Assert they return the same candidate/request, take the same trigger decision, use the same replace operation and continuation suffix, and have identical native-call, policy-step, context-epoch, session-epoch, parser-status, reward, and done accounting. Normalize only the deliberately different memory prompt prefix and `wrapper_evidence.arm` identity.

- [ ] **Step 4: Expand ordinary manager/action/export coverage to all arms**

Run one ordinary shell action in each arm and compare the HTTP request shape, observation, action kind, actor credit, action progress, and native-call accounting. Export the same changed workspace through each arm, assert all `model_patch` bytes are equal, and assert `.agent_memory`, `.agent_logs`, `.agent_receipts`, and `.agent_telemetry` content is absent from every patch.

Also create and reset a manager-backed `amg_compaction_only` slot, use its ordinary shell surface to assert `.agent_memory` is absent at task start, and close it without introducing any dedicated memory lifecycle state.

- [ ] **Step 5: Run the focused tests and confirm the intended red state**

Run:

```bash
PYTHONPATH=agentenv-swebench-verified:agentenv-swesmith:agentenv-agentmemory \
python3 -m unittest -v \
  agentenv-swebench-verified/tests/test_client.py \
  agentenv-swebench-verified/tests/test_environment.py \
  agentenv-swebench-verified/tests/test_exporter.py
```

Expected: failures specifically because `amg_compaction_only` is not yet in `ARMS`/`MODEL_LABELS` and is rejected before compaction or export; no unrelated baseline failure.

### Task 2: Implement the minimal protocol and client split

**Files:**
- Modify: `agentenv-swebench-verified/agentenv_swebench_verified/protocol.py`
- Modify: `agentenv/agentenv/envs/swebench_verified.py`

- [ ] **Step 1: Add the exact frozen arm identities**

Implement:

```python
ARMS = ("native", "amg_compaction_only", "amg_memory")
MODEL_LABELS = {
    "native": "qwen35-4b-native",
    "amg_compaction_only": "qwen35-4b-amg-compaction-only",
    "amg_memory": "qwen35-4b-amg-memory",
}
```

- [ ] **Step 2: Centralize only the compaction capability gate**

Add a client-local immutable set and helper:

```python
SBV_COMPACTION_ARMS = frozenset({"amg_compaction_only", "amg_memory"})

def _compaction_enabled(self) -> bool:
    return self.arm in SBV_COMPACTION_ARMS
```

Use it in `policy_turn_candidate`, `prepare_policy_turn`, and `_complete_context_compaction`. Do not alter the compaction request, pressure arithmetic, replacement messages, task-neutral receipt builder, or action accounting.

- [ ] **Step 3: Preserve the complete memory capability delta**

Keep this gate unchanged in meaning:

```python
SBV_MEMORY_ADDENDUM if self.arm == "amg_memory" else ""
```

Do not add a memory root, mount, endpoint, environment variable, tool schema, parser/dispatch branch, receipt, evidence store, cleanup handle, or any arm-specific server behavior.

- [ ] **Step 4: Run focused tests and confirm green**

Repeat the Task 1 command. Expected: all client, environment, and exporter tests pass.

### Task 3: Document the frozen triad

**Files:**
- Modify: `agentenv-swebench-verified/README.md`

- [ ] **Step 1: Update launch and arm behavior text**

Document exactly three accepted arm values. State that `native` has neither compaction nor AMG memory, `amg_compaction_only` has the same compaction machinery as full AMG but no memory declaration/capability bundle, and `amg_memory` additionally exposes the existing clean per-task `.agent_memory` convention. Retain the statement that ordinary `/testbed` shell/apply-patch and export semantics are common.

- [ ] **Step 2: Update prediction wording and contrasts**

Replace two-arm/native-AMG wording with separate per-arm JSONLs and list the three reportable contrasts. Explicitly state that no memory-only fourth arm exists and no interaction should be inferred.

- [ ] **Step 3: Scan documentation for stale two-arm claims**

Run:

```bash
rg -n "both arms|two-arm|either.*native|native/AMG" agentenv-swebench-verified/README.md
```

Expected: no stale claim that contradicts the triad.

### Task 4: Verify pinned assets and all hardened boundaries

**Files:**
- Read only: `/Users/luolirui.1/Projects/amg-paired-eval-20260815/audits/.swebench-audit.9acrTe/pinned_verified_test.jsonl`
- Read only: `/Users/luolirui.1/Projects/amg-paired-eval-20260815/audits/.swebench-audit.9acrTe/SWE-bench`
- Test: `agentenv-swebench-verified/tests/test_*.py`

- [ ] **Step 1: Run every adapter test**

```bash
PYTHONPATH=agentenv-swebench-verified:agentenv-swesmith:agentenv-agentmemory \
python3 -m unittest discover -v \
  -s agentenv-swebench-verified/tests -p 'test_*.py'
```

Expected: all prior and triad tests pass, including security, slot capability, export, runtime binding, and observation limits.

- [ ] **Step 2: Verify the real 500-row/TestSpec binding read-only**

Run a temporary-script-free `python3 -c` check that reads the pinned JSONL, verifies 500 sorted unique rows and the canonical SHA-256, constructs `VerifiedDataset` with a manifest in a `TemporaryDirectory`, instantiates `OfficialTestSpecResolver` against the pinned clean v4.1.0 checkout, resolves every private row, and verifies all 500 instance IDs plus exact harness commit/tag. Expected summary: `rows=500 bindings=500`.

- [ ] **Step 3: Compile and lint changed Python**

```bash
python3 -m compileall -q \
  agentenv-swebench-verified/agentenv_swebench_verified \
  agentenv-swebench-verified/tests \
  agentenv/agentenv/envs/swebench_verified.py
python3 -m ruff check \
  agentenv-swebench-verified/agentenv_swebench_verified/protocol.py \
  agentenv-swebench-verified/tests/test_client.py \
  agentenv-swebench-verified/tests/test_environment.py \
  agentenv-swebench-verified/tests/test_exporter.py \
  agentenv/agentenv/envs/swebench_verified.py
```

Expected: both commands exit zero. If `ruff` is unavailable, use the repository's installed lint command or report the unavailable tool explicitly while retaining compile/test evidence.

- [ ] **Step 4: Prove scope and protected rollout invariants**

Record `git diff --name-only 6cecfb579e9c38672390602dcc50894e3b2107d0`, reject any path outside the authorized scope, inspect `git diff --check`, and compare the protected shared runner commit/worktree path read-only without modifying it. Confirm `vllm_rollout.py` has zero diff from its recorded runner base and contains no SWE/arm branch introduced by this work.

### Task 5: Independent review, fresh verification, and scoped commit

**Files:**
- Review: every path changed from `6cecfb579e9c38672390602dcc50894e3b2107d0`

- [ ] **Step 1: Dispatch an independent code reviewer**

Provide the approved contract, base SHA, exact diff, test evidence, and protected-boundary requirements. The reviewer must classify findings as Critical, Important, or Minor and explicitly state whether any Critical/Important finding remains.

- [ ] **Step 2: Resolve findings and re-review if necessary**

Fix every valid Critical/Important finding within scope, rerun affected tests, and request a whole-diff re-review. Do not claim closure with an unresolved Critical/Important finding.

- [ ] **Step 3: Freshly rerun all verification**

Repeat Task 4 after review changes. Inspect `git diff --check`, `git status --short`, and the exact staged diff.

- [ ] **Step 4: Commit only scoped files**

```bash
git add \
  agentenv-swebench-verified \
  agentenv/agentenv/envs/swebench_verified.py
git diff --cached --name-only
git commit -m "feat: add SWE-bench compaction-only arm"
```

Expected: the staged list contains only authorized files and the commit succeeds on the existing branch.

- [ ] **Step 5: Confirm final closure**

Run `git status --short --branch`, `git show --stat --oneline --decorate HEAD`, and `git rev-parse HEAD`. Expected: clean worktree, one scoped follow-up commit atop the recorded base, no push performed.
