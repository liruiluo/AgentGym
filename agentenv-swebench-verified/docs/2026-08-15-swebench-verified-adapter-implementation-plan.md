# SWE-bench Verified Thin Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin, full-500 SWE-bench Verified HTTP environment and AgentGym client that generate exact base-commit patches for external official grading while exposing no grader-only data.

**Architecture:** A frozen external-data loader privately retains complete Verified rows but returns an immutable four-field policy projection. A per-episode materializer archives the exact local mirror commit into an isolated persistent workspace and keeps Git export state outside `/testbed`; the existing SWE-smith action parser, patch transaction, bounded observation helpers, and Linux namespace OCI-rootfs sandbox are reused by composition. One manager and HTTP action path serves both arms; only the AgentGym client’s `amg_memory` mode adds `.agent_memory` guidance and task-neutral policy-authored compaction.

**Tech Stack:** Python 3.10+, standard-library threaded HTTP server, AgentGym `BaseEnvClient`, Git plumbing, existing `agentenv_swesmith` and `agentenv_agentmemory` primitives, and `unittest`.

---

## File map

- `agentenv-swebench-verified/agentenv_swebench_verified/protocol.py`: immutable audit pins, arm names, allowed/forbidden fields, and safe policy projection.
- `agentenv-swebench-verified/agentenv_swebench_verified/dataset.py`: external manifest/JSONL validation, exact 500/order/hash proof, and offset-indexed private row loading.
- `agentenv-swebench-verified/agentenv_swebench_verified/testspec.py`: fail-closed import of exact v4.1.0 `make_test_spec` and private image/TestSpec binding.
- `agentenv-swebench-verified/agentenv_swebench_verified/images.py`: external 500-row tag/digest manifest validation and per-TestSpec digest lookup.
- `agentenv-swebench-verified/agentenv_swebench_verified/workspace.py`: exact-base materialization with private Git index state and safe lifecycle.
- `agentenv-swebench-verified/agentenv_swebench_verified/exporter.py`: solution-only staged diff, arm/run-scoped exact-one-row storage, and canonical-order 500-row JSONL assembly.
- `agentenv-swebench-verified/agentenv_swebench_verified/sandbox.py`: Verified-specific wrapper over the shared Linux namespace OCI-rootfs executor.
- `agentenv-swebench-verified/agentenv_swebench_verified/environment.py`: persistent episode state, shared action dispatch, bounded observations, and export-only terminal behavior.
- `agentenv-swebench-verified/agentenv_swebench_verified/server.py`: policy-safe HTTP surface.
- `agentenv-swebench-verified/agentenv_swebench_verified/launch.py`: external runtime configuration and dependency binding.
- `agentenv/agentenv/envs/swebench_verified.py`: two-arm `BaseEnvClient` and task-neutral compaction receipt.
- `agentenv-swebench-verified/tests/`: fixture tests for every contract invariant.

### Task 1: Frozen dataset and safe policy boundary

**Files:**
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/protocol.py`
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/dataset.py`
- Test: `agentenv-swebench-verified/tests/test_dataset.py`

- [ ] Write fixture tests that reject wrong manifest schema/revision/hash/count, unsorted or duplicate IDs, non-canonical JSONL, and rows missing required private TestSpec fields.
- [ ] Prove the loader’s production pins are revision `c104f840...`, 500 rows, JSONL SHA-256 `392529...`, and ID-ledger SHA-256 `a6b0fd...`.
- [ ] Assert the policy projection has exactly `instance_id`, `repo`, `base_commit`, and `problem_statement`, even when the private row includes gold/test patches, F2P/P2P, hints, eval scripts, parser state, and logs.
- [ ] Run `PYTHONPATH=agentenv-swebench-verified pytest -q agentenv-swebench-verified/tests/test_dataset.py`; expect all tests to pass.

### Task 2: Exact TestSpec/image binding and base workspace

**Files:**
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/testspec.py`
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/images.py`
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/workspace.py`
- Test: `agentenv-swebench-verified/tests/test_runtime_binding.py`
- Test: `agentenv-swebench-verified/tests/test_workspace.py`

- [ ] Write tests that require the exact harness Git commit/tag, import `make_test_spec` from only that checkout, invoke it with `namespace="swebench"`, keep TestSpec details private, and require the derived image tag to exist in a complete 500-row digest manifest whose sorted tag-ledger hash is `b69e618...`.
- [ ] Prove the image manifest requires 500 unique expected tags and one valid Linux/amd64 digest per tag while explicitly allowing different tags to share a digest.
- [ ] Write a local Git fixture proving materialization uses the requested 40-character `base_commit`, not mirror HEAD, and puts no Git metadata in the policy root.
- [ ] Implement fail-closed source/path/ref validation, exact `git archive` export, private episode state, ownership handoff, and safe close.
- [ ] Run the two targeted test modules; expect all tests to pass.

### Task 3: Solution-only patch export

**Files:**
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/exporter.py`
- Test: `agentenv-swebench-verified/tests/test_exporter.py`

- [ ] Write tests for modified, deleted, and new solution files; an unchanged empty patch; exclusion of `.agent_memory`, `.agent_logs`, `.agent_receipts`, and `.agent_telemetry`; exact row keys; and exactly-once emission per `(arm, run_id, instance_id)`.
- [ ] Build a fresh private index from the base tree, stage the workspace with reserved artifact pathspec exclusions, and emit `git diff --cached --binary --full-index` against that exact base.
- [ ] Implement atomic prediction storage keyed by `(arm, run_id, data_idx)` with arm-pinned model labels, duplicate rejection inside one arm/run, and deterministic separate native/AMG JSONL assembly in the dataset’s exact 500-ID order.
- [ ] Run the exporter tests; expect all tests to pass.

### Task 4: Shared action environment and HTTP surface

**Files:**
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/sandbox.py`
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/environment.py`
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/server.py`
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/launch.py`
- Create: `agentenv-swebench-verified/agentenv_swebench_verified/__init__.py`
- Create: `agentenv-swebench-verified/pyproject.toml`
- Test: `agentenv-swebench-verified/tests/test_environment.py`
- Test: `agentenv-swebench-verified/tests/test_server.py`

- [ ] Write tests proving both arms enter the same manager dispatch, shell and apply-patch persist, output is bounded, terminal/horizon returns reward zero and one arm/run-scoped prediction, reset starts clean, and policy responses contain no private dataset/TestSpec fields.
- [ ] Reuse the shared parser/patch/sandbox primitives; replace grading with the exporter and keep audit/log/receipt/telemetry roots outside the policy workspace.
- [ ] Expose only health, metadata, create/reset/step/observation/horizon/prediction/prediction-assembly/close endpoints with fail-closed error handling; assembly must reject incomplete or duplicate/foreign ledgers.
- [ ] Bind launch exclusively to external manifest, mirrors, source checkout, OCI cache, image digest ledger, output root, and pinned ripgrep paths.
- [ ] Run environment/server tests; expect all tests to pass.

### Task 5: Two-arm AgentGym client

**Files:**
- Create: `agentenv/agentenv/envs/swebench_verified.py`
- Test: `agentenv-swebench-verified/tests/test_client.py`

- [ ] Write tests that `native` has no memory wording, candidate, or context replacement; `amg_memory` has a clean per-task `.agent_memory` contract and task-neutral compaction; memory written before compaction remains readable afterward; and the same non-memory action produces byte-identical HTTP dispatch in both arms.
- [ ] Prove the external-evaluation budget is 250 unified policy turns in both arms: every native action and policy-authored compaction consumes one policy turn, compaction consumes no HTTP/native call, and horizon export is a wrapper lifecycle call rather than an extra sampled turn.
- [ ] Implement one `BaseEnvClient` with an explicit arm enum, identical reset/step/horizon/export behavior, and the canonical controller context-transition builder.
- [ ] Validate server metadata pins, task count/order contract, the 250-turn external-evaluation budget, exporter contract, arm-pinned model label, and supported arms before creating a slot.
- [ ] Run client tests; expect all tests to pass.

### Task 6: Documentation, verification, review, and commit

**Files:**
- Create: `agentenv-swebench-verified/README.md`
- Modify: this plan to mark completed steps if useful.

- [ ] Document external-only dataset/harness/image inputs, paired-arm invariants, launch variables, prediction schema, and runtime blockers without claiming official grading readiness.
- [ ] Run all new tests with package, AgentGym, SWE-smith, and AgentMemory paths on `PYTHONPATH`.
- [ ] Run `python -m compileall` over both authorized Python roots and `git diff --check`.
- [ ] Audit `git status --short` so every write is within the two authorized paths.
- [ ] Prove `git diff 40f2acc -- vllm_rollout.py '**/vllm_rollout.py'` and shared rollout/controller diff are empty.
- [ ] Dispatch an independent code reviewer against base `40f2acc`, fix every Critical/Important finding, and rerun the full verification suite.
- [ ] Commit only the authorized files on `feat/amg-swebench-verified-adapter-20260815` and record the resulting SHA.
