# GAIA-Text Thin AMG Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leak-safe GAIA-Text HTTP environment and AgentGym client that run frozen native and per-task AMG-memory arms through one domain-action path and emit externally scoreable predictions.

**Architecture:** A strict external-data loader binds the audited 127-row manifest to runner-only questions and a separately hashed browse asset. A single episode manager handles search, visit, answer extraction, and submission for both arms; only the memory arm receives a lazily constructed isolated workspace, while its AgentGym client additionally owns task-neutral policy compaction. The server never loads gold answers or scorer code, and the prediction store atomically publishes the exact two-field JSONL only after all manifest IDs have terminal string-or-null answers.

**Tech Stack:** Python 3.10+, FastAPI/Pydantic, AgentGym `BaseEnvClient`, AgentMemoryGym `PersistentWorkspace`, pytest/unittest, JSONL/SHA-256.

---

## File map

- `agentenv-gaia-text/pyproject.toml`: standalone package metadata and launch entry point.
- `agentenv-gaia-text/README.md`: runtime inputs, security boundary, arm behavior, and verification commands.
- `agentenv-gaia-text/agentenv_gaia_text/contracts.py`: audited constants, arm enum, and injectable protocol contract used only by synthetic tests.
- `agentenv-gaia-text/agentenv_gaia_text/dataset.py`: exact-schema manifest/question loading, canonical count/level/order/hash validation, and safe task projection.
- `agentenv-gaia-text/agentenv_gaia_text/backend.py`: shared search/visit protocol and deterministic external fixture backend with asset digest validation.
- `agentenv-gaia-text/agentenv_gaia_text/submission.py`: one-answer-per-task store, null horizon handling, partial state, atomic final JSONL publication, and path-free receipts.
- `agentenv-gaia-text/agentenv_gaia_text/wrapper.py`: unbound-slot episode lifecycle and the common search/visit/answer/workspace action dispatcher.
- `agentenv-gaia-text/agentenv_gaia_text/server.py`: FastAPI factory with create/reset/step/horizon/close/metadata routes only.
- `agentenv-gaia-text/agentenv_gaia_text/launch.py`: fail-closed environment loader and Linux isolated workspace factory for the memory arm.
- `agentenv-gaia-text/agentenv_gaia_text/__init__.py`: public package exports and CLI launch hook.
- `agentenv/agentenv/envs/gaia_text.py`: arm-aware `BaseEnvClient`, shared prompt core, memory-only affordance, counters, and client-owned replace-messages compaction.
- `agentenv-gaia-text/tests/`: generated 127-row fixture helpers plus loader, privacy, paired-path, submission, client, and HTTP tests; no official rows.

### Task 1: Freeze public contracts and fail-closed private input loading

**Files:**
- Create: `agentenv-gaia-text/agentenv_gaia_text/contracts.py`
- Create: `agentenv-gaia-text/agentenv_gaia_text/dataset.py`
- Test: `agentenv-gaia-text/tests/test_dataset.py`
- Test support: `agentenv-gaia-text/tests/conftest.py`

- [x] **Step 1: Write generated-fixture tests** for exact schemas, sorted unique IDs, 127 count, 42/66/19 levels, manifest SHA, ID SHA, question join, production-contract rejection of synthetic hashes, and rejection of answer/scorer/attachment/annotator fields.
- [x] **Step 2: Run `python -m pytest agentenv-gaia-text/tests/test_dataset.py -q`** and confirm missing-module failures.
- [x] **Step 3: Implement immutable protocol constants, injectable fixture contract, dataclass task projection, byte-exact canonical manifest validation, and runner-question loading.**
- [x] **Step 4: Re-run the dataset tests** and confirm all pass.

### Task 2: Add the shared external browse backend and exact submission store

**Files:**
- Create: `agentenv-gaia-text/agentenv_gaia_text/backend.py`
- Create: `agentenv-gaia-text/agentenv_gaia_text/submission.py`
- Test: `agentenv-gaia-text/tests/test_backend_submission.py`

- [x] **Step 1: Write failing tests** for verified external backend assets, identical search/visit traces, invalid URL/query handling, exact prediction keys and manifest order, duplicate rejection, null horizon entries, partial-before-complete behavior, and path-free receipts.
- [x] **Step 2: Run the focused test module** and confirm failures.
- [x] **Step 3: Implement the deterministic JSON fixture backend and atomic submission writer.** Keep all data paths out of metadata and receipts.
- [x] **Step 4: Re-run the focused tests** and confirm all pass.

### Task 3: Implement the two-arm episode manager and HTTP service

**Files:**
- Create: `agentenv-gaia-text/agentenv_gaia_text/wrapper.py`
- Create: `agentenv-gaia-text/agentenv_gaia_text/server.py`
- Test: `agentenv-gaia-text/tests/test_wrapper.py`
- Test: `agentenv-gaia-text/tests/test_server.py`

- [x] **Step 1: Write failing lifecycle and privacy tests** covering unbound create, explicit reset, unfinished-reset refusal, one action per step, shared search/visit/answer parser, zero reward/no correctness signal, native workspace rejection/no workspace creation, clean per-task memory namespaces, horizon null submission, and absence of detail/gold/scorer routes.
- [x] **Step 2: Run both focused modules** and confirm failures.
- [x] **Step 3: Implement one manager path** with a workspace factory used only for `amg_memory`, public payloads limited to question/task ID/level plus operational receipts, and server error mapping.
- [x] **Step 4: Re-run both modules** and confirm all pass.

### Task 4: Implement the AgentGym client and compaction contract

**Files:**
- Create: `agentenv/agentenv/envs/gaia_text.py`
- Test: `agentenv-gaia-text/tests/test_client.py`

- [x] **Step 1: Write failing client tests** proving prompts share the same base domain instructions, native has no workspace/compaction candidate, memory exposes only its unavoidable affordance, ordinary actions increment native and policy counters, memory compaction increments only policy/context counters and makes no HTTP call, the receipt is task-neutral `replace_messages`, and a note survives compaction then read.
- [x] **Step 2: Run the focused module** and confirm failures.
- [x] **Step 3: Implement `GaiaTextEnvClient`** using the existing `BaseEnvClient` and task-neutral receipt helpers, with arm validation from server metadata and external `/horizon` finalization.
- [x] **Step 4: Re-run the focused module** and confirm all pass.

### Task 5: Add fail-closed runtime assembly and package documentation

**Files:**
- Create: `agentenv-gaia-text/agentenv_gaia_text/launch.py`
- Create: `agentenv-gaia-text/agentenv_gaia_text/__init__.py`
- Create: `agentenv-gaia-text/pyproject.toml`
- Create: `agentenv-gaia-text/README.md`
- Test: `agentenv-gaia-text/tests/test_launch.py`

- [x] **Step 1: Write failing launch tests** for required external paths/hashes, directory separation, forbidden gold/scorer environment variables, native avoiding AgentMemory imports/workspace creation, and memory requiring the formal Linux namespace sandbox inputs.
- [x] **Step 2: Run the launch tests** and confirm failures.
- [x] **Step 3: Implement lazy runtime assembly and documentation.** Never import the memory runtime on the native arm.
- [x] **Step 4: Re-run the launch tests** and confirm all pass.

### Task 6: Verify, review, and commit

**Files:**
- Modify only files listed in the contract write scope.

- [x] **Step 1: Run all package tests** with explicit package source paths.
- [x] **Step 2: Run `python -m compileall agentenv-gaia-text/agentenv_gaia_text agentenv/agentenv/envs/gaia_text.py agentenv-gaia-text/tests`.**
- [x] **Step 3: Run `git diff --check`, source scans for forbidden gold/scorer handling and embedded official rows, and `git status --short`.**
- [x] **Step 4: Dispatch a read-only code reviewer** against base `40f2acc` and the task contract, fix all critical/important findings, and repeat verification.
- [x] **Step 5: Commit the scoped implementation once** on `feat/amg-gaia-text-adapter-20260815` and record the commit SHA.
