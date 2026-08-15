# GAIA-Text thin AMG adapter

This package serves the frozen 127-task attachment-free GAIA validation protocol defined by `gaia_text_2023_validation_no_attachment@682dd723ee1e1697e00360edccf2366dc8418dd9`. It does not contain or download GAIA rows. Runtime task files and browse assets must be supplied externally.

The `native` and `amg_memory` arms use one search/visit/answer dispatcher. Native creates no private workspace and offers no policy compaction. AMG memory creates a new empty `PersistentWorkspace` for every task and enables client-owned, policy-authored task-neutral `replace_messages` compaction. Search, visit, final-answer extraction, zero online reward, and external submission handling are shared.

## Runtime boundary

The inference server requires:

- `GAIA_TEXT_ARM`: `native` or `amg_memory`.
- `GAIA_TEXT_MANIFEST`: external canonical manifest JSONL from the pinned audit.
- `GAIA_TEXT_QUESTIONS` and `GAIA_TEXT_QUESTIONS_SHA256`: external runner-only question JSONL and its staged byte digest.
- `GAIA_TEXT_BACKEND_ASSET` and `GAIA_TEXT_BACKEND_SHA256`: external deterministic browse fixture/corpus asset and its digest.
- `GAIA_TEXT_PREDICTIONS`: fresh external output path.

The memory arm additionally requires `GAIA_TEXT_WORKSPACE_ROOT`, `GAIA_TEXT_RG_BINARY`, and `GAIA_TEXT_RG_SHA256`. Its launcher lazily imports the existing AgentMemoryGym Linux namespace sandbox; native does not import or construct memory state.

The service always binds `127.0.0.1`; `GAIA_TEXT_HOST` is intentionally ignored. `GAIA_TEXT_VISIT_PAGE_CHARS` and `GAIA_TEXT_MAX_POLICY_STEPS` may override their positive-integer defaults. Metadata publishes a canonical, arm-neutral `paired_runtime_contract` and SHA-256 covering the protocol/task/question hashes, backend asset and page size, policy budget, shared domain/answer/reward contracts, and submission format. The common paired runner must require equal digests across arms; it can also pass the frozen digest as `expected_paired_runtime_sha256` to each `GaiaTextEnvClient` for fail-fast pinning.

Do not provide a gold or scorer environment variable. The launcher rejects GAIA-related variable names containing `GOLD` or `SCORER`. The server has no score/detail endpoint, never loads final answers or scorer code, and does not expose host paths. Run the pinned official scorer later in a distinct process with the server stopped.

Before a real memory run, exercise the Linux namespace sandbox preflight on the target Linux host. The macOS unit fixture verifies adapter semantics but cannot validate Linux namespace isolation itself.

While incomplete, predictions are stored at `<GAIA_TEXT_PREDICTIONS>.partial`. Once every manifest ID has exactly one string-or-null answer, the package atomically publishes `<GAIA_TEXT_PREDICTIONS>` as newline-terminated objects with exactly this key order:

```json
{"task_id":"...","model_answer":"..."}
```

The external scorer pins remain revision `9f133d71362e77b3539f1514f31b9c101a545fec` and SHA-256 `0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2`; they are documentation for the separate scoring process, not server inputs.

## Verification

Tests generate 127 synthetic records at runtime; no gated task row is committed.

```bash
PYTHONPATH=agentenv-gaia-text:agentenv:agentenv-agentmemory \
  uv run --no-project --python 3.12 \
  --with pytest --with fastapi --with httpx --with requests --with uvicorn \
  python -m pytest agentenv-gaia-text/tests -q
uvx --python 3.12 ruff check \
  agentenv-gaia-text/agentenv_gaia_text \
  agentenv-gaia-text/tests \
  agentenv/agentenv/envs/gaia_text.py
uv run --no-project --python 3.12 python -m compileall \
  agentenv-gaia-text/agentenv_gaia_text \
  agentenv/agentenv/envs/gaia_text.py \
  agentenv-gaia-text/tests
git diff --check
```
