# GAIA-Text thin AMG triad adapter

This package serves the frozen 127-task attachment-free GAIA validation protocol defined by `gaia_text_2023_validation_no_attachment@682dd723ee1e1697e00360edccf2366dc8418dd9`. It does not contain or download GAIA rows. Runtime task files and retrieval certificates must be supplied externally.

The frozen arm set is exactly `native`, `amg_compaction_only`, and `amg_memory`. All three use one Search/Visit/Answer dispatcher and one arm-neutral production-backend identity. Native has neither policy compaction nor external memory. Compaction-only and full AMG use the same client-owned, policy-authored compaction request, token-pressure trigger, task-neutral `replace_messages` transition, and policy-action accounting. Only full AMG receives the external-memory prompt fragment and a new empty `PersistentWorkspace` for each task.

The triad client is `agentenv_gaia_text.client.GaiaTextEnvClient`. Both compacting arms use one implementation built from the existing task-neutral transition helpers. Its common compaction request and continuation marker are memory-neutral; `amg_compaction_only` therefore never receives workspace instructions or a workspace parser/schema. Search, visit, final-answer extraction, zero online reward, and external submission handling remain shared.

## Runtime boundary

The inference server requires:

- `GAIA_TEXT_ARM`: exactly `native`, `amg_compaction_only`, or `amg_memory`.
- `GAIA_TEXT_BACKEND`: explicit `production` or `fixture`; there is no implicit default.
- `GAIA_TEXT_MANIFEST`: external canonical manifest JSONL from the pinned audit.
- `GAIA_TEXT_QUESTIONS` and `GAIA_TEXT_QUESTIONS_SHA256`: external runner-only question JSONL and its staged byte digest.
- `GAIA_TEXT_PREDICTIONS`: fresh external output path.

`fixture` is only for deterministic tests. It requires `GAIA_TEXT_BACKEND_ASSET` and `GAIA_TEXT_BACKEND_SHA256` and rejects production inputs. A production run must select `production`, which rejects fixture inputs and requires:

- `GAIA_TEXT_LITERESEARCHER_BASE_URL`: one HTTP(S) origin containing no credentials, path, query, or fragment.
- `GAIA_TEXT_LITERESEARCHER_CERTIFICATE` and `GAIA_TEXT_LITERESEARCHER_CERTIFICATE_SHA256`: an external deployment certificate and the SHA-256 of its exact bytes.
- Optional frozen bounds: connect timeout `GAIA_TEXT_LITERESEARCHER_CONNECT_TIMEOUT_MS` (default 2000), read/write timeout `GAIA_TEXT_LITERESEARCHER_READ_TIMEOUT_MS` (30000), `GAIA_TEXT_LITERESEARCHER_RETRY_COUNT` (2), retry backoff `GAIA_TEXT_LITERESEARCHER_RETRY_BACKOFF_MS` (100), `GAIA_TEXT_SEARCH_RESULT_LIMIT` (10, maximum 50), `GAIA_TEXT_VISIT_PAGE_CHARS` (8192), and `GAIA_TEXT_VISIT_PAGE_LIMIT` (256).

The production bridge pins LiteResearcher revision `779e7d5f6a043d4100149ba0992a39507f69a974`. It probes `GET /health`, sends the upstream-native five-field hybrid request to `POST /search`, and sends only `url` to `POST /web_parser`. The full text returned by `/web_parser` is paginated locally under the frozen page bounds. HTTP redirects, inherited proxy environment variables, alternate endpoints, and live-web fallback are disabled. Timeout, connection, HTTP-status, and response-contract failures are separately classified and all terminate the current episode fail-closed.

The external certificate is strict JSON with exactly these fields:

```json
{
  "schema": "gaia_text_literesearcher_service_certificate_v1",
  "upstream_source_revision": "779e7d5f6a043d4100149ba0992a39507f69a974",
  "endpoint_contract_sha256": "b9694a7a90d34522626cb0444d3aca6541ae804923a26a2f555d4d13289fc0b4",
  "endpoint_origin_sha256": "<sha256 of the canonical configured origin>",
  "search_corpus_certificate_sha256": "<sha256>",
  "search_index_certificate_sha256": "<sha256>",
  "browse_store_certificate_sha256": "<sha256>"
}
```

The service deployment owner must provide this certificate only after the pinned source is running with a populated Milvus hybrid index, a live BGE-M3 embedding worker behind Redis, and the PostgreSQL browse store loaded and verified. `/health` proves that Milvus is loaded and Redis responds, but it does not prove that an embedding worker consumes the queue. Certificate issuance therefore also requires a real bounded `/search` probe, plus a known-URL `/web_parser` probe because the pinned health response does not report PostgreSQL. No deployment certificate or corpus is committed to this package.

The memory arm additionally requires `GAIA_TEXT_WORKSPACE_ROOT`, `GAIA_TEXT_RG_BINARY`, and `GAIA_TEXT_RG_SHA256`. Its launcher lazily imports the existing AgentMemoryGym Linux namespace sandbox. Native and compaction-only import no memory package, create no workspace root or cleanup handle, expose no memory runtime metadata, and reject any non-empty `GAIA_TEXT_WORKSPACE_*` or `GAIA_TEXT_RG_*` variable instead of silently retaining a hidden memory path. Workspace-shaped policy actions in either memory-disabled arm follow the ordinary invalid-action path and disclose no host path.

The service always binds `127.0.0.1`; `GAIA_TEXT_HOST` is intentionally ignored. `GAIA_TEXT_MAX_POLICY_STEPS` may override its positive-integer default. Production metadata exposes no configured URL or certificate path. Its `runtime_identity_sha256` (also carried in the legacy `asset_sha256` client field) binds the upstream revision, endpoint contract and origin digests, service/corpus/index/store certificate digests, timeouts, retries, result/page limits, and no-fallback policy. The canonical arm-neutral `paired_runtime_contract` incorporates that identity digest. The common paired runner must require equal digests across arms; it can also pass the frozen digest as `expected_paired_runtime_sha256` to each `GaiaTextEnvClient` for fail-fast pinning.

Do not provide a gold or scorer environment variable. The launcher rejects GAIA-related variable names containing `GOLD` or `SCORER`. The server has no score/detail endpoint, never loads final answers or scorer code, and does not expose host paths. Run the pinned official scorer later in a distinct process with the server stopped.

The lifecycle has distinct terminal and rollback exits. `POST /close` accepts only
a terminal episode. `POST /abort` discards an unfinished, unscored attempt without
creating a null submission. Both exits remove the episode, close any owned
workspace, and release the task claim, so a coordinator can replay an entire
three-arm task after a retryable service or model failure. Submission rollback is
still owned by the external transactional submission controller; aborting an
environment never publishes scorer input.

Before a real memory run, exercise the Linux namespace sandbox preflight on the target Linux host. The macOS unit fixture verifies adapter semantics but cannot validate Linux namespace isolation itself. No memory root or sandbox variables should be supplied to native or compaction-only runs.

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
