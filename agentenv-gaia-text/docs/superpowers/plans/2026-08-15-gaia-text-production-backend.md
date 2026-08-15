# GAIA-Text Production Backend Integration Plan

**Goal:** Extend the clean GAIA-Text adapter with a fail-closed, arm-neutral HTTP bridge to the pinned LiteResearcher local Search/Visit stack while retaining the fixture backend only for explicit deterministic tests.

**Architecture:** Define one `SearchVisitBackend` protocol consumed by the existing shared episode manager. The production implementation validates a byte-digested external service certificate, pins LiteResearcher revision `779e7d5f6a043d4100149ba0992a39507f69a974` and the exact `/health`, `/search`, and `/web_parser` schemas, talks only to one configured origin with redirects and proxy inheritance disabled, and converts the upstream full-text response into bounded deterministic pages. The launcher requires an explicit fixture/production selection, and the paired runtime identity hashes only stable, sanitized backend facts shared by both arms.

**Scope:** Only `agentenv-gaia-text/**`; no runner, controller, rollout, registry, GPU, service deployment, corpus build, scorer, or gated-data changes.

## Task 1: Freeze tests for the production contract

- [x] Add a local fake LiteResearcher HTTP service and certificate builder.
- [x] Prove exact search/web-parser requests, strict response validation, health gating, deterministic visit pagination, and configured limits.
- [x] Prove timeout, connection, HTTP, and protocol-error classification; bounded retries; redirect/no-fallback behavior; and secret/path-free failures and metadata.
- [x] Prove certificate byte-digest and identity-digest mismatch rejection.
- [x] Prove explicit fixture/production selection and identical native/memory metadata plus dispatch.

## Task 2: Implement the backend bridge

- [x] Add the explicit backend protocol and typed infrastructure errors.
- [x] Add strict certificate parsing, origin binding, stable runtime identity, and pinned endpoint-contract digest.
- [x] Add the HTTP client with frozen timeouts/retries/limits, proxy and redirect disabling, exact schemas, bounded response parsing, and no network fallback.
- [x] Preserve the existing policy-facing search/visit result shapes and GAIA answer semantics.

## Task 3: Assemble and document the runtime

- [x] Select the backend through an explicit environment value and reject mixed fixture/production inputs.
- [x] Generalize the paired runtime contract to sanitized backend identity and confirm arm equality.
- [x] Document the external service certificate schema and exact live runtime prerequisites.

## Task 4: Verify, review, and commit

- [x] Run the original 47 tests plus all new regressions, Ruff, compileall, diff-check, and a write-scope scan.
- [x] Dispatch an independent read-only reviewer against `c11ad2c` and the follow-up contract; address findings and rerun verification.
- [x] Commit one scoped follow-up commit without pushing and report its SHA plus live-service-only blockers.
