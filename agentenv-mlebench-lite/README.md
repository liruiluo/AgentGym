# AgentEnv MLE-bench Lite

Matched `native` / `amg_compaction_only` / `amg_memory` environment adapter for
the pinned official MLE-bench Lite split. Formal execution requires an
independently installed, SHA-pinned sandbox runner that attests the exact mount
and isolation contract; there is no host-subprocess fallback.

This package does not contain Kaggle data, private labels, grading code, model
weights, or benchmark scores.

## Frozen benchmark contract

The adapter independently loads the external checkout at commit
`507f92e1138bb6e40dac5c6ee7a6758e6424bf97` and verifies the raw
`experiments/splits/low.txt` hash plus the exact ordered 22 IDs. It does not
import the upstream registry. The public manifest contains a sorted
path/size/SHA-256 inventory for every task and points directly at the official
cache layout:

```text
<data>/<competition>/prepared/public
<data>/<competition>/prepared/private
```

The private sibling is checked for path/symlink isolation, then discarded.
Policy records contain only the selected public source and public tree digest.
No 158 GB copy or alternate aggregate data layout is created.

## Runtime configuration

Launch accepts one strict external JSON file. Unknown/missing fields, relative
paths, symlinks, overlapping roots, hash drift, or an unpinned runner fail
closed. Existing episode and handoff roots must be directories owned by the
service user with exact mode `0700`; the server binds only to loopback.

```json
{
  "schema": "mlebench_lite_runtime_config_v2",
  "upstream_root": "/protected/src/mle-bench",
  "data_root": "/protected/mlebench-data",
  "public_manifest_path": "/protected/manifests/lite-public.json",
  "public_manifest_sha256": "<64 lowercase hex>",
  "episodes_root": "/run/mlebench-lite/episodes",
  "handoff_root": "/protected/mlebench-lite/handoffs",
  "sandbox_runner_path": "/opt/mlebench-lite/bin/sandbox-runner",
  "sandbox_runner_sha256": "<64 lowercase hex>",
  "sandbox_runtime_digest": "<64 lowercase hex>",
  "sandbox_runner_uid": 0,
  "max_actions": 30,
  "max_submission_bytes": 100000000,
  "max_shell_timeout_ms": 3600000,
  "episode_timeout_ms": 86400000,
  "max_total_execution_ms": 72000000,
  "cpu_limit_cores": 36,
  "memory_limit_bytes": 440000000000,
  "pids_limit": 4096,
  "writable_bytes_limit": 500000000000,
  "writable_inodes_limit": 2000000,
  "gpu_count": 1,
  "forbidden_roots": ["/protected/grader", "/protected/leaderboard"]
}
```

```bash
mlebench-lite --config /protected/config/mlebench-lite.json \
  --host 127.0.0.1 --port 9017
```

Server metadata binds the upstream/split/ordered tasks, public manifest,
runner, runtime, and the complete resource contract plus its canonical digest.
That contract covers the episode deadline, total execution time, CPU, memory,
PIDs, writable bytes/inodes, GPU count, action/submission/output limits,
cgroup descendant scope, process-group isolation, networking, and public-data
read-only state. `max_step_response_ms` is conservatively derived as the
episode deadline plus 30 seconds for runner transport and cleanup; the
AgentGym HTTP timeout must be strictly larger. The client verifies every field
before creating a slot, accepts only an absent `data_len` or exact `22`, and
requires identical metadata for all three arms.

`/create` returns a random owner capability token. Reset, step, and close bind
that token to the slot. Every step also carries a canonical UUID4 action ID;
the server caches the exact request payload and public result. A client keeps
the ID pending until the complete response and cumulative counter ledger have
validated, so a lost response—including a successful `submit` response—is
retried without executing the action twice. Reusing an ID with changed bytes
is rejected.

## Isolation and handoff

The configured runner UID is a strict nonnegative integer (`0` is valid). The
runner file and its immediate parent must have that owner; the parent must not
be group/world writable, and the file must be executable, non-writable,
regular, single-link, and match the pinned SHA-256 on every invocation. The
adapter executes the verified open descriptor rather than reopening its
pathname. Every runner request carries the complete canonical resource
contract and digest; the adapter recomputes the digest before sending it, and
the mount attestation must echo both exactly.
The runner must independently create and attest a non-root mount namespace
with networking disabled, a read-only root filesystem, only the selected
public source mounted read-only at `/home/data`, and only the current episode's
workspace/submission roots writable. It must independently verify the public
tree digest, deny host/private/sibling mounts, enforce the mode-specific
external-memory isolation attestation, enforce the pinned cgroup/CPU/RAM/PID/GPU and
writable-volume limits, and return strictly allowlisted execution,
freeze/reap, and teardown receipts. Each execution receipt must prove exact
prior-plus-delta resource counters and zero surviving descendants, including
timeouts. The adapter has no `subprocess(cwd=workspace)` policy fallback.

`native` has no memory prompt, namespace, or compaction candidate.
`amg_compaction_only` has the exact policy-authored task-neutral compaction
trigger, request, transition, and action accounting used by `amg_memory`, but
has no memory prompt, private root, or memory action parser. `amg_memory`
additionally gets a fresh task/reset-local private root mounted read/write only
at `/run/amg_memory`. The native `inspect`, `edit`, and `shell` schemas are
unchanged across all arms. The pinned sandbox runner supplies a strictly
validated structured receipt only when execution actually accesses that mount;
mentioning its pathname is not an access receipt. Compaction is sent to
`/step` first and atomically consumes the same server action budget; the client
emits `replace_messages` only after validating the `n -> n+1` receipt. A
compaction that exhausts the budget is terminal and never replaces context.

`submit` performs public structural CSV checks, freezes/reaps the sandbox,
reopens a no-follow regular single-link bounded file, and stages the exact
bytes outside every policy mount in an owner-only directory. It then tears
down the sandbox, removes the writable episode workspace, writes a protected
manifest binding episode/mode/competition, runner/runtime/resource digests and
freeze/teardown receipts, seals both files to mode `0400`, atomically publishes
the directory, and seals it to `0500`. A publish failure rolls back both the
staging and final name. The policy-visible terminal receipt remains exactly:

```text
competition_id, submission_path, submission_sha256
```

The submission path is always `/home/submission/submission.csv`. Official
grading is a later host-only workflow and is not called by this adapter; no
validity result, score, private path, or grader detail is policy-visible.
The later official grader locates submissions only through the protected
handoff root (or the manager's trusted in-process lookup); no HTTP response or
policy receipt contains the host path. Reset, close, action-budget terminal,
and application shutdown also teardown/reap before removing active episode
workspaces. Successful handoffs are deliberately retained for host grading.

Missing, escaping, symlinked, or non-regular policy paths remain ordinary
`Path is unavailable.` results. Storage faults such as `EIO`, `ENOSPC`, or
`EDQUOT` produce one generic zero-reward infrastructure terminal. Incomplete
episode, failed workspace-creation rollback, and handoff-staging cleanup stays
attached to the slot; reset or close retries the same teardown operation and
the exact manager-tracked creation/staging targets. `/close` returns a generic
`503` while such cleanup is pending and succeeds only after the retry completes.
A successful close marks and removes the slot while holding its lifecycle lock,
so a stale concurrent reset cannot create an unreachable workspace.

The real dynamic namespace/cgroup/process/network probes remain a runtime-host
gate. Unit tests validate the adapter protocol with a synthetic attested
backend; they do not claim that this Mac is a formal MLE-bench execution host.

## Bundled approved-equivalent bridge

`runtime_bridge/` now provides the exact external-runner operations using a
hash-locked MLE-specific supervisor. It preserves the adapter protocol and the
36-CPU / 440,000,000,000-byte memory / 4,096-PID / one-GPU contract; it does
not substitute OpenMLE tasks or grading semantics. The deployment identity
also pins a complete sealed rootfs runtime: native launcher, loader, Python,
ELF dependency closure, mapped-object audit, `nvidia-smi`, GPU UUID and device
minors, plus reusable OpenMLE-v7 isolation provenance.

See [`runtime_bridge/README.md`](runtime_bridge/README.md) for rootfs locking,
private-mount provisioning, reproducible build, and verifier commands. Local
source/protocol success is only a pre-host certificate. Linux/B200 admission,
Kaggle assets plus the 22-task checksum manifest, a matched one-task three-arm
gate, and the official host-only grader remain explicitly pending.
