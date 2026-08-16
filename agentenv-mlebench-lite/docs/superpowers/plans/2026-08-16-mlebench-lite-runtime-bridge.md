# MLE-bench Lite Runtime Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and certify a task- and arm-agnostic Linux runtime bridge that implements the existing MLE-bench Lite external-runner protocol using only the reusable cgroup/namespace/reaping ideas from the admitted OpenMLE v7 runtime.

**Architecture:** Keep the adapter protocol unchanged. Add a hash-locked runtime bundle containing a strict JSON runner, an authenticated per-episode ledger, a Linux cgroup-v1 orchestration module, and a small C namespace supervisor. `attest` creates one exact-size/inode tmpfs at the episode root in the runner's host mount namespace; that mount deliberately survives the short-lived runner process, so every later invocation opens and binds the same workspace, submission and optional memory directories into a fresh per-command namespace. `freeze` reaps descendants and remounts that host mount read-only so the wrapper can read/stage the submission; `teardown` unmounts it and writes an external tombstone after zero-residue proof. The bridge owns isolation and receipts only; wrapper lifecycle stays in the existing adapter and OpenMLE fit/task/private-grader semantics are excluded. Mac tests exercise the protocol and adversarial boundaries with an injected kernel facade; a reproducible live verifier remains fail-closed until run on a coordinator-authorized Linux/B200 host.

**Tech Stack:** Python 3.11+ standard library, Linux cgroup v1, mount/PID/IPC/network/user namespaces, seccomp-BPF, ptrace descendant accounting, GCC/Clang, `unittest`/`pytest`.

**Current status:** Owner-side remediation and fresh pre-host verification are
complete. The final two independent-review blockers now have focused green
coverage: transitive rootfs symlink expansion rejects escape/cycles while
retaining contained chains, and storage-freeing boundaries quiesce all tracees
before writable high-water sampling. The full pinned suite, Ruff, residue-free
compileall, source/diff/shared-rollout audits, GCC 12.5 reproducible builds, and
CPU9N live chained-symlink/concurrent-splice regressions pass. A final bounded
read-only re-review and inner-first/outer-second publication remain blocking.
Detailed evidence and pending external gates are recorded in
`runtime/mlebench-runtime-bridge-20260816/verification-evidence-20260816.md`
in the paired-evaluation control workspace.

---

## File map

- Create `runtime_bridge/runner.py`: executable `attest|execute|freeze|teardown` entrypoint, strict schema validation, artifact verification, HMAC state/replay ledger, exact adapter receipts.
- Create `runtime_bridge/linux_runtime.py`: persistent host quota-mount ownership, reusable OpenMLE-derived cgroup-v1 (including devices), per-command namespace launch, stable dirfd public-tree verification, resource sampling, GPU device verification, freeze/reap/teardown and residue primitives; no benchmark semantics.
- Create `runtime_bridge/sandbox_supervisor.c`: read-only rootfs plus verified mount-FD setup, one-GPU device projection, non-root/user namespace, no-egress seccomp, unrestricted benchmark-native exec surface, ptrace descendant accounting and reaping.
- Create `runtime_bridge/build_bundle.py`: validate deployment input, compile the supervisor on Linux, build the immutable artifact lock, and emit the adapter runtime-config identity fragment.
- Create `runtime_bridge/verify_runtime.py`: static/adversarial verifier everywhere and destructive synthetic live gate only on an explicitly supplied Linux test root/GPU.
- Create `runtime_bridge/deployment.example.json`: exact production resource/rootfs/GPU configuration shape without credentials or task data.
- Create `runtime_bridge/README.md`: ownership boundary, build/install, evidence limits, actual-host admission procedure.
- Create `tests/test_runtime_bridge_protocol.py`: protocol/state/replay/ledger/mount/arm adversarial tests with a fake kernel facade.
- Create `tests/test_runtime_bridge_linux_source.py`: static source/build-policy assertions and live-test skip boundary.
- Modify `README.md`: document the bundled bridge and explicitly retain pending Kaggle/data/actual-host gates.

### Task 1: Freeze protocol constants and request validation

**Files:**
- Create: `runtime_bridge/runner.py`
- Test: `tests/test_runtime_bridge_protocol.py`

- [x] Write failing tests proving duplicate JSON keys, unknown/missing keys, noncanonical UUID4, noncanonical/relative/NUL paths, invalid modes, arm/memory-root mismatches, resource-hash drift, and any value other than 36 CPU / 440,000,000,000 bytes / 4096 pids / one GPU fail closed.
- [x] Run `python -m pytest -q tests/test_runtime_bridge_protocol.py` and require the new tests to fail for missing code.
- [x] Implement `strict_json_loads`, canonical hashing, `_validate_request`, `_validate_resource_contract`, and descriptor-safe root validation. Keep competition and mode opaque except for the optional-memory invariant.
- [x] Re-run the targeted tests and require pass.

Core interface:

```python
class BridgeError(RuntimeError):
    pass

def handle(operation: str, request: dict[str, object], runtime: RuntimeFacade) -> dict[str, object]:
    """Return one exact adapter response or raise BridgeError without details on stdout."""
```

### Task 2: Artifact identity and authenticated episode state

**Files:**
- Modify: `runtime_bridge/runner.py`
- Create: `runtime_bridge/build_bundle.py`
- Create: `runtime_bridge/deployment.example.json`
- Test: `tests/test_runtime_bridge_protocol.py`

- [x] Add failing tests for mutable/unlisted bundle members, artifact-lock drift, unsafe owner/mode/link count, replayed operation IDs with changed requests, cross-episode state reuse, state tampering, and lifecycle calls after teardown.
- [x] Implement canonical artifact-lock hashing and HMAC-sealed per-episode state with `flock`, boot ID, mount identity, request hashes, replayed exact responses, cumulative usage, lifecycle, and a teardown tombstone stored outside the episode mount.
- [x] Persist and fsync a `pending` operation record before each side effect, atomically persist the complete final response before writing stdout, never re-execute an indeterminate `execute`, and reconcile pending freeze/teardown only from observable mount/cgroup/residue facts. Add fault injection immediately before/after side effects, final-state replace/fsync, unmount and stdout.
- [x] Ensure the runtime digest binds runner, Python runtime module, compiled supervisor, deployment config, rootfs digest, GPU UUID/device inventory, and the admitted OpenMLE v7 source identities used as provenance.
- [x] Re-run targeted tests.

### Task 3: Exact attest and receipt schemas

**Files:**
- Modify: `runtime_bridge/runner.py`
- Test: `tests/test_runtime_bridge_protocol.py`

- [x] Add failing exact-equality tests for MLE attestation v3, execution v3, freeze v2 and teardown v2 for all three arms.
- [x] Implement exact responses using the runner SHA, runtime digest, resource-contract hash, canonical mount-attestation hash, command hash, operation UUID, high-water writable ledger, zero descendants and lifecycle flags.
- [x] Prove `native`, `amg_compaction_only`, and `amg_memory` use the same artifact/runtime/resource identity and that the only mount difference is the request-authorized memory root.
- [x] Re-run targeted tests.

### Task 4: Linux cgroup/namespace orchestration

**Files:**
- Create: `runtime_bridge/linux_runtime.py`
- Create: `runtime_bridge/sandbox_supervisor.c`
- Test: `tests/test_runtime_bridge_linux_source.py`

- [x] Add failing tests/static assertions for exact cgroup values, memory+memsw equality, swap zero, pids cap, devices allowlist, private mount propagation, new mount/PID/IPC/network/user namespaces, read-only public bind, absent `/private`/`/host`, one selected GPU compute minor plus only its required control/UVM/caps nodes, capability drop, no-new-privs, an AF_UNIX-only socket allowlist that rejects every other family including AF_VSOCK, closed inherited sockets/device FDs, and no OpenMLE fit/utility/grader symbols.
- [x] Port only the reviewed OpenMLE v7 cgroup setup/stats/cleanup, deadline-bound pipe draining, watchdog, residue scan, read-only overlay, minimal device, privilege-drop and ptrace descendant patterns.
- [x] In `attest`, mount an episode-owned tmpfs in the host mount namespace with exact `size=500000000000,nr_inodes=2000000`; persist its mount ID/device/inode and create only workspace/submission/optional-memory/runtime-temp children. In each command, open those children by no-follow dirfd and bind them to `/home/workspace`, `/home/submission`, optional `/run/amg_memory`; bind `/tmp`/shared-memory from the same quota mount. Freeze remounts the persistent host mount read-only; teardown unmounts it and proves no owned mount remains.
- [x] Recompute `public_tree_sha256` using stable no-follow dirfd inventory matching `dataset.py`'s sorted `{path,size,sha256}` algorithm; reject symlinks, special files, `st_nlink != 1`, subordinate mounts, inode/metadata changes while hashing, and realpath/device/inode aliases or nesting among public/workspace/submission/memory roots. Pass the same opened public dirfd to the supervisor so validation and binding cannot race.
- [x] Project the one pinned compute minor to `/dev/nvidia0`, add only the pinned shared control/UVM/caps nodes, set a devices-cgroup deny-by-default allowlist, close source device FDs after bind, set UUID-bound visibility, and prove changing environment variables cannot access other compute minors.
- [x] Preserve benchmark-native executables and contract resources: count descendants/execs but do not apply OpenMLE's utility allowlist, Python wrapper, fit hook, managed-runtime budget or private grader. Explicitly reject inherited OpenMLE constants (`MAX_PROCS=256`, `RLIMIT_NPROC=64`, one-thread BLAS variables, empty CUDA visibility, `MS_NOEXEC` workspace, and 64/256-MiB tmpfs caps); trace at least 4096 descendants and derive process/thread/file/tmp limits from the MLE contract.
- [x] Re-run source tests and compile the C source where a Linux compiler is available.

### Task 5: Runtime resource and lifecycle enforcement

**Files:**
- Modify: `runtime_bridge/linux_runtime.py`
- Modify: `runtime_bridge/runner.py`
- Test: `tests/test_runtime_bridge_protocol.py`

- [x] Add failing facade tests for CPU/memory/pids/devices setup mismatch, timeout, output cap, process leak, cumulative execution overflow, writable byte/inode overflow, symlink/source/content/metadata swap, GPU UUID/count/minor/control-node drift, freeze failure, teardown residue and pending-operation recovery.
- [x] Implement action deadlines, cgroup CPU usage, wall time, writable capacity/inode high-water sampling, process-start counters, output bounds, process-group/cgroup reaping, and deterministic cleanup.
- [x] Add an integration fixture spanning separate runner processes: `attest → execute/write → execute/read → freeze → host read of submission → teardown`, asserting the persistent quota mount identity, hard byte/inode caps, read-only freeze and final zero mounts.
- [x] Make every infrastructure ambiguity fail closed with nonzero runner exit and no JSON success receipt.
- [x] Re-run targeted tests.

### Task 6: Reproducible build and strongest pre-host verifier

**Files:**
- Modify: `runtime_bridge/build_bundle.py`
- Create: `runtime_bridge/verify_runtime.py`
- Create: `runtime_bridge/README.md`
- Modify: `README.md`
- Test: `tests/test_runtime_bridge_linux_source.py`

- [x] Add failing tests for non-Linux build claims, compiler/hash drift, unsafe install ownership, missing rootfs digest, ambiguous/multiple GPU selection, and accidental live-test execution without explicit flags.
- [x] Implement deterministic compile flags, two-build byte comparison on Linux, artifact-lock generation, source provenance, install receipt and adapter identity fragment.
- [x] Implement a verifier that always runs strict protocol/static tests and, only with explicit `--live-root --gpu-uuid`, creates synthetic public/private fixtures and proves stable public-tree hashing, public read-only, private/host/path/symlink denial, denial of AF_INET/AF_INET6/PACKET/NETLINK/VSOCK and inherited socket FDs, exact CPU/memory/pids/devices cgroups, CUDA and NVML enumeration of exactly the pinned GPU despite environment override attempts, timeout/reap, cross-process persistence, host-readable freeze, teardown and zero residue.
- [x] On Mac, emit `actual_host_admission=pending` rather than a pass.

### Task 7: Full verification and independent review

**Files:**
- Modify only findings in the files above.
- Create evidence under `/Users/luolirui.1/Projects/amg-paired-eval-20260815/runtime/mlebench-runtime-bridge-20260816/`.

- [x] Run the full adapter suite with the repository's pinned command.
- [x] Run targeted bridge tests, `compileall`, `git diff --check`, source artifact-lock verification, and an active-source audit proving no MLE/arm branch was added to shared `vllm_rollout.py`.
- [x] Run the pre-host verifier and record live GPU/cgroup gates as pending unless a coordinator handoff explicitly authorizes a lane.
- [x] Obtain independent review under the written subagent contract; resolve all Critical/Important findings and rerun verification.

Required commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=AgentGym/agentenv-mlebench-lite:AgentGym/agentenv \
  uv run --isolated --no-project --python 3.12 --with pytest --with fastapi \
  --with pydantic --with requests --with uvicorn --with torch --with numpy \
  --with transformers --with openai --with httpx2 \
  python -m pytest -q AgentGym/agentenv-mlebench-lite/tests
python -m compileall -q -f AgentGym/agentenv-mlebench-lite
git diff --check
```

### Task 8: Publish inner first, then outer

**Files:**
- Inner commit: all `agentenv-mlebench-lite/**` changes.
- Outer commit: `AgentGym` gitlink only unless a separately justified MLE deployment file is required.

- [ ] Verify the inner diff contains no data, credential, private grader, heavy artifact or unrelated file.
- [ ] Commit and push `feat/amg-mlebench-runtime-bridge-inner-20260816` to `canonical-github` first; verify the remote SHA.
- [ ] Stage the new inner gitlink in the outer worktree, commit and push `feat/amg-mlebench-runtime-bridge-outer-20260816` to `origin`; verify the remote SHA.
- [ ] Fresh-clone/fetch the outer branch plus submodule and rerun the small identity/verifier gate.
- [ ] Finish the supervision state only after the certificate names every remaining external gate: Kaggle access/assets, 22-task checksum manifest, actual-host admission, one-task three-arm gate, official grader.

## Blocking remediation from independent review

The 2026-08-16 independent review reported zero Critical and eight Important
findings. The owner-side remediation and red/green checks below are complete;
publication remains blocked on independent re-review. Dynamic Linux/B200
assertions remain part of the explicitly pending actual-host admission gate.

- [x] **Optional memory FD and all-arm live execution.** Reproduce the
  non-memory-arm `--memory-fd -1` launch and the verifier's arm-11-only gap.
  Omit the argument entirely for arms 00/10, execute a synthetic command in all
  three arms, and assert identical runtime/resource identities with only arm 11
  carrying the memory mount.
  Evidence: the red baseline failed
  `test_non_memory_supervisor_launch_omits_memory_fd_argument`; the fresh
  15-test matrix passes that test plus
  `test_all_arms_share_identity_and_only_memory_arm_gets_memory_mount`, and
  the live verifier now iterates `LIVE_MODES`. Actual-host execution remains
  pending rather than being inferred from source tests.
- [x] **Bounded immutable identities and end-to-end deadlines.** Reproduce the
  repeated full rootfs/public-tree hashing before timeout accounting. Replace
  it with build/attest-time immutable mount/tree identities plus bounded
  metadata revalidation, start each operation deadline before verification,
  and align the adapter timeout with the bridge's complete operation budget.
  Evidence: fresh Python 3.12 tests
  `test_read_only_tree_anchor_uses_bounded_mount_and_inode_identity` and
  `test_operation_deadline_expires_before_runtime_side_effects` pass; the full
  adapter suite also passes the wrapper's complete-response timeout contract.
- [x] **Runner-death cascade.** Reproduce killing the Python runner while the C
  supervisor and sandbox payload are alive. Add a race-safe native
  runner-to-supervisor-to-PID-namespace-init death cascade and prove the owned
  cgroup/process/mount residue reaches zero without relying on the runner's
  normal cleanup path.
  Evidence: the red baseline failed because the native launcher/death boundary
  was absent; `test_reviewed_runtime_boundaries_are_present_in_source` now
  passes, both native components compile reproducibly on GCC 12.5, and the live
  verifier contains `kill_runner_cascade`. Zero-residue execution remains an
  actual-host admission assertion.
- [x] **Crash-idempotent attest/teardown.** Reproduce a crash in `attesting` and
  a teardown retry under a new operation UUID. Permit exact cleanup from
  `attesting`; accept a new teardown UUID only after a bundle-bound tombstone
  and fresh zero-residue proof, while still rejecting indeterminate execute
  replay.
  Evidence: both red-baseline errors are green in the fresh matrix via
  `test_attesting_episode_can_be_torn_down_exactly` and
  `test_fresh_teardown_id_reconciles_verified_tombstone`; the existing
  indeterminate-execute regression remains green in the full suite.
- [x] **Public topology after namespace creation.** Reproduce a subordinate
  mount appearing between host validation and supervisor bind. Validate the
  public mount topology inside the newly private namespace, use a
  non-recursive bind, and make the resulting tree recursively read-only before
  exec.
  Evidence: `test_reviewed_runtime_boundaries_are_present_in_source` passes
  assertions for in-namespace topology validation and absence of recursive
  bind; the supervisor's current source hash is included in the reproducible
  GCC 12.5 build. Dynamic mount proof remains in actual-host admission.
- [x] **Sealed pre-exec runtime.** Reproduce mutation/replacement of the host
  Python/shebang dependency after bundle verification. Replace the generated
  script with a native launcher and a complete sealed isolated Python runtime
  (loader, libraries, stdlib/extensions, supervisor and required `nvidia-smi`),
  and attest the complete trusted parent/exec chain before any side effect.
  Evidence: the CPU9N reproduction proved glibc default-directory fallback;
  the fresh matrix passes the loader-fallback, ELF-closure and mapped-object
  rejection tests. The launcher and no-libc audit module build pairwise
  byte-identically, and `runtime-audit.so` has zero `DT_NEEDED` entries.
- [x] **Bundle-bound state and tombstones.** Reproduce cross-bundle state and
  teardown reuse. Persist the bundle identity in live state and tombstones and
  compare it before every side effect, recovery decision, replay, and cleanup.
  Evidence: the red baseline raised `KeyError: bundle_identity_sha256`; the
  fresh matrix passes both live-state and tombstone bundle-binding tests.
- [x] **Kernel-backed monotonic writable high-water.** Reproduce
  create-then-delete byte and inode workloads that evade post-exit `statvfs`
  sampling. Enforce and persist a genuinely monotonic kernel-backed byte/inode
  high-water source; timing-dependent polling is not admissible.
  Evidence: the red baseline incorrectly reported a second 60-byte delta; the
  fresh monotonic-ledger, kernel-stats and pre-exec sampling tests pass. CPU9N
  probes also demonstrated why pre-syscall ptrace sampling is required for
  `O_TMPFILE` exec/mmap cases; dynamic enforcement remains an actual-host gate.
- [x] **Dirfd-anchored bundle identity across exec.** Reproduce replacing or
  renaming an ancestor of the admitted bundle after the native launcher opens
  it. Reserve inherited bundle dirfd 197, prove `sandbox-runner` opened
  relative to that directory is the running launcher by device/inode, retain
  the anchor across exec, and load/inventory every bundle member and launch the
  supervisor only through `/proc/self/fd/197` without canonicalizing back to a
  pathname. Preserve the anchor and audit descriptors in every `pass_fds` set.
  Add an adversarial ancestor rename/replacement regression before marking this
  item complete. Evidence: the red test failed because `load_bundle_identity`
  had no bundle-fd API and source still canonicalized the published path. The
  focused green matrix passes a pinned-dirfd ancestor rename/replacement test;
  the Linux locked-entrypoint fixture now performs the rename/replacement after
  verification and reopens the supervisor through `/proc/self/fd/197`. GCC
  12.5 compiled the launcher twice byte-identically. Dynamic execution of that
  fixture remains part of the pending Linux-capability gate rather than being
  inferred from the macOS skip.
- [x] **Rootfs symlink containment.** Reproduce absolute and multi-`..`
  rootfs symlinks escaping the locked tree. Reject absolute link targets and
  lexically reject relative targets that traverse above the root while
  retaining valid contained relative links. Exercise the same policy in lock
  construction and verification before marking this item complete. Evidence:
  the red regression accepted `usr/escape -> ../../outside`; the focused green
  matrix rejects it and `/tmp` while accepting `bin -> usr/bin`. Lock creation
  and verification share the same fd-relative inventory function.

For each checkbox, record the failing command/output before implementation and
the passing command/output after implementation in the runtime certificate.
Do not weaken the fixed 36 CPU / 440,000,000,000-byte RAM / 4096 PID /
500,000,000,000 writable-byte / 2,000,000-inode / one-GPU contract to make a
regression green.

## Final blocking remediation from independent re-review

- [x] **Transitive rootfs symlink containment.** Preserve the red regression
  `sub/a -> ..`, `sub/x -> a/../..`, which escapes only after component-wise
  expansion of the first link. Build the complete fd-relative rootfs symlink
  map before validation, resolve every target component through that map, and
  reject absolute targets, traversal above the sealed root, cycles, and
  excessive expansion while retaining contained links such as
  `bin -> usr/bin`. Exercise the same inventory path during lock construction
  and lock verification.
- [x] **Stop-the-world writable high-water sampling.** Preserve the red source
  and live-verifier expectations for a concurrent producer splicing into an
  anonymous tmpfile while another tracee closes its descriptor. Replace the
  trace-me startup with a race-safe `PTRACE_SEIZE`/`PTRACE_INTERRUPT`
  lifecycle, quiesce every active tracee before each storage-freeing boundary,
  sample only while the tracee set is stopped, then resume without deadlocking
  a blocking producer/consumer pair. Handle fork, exit, signal and unexpected
  stop races fail closed, add the bounded `concurrent_close_splice_high_water`
  live adversary, and retain exact termination/resource targets.
- [x] Run the focused red-to-green matrix after each fix, then rerun the full
  pinned Python 3.12 suite, Ruff, residue-free compileall, diff/source/shared-
  rollout audits, reproducible GCC 12.5 builds, and prehost certificate.
- [x] Obtain an independent read-only re-review with zero Critical/Important
  findings.

## Four-finding remediation from final re-review

- [x] **Authenticate the configured runtime digest before Python.** The
  adapter now supplies `--expected-runtime-digest`; the native launcher opens
  the admitted bundle through dirfd 197, hashes the sealed artifact lock, and
  verifies compiled-in hashes for `runner.py`, `linux_runtime.py`, and the
  loader-audit module before executing the pinned Python runtime.
- [x] **Keep the 440,000,000,000-byte memory limit hierarchical for the whole
  episode.** Attest creates one exact cgroup-v1 memory parent with hierarchy,
  memory+swap, swappiness, OOM, usage, peak, and fail counters validated.
  Every execution gets an operation child below it; freeze retains the empty
  parent and teardown removes it only after unmount and operation cleanup.
- [x] **Accept only loader objects beneath the identity-checked rootfs FD.**
  Reserve inherited rootfs FD 200, construct library search paths below
  `/proc/self/fd/200`, and restrict the no-libc audit module to the sealed
  rootfs pathname or that exact FD prefix. CPU9N's real glibc 2.42 loader
  handshake passes through the FD-relative path with zero `DT_NEEDED` entries.
- [x] **Drain multithreaded `execve` and `exit_group` correctly.** Track TGIDs,
  serialize group-destructive mutations, remap a nonleader exec survivor with
  `PTRACE_GETEVENTMSG`, accept only expected sibling exit-event stops, hold the
  exec survivor until the group drains, clear state on failed exec, and avoid
  classifying killed sibling threads as background processes. The live
  verifier now runs named nonleader exec, failed-exec, and exit-group probes;
  the CPU9N exact-state-machine harness also retains a real background-child
  rejection control.
- [x] Rerun the focused and full pinned suites, Ruff, residue-free compileall,
  diff/scope/credential/shared-rollout audits, pairwise GCC builds, glibc
  loader audit, chained-symlink, concurrent-splice, and thread-group live
  regressions; refresh the prehost certificate with actual-host admission
  still truthfully pending.
- [x] Reactivate the same bounded read-only reviewer and require
  `Critical=0`, `Important=0` before publication.

## Three-finding remediation from zero-gate review

- [x] **Enforce final memory/pids event counters.** Compare child and persistent
  episode cgroup baselines before constructing a receipt. Reject backward or
  over-limit usage/peak accounting and every new child/episode failcnt, OOM
  kill, or pids-max event rather than returning a normal exit-137 receipt.
- [x] **Use the hierarchical memory path in runner-death admission.** Export
  the production operation-cgroup path derivation to the live verifier, wait
  on `memory/mlebridge-<episode>/<operation>`, read `pids/cgroup.procs`, and
  audit both current and legacy operation residue without requiring removal of
  the intentionally persistent episode parent.
- [x] **Reject lexical loader-audit escapes.** Require nonempty path components
  under both the sealed rootfs pathname and exact FD-200 prefix; reject `.`,
  `..`, empty components and wrong FDs. Exercise the actual C predicates on
  CPU9N and retain zero `DT_NEEDED` in the production audit module.
- [x] Rerun focused/full pinned tests, scoped Ruff, residue-free compileall,
  reproducible GCC 12.5 builds, FD-200 glibc audit, certificate and diff/scope
  audits; obtain final independent `APPROVED`, `Critical=0`, `Important=0`.
