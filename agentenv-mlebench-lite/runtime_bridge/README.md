# MLE-bench Lite runtime bridge

This directory contains the task-agnostic Linux executor for the existing
`ExternalSandboxRunnerBackend` protocol. It implements only isolation,
resource accounting, crash-consistent operation receipts, freeze/reap, and
teardown. Competition selection, action parsing, compaction, external-memory
lifecycle, submission handoff, and official grading remain in the MLE wrapper.

The implementation reuses only the admitted OpenMLE v7 cgroup-v1, namespace,
seccomp, descendant-tracing, and cleanup patterns. It does not run OpenMLE
tasks and does not contain its fit hook, utility allowlist, managed-runtime
budget, task packages, private worker, or grader. The OpenMLE artifact hashes
in `deployment.example.json` record provenance; they are not a claim that the
OpenMLE runner itself satisfies the MLE protocol.

## Fixed acceptance envelope

Every request must carry the adapter's exact canonical resource contract. The
bridge rejects any value other than:

- 36 CPU cores (`cpu.cfs_quota_us=3600000` at a 100000-us period)
- 440,000,000,000 bytes of memory and the same memory+swap limit, with
  swappiness zero. One hierarchical memory-cgroup parent persists from attest
  through teardown, and every execution runs in a child of that parent, so
  concurrent operations cannot each acquire a fresh episode-sized allowance.
- 4,096 concurrent PIDs
- one deployment-pinned GPU UUID and compute minor
- a 500,000,000,000-byte tmpfs (allowing only kernel page rounding) with
  exactly 2,000,000 inodes

The public source must already live on a read-only host mount. It is hashed
once during attest, persisted as a mount/inode anchor, passed by a no-follow
directory FD, non-recursively bound, topology-checked inside the private
namespace, and recursively sealed read-only. The rootfs follows the same
build/attest hash plus later bounded-anchor rule. The workspace, submission,
optional AMG memory, `/tmp`, and
`/dev/shm` all live on the one quota tmpfs. A frozen episode retains one
read-only mount for host submission staging; teardown requires zero owned
mounts, cgroups, and descendants.

All three arms use the same runner and runtime digest. The only arm-dependent
mount is `/run/amg_memory`, present for `amg_memory` and absent for `native`
and `amg_compaction_only`. The supervisor does not branch on competition ID or
arm.

## Host provisioning prerequisites

Production build and execution require Linux/x86_64, root, cgroup v1
`cpu,cpuacct`, `memory`, `pids`, and `devices` controllers, and an NVIDIA host.
Provisioning must complete these steps before building the bundle:

1. Install an MLE-compatible rootfs as a dedicated root-owned, read-only mount.
   It must contain no subordinate mounts, `/host`, or `/private`, and every
   fixed mount target must be a real directory or absent—not a symlink.
2. Make the mount containing `episodes_root` private. Shared, slave, or
   propagating mount ancestry is rejected so episode tmpfs mounts cannot leak
   into peer namespaces.
3. Create `state_root` and `episodes_root` as root-owned mode-`0700`
   directories. Reserve a dedicated nonzero host UID/GID for sandbox UID 1000.
4. Record the real loader, Python, Python home, library directories, and
   `nvidia-smi` paths inside that sealed rootfs. The native launcher opens the
   loader/Python without following symlinks, disables the host loader cache,
   and loads a no-libc audit module that exits before relocation/constructors
   if any mapped object resolves outside the sealed rootfs. The builder also
   freezes the loader-reported Python and `nvidia-smi` ELF closures; the runner
   independently verifies the loader, Python, extension-library, and audit
   mappings before lifecycle side effects. The GPU inventory subprocess uses
   the same audit and a mandatory initialization handshake. List only the
   selected compute device plus its required control/UVM/capability devices;
   device sources, targets, and major/minor pairs must all be unique.
5. Present every policy-visible public tree through a dedicated host-side
   read-only mount before attest. A writable source mount is rejected even
   though the in-sandbox bind would itself be read-only.
6. Fill a canonical deployment JSON using `deployment.example.json` as the
   schema. The example hashes and GPU UUID are placeholders and must never be
   used as an admission artifact.

Generate the rootfs tree lock only after the rootfs mount is read-only:

```bash
python3 runtime_bridge/build_bundle.py \
  --rootfs-lock-source /opt/mlebench-lite/rootfs \
  --rootfs-lock-output /opt/mlebench-lite/rootfs-tree-lock.json
```

Copy the returned `rootfs_digest` and `rootfs_tree_lock_sha256` into the
deployment JSON. Build the immutable bundle; the builder compiles the native
runner launcher, loader-audit module, and namespace supervisor twice and
requires every pair to be byte-identical. The launcher executes only the
sealed rootfs loader and Python runtime:

```bash
python3 runtime_bridge/build_bundle.py \
  --deployment /protected/mlebridge-deployment.json \
  --output /opt/mlebench-lite/runtime-bundle \
  --compiler /usr/bin/gcc
```

Use the receipt's `runner_path`, `runner_sha256`, and `runtime_digest` in the
adapter runtime configuration. The adapter passes that digest to the native
launcher, which hashes the anchored artifact lock and the compiled-in Python,
runtime, and audit members before starting Python. Do not edit an installed
bundle; every member, including deployment and build provenance, is hash
locked and read-only.

## Verification and evidence boundary

The non-destructive pre-host verifier runs the protocol and source tests and
always records actual-host admission as pending:

```bash
PYTHONPATH=agentenv-mlebench-lite \
python3 agentenv-mlebench-lite/runtime_bridge/verify_runtime.py \
  --output /protected/evidence/prehost-certificate.json
```

The synthetic live gate is deliberately inaccessible unless all four live
arguments are present. `LIVE_ROOT` must be an empty root-owned mode-`0700`
directory named `mlebridge-live-*`; the command creates and removes only its
synthetic fixtures and configured episode mounts:

```bash
PYTHONPATH=agentenv-mlebench-lite \
python3 agentenv-mlebench-lite/runtime_bridge/verify_runtime.py \
  --bundle /opt/mlebench-lite/runtime-bundle/sandbox-runner \
  --live-root "$LIVE_ROOT" \
  --gpu-uuid "$GPU_UUID" \
  --allow-live-destructive \
  --output /protected/evidence/live-admission.json
```

This gate uses no Kaggle credentials or competition data. It proves synthetic
public/private isolation, socket-family denial (including x32), exact writable
capacity, create-then-delete and concurrent close/splice byte/inode high-water,
multithreaded nonleader `execve`/failed-`execve`/`exit_group`, runner-death
cascade, persistence, one-GPU visibility under environment overrides,
timeout/reaping, freeze readback, teardown, and full execution under one shared
runtime identity in all three arms on the actual host.

A pre-host certificate is not benchmark readiness. The following gates remain
external until separately recorded: coordinator-authorized Linux/B200
admission, Kaggle access and the official 22-task checksum manifest, a one-task
matched three-arm run, and the official host-only grader.
