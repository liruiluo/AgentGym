# MLE-bench Lite Adapter Implementation Plan

> **For agentic workers:** Execute the checked contract inline with regression
> tests before implementation. Keep the work uncommitted for root review.

**Goal:** Build a fail-closed paired MLE-bench Lite adapter whose lifecycle,
resource, replay, filesystem, and host-handoff evidence is independently
verifiable.

**Architecture:** The environment wrapper owns every benchmark lifecycle and
action transition. A SHA-pinned external runner owns sandbox/cgroup execution;
the shared policy rollout remains unchanged and consumes only ordinary
`env.step` results plus task-neutral context-transition receipts.

**Tech Stack:** Python 3.11, FastAPI/Pydantic, an external JSON sandbox protocol,
pytest, and AgentGym's existing task-neutral client contract.

---

## Goal and boundaries

Implement a paired `native` / `amg_memory` AgentGym environment for the
official MLE-bench Lite split without changing the shared policy rollout.
Only `agentenv-mlebench-lite/**` and
`agentenv/agentenv/envs/mlebench_lite.py` may change.

The implementation is a local adapter and isolation contract. It does not
download Kaggle data, run a GPU workload, invoke an official private grader,
or claim that the current host is ready for a formal evaluation.

## Frozen upstream identity

- MLE-bench commit:
  `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`
- Split: `experiments/splits/low.txt`
- Raw split SHA-256:
  `590270f007fa96b4060f59f3861500159c73ca50f7f30ff6bd38303c236c799b`
- Membership: the exact ordered 22 competition IDs recorded in
  `mlebench-lite-audit.md`
- Policy submission path: `/home/submission/submission.csv`

Startup must load these values from an external pinned checkout and fail
closed on any commit, hash, order, count, duplicate, path, or symlink drift.

## Ownership matrix

| Concern | Owner |
| --- | --- |
| Policy sampling, tokens, log-probabilities, PPO packing | Existing shared task-neutral rollout; zero diff |
| Action parsing and total action budget | MLE-bench Lite server environment |
| Context-pressure decision and compaction request | MLE-bench Lite AgentGym client |
| Context replacement | Existing task-neutral `replace_messages` receipt |
| Public-data mount and writable workspace mounts | Attested external sandbox runner |
| Private labels and official grade | Later host workflow, outside the adapter and every policy mount |
| Policy-visible terminal receipt | MLE-bench Lite server environment |

Compaction is an ordinary sampled policy turn. The client dispatches it to
the server before emitting `replace_messages`, so it consumes the same total
action budget as inspect, edit, shell, parser errors, and submit.

## Files and interfaces

- `agentenv_mlebench_lite/identity.py`
  - frozen constants and `load_official_lite_identity(upstream_root)`
- `agentenv_mlebench_lite/dataset.py`
  - public/private root validation, task manifest loading, public-only records
- `agentenv_mlebench_lite/workspace.py`
  - unique reset workspace, virtual policy paths, memory namespace isolation,
    and protected host-only submission handoff
- `agentenv_mlebench_lite/actions.py`
  - one exact parser shared by both modes
- `agentenv_mlebench_lite/executor.py`
  - sandbox-runner attestation protocol; formal construction rejects local
    `subprocess(cwd=workspace)` execution and incomplete mount isolation
- `agentenv_mlebench_lite/environment.py`
  - reset/step lifecycle, single action budget, grading boundary, allowlisted
    terminal receipt
- `agentenv_mlebench_lite/server.py`
  - small FastAPI transport around the manager
- `agentenv_mlebench_lite/launch.py`
  - fail-closed configuration and server launch
- `agentenv/agentenv/envs/mlebench_lite.py`
  - AgentGym client, native/memory framing, pressure-triggered compaction, and
    task-neutral transition receipts

Public data is never copied into an episode. The loader consumes the official
cache layout `<data>/<competition>/prepared/public` directly and validates its
private sibling without retaining that private path in a policy record. A
reset creates only a small writable workspace and submission directory. The
formal runner must attest that the current task's public source is mounted
read-only at `/home/data`,
the episode workspace and submission directory are the only writable mounts,
the root filesystem is read-only, networking is disabled, the process is
non-root, and no private/host/task-crossing mount exists. Missing or unequal
attestation is an infrastructure error, never a local-execution fallback.

`submit` performs only a public structural handoff: it requires the official
path to be a non-symlink regular file and hashes its bytes. It never calls a
grader, validates competition-specific contents, or derives reward from
private state. Official grading happens later in a host-only workflow.

## Red/green sequence

1. Add identity and dataset tests for commit/hash/order/count and root/symlink
   rejection; run them to observe import/contract failures.
2. Add workspace and executor tests for reset isolation, virtual-path escape
   denial, read-only public mounts, and incomplete sandbox attestation.
3. Add action/environment tests for the shared parser, all charged action
   kinds, identical non-memory dispatch, private negative probes, submission
   isolation, and receipt allowlisting.
4. Add client tests for native memory absence and
   write -> compaction -> later read with `replace_messages`. The server must
   atomically return action count `n + 1` plus zero execution/native/grading
   deltas for compaction; the client emits replacement only after validating
   that receipt. A budget-terminal compaction must not replace context.
5. Implement the smallest modules that make each group pass.

## Hardening closure (root review, 2026-08-15)

- [x] Add staged-fault reset tests proving partial directory rollback, supervised
  cleanup after preflight failure, and retryable reset after cleanup failure.
- [x] Cache the validated preflight attestation digest. Bind resource digests
  and retry-stable operation UUIDs into execute/freeze/teardown receipts; prove
  descendant reap, mount release, and sandbox absence with negative probes.
- [x] Classify runner, receipt, storage, and cleanup failures as terminal
  infrastructure outcomes. Return one generic zero-reward terminal step, never
  a normal execution observation or a context replacement.
- [x] Give every `/step` payload a strict UUID action ID. Cache its exact payload
  and public result/error server-side; keep the ID pending client-side until the
  full response validates so a lost submit response can be replayed exactly.
- [x] Stage submissions outside policy mounts, then freeze, read/hash, teardown,
  remove, and atomically publish a sealed owner-only directory. Bind episode,
  mode, task, runner/runtime/resource digests, and lifecycle receipts in the
  protected manifest; reopen no-follow and rehash on host lookup.
- [x] Pin the complete identical resource contract for both arms: episode wall
  deadline, CPU, RAM, PIDs, writable bytes/inodes, GPUs, total execution time,
  descendant/cgroup scope, and a shell timeout configurable up to the episode
  budget. Validate cumulative resource counters as exact prior plus delta.
- [x] Require cgroup/PGID containment and zero descendants after every execution.
  Replace host inspect/edit path handling with dirfd/openat no-follow traversal
  and adversarial symlink-parent/race tests.
- [x] Reject `data_len=21`; only absent or exact 22 is valid. Keep the successful
  policy terminal receipt exactly `competition_id`, `submission_path`, and
  `submission_sha256`, while the host manifest remains richer.
- [x] Reject unsafe group/world-writable or foreign-owned episode/handoff roots;
  keep staging/final handoffs outside policy reach and owner-only throughout.

## Verification

Run from the repository root:

```bash
set -euo pipefail
python -m pytest -q agentenv-mlebench-lite/tests
python -m compileall -q \
  agentenv-mlebench-lite/agentenv_mlebench_lite \
  agentenv/agentenv/envs/mlebench_lite.py
git diff --check
git status --short
git diff --name-only 40f2accb3a6a1f5b4b300361cad3abc081e177ba --
test "$(git diff --name-only 40f2accb3a6a1f5b4b300361cad3abc081e177ba -- | wc -l | tr -d ' ')" -eq 0
git ls-files --others --exclude-standard
test "$(git ls-files --others --exclude-standard | wc -l | tr -d ' ')" -eq 24
diff -u \
  <(printf '%s\n' \
    agentenv-mlebench-lite/.gitignore \
    agentenv-mlebench-lite/README.md \
    agentenv-mlebench-lite/agentenv_mlebench_lite/__init__.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/actions.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/config.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/dataset.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/environment.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/executor.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/identity.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/launch.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/resources.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/server.py \
    agentenv-mlebench-lite/agentenv_mlebench_lite/workspace.py \
    agentenv-mlebench-lite/docs/plans/2026-08-15-mlebench-lite-adapter.md \
    agentenv-mlebench-lite/pyproject.toml \
    agentenv-mlebench-lite/tests/__init__.py \
    agentenv-mlebench-lite/tests/support.py \
    agentenv-mlebench-lite/tests/test_actions_environment.py \
    agentenv-mlebench-lite/tests/test_client.py \
    agentenv-mlebench-lite/tests/test_config.py \
    agentenv-mlebench-lite/tests/test_identity_dataset.py \
    agentenv-mlebench-lite/tests/test_server.py \
    agentenv-mlebench-lite/tests/test_workspace_executor.py \
    agentenv/agentenv/envs/mlebench_lite.py) \
  <(git ls-files --others --exclude-standard | LC_ALL=C sort)
while IFS= read -r -d '' untracked_file; do
  if whitespace_output=$(git diff --no-index --check -- /dev/null "$untracked_file" 2>&1); then
    whitespace_rc=0
  else
    whitespace_rc=$?
  fi
  test "$whitespace_rc" -eq 1 && test -z "$whitespace_output" || {
    printf '%s\n' "$whitespace_output" >&2
    exit 1
  }
done < <(git ls-files --others --exclude-standard -z)
git diff --exit-code 40f2accb3a6a1f5b4b300361cad3abc081e177ba -- \
  ':!agentenv-mlebench-lite/**' \
  ':!agentenv/agentenv/envs/mlebench_lite.py'
```

The count and exact-list guards pin this pre-commit delivery to zero tracked
changes and the 24 expected untracked files. The no-index loop extends the
whitespace check to those untracked files, which `git diff --check` does not
inspect. The final command is the baseline changed-path guard: this task must
leave all shared rollout/controller code unchanged.

## Future integration step

After local verification, the root task owner reviews this uncommitted diff,
requests any repairs, and performs the final commit. Formal host integration
then installs a real attested sandbox runner, attaches protected prepared data,
and runs grading only after the policy container has terminated.
