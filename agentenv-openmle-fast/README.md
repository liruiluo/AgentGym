# OpenMLE-fast AgentEnv

This package implements the `openmle_fast` environment boundary. A policy sees
only a fresh `/workspace` containing read-only `TASK.md`, read-only copied
public data, and files it creates. It has exactly three actions
(`shell_command`, `apply_patch`, and `submit`) and 30 total actions. Before
submitting, the policy may repeatedly inspect public data, edit and run code,
compute its own validation metric from public labelled training data, and keep
an ordinary filesystem experiment log across context compaction. The first
`submit` alone queries protected private data and is terminal; there is no
repeatable private-score action or automatic submission. A non-submit action 30
executes and then terminates with reward `-1`. Reset and every charged action
observation expose the completed action number and remaining shared budget,
including the observation retained by a context compaction.

Private answers and native scoring code belong exclusively to the separately
launched `openmle-fast-private-grader`. The public service carries only an
authenticated grader client and sends bounded submission bytes plus their
SHA-256 over a mode-`0600` Unix socket. There is no public detail route.
The authenticated broker never imports native metric code: each grade is sent
to one fresh, digest-pinned private runner worker through selected-task file
descriptors, and the worker must attest its namespaces, cgroups, no-network
policy, read-only mounts, sanitized result IPC, hard wall, and full teardown.
The frozen grader wall is five seconds end to end from socket acceptance through
authenticated receive/decode, verified staging, the at-most-four-second worker,
audit fsync, encoding, and response send. Admission is bounded at eight active
requests. The listener applies backpressure before `accept`, so a request's
five-second grader deadline starts only after an execution slot is available.

## Runtime modes

`LocalCPUExecutionBackend` exists only for CPU/unit tests and reports
`formal_eligible=false`, partial execution counters, no namespace coverage, and
no cgroup coverage. The public launcher accepts only an independently installed
runner that attests the frozen Linux namespace, cgroup-v2, seccomp, read-only
rootfs, no-network, noexec-workspace, instrumented-Python, and resource-limit
contract. Missing or inconsistent attestation fails startup.

`LocalCPUPrivateGraderBackend` likewise exists only for unit tests. It uses a
fresh sanitized subprocess and hard parent wall timeout but reports no namespace,
cgroup, or network isolation. Formal private-grader startup instead requires
`OPENMLE_FAST_PRIVATE_RUNNER` and its SHA-256; the runner's runtime digest and
complete private isolation attestation must match the frozen manifest and caps.

The formal executor request carries the remaining cumulative managed-runtime
budget separately from the ordinary shell deadline. A runner must attest that
it enforces that budget across every managed descendant; an older runner that
only accepts a per-action timeout is rejected at startup.

Both launchers require explicit environment configuration; they have no
resource defaults. `agentenv_openmle_fast.launch._LIMIT_ENVIRONMENT` is the
authoritative list of frozen v1 limit variables. Public startup additionally
requires the frozen task-manifest identity, release/role, package/archive and
episode roots, materializer/action-parser hashes, attested runner identity,
grader socket/credential, audit root, runtime commits, observation-token cap,
and checked client/grader timeout margins. Private-grader startup requires its
own manifest/hash, package/archive roots, runtime digest, socket/credential,
audit root, and explicit private caps.

The thin client also requires those identity values as explicit constructor
arguments or `OPENMLE_FAST_*` environment pins. It never treats values returned
by `/metadata` as their own trust anchor.

Every policy action is charged before parsing. The 180-second episode deadline
is absolute from reset start, so an already-expired submit cannot grade. Shell
snapshot-before, sandbox execution, snapshot-after, and receipt construction
share one 20-second monotonic deadline, further capped by the episode deadline;
the sandbox runner receives only the milliseconds still remaining. Submission
freeze, bounded read/hash, grader IPC, and response validation are likewise
capped by the remaining episode time. No runner adapter adds an unchecked grace
period.

## Local CPU tests

From the AgentGym checkout:

```bash
PYTHONDONTWRITEBYTECODE=1 \
uv run --no-project --with pytest --with fastapi --with httpx \
  --with pydantic --with uvicorn --with pandas --with numpy --with requests \
  pytest -q agentenv-openmle-fast/tests
```

These tests establish the inner mechanism and security protocol on CPU. They do
not claim the Linux exact-runtime, remote panel, GPU, or PPO acceptance gates.
