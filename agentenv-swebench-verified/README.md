# AgentEnv SWE-bench Verified

This package is a thin policy-generation adapter for the frozen full-500
SWE-bench Verified protocol. It does not grade patches. It exposes a persistent
`/testbed` workspace through the existing Codex-style `shell_command` and
`apply_patch` action grammar, then exports one exact-base solution diff for the
separate official v4.1.0 harness.

## Frozen boundary

The production loader fails closed unless all of these identities match:

- dataset `princeton-nlp/SWE-bench_Verified` at
  `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`;
- 500 sorted unique test rows;
- canonical JSONL SHA-256
  `392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb`;
- instance-ID ledger SHA-256
  `a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9`;
- harness `SWE-bench/SWE-bench` tag `v4.1.0`, commit
  `726c5461e2ef52d83cf1ea2107870a8bb3328d57`;
- the exact 500-tag `swebench` Linux/x86_64 image ledger, SHA-256
  `b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a`.

Complete dataset rows stay server-private for pinned `make_test_spec` binding.
The policy projection contains exactly `instance_id`, `repo`, `base_commit`, and
`problem_statement`. Gold/test patches, F2P/P2P, hints, eval scripts, parsers,
and grader evidence never enter an HTTP response or policy workspace.

## External inputs

No dataset, harness, repository, image, or OCI rootfs is downloaded by this
package. The dataset manifest is external JSON:

```json
{
  "schema_version": "swebench_verified_frozen_jsonl_manifest_v1",
  "dataset": {
    "repository": "princeton-nlp/SWE-bench_Verified",
    "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    "split": "test"
  },
  "canonical_jsonl": {
    "path": "/absolute/path/swebench-verified-test-500.jsonl",
    "sha256": "392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb",
    "rows": 500,
    "id_ledger_sha256": "a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9"
  }
}
```

The image input is the audit-defined sorted TSV, one
`instance_image_key<TAB>sha256:digest` row per task. Tags must be unique; distinct
tags may share a digest. Repository mirrors use `owner__repo` directory names
under the mirrors root and must contain every exact `base_commit`.

The OCI cache uses the existing `agentenv_swesmith` digest-pinned rootfs format.
Each selected official instance image must already be materialized there. The
policy sees only the isolated archived base tree bound at `/testbed`; private Git
metadata and export indexes remain outside that mount.

## Launch

Install or place these sibling packages on `PYTHONPATH`:
`agentenv-swebench-verified`, `agentenv-swesmith`, and
`agentenv-agentmemory`. Then set:

```text
SWEBENCH_VERIFIED_DATASET_MANIFEST
SWEBENCH_VERIFIED_HARNESS_ROOT
SWEBENCH_VERIFIED_IMAGE_DIGESTS
SWEBENCH_VERIFIED_IMAGE_DIGESTS_SHA256
SWEBENCH_VERIFIED_MIRRORS_ROOT
SWEBENCH_VERIFIED_EPISODES_ROOT
SWEBENCH_VERIFIED_PREDICTIONS_ROOT
SWEBENCH_VERIFIED_OCI_CACHE_ROOT
SWEBENCH_VERIFIED_RG_BINARY
SWEBENCH_VERIFIED_RG_SHA256
```

Run `python -m agentenv_swebench_verified.launch` through the package entry
point `swebench-verified`, or call `agentenv_swebench_verified.launch()`.
Optional host/port variables are `SWEBENCH_VERIFIED_HOST` and
`SWEBENCH_VERIFIED_PORT`.

The AgentGym client is
`agentenv.envs.swebench_verified.SwebenchVerifiedEnvClient`. Pass the
same server URL, run ID, runtime-pinned `image_manifest_sha256`, a mandatory
unpredictable `run_capability`, and exactly one of `arm="native"`,
`arm="amg_compaction_only"`, or `arm="amg_memory"`. Generate 32 random bytes
for each `(arm, run_id)` namespace and reuse that bearer only for parallel
clients writing that same namespace:

```python
import secrets

from agentenv.envs.swebench_verified import SwebenchVerifiedEnvClient

run_capability = secrets.token_urlsafe(32)
client = SwebenchVerifiedEnvClient(
    "http://127.0.0.1:8000",
    arm="native",
    run_id="paired-20260815",
    run_capability=run_capability,
    image_manifest_sha256="<64 lowercase hex characters>",
)
```

The client sends the run bearer only as `Authorization: Bearer ...` on
`POST /create`; it is never placed in a URL, query string, or JSON body. The
server persists only its SHA-256 digest in the private prediction namespace, so
the ownership claim survives a server restart. A second bearer cannot claim an
existing `(arm, run_id)`. The server returns a separate per-slot bearer for all
later reset/step/export/close requests. Do not log either bearer. No
environment-registry or shared-rollout edit is included.

All three arms use the same task order, 30 unified policy-turn cap, action HTTP
path, `/testbed` shell/apply-patch surface, workspace materializer, observation
budget, runtime identity, and patch exporter. Policy-visible output is capped at
6,144 bytes with 3,072 bytes per stdout/stderr stream; shell commands have a
120,000 ms default and hard maximum; model-visible thinking is disabled. Native
has no AMG memory or
compaction candidate. `amg_compaction_only` and `amg_memory` use the exact same
compaction request, token-pressure trigger, task-neutral `replace_messages`
transition, and action accounting. Only `amg_memory` receives the durable-memory
prompt convention and a private per-task root mounted at `/run/amg_memory`
outside `/testbed`. It uses the same native `shell_command` schema as every
other arm; the sandbox emits structured read/write evidence from actual
filesystem events, not command text. Compaction-only has
no dedicated memory namespace, root, mount, endpoint, environment variable,
prompt declaration, tool schema, parser/dispatch path, memory action receipt,
private evidence store, or cleanup handle. A compaction consumes one policy turn
and no native HTTP call; horizon finalization consumes no sampled turn and never
exports an unsubmitted workspace.

The observation byte limit applies to the complete policy-visible message,
including the initial issue, shell framing, changed-path summary, patch result,
and parser errors. Oversized messages retain bounded head/tail evidence with an
explicit truncation marker.

## Predictions and grading

Only a successful, non-timeout shell command whose first stdout line is
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` exports the current workspace. Arbitrary
prose, including literal `final`, is a charged parser error. Policy horizon,
native-action cap, reset, and close never export. When the official harness
needs a row for a terminal episode without submission, the controller must call
the separate authenticated `POST /no-submission` endpoint; it writes an explicit
empty-patch row without inspecting or exporting the workspace. Assembly rejects
incomplete ledgers and writes a separate JSONL for each arm in canonical dataset
order.

The diff is produced against the exact base commit with a private Git index and
private object database. It includes modified, deleted, binary, executable-mode,
ignored-but-created, and newly created solution files. Reserved `.agent_logs`,
`.agent_receipts`, and `.agent_telemetry` roots are excluded; external memory
never enters `/testbed`. Unsupported nested Git metadata or base-gitlink replacement yields
the required explicit empty prediction row rather than an unusable gitlink patch.
Model patches are capped at 16 MiB and Git export has a five-minute timeout.
Official test patches, eval scripts, parsers, and scoring remain entirely in the
external v4.1.0 grader.

The frozen reportable contrasts are `amg_compaction_only - native` for the
compaction effect, `amg_memory - amg_compaction_only` for the incremental
external-memory effect, and `amg_memory - native` for the full AMG effect. There
is no memory-only fourth arm, so this triad does not identify a
compaction-by-memory interaction.

## Verification

```bash
PYTHONPATH=agentenv-swebench-verified:agentenv-swesmith:agentenv-agentmemory \
  python3 -m unittest discover -v \
  -s agentenv-swebench-verified/tests -p 'test_*.py'
```

Formal full-500 execution still requires a Docker-capable Linux/x86_64 host,
at least 2,000,000,000,000 free bytes on the actual Docker data root, all 500
official instance images frozen by Linux/amd64 digest and retained locally, and
the external canonical dataset/harness/repository/OCI-cache assets. These are
runtime substrate requirements, not GPU requirements; this adapter performs no
GPU launch or official grading.

Formal use also remains blocked until the inherited namespace executor is placed
on an immutable or fully content-attested extracted-rootfs store and the host
enforces aggregate cgroup-v2 memory/process limits plus filesystem byte/inode
quotas. Its current manifest/config/key-executable checks and post-command
workspace validation do not by themselves close those host-runtime guarantees.
