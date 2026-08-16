from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest import mock

import runtime_bridge.runner as runner_module
from runtime_bridge.runner import (
    BridgeEngine,
    BridgeError,
    BridgeIdentity,
    ExecutionOutcome,
    LifecycleOutcome,
    MemoryStateStore,
    RuntimeAttestation,
    SealedFileStateStore,
    canonical_sha256,
    strict_json_loads,
)

RUNNER_SHA256 = "a" * 64
RUNTIME_DIGEST = "b" * 64


class FakeRuntime:
    def __init__(self) -> None:
        self.attestations: list[dict[str, object]] = []
        self.executions: list[dict[str, object]] = []
        self.freezes: list[dict[str, object]] = []
        self.teardowns: list[dict[str, object]] = []
        self.attestation = RuntimeAttestation(
            cpu_limit_cores=36,
            memory_limit_bytes=440_000_000_000,
            pids_limit=4096,
            gpu_count=1,
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
            mount_namespace=True,
            network_disabled=True,
            non_root=True,
            read_only_rootfs=True,
            runtime_identity={
                "schema": "mlebench_lite_fake_runtime_identity_v1",
                "mount_id": 41,
            },
        )
        self.next_execution = ExecutionOutcome(
            returncode=0,
            stdout="ok\n",
            stderr="",
            timed_out=False,
            execution_time_ms=11,
            cpu_time_ms=7,
            writable_bytes=13,
            writable_inodes=2,
            processes_started=3,
            descendant_process_count=0,
        )
        self.freeze_outcome = LifecycleOutcome(
            processes_reaped=True,
            workspace_frozen=True,
            mounts_released=False,
            descendant_process_count=0,
            mount_count=1,
            sandbox_present=True,
        )
        self.teardown_outcome = LifecycleOutcome(
            processes_reaped=True,
            workspace_frozen=False,
            mounts_released=True,
            descendant_process_count=0,
            mount_count=0,
            sandbox_present=False,
        )

    def attest(self, request, state=None):
        del state
        self.attestations.append(dict(request))
        return self.attestation

    def execute(self, request, state):
        del state
        self.executions.append(dict(request))
        return self.next_execution

    def freeze(self, request, state):
        del state
        self.freezes.append(dict(request))
        return self.freeze_outcome

    def teardown(self, request, state):
        del state
        self.teardowns.append(dict(request))
        return self.teardown_outcome

    def reconcile(self, operation, request, state):
        if operation == "freeze":
            return self.freeze(request, state)
        if operation == "teardown":
            return self.teardown(request, state)
        raise AssertionError(operation)


class RuntimeBridgeProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mle-runtime-bridge-")
        root = Path(self.temporary.name)
        self.public = root / "prepared" / "public"
        self.episode = root / "episodes" / ("e" * 32)
        self.workspace = self.episode / "workspace"
        self.submission = self.episode / "submission"
        self.memory = self.episode / "external-memory"
        for path in (
            self.public,
            self.workspace,
            self.submission,
            self.memory,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.contract = {
            "schema": "mlebench_lite_resource_contract_v2",
            "max_actions": 30,
            "max_submission_bytes": 100_000_000,
            "max_shell_timeout_ms": 3_600_000,
            "max_visible_output_bytes": 65_536,
            "episode_timeout_ms": 86_400_000,
            "max_total_execution_ms": 72_000_000,
            "cpu_limit_cores": 36,
            "memory_limit_bytes": 440_000_000_000,
            "pids_limit": 4096,
            "writable_bytes_limit": 500_000_000_000,
            "writable_inodes_limit": 2_000_000,
            "gpu_count": 1,
            "max_step_response_ms": 86_430_000,
            "submission_path": "/home/submission/submission.csv",
            "network_disabled": True,
            "read_only_public_data": True,
            "process_scope": "episode_cgroup_descendants",
            "cgroup_required": True,
            "isolated_process_group_required": True,
        }
        self.base = {
            "schema": "mlebench_lite_sandbox_request_v3",
            "episode_id": "e" * 32,
            "competition_id": "synthetic-task",
            "mode": "native",
            "resource_contract": self.contract,
            "resource_contract_sha256": canonical_sha256(self.contract),
            "public_root": str(self.public),
            "public_tree_sha256": "c" * 64,
            "workspace_root": str(self.workspace),
            "submission_root": str(self.submission),
        }
        self.runtime = FakeRuntime()
        self.store = MemoryStateStore()
        self.engine = BridgeEngine(
            BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST),
            self.runtime,
            self.store,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def execute_request(self, **overrides):
        return {
            **copy.deepcopy(self.base),
            "operation_id": str(uuid.uuid4()),
            "command": "python train.py",
            "timeout_ms": 10_000,
            **overrides,
        }

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self) -> None:
        with self.assertRaises(ValueError):
            strict_json_loads(b'{"a":1,"a":2}')
        for payload in (b'{"a":NaN}', b'{"a":Infinity}', b'{"a":-Infinity}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                strict_json_loads(payload)

    def test_attestation_is_exact_and_binds_mount_and_resource_identity(self) -> None:
        result = self.engine.handle("attest", copy.deepcopy(self.base))
        self.assertEqual(
            result,
            {
                "schema": "mlebench_lite_sandbox_attestation_v3",
                "runner_sha256": RUNNER_SHA256,
                "runtime_digest": RUNTIME_DIGEST,
                "resource_contract": self.contract,
                "resource_contract_sha256": canonical_sha256(self.contract),
                "mount_namespace": True,
                "network_disabled": True,
                "non_root": True,
                "read_only_rootfs": True,
                "execution_scope": {
                    "scope": "episode_cgroup_descendants",
                    "cgroup_enforced": True,
                    "isolated_process_group": True,
                },
                "external_memory_isolation": {
                    "sandbox_access": "none",
                    "native_tool_surface": "inspect_edit_shell_v1",
                    "private_root_state": "absent",
                },
                "mounts": [
                    {
                        "source": str(self.public),
                        "target": "/home/data",
                        "read_only": True,
                        "source_tree_sha256": "c" * 64,
                    },
                    {
                        "source": str(self.workspace),
                        "target": "/home/workspace",
                        "read_only": False,
                    },
                    {
                        "source": str(self.submission),
                        "target": "/home/submission",
                        "read_only": False,
                    },
                ],
                "denied_mount_prefixes": ["/host", "/private"],
            },
        )

    def test_all_arms_share_identity_and_only_memory_arm_gets_memory_mount(self) -> None:
        results = {}
        for index, mode in enumerate(
            ("native", "amg_compaction_only", "amg_memory")
        ):
            request = copy.deepcopy(self.base)
            request["episode_id"] = f"{index + 1:032x}"
            episode = self.episode.parent / request["episode_id"]
            request["workspace_root"] = str(episode / "workspace")
            request["submission_root"] = str(episode / "submission")
            request["mode"] = mode
            if mode == "amg_memory":
                request["external_memory_root"] = str(episode / "external-memory")
            results[mode] = BridgeEngine(
                BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST),
                FakeRuntime(),
                MemoryStateStore(),
            ).handle("attest", request)
        identities = {
            (item["runner_sha256"], item["runtime_digest"], item["resource_contract_sha256"])
            for item in results.values()
        }
        self.assertEqual(len(identities), 1)
        self.assertEqual(len(results["native"]["mounts"]), 3)
        self.assertEqual(len(results["amg_compaction_only"]["mounts"]), 3)
        self.assertEqual(results["amg_memory"]["mounts"][-1]["target"], "/run/amg_memory")

    def test_request_fields_resource_target_and_memory_capability_are_strict(self) -> None:
        variants = []
        missing = copy.deepcopy(self.base)
        missing.pop("public_tree_sha256")
        variants.append(missing)
        variants.append({**copy.deepcopy(self.base), "unexpected": True})
        variants.append({**copy.deepcopy(self.base), "mode": "arm11"})
        variants.append({**copy.deepcopy(self.base), "external_memory_root": str(self.memory)})
        memory_missing = {**copy.deepcopy(self.base), "mode": "amg_memory"}
        variants.append(memory_missing)
        for field, value in (
            ("cpu_limit_cores", 35),
            ("memory_limit_bytes", 439_999_999_999),
            ("pids_limit", 4095),
            ("gpu_count", 2),
        ):
            request = copy.deepcopy(self.base)
            request["resource_contract"][field] = value
            request["resource_contract_sha256"] = canonical_sha256(
                request["resource_contract"]
            )
            variants.append(request)
        for request in variants:
            with self.subTest(request=request), self.assertRaises(BridgeError):
                BridgeEngine(
                    BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST),
                    FakeRuntime(),
                    MemoryStateStore(),
                ).handle("attest", request)

    def test_paths_must_be_canonical_disjoint_and_episode_local(self) -> None:
        variants = []
        for field, value in (
            ("public_root", "relative/public"),
            ("workspace_root", str(self.episode / "x" / ".." / "workspace")),
            ("submission_root", str(self.workspace)),
            ("workspace_root", "/private/workspace"),
            ("public_root", "/host/public"),
        ):
            variants.append({**copy.deepcopy(self.base), field: value})
        for request in variants:
            with self.subTest(request=request), self.assertRaises(BridgeError):
                self.engine.handle("attest", request)

    def test_execute_response_and_resource_ledger_match_adapter_schema(self) -> None:
        attestation = self.engine.handle("attest", copy.deepcopy(self.base))
        request = self.execute_request()
        result = self.engine.handle("execute", request)
        self.assertEqual(set(result), {"returncode", "stdout", "stderr", "timed_out", "receipt"})
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"], "ok\n")
        self.assertEqual(
            result["receipt"],
            {
                "schema": "mlebench_lite_sandbox_execution_v3",
                "operation_id": request["operation_id"],
                "runner_sha256": RUNNER_SHA256,
                "runtime_digest": RUNTIME_DIGEST,
                "resource_contract_sha256": self.base["resource_contract_sha256"],
                "mount_attestation_sha256": canonical_sha256(attestation),
                "command_sha256": hashlib.sha256(b"python train.py").hexdigest(),
                "timeout_ms": 10_000,
                "returncode": 0,
                "timed_out": False,
                "resource_delta": {
                    "execution_time_ms": 11,
                    "cpu_time_ms": 7,
                    "writable_bytes": 13,
                    "writable_inodes": 2,
                    "processes_started": 3,
                },
                "resource_cumulative": {
                    "execution_time_ms": 11,
                    "cpu_time_ms": 7,
                    "writable_bytes": 13,
                    "writable_inodes": 2,
                    "processes_started": 3,
                },
                "containment": {
                    "scope": "episode_cgroup_descendants",
                    "cgroup_enforced": True,
                    "isolated_process_group": True,
                    "descendant_process_count": 0,
                },
            },
        )

    def test_operation_replay_is_exact_and_changed_payload_is_rejected(self) -> None:
        self.engine.handle("attest", copy.deepcopy(self.base))
        request = self.execute_request()
        first = self.engine.handle("execute", copy.deepcopy(request))
        second = self.engine.handle("execute", copy.deepcopy(request))
        self.assertEqual(second, first)
        self.assertEqual(len(self.runtime.executions), 1)
        changed = {**copy.deepcopy(request), "command": "python other.py"}
        with self.assertRaises(BridgeError):
            self.engine.handle("execute", changed)

    def test_execute_rejects_noncanonical_uuid_timeout_and_nul_command(self) -> None:
        self.engine.handle("attest", copy.deepcopy(self.base))
        variants = (
            self.execute_request(operation_id=uuid.uuid4().hex),
            self.execute_request(operation_id=str(uuid.uuid1())),
            self.execute_request(timeout_ms=3_600_001),
            self.execute_request(command="bad\x00command"),
        )
        for request in variants:
            with self.subTest(request=request), self.assertRaises(BridgeError):
                self.engine.handle("execute", request)

    def test_runtime_capability_drift_and_descendant_leak_fail_closed(self) -> None:
        for field, value in (
            ("gpu_count", 0),
            ("cpu_limit_cores", 2),
            ("network_disabled", False),
        ):
            runtime = FakeRuntime()
            runtime.attestation = RuntimeAttestation(
                **{
                    **runtime.attestation.__dict__,
                    field: value,
                }
            )
            with self.subTest(field=field), self.assertRaises(BridgeError):
                BridgeEngine(
                    BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST),
                    runtime,
                    MemoryStateStore(),
                ).handle("attest", copy.deepcopy(self.base))

        self.engine.handle("attest", copy.deepcopy(self.base))
        self.runtime.next_execution = ExecutionOutcome(
            **{
                **self.runtime.next_execution.__dict__,
                "descendant_process_count": 1,
            }
        )
        with self.assertRaises(BridgeError):
            self.engine.handle("execute", self.execute_request())

    def test_output_and_cumulative_resource_overflow_fail_closed(self) -> None:
        self.engine.handle("attest", copy.deepcopy(self.base))
        self.runtime.next_execution = ExecutionOutcome(
            **{
                **self.runtime.next_execution.__dict__,
                "stdout": "x" * 65_537,
            }
        )
        with self.assertRaises(BridgeError):
            self.engine.handle("execute", self.execute_request())

        runtime = FakeRuntime()
        runtime.next_execution = ExecutionOutcome(
            **{
                **runtime.next_execution.__dict__,
                "writable_bytes": 500_000_000_001,
            }
        )
        engine = BridgeEngine(
            BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST),
            runtime,
            MemoryStateStore(),
        )
        engine.handle("attest", copy.deepcopy(self.base))
        with self.assertRaises(BridgeError):
            engine.handle("execute", self.execute_request())

    def test_memory_access_receipt_is_only_valid_for_memory_arm(self) -> None:
        self.engine.handle("attest", copy.deepcopy(self.base))
        self.runtime.next_execution = ExecutionOutcome(
            **{
                **self.runtime.next_execution.__dict__,
                "external_memory_access": "read",
            }
        )
        with self.assertRaises(BridgeError):
            self.engine.handle("execute", self.execute_request())

        request = copy.deepcopy(self.base)
        request["episode_id"] = "f" * 32
        request["mode"] = "amg_memory"
        episode = self.episode.parent / request["episode_id"]
        request["workspace_root"] = str(episode / "workspace")
        request["submission_root"] = str(episode / "submission")
        request["external_memory_root"] = str(episode / "external-memory")
        runtime = FakeRuntime()
        runtime.next_execution = ExecutionOutcome(
            **{
                **runtime.next_execution.__dict__,
                "external_memory_access": "read_write",
            }
        )
        engine = BridgeEngine(
            BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST),
            runtime,
            MemoryStateStore(),
        )
        engine.handle("attest", copy.deepcopy(request))
        result = engine.handle(
            "execute",
            {
                **copy.deepcopy(request),
                "operation_id": str(uuid.uuid4()),
                "command": "python train.py",
                "timeout_ms": 10_000,
            },
        )
        self.assertEqual(
            result["receipt"]["external_memory_access"],
            {
                "schema": "amg_external_memory_access_v1",
                "operation": "read_write",
            },
        )

    def test_freeze_and_teardown_are_exact_idempotent_and_ordered(self) -> None:
        attestation = self.engine.handle("attest", copy.deepcopy(self.base))
        self.engine.handle("execute", self.execute_request())
        freeze_request = {
            **copy.deepcopy(self.base),
            "operation_id": str(uuid.uuid4()),
        }
        freeze = self.engine.handle("freeze", freeze_request)
        self.assertEqual(
            freeze,
            {
                "schema": "mlebench_lite_sandbox_freeze_v2",
                "operation_id": freeze_request["operation_id"],
                "runner_sha256": RUNNER_SHA256,
                "runtime_digest": RUNTIME_DIGEST,
                "resource_contract_sha256": self.base["resource_contract_sha256"],
                "mount_attestation_sha256": canonical_sha256(attestation),
                "resource_cumulative": {
                    "execution_time_ms": 11,
                    "cpu_time_ms": 7,
                    "writable_bytes": 13,
                    "writable_inodes": 2,
                    "processes_started": 3,
                },
                "processes_reaped": True,
                "workspace_frozen": True,
                "descendant_process_count": 0,
            },
        )
        self.assertEqual(self.engine.handle("freeze", copy.deepcopy(freeze_request)), freeze)
        with self.assertRaises(BridgeError):
            self.engine.handle("execute", self.execute_request())

        teardown_request = {
            **copy.deepcopy(self.base),
            "operation_id": str(uuid.uuid4()),
        }
        teardown = self.engine.handle("teardown", teardown_request)
        self.assertEqual(teardown["mounts_released"], True)
        self.assertEqual(teardown["mount_count"], 0)
        self.assertEqual(teardown["sandbox_present"], False)
        self.assertEqual(
            self.engine.handle("teardown", copy.deepcopy(teardown_request)), teardown
        )
        self.assertEqual(len(self.runtime.freezes), 1)
        self.assertEqual(len(self.runtime.teardowns), 1)

    def test_lifecycle_runtime_failure_never_fabricates_success(self) -> None:
        self.engine.handle("attest", copy.deepcopy(self.base))
        self.runtime.freeze_outcome = LifecycleOutcome(
            processes_reaped=False,
            workspace_frozen=False,
            mounts_released=False,
            descendant_process_count=1,
            mount_count=1,
            sandbox_present=True,
        )
        with self.assertRaises(BridgeError):
            self.engine.handle(
                "freeze",
                {**copy.deepcopy(self.base), "operation_id": str(uuid.uuid4())},
            )

    def test_file_state_store_hmac_detects_tamper_and_supports_exact_replay(self) -> None:
        state_root = Path(self.temporary.name) / "sealed-state"
        state_root.mkdir(mode=0o700)
        store = SealedFileStateStore(state_root, expected_uid=os.geteuid())
        engine = BridgeEngine(
            BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST),
            self.runtime,
            store,
        )
        with store.episode_lock(self.base["episode_id"]):
            engine.handle("attest", copy.deepcopy(self.base))
            request = self.execute_request()
            first = engine.handle("execute", copy.deepcopy(request))
        with store.episode_lock(self.base["episode_id"]):
            second = engine.handle("execute", copy.deepcopy(request))
        self.assertEqual(second, first)
        state_path = next(state_root.glob("state-*.json"))
        envelope = json.loads(state_path.read_text(encoding="utf-8"))
        envelope["body"]["lifecycle"] = "torn_down"
        state_path.write_text(json.dumps(envelope), encoding="utf-8")
        with store.episode_lock(self.base["episode_id"]), self.assertRaises(
            BridgeError
        ):
            engine.handle("execute", copy.deepcopy(request))

    def test_file_state_store_requires_owner_only_real_directory(self) -> None:
        state_root = Path(self.temporary.name) / "unsafe-state"
        state_root.mkdir(mode=0o700)
        state_root.chmod(0o770)
        with self.assertRaises(BridgeError):
            SealedFileStateStore(state_root, expected_uid=os.geteuid())

    def test_owner_key_is_atomically_published_across_concurrent_stores(self) -> None:
        state_root = Path(self.temporary.name) / "concurrent-key-state"
        state_root.mkdir(mode=0o700)
        writer_blocked = threading.Event()
        release_writer = threading.Event()
        keys: list[bytes] = []
        errors: list[BaseException] = []
        original_write_all = runner_module._write_all

        def blocking_first_writer(descriptor: int, payload: bytes) -> None:
            if threading.current_thread().name == "key-winner":
                writer_blocked.set()
                if not release_writer.wait(timeout=5):
                    raise RuntimeError("timed out waiting to publish owner key")
            original_write_all(descriptor, payload)

        def construct_store() -> None:
            try:
                store = SealedFileStateStore(state_root, expected_uid=os.geteuid())
                keys.append(store._key)
                store.close()
            except BaseException as exc:  # noqa: BLE001 - asserted by parent thread.
                errors.append(exc)

        with mock.patch.object(runner_module, "_write_all", blocking_first_writer):
            first = threading.Thread(target=construct_store, name="key-winner")
            second = threading.Thread(target=construct_store, name="key-contender")
            first.start()
            self.assertTrue(writer_blocked.wait(timeout=5))
            second.start()
            second.join(timeout=1)
            release_writer.set()
            first.join(timeout=5)
            second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], keys[1])
        self.assertEqual((state_root / "owner.key").stat().st_size, 32)

    def test_pending_attestation_is_recovered_only_with_same_runtime_identity(self) -> None:
        class LostAttestationRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def attest(self, request, state=None):
                self.calls += 1
                if self.calls == 1:
                    raise BridgeError("injected response loss after mount")
                return super().attest(request, state)

        runtime = LostAttestationRuntime()
        store = MemoryStateStore()
        engine = BridgeEngine(
            BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST), runtime, store
        )
        with self.assertRaises(BridgeError):
            engine.handle("attest", copy.deepcopy(self.base))
        pending = store.get(self.base["episode_id"])
        self.assertEqual(pending["lifecycle"], "attesting")
        self.assertIsNone(pending["runtime_identity"])
        result = engine.handle("attest", copy.deepcopy(self.base))
        self.assertEqual(result["schema"], "mlebench_lite_sandbox_attestation_v3")
        self.assertEqual(store.get(self.base["episode_id"])["lifecycle"], "active")
        self.assertEqual(runtime.calls, 2)

    def test_indeterminate_execute_is_never_reexecuted_or_bypassed(self) -> None:
        class LostExecuteRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def execute(self, request, state):
                del request, state
                self.calls += 1
                raise BridgeError("injected response loss after command")

        runtime = LostExecuteRuntime()
        store = MemoryStateStore()
        engine = BridgeEngine(
            BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST), runtime, store
        )
        engine.handle("attest", copy.deepcopy(self.base))
        request = self.execute_request()
        with self.assertRaises(BridgeError):
            engine.handle("execute", copy.deepcopy(request))
        with self.assertRaises(BridgeError):
            engine.handle("execute", copy.deepcopy(request))
        with self.assertRaises(BridgeError):
            engine.handle("execute", self.execute_request())
        self.assertEqual(runtime.calls, 1)
        pending = store.get(self.base["episode_id"])["operations"]
        self.assertEqual(pending[request["operation_id"]]["status"], "pending")

    def test_pending_lifecycle_reconciles_from_runtime_facts_without_replay(self) -> None:
        class LostFreezeRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.freeze_calls = 0
                self.reconcile_calls = 0

            def freeze(self, request, state):
                del request, state
                self.freeze_calls += 1
                raise BridgeError("injected response loss after remount")

            def reconcile(self, operation, request, state):
                del request, state
                self.reconcile_calls += 1
                if operation != "freeze":
                    raise AssertionError(operation)
                return self.freeze_outcome

        runtime = LostFreezeRuntime()
        store = MemoryStateStore()
        engine = BridgeEngine(
            BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST), runtime, store
        )
        engine.handle("attest", copy.deepcopy(self.base))
        request = {**copy.deepcopy(self.base), "operation_id": str(uuid.uuid4())}
        with self.assertRaises(BridgeError):
            engine.handle("freeze", copy.deepcopy(request))
        result = engine.handle("freeze", copy.deepcopy(request))
        self.assertTrue(result["workspace_frozen"])
        self.assertEqual(runtime.freeze_calls, 1)
        self.assertEqual(runtime.reconcile_calls, 1)
        self.assertEqual(
            store.get(self.base["episode_id"])["operations"][
                request["operation_id"]
            ]["status"],
            "final",
        )

    def test_state_is_bound_to_bundle_identity_before_runtime_side_effects(self) -> None:
        self.engine.handle("attest", copy.deepcopy(self.base))
        state = self.store.get(self.base["episode_id"])
        self.assertEqual(
            state["bundle_identity_sha256"],
            canonical_sha256(
                {
                    "runner_sha256": RUNNER_SHA256,
                    "runtime_digest": RUNTIME_DIGEST,
                }
            ),
        )

        foreign_runtime = FakeRuntime()
        foreign = BridgeEngine(
            BridgeIdentity("d" * 64, "f" * 64),
            foreign_runtime,
            self.store,
        )
        with self.assertRaises(BridgeError):
            foreign.handle("execute", self.execute_request())
        self.assertEqual(foreign_runtime.executions, [])

    def test_attesting_episode_can_be_torn_down_exactly(self) -> None:
        class LostAttestationRuntime(FakeRuntime):
            def attest(self, request, state=None):
                del request, state
                raise BridgeError("injected crash after attest side effect")

        runtime = LostAttestationRuntime()
        store = MemoryStateStore()
        engine = BridgeEngine(
            BridgeIdentity(RUNNER_SHA256, RUNTIME_DIGEST), runtime, store
        )
        with self.assertRaises(BridgeError):
            engine.handle("attest", copy.deepcopy(self.base))
        self.assertEqual(store.get(self.base["episode_id"])["lifecycle"], "attesting")

        request = {**copy.deepcopy(self.base), "operation_id": str(uuid.uuid4())}
        response = engine.handle("teardown", request)
        self.assertTrue(response["mounts_released"])
        self.assertEqual(response["mount_count"], 0)
        self.assertEqual(store.get(self.base["episode_id"])["lifecycle"], "torn_down")

    def test_fresh_teardown_id_reconciles_verified_tombstone(self) -> None:
        self.engine.handle("attest", copy.deepcopy(self.base))
        first_request = {
            **copy.deepcopy(self.base),
            "operation_id": str(uuid.uuid4()),
        }
        self.engine.handle("teardown", first_request)
        second_request = {
            **copy.deepcopy(self.base),
            "operation_id": str(uuid.uuid4()),
        }
        second = self.engine.handle("teardown", second_request)
        self.assertEqual(second["operation_id"], second_request["operation_id"])
        self.assertEqual(second["mount_count"], 0)
        self.assertEqual(
            self.store.get(self.base["episode_id"])["operations"][
                second_request["operation_id"]
            ]["status"],
            "final",
        )

    def test_writable_usage_is_an_absolute_monotonic_high_water(self) -> None:
        self.engine.handle("attest", copy.deepcopy(self.base))
        self.runtime.next_execution = ExecutionOutcome(
            **{
                **self.runtime.next_execution.__dict__,
                "writable_bytes": 100,
                "writable_inodes": 10,
            }
        )
        first = self.engine.handle("execute", self.execute_request())
        self.assertEqual(first["receipt"]["resource_delta"]["writable_bytes"], 100)
        self.assertEqual(first["receipt"]["resource_cumulative"]["writable_inodes"], 10)

        self.runtime.next_execution = ExecutionOutcome(
            **{
                **self.runtime.next_execution.__dict__,
                "writable_bytes": 60,
                "writable_inodes": 5,
            }
        )
        second = self.engine.handle("execute", self.execute_request())
        self.assertEqual(second["receipt"]["resource_delta"]["writable_bytes"], 0)
        self.assertEqual(second["receipt"]["resource_delta"]["writable_inodes"], 0)
        self.assertEqual(second["receipt"]["resource_cumulative"]["writable_bytes"], 100)
        self.assertEqual(second["receipt"]["resource_cumulative"]["writable_inodes"], 10)


if __name__ == "__main__":
    unittest.main()
