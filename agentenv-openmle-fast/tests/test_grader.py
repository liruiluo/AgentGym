from __future__ import annotations

import hashlib
import json
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_openmle_fast.deadline import DeadlineExceeded, MonotonicDeadline
from agentenv_openmle_fast.grader_client import (
    PrivateGraderClient,
    PrivateGraderClientError,
)
from agentenv_openmle_fast.grader_protocol import (
    GraderProtocolError,
    receive_frame,
    verify_authenticated_message,
)
from agentenv_openmle_fast.private_grader import (
    PrivateGraderError,
    PrivateGraderService,
)
from agentenv_openmle_fast.private_grader_runner import (
    PRIVATE_RUNNER_COMPLETION_GRACE_MS,
    PRIVATE_RUNNER_CONTRACT,
    ExternalPrivateGraderRunnerBackend,
    LocalCPUPrivateGraderBackend,
    PrivateGradeExecution,
    PrivateGradeExecutionRequest,
    PrivateGraderLimits,
    PrivateGraderRunnerError,
)
from tests.support import (
    PRIVATE_CANARY,
    PRIVATE_RUNTIME_DIGEST,
    RELEASE_REVISION,
    TASK_ID,
    GraderServiceThread,
    create_fixture,
    sha256_file,
)


class _RecordingPrivateBackend:
    def __init__(self, limits: PrivateGraderLimits) -> None:
        self.limits = limits
        self.timeout_ms: int | None = None

    @property
    def metadata(self):
        return {
            "contract": "recording-private-test-backend",
            "formal_eligible": False,
            "resource_limits": self.limits.as_dict(),
        }

    def grade(self, request, *, timeout_ms: int):
        self.timeout_ms = timeout_ms
        return PrivateGradeExecution(
            classification="infrastructure_fault",
            native_score=None,
            higher_is_better=request.higher_is_better,
        )


class _BlockingPrivateBackend(_RecordingPrivateBackend):
    def __init__(self, limits: PrivateGraderLimits) -> None:
        super().__init__(limits)
        self.entered = threading.Event()
        self.release = threading.Event()

    def grade(self, request, *, timeout_ms: int):
        self.timeout_ms = timeout_ms
        self.entered.set()
        self.release.wait(timeout_ms / 1000.0)
        return PrivateGradeExecution(
            classification="infrastructure_fault",
            native_score=None,
            higher_is_better=request.higher_is_better,
        )


class OpenMLEFastPrivateGraderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="openmle-grader-test-")
        self.root = Path(self.temporary.name)
        self.fixture = create_fixture(self.root)
        self.socket_path = self.root / "grader.sock"
        self.service = PrivateGraderService(
            private_manifest_path=Path(self.fixture["private_manifest"]),
            expected_manifest_sha256=str(self.fixture["private_manifest_sha256"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=self.socket_path,
            credential_path=Path(self.fixture["credential"]),
            audit_root=Path(self.fixture["audit_root"]),
            total_wall_ms=5_000,
            max_concurrent_requests=2,
            backend=LocalCPUPrivateGraderBackend(PrivateGraderLimits.frozen_v1()),
        )
        self.thread = GraderServiceThread(self.service)
        self.thread.__enter__()
        self.client = PrivateGraderClient(
            endpoint=self.socket_path,
            credential_path=Path(self.fixture["credential"]),
            timeout_seconds=5.0,
        )

    def tearDown(self) -> None:
        self.thread.__exit__(None, None, None)
        self.temporary.cleanup()

    def grade(self, payload: bytes):
        task = self.fixture["task"]
        assert isinstance(task, dict)
        return self.client.grade(
            request_id="request-1",
            episode_id="episode-1",
            task_id=TASK_ID,
            grader_binding_sha256=str(task["private_grader_binding_sha256"]),
            package_identity_sha256=str(task["package_identity_sha256"]),
            baseline_score=float(task["baseline_score"]),
            ideal_score=float(task["ideal_score"]),
            higher_is_better=bool(task["higher_is_better"]),
            submission=payload,
        )

    def grade_kwargs(self, request_id: str) -> dict[str, object]:
        task = self.fixture["task"]
        assert isinstance(task, dict)
        return {
            "request_id": request_id,
            "episode_id": "episode-deadline",
            "task_id": TASK_ID,
            "grader_binding_sha256": str(task["private_grader_binding_sha256"]),
            "package_identity_sha256": str(task["package_identity_sha256"]),
            "baseline_score": float(task["baseline_score"]),
            "ideal_score": float(task["ideal_score"]),
            "higher_is_better": bool(task["higher_is_better"]),
            "submission": b"id,target\n3,1\n4,2\n",
        }

    def test_baseline_oracle_and_invalid_reward_contract(self) -> None:
        baseline = self.grade(b"id,target\n3,5\n4,6\n")
        self.assertTrue(baseline.submission_valid)
        self.assertEqual(baseline.native_score, 4.0)
        self.assertEqual(baseline.normalized_reward, 0.0)
        self.assertFalse(baseline.improved_over_baseline)

        oracle = self.grade(b"id,target\n3,1\n4,2\n")
        self.assertEqual(oracle.native_score, 0.0)
        self.assertEqual(oracle.normalized_reward, 1.0)
        self.assertTrue(oracle.improved_over_baseline)

        invalid = self.grade(b"wrong,target\n3,1\n4,2\n")
        self.assertFalse(invalid.submission_valid)
        self.assertEqual(invalid.normalized_reward, -1.0)
        self.assertEqual(invalid.terminal_reason, "invalid_submission")

        serialized = json.dumps(invalid.as_dict(), sort_keys=True)
        self.assertNotIn(PRIVATE_CANARY, serialized)
        self.assertNotIn(str(self.root), serialized)
        self.assertNotIn("answer", serialized.lower())
        self.assertNotIn("metric", serialized.lower())

    def test_empty_submission_is_rejected_before_native_metric_execution(self) -> None:
        with patch.object(
            self.service.backend,
            "grade",
            side_effect=AssertionError("empty submission reached native metric worker"),
        ) as grade:
            result = self.grade(b"")
        grade.assert_not_called()
        self.assertEqual(result.classification, "invalid_submission")
        self.assertEqual(result.normalized_reward, -1.0)

    def test_authentication_fails_closed(self) -> None:
        wrong = self.root / "wrong.credential"
        wrong.write_bytes(b"x" * 32)
        wrong.chmod(0o600)
        client = PrivateGraderClient(
            endpoint=self.socket_path,
            credential_path=wrong,
            timeout_seconds=5.0,
        )
        with self.assertRaises(PrivateGraderClientError):
            task = self.fixture["task"]
            assert isinstance(task, dict)
            client.grade(
                request_id="request-2",
                episode_id="episode-1",
                task_id=TASK_ID,
                grader_binding_sha256=str(task["private_grader_binding_sha256"]),
                package_identity_sha256=str(task["package_identity_sha256"]),
                baseline_score=float(task["baseline_score"]),
                ideal_score=float(task["ideal_score"]),
                higher_is_better=bool(task["higher_is_better"]),
                submission=b"id,target\n3,1\n4,2\n",
            )

    def test_authenticated_protocol_rejects_duplicate_json_keys(self) -> None:
        with self.assertRaises(GraderProtocolError):
            verify_authenticated_message(
                b'{"schema":"a","schema":"b","payload":{},"hmac_sha256":"'
                + b"0" * 64
                + b'"}',
                b"x" * 32,
            )

    def test_protocol_receive_uses_one_absolute_deadline(self) -> None:
        receiver, sender = socket.socketpair()

        def slow_sender() -> None:
            try:
                sender.sendall(struct.pack("!I", 3) + b"a")
                time.sleep(0.07)
                sender.sendall(b"b")
                time.sleep(0.07)
                sender.sendall(b"c")
            except OSError:
                pass
            finally:
                sender.close()

        thread = threading.Thread(target=slow_sender)
        thread.start()
        started = time.monotonic()
        try:
            with self.assertRaises(DeadlineExceeded):
                receive_frame(
                    receiver,
                    deadline=MonotonicDeadline.after_ms(100),
                )
        finally:
            receiver.close()
            thread.join(1.0)
        self.assertLess(time.monotonic() - started, 0.25)

    def test_public_private_binding_mismatch_is_infrastructure_fault(self) -> None:
        task = self.fixture["task"]
        assert isinstance(task, dict)
        result = self.client.grade(
            request_id="binding-mismatch",
            episode_id="episode-1",
            task_id=TASK_ID,
            grader_binding_sha256="f" * 64,
            package_identity_sha256=str(task["package_identity_sha256"]),
            baseline_score=float(task["baseline_score"]),
            ideal_score=float(task["ideal_score"]),
            higher_is_better=bool(task["higher_is_better"]),
            submission=b"id,target\n3,1\n4,2\n",
        )
        self.assertEqual(result.classification, "infrastructure_fault")
        self.assertIsNone(result.normalized_reward)

    def test_total_deadline_is_passed_to_private_backend(self) -> None:
        limits = PrivateGraderLimits.frozen_v1()
        backend = _RecordingPrivateBackend(limits)
        socket_path = self.root / "deadline-grader.sock"
        service = PrivateGraderService(
            private_manifest_path=Path(self.fixture["private_manifest"]),
            expected_manifest_sha256=str(self.fixture["private_manifest_sha256"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=socket_path,
            credential_path=Path(self.fixture["credential"]),
            audit_root=self.root / "deadline-audit",
            total_wall_ms=500,
            max_concurrent_requests=1,
            backend=backend,
        )
        client = PrivateGraderClient(
            endpoint=socket_path,
            credential_path=Path(self.fixture["credential"]),
            timeout_seconds=1.0,
        )
        with GraderServiceThread(service):
            result = client.grade(**self.grade_kwargs("deadline-1"))
        self.assertEqual(result.classification, "infrastructure_fault")
        self.assertIsNotNone(backend.timeout_ms)
        self.assertGreater(backend.timeout_ms, 0)
        self.assertLess(backend.timeout_ms, 500)

    def test_saturated_grader_backpressures_before_socket_acceptance(self) -> None:
        limits = PrivateGraderLimits.frozen_v1()
        backend = _BlockingPrivateBackend(limits)
        socket_path = self.root / "bounded-grader.sock"
        service = PrivateGraderService(
            private_manifest_path=Path(self.fixture["private_manifest"]),
            expected_manifest_sha256=str(self.fixture["private_manifest_sha256"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=socket_path,
            credential_path=Path(self.fixture["credential"]),
            audit_root=self.root / "bounded-audit",
            total_wall_ms=500,
            max_concurrent_requests=1,
            backend=backend,
        )
        client = PrivateGraderClient(
            endpoint=socket_path,
            credential_path=Path(self.fixture["credential"]),
            timeout_seconds=1.0,
        )
        first_errors: list[BaseException] = []

        def first_grade() -> None:
            try:
                client.grade(**self.grade_kwargs("bounded-1"))
            except BaseException as exc:  # noqa: BLE001 - asserted below
                first_errors.append(exc)

        with GraderServiceThread(service):
            first = threading.Thread(target=first_grade)
            first.start()
            self.assertTrue(backend.entered.wait(1.0))
            started = time.monotonic()
            releaser = threading.Timer(0.1, backend.release.set)
            releaser.start()
            second = client.grade(**self.grade_kwargs("bounded-2"))
            elapsed = time.monotonic() - started
            releaser.join(1.0)
            first.join(1.0)
        self.assertGreaterEqual(elapsed, 0.05)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(second.classification, "infrastructure_fault")
        self.assertFalse(first.is_alive())
        self.assertEqual(first_errors, [])

    def test_external_private_runner_has_bounded_completion_grace(self) -> None:
        limits = PrivateGraderLimits.frozen_v1()
        backend = object.__new__(ExternalPrivateGraderRunnerBackend)
        backend.runner_path = self.root / "private-runner"
        backend.limits = limits
        backend.expected_runtime_digest = PRIVATE_RUNTIME_DIGEST
        metric = b"metric"
        answer = b"answer"
        request = PrivateGradeExecutionRequest(
            task_id=TASK_ID,
            grader_binding_sha256="a" * 64,
            package_identity_sha256="b" * 64,
            metric_sha256=hashlib.sha256(metric).hexdigest(),
            answer_sha256=hashlib.sha256(answer).hexdigest(),
            higher_is_better=False,
            validator_success_forms=(),
            metric=metric,
            answer=answer,
            submission=b"submission",
        )
        with patch(
            "agentenv_openmle_fast.private_grader_runner.subprocess.run",
            side_effect=subprocess.TimeoutExpired("private-runner", 0.1),
        ) as run:
            result = backend.grade(request, timeout_ms=1_000)
        self.assertLessEqual(
            run.call_args.kwargs["timeout"],
            (1_000 + PRIVATE_RUNNER_COMPLETION_GRACE_MS) / 1000.0,
        )
        payload = json.loads(run.call_args.kwargs["input"])
        self.assertLessEqual(payload["timeout_ms"], 1_000)
        self.assertEqual(result.classification, "infrastructure_fault")

    def test_workspace_metric_shadow_cannot_affect_private_import(self) -> None:
        shadow = self.root / "policy-workspace" / "utils"
        shadow.mkdir(parents=True)
        (shadow / "metric.py").write_text("raise RuntimeError('shadow')\n")
        grade = self.grade(b"id,target\n3,1\n4,2\n")
        self.assertTrue(grade.submission_valid)
        self.assertEqual(grade.normalized_reward, 1.0)

    def test_metric_worker_cannot_inherit_service_secret(self) -> None:
        source = """import os
class SecretProbeMetric:
    def __init__(self):
        self.higher_is_better = False
    def validate_submission(self, pred, truth):
        return True
    def evaluate(self, y_true, y_pred):
        return 999.0 if os.environ.get('OPENMLE_TEST_GRADER_SECRET') else 0.0
"""
        probe_root = self.root / "secret-probe"
        fixture = create_fixture(probe_root, metric_source=source)
        service = PrivateGraderService(
            private_manifest_path=Path(fixture["private_manifest"]),
            expected_manifest_sha256=str(fixture["private_manifest_sha256"]),
            package_root=Path(fixture["package_root"]),
            archive_root=Path(fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=probe_root / "grader.sock",
            credential_path=Path(fixture["credential"]),
            audit_root=Path(fixture["audit_root"]),
            total_wall_ms=5_000,
            max_concurrent_requests=2,
            backend=LocalCPUPrivateGraderBackend(PrivateGraderLimits.frozen_v1()),
        )
        client = PrivateGraderClient(
            endpoint=probe_root / "grader.sock",
            credential_path=Path(fixture["credential"]),
            timeout_seconds=5.0,
        )
        with (
            patch.dict("os.environ", {"OPENMLE_TEST_GRADER_SECRET": "canary"}),
            GraderServiceThread(service),
        ):
            result = client.grade(
                request_id="secret-probe",
                episode_id="episode-probe",
                task_id=TASK_ID,
                grader_binding_sha256=str(
                    fixture["task"]["private_grader_binding_sha256"]
                ),
                package_identity_sha256=str(fixture["task"]["package_identity_sha256"]),
                baseline_score=4.0,
                ideal_score=0.0,
                higher_is_better=False,
                submission=b"id,target\n3,1\n4,2\n",
            )
        self.assertEqual(result.native_score, 0.0)

    def test_metric_process_exit_is_contained_and_service_survives(self) -> None:
        source = """import os
class ExitMetric:
    def __init__(self):
        self.higher_is_better = False
    def validate_submission(self, pred, truth):
        return True
    def evaluate(self, y_true, y_pred):
        os._exit(17)
"""
        probe_root = self.root / "exit-probe"
        fixture = create_fixture(probe_root, metric_source=source)
        service = PrivateGraderService(
            private_manifest_path=Path(fixture["private_manifest"]),
            expected_manifest_sha256=str(fixture["private_manifest_sha256"]),
            package_root=Path(fixture["package_root"]),
            archive_root=Path(fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=probe_root / "grader.sock",
            credential_path=Path(fixture["credential"]),
            audit_root=Path(fixture["audit_root"]),
            total_wall_ms=5_000,
            max_concurrent_requests=2,
            backend=LocalCPUPrivateGraderBackend(PrivateGraderLimits.frozen_v1()),
        )
        client = PrivateGraderClient(
            endpoint=probe_root / "grader.sock",
            credential_path=Path(fixture["credential"]),
            timeout_seconds=5.0,
        )
        kwargs = {
            "episode_id": "episode-exit",
            "task_id": TASK_ID,
            "grader_binding_sha256": str(
                fixture["task"]["private_grader_binding_sha256"]
            ),
            "package_identity_sha256": str(fixture["task"]["package_identity_sha256"]),
            "baseline_score": 4.0,
            "ideal_score": 0.0,
            "higher_is_better": False,
            "submission": b"id,target\n3,1\n4,2\n",
        }
        with GraderServiceThread(service):
            first = client.grade(request_id="exit-1", **kwargs)
            second = client.grade(request_id="exit-2", **kwargs)
        self.assertEqual(first.classification, "infrastructure_fault")
        self.assertEqual(second.classification, "infrastructure_fault")

    def test_hung_metric_hits_worker_wall_and_service_survives(self) -> None:
        source = """class HungMetric:
    def __init__(self):
        self.higher_is_better = False
    def validate_submission(self, pred, truth):
        return True
    def evaluate(self, y_true, y_pred):
        while True:
            pass
"""
        probe_root = self.root / "hang-probe"
        fixture = create_fixture(probe_root, metric_source=source)
        limits = PrivateGraderLimits(
            cpu_vcpus=1,
            memory_bytes=2 * 1024**3,
            max_processes=32,
            wall_ms=200,
            input_bytes=64 * 1024**2,
        )
        service = PrivateGraderService(
            private_manifest_path=Path(fixture["private_manifest"]),
            expected_manifest_sha256=str(fixture["private_manifest_sha256"]),
            package_root=Path(fixture["package_root"]),
            archive_root=Path(fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=probe_root / "grader.sock",
            credential_path=Path(fixture["credential"]),
            audit_root=Path(fixture["audit_root"]),
            total_wall_ms=1_000,
            max_concurrent_requests=2,
            backend=LocalCPUPrivateGraderBackend(limits),
        )
        client = PrivateGraderClient(
            endpoint=probe_root / "grader.sock",
            credential_path=Path(fixture["credential"]),
            timeout_seconds=2.0,
        )
        task = fixture["task"]
        assert isinstance(task, dict)
        kwargs = {
            "episode_id": "episode-hang",
            "task_id": TASK_ID,
            "grader_binding_sha256": str(task["private_grader_binding_sha256"]),
            "package_identity_sha256": str(task["package_identity_sha256"]),
            "baseline_score": float(task["baseline_score"]),
            "ideal_score": float(task["ideal_score"]),
            "higher_is_better": bool(task["higher_is_better"]),
            "submission": b"id,target\n3,1\n4,2\n",
        }
        with GraderServiceThread(service):
            first = client.grade(request_id="hang-1", **kwargs)
            second = client.grade(request_id="hang-2", **kwargs)
        self.assertEqual(first.classification, "infrastructure_fault")
        self.assertEqual(second.classification, "infrastructure_fault")

    def test_formal_private_runner_rejects_partial_isolation_attestation(self) -> None:
        runner = self.root / "private-runner"
        runner.write_bytes(b"private-runner")
        runner.chmod(0o700)
        limits = PrivateGraderLimits.frozen_v1()
        metadata = {
            "contract": PRIVATE_RUNNER_CONTRACT,
            "runtime_digest": PRIVATE_RUNTIME_DIGEST,
            "resource_limits": limits.as_dict(),
            "formal_eligible": True,
            "fresh_worker_per_grade": True,
            "selected_task_only_mounts": True,
        }
        completed = subprocess.CompletedProcess(
            args=[str(runner), "metadata"],
            returncode=0,
            stdout=json.dumps(metadata).encode(),
            stderr=b"",
        )
        with (
            patch(
                "agentenv_openmle_fast.private_grader_runner.subprocess.run",
                return_value=completed,
            ),
            self.assertRaises(PrivateGraderRunnerError),
        ):
            ExternalPrivateGraderRunnerBackend(
                runner_path=runner,
                expected_runner_sha256=hashlib.sha256(b"private-runner").hexdigest(),
                expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
                expected_artifact_lock_sha256="a" * 64,
                limits=limits,
            )

    def test_private_manifest_binds_exact_runtime_and_all_public_manifests(
        self,
    ) -> None:
        for mutation, message in (
            (
                lambda value: value.__setitem__("runtime_digest", "sha256:" + "8" * 64),
                "runtime digest",
            ),
            (
                lambda value: value["public_manifest_sha256"].pop("heldout"),
                "public-manifest",
            ),
        ):
            with self.subTest(message=message):
                root = self.root / ("bad-" + message.replace("-", "_"))
                fixture = create_fixture(root)
                manifest = Path(fixture["private_manifest"])
                value = json.loads(manifest.read_text(encoding="utf-8"))
                mutation(value)
                manifest.write_text(
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    PrivateGraderError, message.replace("-", "[- ]?")
                ):
                    PrivateGraderService(
                        private_manifest_path=manifest,
                        expected_manifest_sha256=sha256_file(manifest),
                        package_root=Path(fixture["package_root"]),
                        archive_root=Path(fixture["archive_root"]),
                        expected_release_revision=RELEASE_REVISION,
                        expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
                        socket_path=root / "grader.sock",
                        credential_path=Path(fixture["credential"]),
                        audit_root=Path(fixture["audit_root"]),
                        total_wall_ms=5_000,
                        max_concurrent_requests=2,
                        backend=LocalCPUPrivateGraderBackend(
                            PrivateGraderLimits.frozen_v1()
                        ),
                    )

    def test_private_client_rejects_nonfinite_timeout(self) -> None:
        for timeout in (float("nan"), float("inf")):
            with (
                self.subTest(timeout=timeout),
                self.assertRaisesRegex(ValueError, "finite"),
            ):
                PrivateGraderClient(
                    endpoint=self.socket_path,
                    credential_path=Path(self.fixture["credential"]),
                    timeout_seconds=timeout,
                )

    def test_exact_private_runner_accepts_truthful_v1_and_pins_artifact_lock(
        self,
    ) -> None:
        runner = self.root / "private-runner-v1"
        runner.write_bytes(b"private-runner-v1")
        runner.chmod(0o700)
        limits = PrivateGraderLimits.frozen_v1()
        artifact_lock = "a" * 64
        true_fields = (
            "formal_eligible",
            "fresh_worker_per_grade",
            "selected_task_only_mounts",
            "submission_passed_by_fd",
            "all_task_inputs_passed_by_fd",
            "result_sanitized_ipc",
            "service_environment_hidden",
            "network_namespace",
            "network_no_egress",
            "dns_disabled",
            "metadata_service_blocked",
            "external_unix_sockets_blocked",
            "pid_namespace",
            "ipc_namespace",
            "mount_namespace",
            "fresh_unprivileged_uid_gid",
            "capabilities_dropped",
            "no_new_privs",
            "seccomp",
            "read_only_rootfs",
            "isolated_proc",
            "minimal_devices",
            "gpu_devices_absent",
            "core_dumps_disabled",
            "hard_wall_supervision",
            "descendant_kill_reap",
            "parent_death_cleanup_watchdog",
            "worker_teardown_verified",
            "validate_submission_once",
            "evaluate_once_after_validation",
        )
        metadata = {field: True for field in true_fields}
        metadata.update(
            {
                "contract": PRIVATE_RUNNER_CONTRACT,
                "runtime_digest": PRIVATE_RUNTIME_DIGEST,
                "resource_limits": limits.as_dict(),
                "cgroup_version": "v1",
                "cgroup_controller_attestation": {"version": "v1"},
                "cgroup_v1_cpu": True,
                "cgroup_v1_memory": True,
                "cgroup_v1_pids": True,
                "cgroup_v2_cpu": False,
                "cgroup_v2_memory": False,
                "cgroup_v2_pids": False,
                "active_verification": {
                    "admission_stamp_valid": True,
                    "all_checks_pass": True,
                },
                "artifact_identity": {
                    "artifact_lock_sha256": artifact_lock,
                    "artifact_lock_expected_sha256": artifact_lock,
                },
            }
        )
        completed = subprocess.CompletedProcess(
            args=[str(runner), "metadata"],
            returncode=0,
            stdout=json.dumps(metadata).encode(),
            stderr=b"",
        )
        with patch(
            "agentenv_openmle_fast.private_grader_runner.subprocess.run",
            return_value=completed,
        ):
            backend = ExternalPrivateGraderRunnerBackend(
                runner_path=runner,
                expected_runner_sha256=hashlib.sha256(b"private-runner-v1").hexdigest(),
                expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
                expected_artifact_lock_sha256=artifact_lock,
                limits=limits,
            )
        self.assertEqual(backend.metadata["cgroup_version"], "v1")


if __name__ == "__main__":
    unittest.main()
