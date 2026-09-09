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

import agentenv_openmle_fast.private_grader as private_grader_module
from agentenv_openmle_fast.deadline import DeadlineExceeded, MonotonicDeadline
from agentenv_openmle_fast.grader_client import (
    PrivateGraderClient,
    PrivateGraderClientError,
    PrivateGraderTransportError,
)
from agentenv_openmle_fast.grader_protocol import (
    GradeRequest,
    GradeResult,
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
        self.requests: list[PrivateGradeExecutionRequest] = []

    def grade(self, request, *, timeout_ms: int):
        self.timeout_ms = timeout_ms
        self.requests.append(request)
        self.entered.set()
        self.release.wait(timeout_ms / 1000.0)
        return PrivateGradeExecution(
            classification="infrastructure_fault",
            native_score=None,
            higher_is_better=request.higher_is_better,
        )


class _SequencedPrivateBackend(_RecordingPrivateBackend):
    def __init__(
        self,
        limits: PrivateGraderLimits,
        classifications: tuple[str, ...],
    ) -> None:
        super().__init__(limits)
        self.classifications = classifications
        self.requests: list[PrivateGradeExecutionRequest] = []

    def grade(self, request, *, timeout_ms: int):
        self.timeout_ms = timeout_ms
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.classifications) - 1)
        classification = self.classifications[index]
        return PrivateGradeExecution(
            classification=classification,
            native_score=0.0 if classification == "graded" else None,
            higher_is_better=request.higher_is_better,
        )


class OpenMLEFastPrivateGraderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="openmle-grader-test-")
        self.root = Path(self.temporary.name)
        self.fixture = create_fixture(self.root)
        self.socket_path = self.root / "grader.sock"
        self._request_counter = 0
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
        self._request_counter += 1
        return self.client.grade(
            request_id=f"request-{self._request_counter}",
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

    def _grade_with_backend_sequence(
        self,
        classifications: tuple[str, ...],
        *,
        retries: int,
    ) -> tuple[GradeResult, _SequencedPrivateBackend, list[dict[str, object]]]:
        limits = PrivateGraderLimits.frozen_v1()
        backend = _SequencedPrivateBackend(limits, classifications)
        socket_path = self.root / f"retry-{len(classifications)}-{retries}.sock"
        audit_root = self.root / f"retry-audit-{len(classifications)}-{retries}"
        service = PrivateGraderService(
            private_manifest_path=Path(self.fixture["private_manifest"]),
            expected_manifest_sha256=str(self.fixture["private_manifest_sha256"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=socket_path,
            credential_path=Path(self.fixture["credential"]),
            audit_root=audit_root,
            total_wall_ms=5_000,
            max_concurrent_requests=1,
            backend=backend,
        )
        with GraderServiceThread(service):
            client = PrivateGraderClient(
                endpoint=socket_path,
                credential_path=Path(self.fixture["credential"]),
                timeout_seconds=5.0,
                max_infrastructure_retries=retries,
            )
            result = client.grade(**self.grade_kwargs("stable-retry-request"))
        audits = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in audit_root.glob("grade-*.json")
        ]
        return result, backend, audits

    def test_retry_count_rejects_negative_bool_and_non_integer_values(self) -> None:
        for value in (-1, True, 1.0, "1"):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "non-negative integer"),
            ):
                PrivateGraderClient(
                    endpoint=self.socket_path,
                    credential_path=Path(self.fixture["credential"]),
                    timeout_seconds=5.0,
                    max_infrastructure_retries=value,
                )

    def test_retryable_transport_failure_reuses_exact_request(self) -> None:
        client = PrivateGraderClient(
            endpoint=self.socket_path,
            credential_path=Path(self.fixture["credential"]),
            timeout_seconds=5.0,
            max_infrastructure_retries=1,
        )
        request = self.grade_kwargs("stable-transport-request")
        expected = GradeResult(
            request_id=str(request["request_id"]),
            episode_id=str(request["episode_id"]),
            task_id=str(request["task_id"]),
            grader_binding_sha256=str(request["grader_binding_sha256"]),
            package_identity_sha256=str(request["package_identity_sha256"]),
            baseline_score=float(request["baseline_score"]),
            ideal_score=float(request["ideal_score"]),
            submission_sha256=hashlib.sha256(request["submission"]).hexdigest(),
            submission_valid=True,
            native_score=0.0,
            higher_is_better=bool(request["higher_is_better"]),
            normalized_reward=1.0,
            improved_over_baseline=True,
            runtime_success=True,
            terminal_reason="graded_submission",
            classification="graded",
            audit_digest="0" * 64,
        )
        with patch.object(
            client,
            "_grade_once",
            side_effect=[
                PrivateGraderTransportError("transient transport fault"),
                expected,
            ],
        ) as grade_once:
            actual = client.grade(**request)

        self.assertIs(actual, expected)
        self.assertEqual(grade_once.call_count, 2)
        first_request = grade_once.call_args_list[0].args[0]
        second_request = grade_once.call_args_list[1].args[0]
        self.assertIs(first_request, second_request)

    def test_retries_share_one_absolute_operation_deadline(self) -> None:
        client = PrivateGraderClient(
            endpoint=self.socket_path,
            credential_path=Path(self.fixture["credential"]),
            timeout_seconds=5.0,
            max_infrastructure_retries=1,
        )
        kwargs = self.grade_kwargs("shared-deadline-request")
        operation_deadline = MonotonicDeadline.after_ms(5_000)
        observed_deadlines: list[MonotonicDeadline] = []
        observed_requests: list[GradeRequest] = []

        def result_for(request: GradeRequest, classification: str) -> GradeResult:
            graded = classification == "graded"
            return GradeResult(
                request_id=request.request_id,
                episode_id=request.episode_id,
                task_id=request.task_id,
                grader_binding_sha256=request.grader_binding_sha256,
                package_identity_sha256=request.package_identity_sha256,
                baseline_score=request.baseline_score,
                ideal_score=request.ideal_score,
                submission_sha256=request.submission_sha256,
                submission_valid=graded,
                native_score=0.0 if graded else None,
                higher_is_better=request.higher_is_better,
                normalized_reward=1.0 if graded else None,
                improved_over_baseline=graded,
                runtime_success=graded,
                terminal_reason=(
                    "graded_submission" if graded else "grader_infrastructure_fault"
                ),
                classification=classification,
                audit_digest="0" * 64,
            )

        def grade_once(request: GradeRequest, *, deadline: MonotonicDeadline):
            observed_requests.append(request)
            observed_deadlines.append(deadline)
            classification = "infrastructure_fault" if len(observed_requests) == 1 else "graded"
            return result_for(request, classification)

        with (
            patch.object(
                MonotonicDeadline, "after_ms", return_value=operation_deadline
            ) as make_deadline,
            patch.object(client, "_grade_once", side_effect=grade_once),
        ):
            result = client.grade(**kwargs)

        self.assertEqual(result.classification, "graded")
        make_deadline.assert_called_once()
        self.assertEqual(len(observed_deadlines), 2)
        self.assertIs(observed_deadlines[0], operation_deadline)
        self.assertIs(observed_deadlines[1], operation_deadline)
        self.assertNotEqual(
            observed_requests[0].request_id, observed_requests[1].request_id
        )
        self.assertEqual(
            observed_requests[0].submission, observed_requests[1].submission
        )

    def test_expired_shared_deadline_does_not_start_new_grader_execution(self) -> None:
        client = PrivateGraderClient(
            endpoint=self.socket_path,
            credential_path=Path(self.fixture["credential"]),
            timeout_seconds=5.0,
            max_infrastructure_retries=1,
        )
        operation_deadline = MonotonicDeadline.after_ms(5_000)
        observed_requests: list[GradeRequest] = []

        def grade_once(request: GradeRequest, *, deadline: MonotonicDeadline):
            observed_requests.append(request)
            object.__setattr__(deadline, "expires_at", time.monotonic() - 1.0)
            return GradeResult(
                request_id=request.request_id,
                episode_id=request.episode_id,
                task_id=request.task_id,
                grader_binding_sha256=request.grader_binding_sha256,
                package_identity_sha256=request.package_identity_sha256,
                baseline_score=request.baseline_score,
                ideal_score=request.ideal_score,
                submission_sha256=request.submission_sha256,
                submission_valid=False,
                native_score=None,
                higher_is_better=request.higher_is_better,
                normalized_reward=None,
                improved_over_baseline=False,
                runtime_success=False,
                terminal_reason="grader_infrastructure_fault",
                classification="infrastructure_fault",
                audit_digest="0" * 64,
            )

        with (
            patch.object(
                MonotonicDeadline, "after_ms", return_value=operation_deadline
            ) as make_deadline,
            patch.object(client, "_grade_once", side_effect=grade_once),
        ):
            result = client.grade(**self.grade_kwargs("expired-shared-deadline"))

        self.assertEqual(result.classification, "infrastructure_fault")
        make_deadline.assert_called_once()
        self.assertEqual(len(observed_requests), 1)

    def test_identity_mismatch_is_not_retried(self) -> None:
        client = PrivateGraderClient(
            endpoint=self.socket_path,
            credential_path=Path(self.fixture["credential"]),
            timeout_seconds=5.0,
            max_infrastructure_retries=1,
        )
        with patch.object(
            client,
            "_grade_once",
            side_effect=[
                PrivateGraderClientError(
                    "private grader response identity mismatch"
                ),
                AssertionError("identity mismatch was retried"),
            ],
        ) as grade_once:
            with self.assertRaisesRegex(
                PrivateGraderClientError, "response identity mismatch"
            ):
                client.grade(**self.grade_kwargs("mismatched-response"))

        self.assertEqual(grade_once.call_count, 1)

    def test_transient_infrastructure_fault_retries_same_submission(self) -> None:
        result, backend, audits = self._grade_with_backend_sequence(
            ("infrastructure_fault", "graded"),
            retries=1,
        )

        self.assertEqual(result.classification, "graded")
        self.assertEqual(len(backend.requests), 2)
        self.assertNotEqual(
            backend.requests[0].request_id,
            backend.requests[1].request_id,
        )
        self.assertEqual(
            backend.requests[0].episode_id,
            backend.requests[1].episode_id,
        )
        self.assertEqual(
            backend.requests[0].submission,
            backend.requests[1].submission,
        )
        self.assertEqual(
            len({record["request_id"] for record in audits}),
            2,
        )
        self.assertEqual(
            len({record["submission_sha256"] for record in audits}),
            1,
        )
        self.assertEqual(
            {record["classification"] for record in audits},
            {"infrastructure_fault", "graded"},
        )

    def test_response_loss_replays_result_without_duplicate_backend_execution(self) -> None:
        limits = PrivateGraderLimits.frozen_v1()
        backend = _SequencedPrivateBackend(limits, ("graded",))
        socket_path = self.root / "response-loss-replay.sock"
        audit_root = self.root / "response-loss-replay-audit"
        service = PrivateGraderService(
            private_manifest_path=Path(self.fixture["private_manifest"]),
            expected_manifest_sha256=str(self.fixture["private_manifest_sha256"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=socket_path,
            credential_path=Path(self.fixture["credential"]),
            audit_root=audit_root,
            total_wall_ms=5_000,
            max_concurrent_requests=2,
            backend=backend,
        )
        real_send_frame = private_grader_module.send_frame
        send_calls = 0

        def drop_first_response(*args, **kwargs):
            nonlocal send_calls
            send_calls += 1
            if send_calls == 1:
                raise OSError("simulated response loss")
            return real_send_frame(*args, **kwargs)

        with (
            GraderServiceThread(service),
            patch.object(
                private_grader_module,
                "send_frame",
                side_effect=drop_first_response,
            ),
        ):
            client = PrivateGraderClient(
                endpoint=socket_path,
                credential_path=Path(self.fixture["credential"]),
                timeout_seconds=5.0,
                max_infrastructure_retries=1,
            )
            result = client.grade(**self.grade_kwargs("response-loss-request"))

        self.assertEqual(result.classification, "graded")
        self.assertEqual(send_calls, 2)
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(len(list(audit_root.glob("grade-*.json"))), 1)

    def test_request_id_reuse_with_different_content_fails_before_backend(self) -> None:
        limits = PrivateGraderLimits.frozen_v1()
        backend = _SequencedPrivateBackend(limits, ("graded",))
        service = PrivateGraderService(
            private_manifest_path=Path(self.fixture["private_manifest"]),
            expected_manifest_sha256=str(self.fixture["private_manifest_sha256"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=self.root / "identity-conflict.sock",
            credential_path=Path(self.fixture["credential"]),
            audit_root=self.root / "identity-conflict-audit",
            total_wall_ms=5_000,
            max_concurrent_requests=2,
            backend=backend,
        )
        kwargs = self.grade_kwargs("conflicting-request")
        first = GradeRequest.build(**kwargs)
        second = GradeRequest.build(**{**kwargs, "submission": b"different"})
        service._grade_or_replay(first, MonotonicDeadline.after_ms(5_000))
        with self.assertRaisesRegex(PrivateGraderError, "different content"):
            service._grade_or_replay(second, MonotonicDeadline.after_ms(5_000))
        self.assertEqual(len(backend.requests), 1)

    def test_concurrent_identical_request_is_single_flight(self) -> None:
        limits = PrivateGraderLimits.frozen_v1()
        backend = _BlockingPrivateBackend(limits)
        service = PrivateGraderService(
            private_manifest_path=Path(self.fixture["private_manifest"]),
            expected_manifest_sha256=str(self.fixture["private_manifest_sha256"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_release_revision=RELEASE_REVISION,
            expected_runtime_digest=PRIVATE_RUNTIME_DIGEST,
            socket_path=self.root / "single-flight.sock",
            credential_path=Path(self.fixture["credential"]),
            audit_root=self.root / "single-flight-audit",
            total_wall_ms=5_000,
            max_concurrent_requests=2,
            backend=backend,
        )
        request = GradeRequest.build(**self.grade_kwargs("single-flight-request"))
        results: list[GradeResult] = []
        errors: list[BaseException] = []

        def run() -> None:
            try:
                results.append(
                    service._grade_or_replay(
                        request,
                        MonotonicDeadline.after_ms(5_000),
                    )
                )
            except BaseException as exc:  # pragma: no cover - assertion below
                errors.append(exc)

        first = threading.Thread(target=run)
        second = threading.Thread(target=run)
        first.start()
        self.assertTrue(backend.entered.wait(1.0))
        second.start()
        time.sleep(0.05)
        backend.release.set()
        first.join(2.0)
        second.join(2.0)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(
            len(list((self.root / "single-flight-audit").glob("grade-*.json"))),
            1,
        )

    def test_persistent_infrastructure_fault_remains_bounded_and_fail_closed(self) -> None:
        result, backend, audits = self._grade_with_backend_sequence(
            ("infrastructure_fault", "infrastructure_fault", "graded"),
            retries=1,
        )

        self.assertEqual(result.classification, "infrastructure_fault")
        self.assertEqual(len(backend.requests), 2)
        self.assertNotEqual(
            backend.requests[0].request_id, backend.requests[1].request_id
        )
        self.assertEqual(
            backend.requests[0].episode_id, backend.requests[1].episode_id
        )
        self.assertEqual(
            backend.requests[0].submission, backend.requests[1].submission
        )
        self.assertEqual(len(audits), 2)
        self.assertEqual(
            {record["classification"] for record in audits},
            {"infrastructure_fault"},
        )

    def test_invalid_submission_is_not_retried(self) -> None:
        result, backend, audits = self._grade_with_backend_sequence(
            ("invalid_submission", "graded"),
            retries=1,
        )

        self.assertEqual(result.classification, "invalid_submission")
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(len(audits), 1)

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
        backend.fault_audit_root = None
        backend.process_owner = None
        backend.run_id = None
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

    def test_exact_private_runner_receives_bounded_formal_fault_context(self) -> None:
        runner = self.root / "private-runner-fault-context"
        runner.write_bytes(b"private-runner-fault-context")
        runner.chmod(0o700)
        audit_root = self.root / "formal-private-audit"
        audit_root.mkdir(mode=0o700)
        backend = object.__new__(ExternalPrivateGraderRunnerBackend)
        backend.runner_path = runner
        backend.limits = PrivateGraderLimits.frozen_v1()
        backend.fault_audit_root = audit_root
        backend.process_owner = "joint-r125-owner"
        backend.run_id = "joint-r125-run"
        request = PrivateGradeExecutionRequest(
            task_id=TASK_ID,
            grader_binding_sha256="1" * 64,
            package_identity_sha256="2" * 64,
            metric_sha256=hashlib.sha256(b"metric").hexdigest(),
            answer_sha256=hashlib.sha256(b"answer").hexdigest(),
            higher_is_better=False,
            validator_success_forms=(),
            metric=b"metric",
            answer=b"answer",
            submission=b"submission",
            request_id="request-with-private-context",
            episode_id="episode-with-private-context",
        )
        completed = subprocess.CompletedProcess(
            args=[str(runner), "grade"],
            returncode=0,
            stdout=json.dumps(
                {
                    "schema": "openmle_fast_private_worker_result_v1",
                    "classification": "infrastructure_fault",
                    "native_score": None,
                    "higher_is_better": False,
                }
            ).encode(),
            stderr=b"",
        )
        with patch(
            "agentenv_openmle_fast.private_grader_runner.subprocess.run",
            return_value=completed,
        ) as run:
            result = backend.grade(request, timeout_ms=1_000)

        self.assertEqual(result.classification, "infrastructure_fault")
        environment = run.call_args.kwargs["env"]
        self.assertEqual(
            environment["OPENMLE_FAST_PRIVATE_FAULT_AUDIT_ROOT"], str(audit_root)
        )
        self.assertEqual(environment["OPENMLE_FAST_PROCESS_OWNER"], "joint-r125-owner")
        self.assertEqual(environment["OPENMLE_FAST_RUN_ID"], "joint-r125-run")
        self.assertEqual(
            environment["OPENMLE_FAST_PRIVATE_REQUEST_ID_SHA256"],
            hashlib.sha256(b"request-with-private-context").hexdigest(),
        )
        self.assertEqual(
            environment["OPENMLE_FAST_PRIVATE_EPISODE_ID_SHA256"],
            hashlib.sha256(b"episode-with-private-context").hexdigest(),
        )
        self.assertNotIn("request-with-private-context", environment.values())
        self.assertNotIn("episode-with-private-context", environment.values())

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
