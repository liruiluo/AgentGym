from __future__ import annotations

import hashlib
import math
import os
import socket
import stat
from pathlib import Path

from .deadline import MonotonicDeadline
from .grader_protocol import (
    GradeRequest,
    GradeResult,
    GraderProtocolError,
    authenticated_message,
    receive_frame,
    send_frame,
    verify_authenticated_message,
)


class PrivateGraderClientError(RuntimeError):
    pass


class PrivateGraderTransportError(PrivateGraderClientError):
    pass


def _retry_request_id(request_id: str, attempt_index: int) -> str:
    """Derive a stable ID for a new infrastructure retry attempt.

    Transport retries keep the same request object so the grader can replay a
    completed result.  A returned infrastructure fault, on the other hand,
    authorizes one new grader execution and therefore needs a distinct request
    ID while retaining the same episode and submission identity.
    """

    material = f"{request_id}\0{attempt_index}".encode("utf-8")
    return f"retry-{hashlib.sha256(material).hexdigest()}"


class PrivateGraderClient:
    """The environment-side holder of the authenticated private IPC boundary."""

    def __init__(
        self,
        *,
        endpoint: Path | str,
        credential_path: Path | str,
        timeout_seconds: float,
        max_infrastructure_retries: int = 0,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("private-grader timeout must be finite and positive")
        if (
            isinstance(max_infrastructure_retries, bool)
            or not isinstance(max_infrastructure_retries, int)
            or max_infrastructure_retries < 0
        ):
            raise ValueError(
                "private-grader infrastructure retries must be a non-negative integer"
            )
        self.endpoint = Path(endpoint).expanduser().absolute()
        self.credential = read_credential(Path(credential_path))
        self.timeout_seconds = float(timeout_seconds)
        self.max_infrastructure_retries = max_infrastructure_retries

    def grade(
        self,
        *,
        request_id: str,
        episode_id: str,
        task_id: str,
        grader_binding_sha256: str,
        package_identity_sha256: str,
        baseline_score: float,
        ideal_score: float,
        higher_is_better: bool,
        submission: bytes,
        deadline: MonotonicDeadline | None = None,
    ) -> GradeResult:
        operation_deadline = MonotonicDeadline.after_ms(
            max(1, math.floor(self.timeout_seconds * 1000.0)),
            cap=deadline,
        )
        operation_deadline.check()

        def build_request(attempt_index: int) -> GradeRequest:
            attempt_request_id = (
                request_id
                if attempt_index == 0
                else _retry_request_id(request_id, attempt_index)
            )
            return GradeRequest.build(
                request_id=attempt_request_id,
                episode_id=episode_id,
                task_id=task_id,
                grader_binding_sha256=grader_binding_sha256,
                package_identity_sha256=package_identity_sha256,
                baseline_score=baseline_score,
                ideal_score=ideal_score,
                higher_is_better=higher_is_better,
                submission=submission,
                deadline=operation_deadline,
            )

        request = build_request(0)
        retries_used = 0
        while True:
            try:
                result = self._grade_once(request, deadline=operation_deadline)
            except PrivateGraderTransportError as exc:
                if retries_used >= self.max_infrastructure_retries:
                    raise
                retries_used += 1
                if operation_deadline.expired():
                    raise exc
                # The service may have completed the request before the
                # response path failed.  Reuse the exact request ID so its
                # single-flight cache can replay rather than execute twice.
                continue
            if result.classification != "infrastructure_fault":
                return result
            if retries_used >= self.max_infrastructure_retries:
                return result
            retries_used += 1
            if operation_deadline.expired():
                return result
            # A delivered infrastructure result proves that this attempt
            # completed.  Use a new, deterministic attempt ID to authorize one
            # new backend execution without resampling the policy trajectory.
            request = build_request(retries_used)

    def _grade_once(
        self,
        request: GradeRequest,
        *,
        deadline: MonotonicDeadline,
    ) -> GradeResult:
        try:
            deadline.check()
            info = os.stat(self.endpoint, follow_symlinks=False)
            if not stat.S_ISSOCK(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                raise PrivateGraderClientError(
                    "private-grader endpoint is not a private Unix socket"
                )
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(deadline.remaining_seconds())
                connection.connect(str(self.endpoint))
                deadline.check()
                message = authenticated_message(
                    request.payload(deadline=deadline),
                    self.credential,
                    deadline=deadline,
                )
                deadline.check()
                connection.settimeout(deadline.remaining_seconds())
                send_frame(
                    connection,
                    message,
                    deadline=deadline,
                )
                deadline.check()
                connection.settimeout(deadline.remaining_seconds())
                response_payload = verify_authenticated_message(
                    receive_frame(connection, deadline=deadline),
                    self.credential,
                    deadline=deadline,
                )
                deadline.check()
            result = GradeResult.from_payload(response_payload)
            deadline.check()
        except PrivateGraderClientError:
            raise
        except (OSError, TimeoutError, GraderProtocolError) as exc:
            raise PrivateGraderTransportError(
                "private grader IPC failed closed"
            ) from exc
        if (
            result.request_id != request.request_id
            or result.episode_id != request.episode_id
            or result.task_id != request.task_id
            or result.grader_binding_sha256 != request.grader_binding_sha256
            or result.package_identity_sha256 != request.package_identity_sha256
            or result.baseline_score != request.baseline_score
            or result.ideal_score != request.ideal_score
            or result.higher_is_better != request.higher_is_better
            or result.submission_sha256 != request.submission_sha256
        ):
            raise PrivateGraderClientError("private grader response identity mismatch")
        deadline.check()
        return result


def read_credential(path: Path) -> bytes:
    absolute = path.expanduser().absolute()
    try:
        info = os.stat(absolute, follow_symlinks=False)
    except OSError as exc:
        raise PrivateGraderClientError("grader credential is unavailable") from exc
    if (
        absolute.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) & 0o077
        or not 32 <= info.st_size <= 4096
    ):
        raise PrivateGraderClientError(
            "grader credential must be a private regular file"
        )
    try:
        return absolute.read_bytes()
    except OSError as exc:
        raise PrivateGraderClientError("cannot read grader credential") from exc
