from __future__ import annotations

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


class PrivateGraderClient:
    """The environment-side holder of the authenticated private IPC boundary."""

    def __init__(
        self,
        *,
        endpoint: Path | str,
        credential_path: Path | str,
        timeout_seconds: float,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("private-grader timeout must be finite and positive")
        self.endpoint = Path(endpoint).expanduser().absolute()
        self.credential = read_credential(Path(credential_path))
        self.timeout_seconds = float(timeout_seconds)

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
        request = GradeRequest.build(
            request_id=request_id,
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
        operation_deadline.check()
        try:
            info = os.stat(self.endpoint, follow_symlinks=False)
            if not stat.S_ISSOCK(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
                raise PrivateGraderClientError(
                    "private-grader endpoint is not a private Unix socket"
                )
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(operation_deadline.remaining_seconds())
                connection.connect(str(self.endpoint))
                operation_deadline.check()
                message = authenticated_message(
                    request.payload(deadline=operation_deadline),
                    self.credential,
                    deadline=operation_deadline,
                )
                operation_deadline.check()
                connection.settimeout(operation_deadline.remaining_seconds())
                send_frame(
                    connection,
                    message,
                    deadline=operation_deadline,
                )
                operation_deadline.check()
                connection.settimeout(operation_deadline.remaining_seconds())
                response_payload = verify_authenticated_message(
                    receive_frame(connection, deadline=operation_deadline),
                    self.credential,
                    deadline=operation_deadline,
                )
                operation_deadline.check()
            result = GradeResult.from_payload(response_payload)
            operation_deadline.check()
        except PrivateGraderClientError:
            raise
        except (OSError, TimeoutError, GraderProtocolError) as exc:
            raise PrivateGraderClientError("private grader IPC failed closed") from exc
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
        operation_deadline.check()
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
