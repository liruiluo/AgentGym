from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import socket
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .deadline import DeadlineExceeded, MonotonicDeadline

CONTRACT_VERSION = "openmle_fast_v1"
REQUEST_SCHEMA = "openmle_fast_grade_request_v1"
RESPONSE_SCHEMA = "openmle_fast_grade_response_v1"
ENVELOPE_SCHEMA = "openmle_fast_authenticated_envelope_v1"
GRADER_BOUNDARY_CONTRACT = "openmle_fast_authenticated_private_ipc_v1"
MAX_PROTOCOL_BYTES = 96 * 1024 * 1024
_IDENTITY_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:@/+~-]{0,255}\Z")


class GraderProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class GradeRequest:
    request_id: str
    episode_id: str
    task_id: str
    grader_binding_sha256: str
    package_identity_sha256: str
    baseline_score: float
    ideal_score: float
    higher_is_better: bool
    submission: bytes
    submission_sha256: str

    @classmethod
    def build(
        cls,
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
    ) -> GradeRequest:
        if deadline is not None:
            deadline.check()
        if not isinstance(submission, bytes):
            raise TypeError("grader submission must be bytes")
        if type(higher_is_better) is not bool:
            raise GraderProtocolError("higher_is_better must be Boolean")
        return cls(
            request_id=_identity(request_id, "request_id"),
            episode_id=_identity(episode_id, "episode_id"),
            task_id=_identity(task_id, "task_id"),
            grader_binding_sha256=_sha256(
                grader_binding_sha256, "grader_binding_sha256"
            ),
            package_identity_sha256=_sha256(
                package_identity_sha256, "package_identity_sha256"
            ),
            baseline_score=_finite(baseline_score, "baseline_score"),
            ideal_score=_finite(ideal_score, "ideal_score"),
            higher_is_better=higher_is_better,
            submission=submission,
            submission_sha256=_payload_sha256(submission, deadline=deadline),
        )

    def payload(
        self,
        *,
        deadline: MonotonicDeadline | None = None,
    ) -> dict[str, Any]:
        if deadline is not None:
            deadline.check()
        encoded = base64.b64encode(self.submission).decode("ascii")
        if deadline is not None:
            deadline.check()
        return {
            "schema": REQUEST_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "grader_binding_sha256": self.grader_binding_sha256,
            "package_identity_sha256": self.package_identity_sha256,
            "baseline_score": self.baseline_score,
            "ideal_score": self.ideal_score,
            "higher_is_better": self.higher_is_better,
            "submission_b64": encoded,
            "submission_sha256": self.submission_sha256,
        }

    @classmethod
    def from_payload(
        cls,
        value: Any,
        *,
        max_submission_bytes: int,
        deadline: MonotonicDeadline | None = None,
    ) -> GradeRequest:
        if deadline is not None:
            deadline.check()
        required = {
            "schema",
            "contract_version",
            "request_id",
            "episode_id",
            "task_id",
            "grader_binding_sha256",
            "package_identity_sha256",
            "baseline_score",
            "ideal_score",
            "higher_is_better",
            "submission_b64",
            "submission_sha256",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise GraderProtocolError("grade request has unexpected fields")
        if (
            value["schema"] != REQUEST_SCHEMA
            or value["contract_version"] != CONTRACT_VERSION
        ):
            raise GraderProtocolError("grade request contract mismatch")
        if type(value["higher_is_better"]) is not bool:
            raise GraderProtocolError("grade request direction is invalid")
        encoded = value["submission_b64"]
        if not isinstance(encoded, str):
            raise GraderProtocolError("submission encoding must be text")
        try:
            submission = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise GraderProtocolError("submission encoding is invalid") from exc
        if len(submission) > max_submission_bytes:
            raise GraderProtocolError("submission exceeds the private-grader input cap")
        if deadline is not None:
            deadline.check()
        digest = _sha256(value["submission_sha256"], "submission_sha256")
        if not hmac.compare_digest(
            _payload_sha256(submission, deadline=deadline), digest
        ):
            raise GraderProtocolError("submission SHA256 mismatch")
        return cls(
            request_id=_identity(value["request_id"], "request_id"),
            episode_id=_identity(value["episode_id"], "episode_id"),
            task_id=_identity(value["task_id"], "task_id"),
            grader_binding_sha256=_sha256(
                value["grader_binding_sha256"], "grader_binding_sha256"
            ),
            package_identity_sha256=_sha256(
                value["package_identity_sha256"], "package_identity_sha256"
            ),
            baseline_score=_finite(value["baseline_score"], "baseline_score"),
            ideal_score=_finite(value["ideal_score"], "ideal_score"),
            higher_is_better=value["higher_is_better"],
            submission=submission,
            submission_sha256=digest,
        )


@dataclass(frozen=True)
class GradeResult:
    request_id: str
    episode_id: str
    task_id: str
    grader_binding_sha256: str
    package_identity_sha256: str
    baseline_score: float
    ideal_score: float
    submission_sha256: str
    submission_valid: bool
    native_score: float | None
    higher_is_better: bool
    normalized_reward: float | None
    improved_over_baseline: bool
    runtime_success: bool
    terminal_reason: str
    classification: str
    audit_digest: str

    def as_dict(self) -> dict[str, Any]:
        return self.payload()

    def payload(self) -> dict[str, Any]:
        return {
            "schema": RESPONSE_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "grader_binding_sha256": self.grader_binding_sha256,
            "package_identity_sha256": self.package_identity_sha256,
            "baseline_score": self.baseline_score,
            "ideal_score": self.ideal_score,
            "submission_sha256": self.submission_sha256,
            "submission_valid": self.submission_valid,
            "native_score": self.native_score,
            "higher_is_better": self.higher_is_better,
            "normalized_reward": self.normalized_reward,
            "improved_over_baseline": self.improved_over_baseline,
            "runtime_success": self.runtime_success,
            "terminal_reason": self.terminal_reason,
            "classification": self.classification,
            "audit_digest": self.audit_digest,
        }

    @classmethod
    def from_payload(cls, value: Any) -> GradeResult:
        required = {
            "schema",
            "contract_version",
            "request_id",
            "episode_id",
            "task_id",
            "grader_binding_sha256",
            "package_identity_sha256",
            "baseline_score",
            "ideal_score",
            "submission_sha256",
            "submission_valid",
            "native_score",
            "higher_is_better",
            "normalized_reward",
            "improved_over_baseline",
            "runtime_success",
            "terminal_reason",
            "classification",
            "audit_digest",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise GraderProtocolError("grade response has unexpected fields")
        if (
            value["schema"] != RESPONSE_SCHEMA
            or value["contract_version"] != CONTRACT_VERSION
        ):
            raise GraderProtocolError("grade response contract mismatch")
        booleans = (
            "submission_valid",
            "higher_is_better",
            "improved_over_baseline",
            "runtime_success",
        )
        if any(type(value[key]) is not bool for key in booleans):
            raise GraderProtocolError("grade response Boolean fields are invalid")
        native_score = _nullable_finite(value["native_score"], "native_score")
        normalized_reward = _nullable_finite(
            value["normalized_reward"], "normalized_reward"
        )
        baseline_score = _finite(value["baseline_score"], "baseline_score")
        ideal_score = _finite(value["ideal_score"], "ideal_score")
        classification = value["classification"]
        if classification not in {
            "graded",
            "invalid_submission",
            "infrastructure_fault",
        }:
            raise GraderProtocolError("grade response classification is invalid")
        terminal_reason = value["terminal_reason"]
        if terminal_reason not in {
            "graded_submission",
            "invalid_submission",
            "grader_infrastructure_fault",
        }:
            raise GraderProtocolError("grade terminal reason is invalid")
        if classification == "infrastructure_fault" and normalized_reward is not None:
            raise GraderProtocolError("infrastructure fault must carry null reward")
        if classification != "infrastructure_fault" and normalized_reward is None:
            raise GraderProtocolError("completed grade must carry a reward")
        if normalized_reward is not None and not -1.0 <= normalized_reward <= 1.0:
            raise GraderProtocolError("normalized reward is outside [-1, 1]")
        if classification == "graded" and (
            value["submission_valid"] is not True
            or native_score is None
            or normalized_reward is None
            or value["runtime_success"] is not True
            or terminal_reason != "graded_submission"
        ):
            raise GraderProtocolError("graded response fields are inconsistent")
        if classification == "invalid_submission" and (
            value["submission_valid"] is not False
            or native_score is not None
            or normalized_reward != -1.0
            or value["improved_over_baseline"] is not False
            or value["runtime_success"] is not False
            or terminal_reason != "invalid_submission"
        ):
            raise GraderProtocolError(
                "invalid-submission response fields are inconsistent"
            )
        if classification == "infrastructure_fault" and (
            value["submission_valid"] is not False
            or native_score is not None
            or value["improved_over_baseline"] is not False
            or value["runtime_success"] is not False
            or terminal_reason != "grader_infrastructure_fault"
        ):
            raise GraderProtocolError(
                "infrastructure-fault response fields are inconsistent"
            )
        if value["improved_over_baseline"] and not value["submission_valid"]:
            raise GraderProtocolError("invalid submission cannot improve over baseline")
        return cls(
            request_id=_identity(value["request_id"], "request_id"),
            episode_id=_identity(value["episode_id"], "episode_id"),
            task_id=_identity(value["task_id"], "task_id"),
            grader_binding_sha256=_sha256(
                value["grader_binding_sha256"], "grader_binding_sha256"
            ),
            package_identity_sha256=_sha256(
                value["package_identity_sha256"], "package_identity_sha256"
            ),
            baseline_score=baseline_score,
            ideal_score=ideal_score,
            submission_sha256=_sha256(value["submission_sha256"], "submission_sha256"),
            submission_valid=value["submission_valid"],
            native_score=native_score,
            higher_is_better=value["higher_is_better"],
            normalized_reward=normalized_reward,
            improved_over_baseline=value["improved_over_baseline"],
            runtime_success=value["runtime_success"],
            terminal_reason=terminal_reason,
            classification=classification,
            audit_digest=_sha256(value["audit_digest"], "audit_digest"),
        )


def authenticated_message(
    payload: Mapping[str, Any],
    credential: bytes,
    *,
    deadline: MonotonicDeadline | None = None,
) -> bytes:
    canonical = _canonical(payload, deadline=deadline)
    envelope = {
        "schema": ENVELOPE_SCHEMA,
        "payload": dict(payload),
        "hmac_sha256": _hmac_sha256(credential, canonical, deadline=deadline),
    }
    return _canonical(envelope, deadline=deadline)


def verify_authenticated_message(
    raw: bytes,
    credential: bytes,
    *,
    deadline: MonotonicDeadline | None = None,
) -> Mapping[str, Any]:
    if deadline is not None:
        deadline.check()
    try:
        envelope = _strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraderProtocolError("authenticated envelope is invalid JSON") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema",
        "payload",
        "hmac_sha256",
    }:
        raise GraderProtocolError("authenticated envelope has unexpected fields")
    if envelope["schema"] != ENVELOPE_SCHEMA or not isinstance(
        envelope["payload"], dict
    ):
        raise GraderProtocolError("authenticated envelope schema mismatch")
    if deadline is not None:
        deadline.check()
    supplied = _sha256(envelope["hmac_sha256"], "hmac_sha256")
    expected = _hmac_sha256(
        credential,
        _canonical(envelope["payload"], deadline=deadline),
        deadline=deadline,
    )
    if not hmac.compare_digest(supplied, expected):
        raise GraderProtocolError("authenticated envelope MAC mismatch")
    return envelope["payload"]


def send_frame(
    connection: socket.socket,
    payload: bytes,
    *,
    deadline: MonotonicDeadline | None = None,
) -> None:
    if len(payload) > MAX_PROTOCOL_BYTES:
        raise GraderProtocolError("grader protocol frame is too large")
    frame = struct.pack("!I", len(payload)) + payload
    if deadline is None:
        connection.sendall(frame)
        return
    remaining = memoryview(frame)
    while remaining:
        deadline.check()
        connection.settimeout(deadline.remaining_seconds())
        try:
            sent = connection.send(remaining[: 1024 * 1024])
        except TimeoutError as exc:
            raise DeadlineExceeded("grader protocol send deadline expired") from exc
        if sent <= 0:
            raise GraderProtocolError("grader protocol connection closed early")
        remaining = remaining[sent:]
    deadline.check()


def receive_frame(
    connection: socket.socket,
    *,
    deadline: MonotonicDeadline | None = None,
) -> bytes:
    size_raw = _receive_exact(connection, 4, deadline=deadline)
    size = struct.unpack("!I", size_raw)[0]
    if size == 0 or size > MAX_PROTOCOL_BYTES:
        raise GraderProtocolError("grader protocol frame size is invalid")
    return _receive_exact(connection, size, deadline=deadline)


def _receive_exact(
    connection: socket.socket,
    size: int,
    *,
    deadline: MonotonicDeadline | None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        if deadline is not None:
            deadline.check()
            connection.settimeout(deadline.remaining_seconds())
        try:
            chunk = connection.recv(min(remaining, 1024 * 1024))
        except TimeoutError as exc:
            if deadline is None:
                raise
            raise DeadlineExceeded("grader protocol receive deadline expired") from exc
        if not chunk:
            raise GraderProtocolError("grader protocol connection closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    if deadline is not None:
        deadline.check()
    return b"".join(chunks)


def _canonical(
    value: Any,
    *,
    deadline: MonotonicDeadline | None = None,
) -> bytes:
    if deadline is not None:
        deadline.check()
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if deadline is not None:
        deadline.check()
    return payload


def _payload_sha256(
    payload: bytes,
    *,
    deadline: MonotonicDeadline | None,
) -> str:
    digest = hashlib.sha256()
    view = memoryview(payload)
    for offset in range(0, len(view), 1024 * 1024):
        if deadline is not None:
            deadline.check()
        digest.update(view[offset : offset + 1024 * 1024])
    if deadline is not None:
        deadline.check()
    return digest.hexdigest()


def _hmac_sha256(
    credential: bytes,
    payload: bytes,
    *,
    deadline: MonotonicDeadline | None,
) -> str:
    digest = hmac.new(credential, digestmod=hashlib.sha256)
    view = memoryview(payload)
    for offset in range(0, len(view), 1024 * 1024):
        if deadline is not None:
            deadline.check()
        digest.update(view[offset : offset + 1024 * 1024])
    if deadline is not None:
        deadline.check()
    return digest.hexdigest()


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise GraderProtocolError(f"{label} is not a valid opaque identity")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise GraderProtocolError(f"{label} must be text")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise GraderProtocolError(f"{label} is not a SHA256 digest")
    return normalized


def _nullable_finite(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraderProtocolError(f"{label} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise GraderProtocolError(f"{label} must be finite")
    return result


def _finite(value: Any, label: str) -> float:
    result = _nullable_finite(value, label)
    if result is None:
        raise GraderProtocolError(f"{label} must not be null")
    return result


def _strict_json_loads(raw: bytes | str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GraderProtocolError("grader JSON contains a duplicate key")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicates)
