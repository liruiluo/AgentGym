from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AUDIT_CONTRACT = "openmle_fast_append_only_episode_audit_v1"


class OpenMLEFastAuditError(RuntimeError):
    pass


class OpenMLEFastAuditSink:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise OpenMLEFastAuditError(
                "environment audit root must be a real directory"
            )
        self.root.chmod(0o700)
        self._lock = threading.Lock()

    def emit(
        self,
        *,
        event: str,
        episode_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        record = {
            "schema": "openmle_fast_audit_record_v1",
            "contract": AUDIT_CONTRACT,
            "event": event,
            "episode_id": episode_id,
            "payload": dict(payload),
        }
        canonical = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        serialized = (
            json.dumps(
                {**record, "audit_digest": digest},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        destination = self.root / f"event-{uuid.uuid4().hex}.json"
        with self._lock:
            try:
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                try:
                    with os.fdopen(descriptor, "wb", closefd=False) as handle:
                        handle.write(serialized)
                        handle.flush()
                        os.fsync(handle.fileno())
                finally:
                    os.close(descriptor)
            except OSError as exc:
                raise OpenMLEFastAuditError(
                    "environment audit write failed closed"
                ) from exc
        return digest
