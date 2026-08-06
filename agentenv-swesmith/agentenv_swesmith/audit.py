from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Mapping


AUDIT_SCHEMA = "agentmemory_swesmith_private_episode_audit_v1"
_AUDIT_ID_RE = re.compile(r"\A[0-9a-f]{32}\Z")


class SwesmithAuditError(RuntimeError):
    pass


class SwesmithEpisodeAuditSink:
    """Persist private episode evidence without exposing it through the policy API."""

    def __init__(self, root: Path | str) -> None:
        path = Path(root).expanduser()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.is_symlink() or not path.is_dir():
            raise SwesmithAuditError(f"SWE-smith audit root must be a real directory: {path}")
        self.root = path.resolve(strict=True)
        os.chmod(self.root, 0o700)

    def write(self, *, audit_id: str, payload: Mapping[str, Any]) -> Path:
        if not _AUDIT_ID_RE.fullmatch(audit_id):
            raise SwesmithAuditError("SWE-smith audit id must be 32 lowercase hex characters")
        document = dict(payload)
        document["schema"] = AUDIT_SCHEMA
        document["audit_id"] = audit_id
        try:
            encoded = (
                json.dumps(
                    document,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise SwesmithAuditError("SWE-smith audit evidence is not JSON serializable") from exc

        target = self.root / f"episode-{audit_id}.json"
        temporary = self.root / (
            f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        if target.exists() or target.is_symlink():
            raise SwesmithAuditError(f"SWE-smith audit already exists: {target.name}")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
            directory = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise SwesmithAuditError(f"failed to persist SWE-smith audit {audit_id}") from exc
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        return target
