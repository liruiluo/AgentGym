from __future__ import annotations

import hashlib
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from .protocol import require_sha256
from .testspec import VerifiedTestSpecBinding


IMAGE_MANIFEST_CONTRACT = "swebench_verified_linux_amd64_digest_tsv_v1"
_IMAGE_TAG_RE = re.compile(
    r"\Aswebench/sweb\.eval\.x86_64\.[a-z0-9_.-]+:latest\Z"
)
_DIGEST_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")


class VerifiedImageManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenImagePins:
    tag_count: int
    tag_ledger_sha256: str


PRODUCTION_IMAGE_PINS = FrozenImagePins(
    tag_count=500,
    tag_ledger_sha256=(
        "b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a"
    ),
)


class VerifiedImageManifest:
    """Validate the externally frozen tag-to-linux/amd64-digest ledger."""

    def __init__(
        self,
        path: Path | str,
        *,
        expected_manifest_sha256: str,
        pins: FrozenImagePins = PRODUCTION_IMAGE_PINS,
    ) -> None:
        self.path = require_regular_file(Path(path).expanduser())
        payload = self.path.read_bytes()
        try:
            expected_sha256 = require_sha256(
                expected_manifest_sha256,
                "expected image manifest SHA-256",
            )
        except ValueError as exc:
            raise VerifiedImageManifestError(str(exc)) from exc
        actual_manifest_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_manifest_sha256 != expected_sha256:
            raise VerifiedImageManifestError(
                "image manifest SHA-256 does not match the runtime pin"
            )
        if not payload or not payload.endswith(b"\n"):
            raise VerifiedImageManifestError(
                "image digest manifest must be non-empty and newline terminated"
            )
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise VerifiedImageManifestError(
                "image digest manifest must be UTF-8"
            ) from exc
        if len(lines) != pins.tag_count:
            raise VerifiedImageManifestError(
                f"image digest row count mismatch: expected {pins.tag_count}, "
                f"got {len(lines)}"
            )

        rows: list[tuple[str, str]] = []
        for line_number, line in enumerate(lines, start=1):
            fields = line.split("\t")
            if len(fields) != 2:
                raise VerifiedImageManifestError(
                    f"image digest row {line_number} must have two TSV fields"
                )
            tag, digest = fields
            if _IMAGE_TAG_RE.fullmatch(tag) is None:
                raise VerifiedImageManifestError(
                    f"image tag {line_number} is not an expected swebench tag"
                )
            if _DIGEST_RE.fullmatch(digest) is None:
                raise VerifiedImageManifestError(
                    f"image digest {line_number} is not a lowercase sha256 digest"
                )
            rows.append((tag, digest))

        tags = [tag for tag, _digest in rows]
        if len(set(tags)) != len(tags):
            raise VerifiedImageManifestError("image tags must be unique")
        if tags != sorted(tags):
            raise VerifiedImageManifestError("image tags must be sorted")
        ledger = "".join(f"{tag}\n" for tag in tags).encode("utf-8")
        ledger_sha256 = hashlib.sha256(ledger).hexdigest()
        if ledger_sha256 != pins.tag_ledger_sha256:
            raise VerifiedImageManifestError(
                "image tag ledger SHA-256 does not match the frozen pins"
            )

        self._digests = dict(rows)
        tags_by_digest: dict[str, list[str]] = {}
        for tag, digest in rows:
            tags_by_digest.setdefault(digest, []).append(tag)
        self._tags_by_digest = {
            digest: tuple(tags) for digest, tags in tags_by_digest.items()
        }
        self.tag_ledger_sha256 = ledger_sha256
        self.manifest_sha256 = actual_manifest_sha256
        self.unique_digest_count = len(set(self._digests.values()))

    def resolve(self, binding: VerifiedTestSpecBinding) -> str:
        if binding.platform != "linux/x86_64" or binding.namespace != "swebench":
            raise VerifiedImageManifestError(
                "TestSpec binding is not the pinned swebench linux/x86_64 profile"
            )
        try:
            return self._digests[binding.instance_image_key]
        except KeyError as exc:
            raise VerifiedImageManifestError(
                "TestSpec image tag is absent from the frozen digest manifest"
            ) from exc

    def aliases_for_digest(self, digest: str) -> tuple[str, ...]:
        if _DIGEST_RE.fullmatch(digest) is None:
            raise VerifiedImageManifestError("image digest is not a valid sha256")
        try:
            return self._tags_by_digest[digest]
        except KeyError as exc:
            raise VerifiedImageManifestError(
                "image digest is absent from the frozen manifest"
            ) from exc

    def public_metadata(self) -> dict[str, object]:
        return {
            "contract": IMAGE_MANIFEST_CONTRACT,
            "tag_count": len(self._digests),
            "tag_ledger_sha256": self.tag_ledger_sha256,
            "manifest_sha256": self.manifest_sha256,
            "unique_digest_count": self.unique_digest_count,
        }


def require_regular_file(path: Path) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise VerifiedImageManifestError(
            f"image digest manifest is unavailable: {path}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise VerifiedImageManifestError(
            "image digest manifest must be a real regular file"
        )
    return path.resolve(strict=True)
