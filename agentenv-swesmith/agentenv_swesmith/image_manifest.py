from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


IMAGE_MANIFEST_SCHEMA = "swesmith_oci_image_manifest_v1"
UPSTREAM_REPOSITORY = "SWE-bench/SWE-smith"
_DIGEST_RE = re.compile(r"\Asha256:[0-9a-f]{64}\Z")


class SwesmithImageManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class SwesmithImageBinding:
    image: str
    digest: str


class SwesmithImageManifest:
    """Frozen mapping from an official profile image name to one OCI digest."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file() or self.path.is_symlink():
            raise SwesmithImageManifestError(
                f"image manifest must be a real file: {self.path}"
            )
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        try:
            manifest = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SwesmithImageManifestError(
                "image manifest is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(manifest, dict):
            raise SwesmithImageManifestError("image manifest must be an object")
        if manifest.get("schema_version") != IMAGE_MANIFEST_SCHEMA:
            raise SwesmithImageManifestError("unsupported image manifest schema")
        upstream = manifest.get("upstream")
        if not isinstance(upstream, dict):
            raise SwesmithImageManifestError("image manifest upstream must be an object")
        if upstream.get("repository") != UPSTREAM_REPOSITORY:
            raise SwesmithImageManifestError("unexpected image manifest repository")
        # ``revision`` was the original field name and denotes the frozen
        # Hugging Face dataset snapshot.  Keep accepting it for old manifests,
        # while exposing the meaning explicitly for new launchers.
        raw_dataset_revision = upstream.get("dataset_revision", upstream.get("revision"))
        self.dataset_revision = _git_revision(raw_dataset_revision)
        raw_source_revision = upstream.get("source_revision")
        self.source_revision = (
            None if raw_source_revision is None else _git_revision(raw_source_revision)
        )
        self.upstream_revision = self.dataset_revision
        entries = manifest.get("images")
        if not isinstance(entries, list) or not entries:
            raise SwesmithImageManifestError("image manifest must contain images")
        bindings: dict[str, SwesmithImageBinding] = {}
        digests: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise SwesmithImageManifestError(f"image entry {index} must be an object")
            image = _text(entry, "image")
            digest = _text(entry, "digest").lower()
            if _DIGEST_RE.fullmatch(digest) is None:
                raise SwesmithImageManifestError(
                    f"image entry {index} has an invalid OCI digest"
                )
            if image in bindings:
                raise SwesmithImageManifestError(f"duplicate image name: {image}")
            if digest in digests:
                raise SwesmithImageManifestError(f"duplicate image digest: {digest}")
            bindings[image] = SwesmithImageBinding(image=image, digest=digest)
            digests.add(digest)
        self._bindings = bindings

    def resolve(self, image: str) -> SwesmithImageBinding:
        try:
            return self._bindings[image]
        except KeyError as exc:
            raise SwesmithImageManifestError(
                f"profile image is absent from the frozen image manifest: {image}"
            ) from exc

    def public_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": IMAGE_MANIFEST_SCHEMA,
            "dataset_revision": self.dataset_revision,
            "source_revision": self.source_revision,
            # Compatibility alias for older metadata consumers.
            "upstream_revision": self.dataset_revision,
            "manifest_sha256": self.sha256,
            "image_count": len(self._bindings),
        }


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SwesmithImageManifestError(f"image manifest field {key!r} is empty")
    return value.strip()


def _git_revision(value: Any) -> str:
    if not isinstance(value, str):
        raise SwesmithImageManifestError("upstream revision must be text")
    normalized = value.lower()
    if len(normalized) != 40 or any(c not in "0123456789abcdef" for c in normalized):
        raise SwesmithImageManifestError(
            "upstream revision must be a full 40-character Git commit"
        )
    return normalized
