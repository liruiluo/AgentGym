from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_COMMIT = "507f92e1138bb6e40dac5c6ee7a6758e6424bf97"
SPLIT_RELATIVE_PATH = Path("experiments/splits/low.txt")
SPLIT_SHA256 = "590270f007fa96b4060f59f3861500159c73ca50f7f30ff6bd38303c236c799b"
LITE_COMPETITION_IDS = (
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
)


class MLEBenchLiteIdentityError(RuntimeError):
    """The external MLE-bench checkout does not match the frozen identity."""


@dataclass(frozen=True)
class OfficialLiteIdentity:
    upstream_root: Path
    upstream_commit: str
    split_path: Path
    split_sha256: str
    competition_ids: tuple[str, ...]


CommitResolver = Callable[[Path], str]


def load_official_lite_identity(
    upstream_root: Path,
    *,
    commit_resolver: CommitResolver | None = None,
) -> OfficialLiteIdentity:
    """Load the official split from a pinned external checkout.

    This module intentionally does not import the upstream Python package or
    call its registry. The raw split bytes and ordered IDs are independently
    bound here.
    """

    root = _regular_directory(Path(upstream_root), "upstream root")
    resolver = _git_head if commit_resolver is None else commit_resolver
    try:
        commit = str(resolver(root)).strip()
    except Exception as exc:
        raise MLEBenchLiteIdentityError("cannot attest upstream commit") from exc
    if commit != UPSTREAM_COMMIT:
        raise MLEBenchLiteIdentityError("upstream commit does not match the pin")

    split_path = root / SPLIT_RELATIVE_PATH
    _reject_symlink_below(root, split_path)
    try:
        mode = split_path.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise MLEBenchLiteIdentityError("official split is not a regular file")
        payload = split_path.read_bytes()
    except OSError as exc:
        raise MLEBenchLiteIdentityError("cannot read official split") from exc

    digest = hashlib.sha256(payload).hexdigest()
    if digest != SPLIT_SHA256:
        raise MLEBenchLiteIdentityError("official split SHA256 does not match")
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MLEBenchLiteIdentityError("official split is not UTF-8") from exc
    competition_ids = tuple(decoded.splitlines())
    if len(competition_ids) != 22 or len(set(competition_ids)) != 22:
        raise MLEBenchLiteIdentityError("official split must contain 22 unique IDs")
    if competition_ids != LITE_COMPETITION_IDS:
        raise MLEBenchLiteIdentityError("official split membership or order drifted")
    return OfficialLiteIdentity(
        upstream_root=root,
        upstream_commit=commit,
        split_path=split_path,
        split_sha256=digest,
        competition_ids=competition_ids,
    )


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=10,
    )
    return completed.stdout.strip()


def _regular_directory(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise MLEBenchLiteIdentityError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise MLEBenchLiteIdentityError(f"{label} must be a non-symlink directory")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise MLEBenchLiteIdentityError(f"{label} cannot be resolved") from exc


def _reject_symlink_below(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise MLEBenchLiteIdentityError(
            "identity path escaped the upstream root"
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        try:
            if stat.S_ISLNK(current.lstat().st_mode):
                raise MLEBenchLiteIdentityError("identity path contains a symlink")
        except FileNotFoundError as exc:
            raise MLEBenchLiteIdentityError("identity path is unavailable") from exc
