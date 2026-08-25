from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_SCRAPY_SYBIL_IMAGE = "swebench/swesmith.x86_64.scrapy_1776_scrapy.35212ec5"


class SwesmithProfileError(RuntimeError):
    """Raised when the frozen SWE-smith profile contract cannot be loaded."""


@dataclass(frozen=True)
class SwesmithProfileBinding:
    """Policy-private profile information needed to materialize and grade one row."""

    repo: str
    image: str
    f2p_test_paths: tuple[str, ...]
    p2p_test_paths: tuple[str, ...]
    f2p_command: str
    full_command: str
    log_parser: Callable[[str], Mapping[str, str]]
    get_eval_tests_report: Callable[..., Mapping[str, Any]]
    get_resolution_status: Callable[[Mapping[str, Any]], str]
    full_resolution_status: str
    source_full_command: str | None = None
    command_corrections: tuple[str, ...] = ()
    profile_contract: str = "swesmith_official_repo_profile_v1"

    @property
    def all_test_paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.f2p_test_paths, *self.p2p_test_paths)))

    def as_private_metadata(self) -> dict[str, Any]:
        return {
            "contract": self.profile_contract,
            "repo": self.repo,
            "image": self.image,
            "f2p_test_paths": list(self.f2p_test_paths),
            "p2p_test_paths": list(self.p2p_test_paths),
            "f2p_command": self.f2p_command,
            "full_command": self.full_command,
            "source_full_command": self.source_full_command or self.full_command,
            "command_corrections": list(self.command_corrections),
        }


class OfficialSwesmithProfileResolver:
    """Resolve official SWE-smith profiles from one pinned source checkout.

    The resolver deliberately imports the upstream registry lazily.  Unit tests
    can use a fake resolver without installing SWE-smith, while formal launchers
    must provide the exact source revision through ``source_root`` or the
    ``SWESMITH_SOURCE_ROOT`` environment variable.
    """

    def __init__(
        self,
        *,
        source_root: Path | str | None = None,
        expected_revision: str | None = None,
    ) -> None:
        raw_root = source_root or os.environ.get("SWESMITH_SOURCE_ROOT")
        self.source_root = None if raw_root is None else Path(raw_root).expanduser().resolve()
        self.expected_revision = expected_revision or os.environ.get(
            "SWESMITH_SOURCE_REVISION"
        )
        self._loaded = False
        self._registry: Any = None
        self._get_eval_tests_report: Callable[..., Mapping[str, Any]] | None = None
        self._get_resolution_status: Callable[[Mapping[str, Any]], str] | None = None
        self._full_resolution_status: str | None = None

    def resolve(self, instance: Mapping[str, Any]) -> SwesmithProfileBinding:
        self._load()
        if not isinstance(instance, Mapping):
            raise SwesmithProfileError("SWE-smith instance must be a mapping")
        try:
            profile = self._registry.get_from_inst(dict(instance))
            f2p, p2p = profile.get_test_files(dict(instance))
            f2p_command, _ = profile.get_test_cmd(dict(instance), f2p_only=True)
            source_full_command, source_full_paths = profile.get_test_cmd(
                dict(instance), f2p_only=False
            )
        except Exception as exc:
            raise SwesmithProfileError(
                f"failed to resolve official profile for {instance.get('instance_id')!r}"
            ) from exc
        f2p_paths = _normalize_paths(f2p, "FAIL_TO_PASS test paths")
        p2p_paths = _normalize_paths(p2p, "PASS_TO_PASS test paths")
        if not f2p_paths:
            raise SwesmithProfileError("official profile returned no F2P test files")
        if not isinstance(f2p_command, str) or not f2p_command.strip():
            raise SwesmithProfileError("official profile returned an empty F2P command")
        if (
            not isinstance(source_full_command, str)
            or not source_full_command.strip()
        ):
            raise SwesmithProfileError("official profile returned an empty full command")
        image = _required_profile_text(profile, "image_name")
        full_command, command_corrections = _effective_full_command(
            source_full_command,
            source_paths=tuple(str(path) for path in source_full_paths),
            expected_paths=(*f2p_paths, *p2p_paths),
            image=image,
        )
        return SwesmithProfileBinding(
            repo=_required_text(instance, "repo"),
            image=image,
            f2p_test_paths=f2p_paths,
            p2p_test_paths=p2p_paths,
            f2p_command=f2p_command,
            full_command=full_command,
            log_parser=profile.log_parser,
            get_eval_tests_report=self._get_eval_tests_report,  # type: ignore[arg-type]
            get_resolution_status=self._get_resolution_status,  # type: ignore[arg-type]
            full_resolution_status=self._full_resolution_status or "FULL",
            source_full_command=source_full_command,
            command_corrections=command_corrections,
        )

    def _load(self) -> None:
        if self._loaded:
            return
        if self.source_root is not None:
            if not self.source_root.is_dir() or self.source_root.is_symlink():
                raise SwesmithProfileError(
                    f"SWE-smith source root is not a real directory: {self.source_root}"
                )
            if self.expected_revision:
                actual = _git_head(self.source_root)
                if actual != self.expected_revision.lower():
                    raise SwesmithProfileError(
                        "SWE-smith source revision mismatch: "
                        f"expected {self.expected_revision.lower()}, got {actual}"
                    )
            source_text = str(self.source_root)
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
        try:
            from swesmith.harness.grading import get_eval_tests_report
            from swesmith.profiles import registry
            from swebench.harness.constants import ResolvedStatus
            from swebench.harness.grading import get_resolution_status
        except Exception as exc:
            raise SwesmithProfileError(
                "formal SWE-smith requires the pinned swesmith and swebench packages"
            ) from exc
        self._registry = registry
        self._get_eval_tests_report = get_eval_tests_report
        self._get_resolution_status = get_resolution_status
        self._full_resolution_status = ResolvedStatus.FULL.value
        self._loaded = True


def _normalize_paths(values: Sequence[Any], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise SwesmithProfileError(f"{label} must be a sequence")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, (str, Path)):
            raise SwesmithProfileError(f"{label} contains a non-path value")
        text = str(value).strip()
        if not text or "\x00" in text:
            raise SwesmithProfileError(f"{label} contains an invalid path")
        path = Path(text)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise SwesmithProfileError(f"{label} must contain relative paths: {text!r}")
        normalized.append(path.as_posix())
    return tuple(dict.fromkeys(normalized))


def _effective_full_command(
    command: str,
    *,
    source_paths: Sequence[str],
    expected_paths: Sequence[str],
    image: str,
) -> tuple[str, tuple[str, ...]]:
    """Repair selection-only defects without weakening the declared test set."""

    source = tuple(source_paths)
    expected = tuple(expected_paths)
    if source != expected:
        raise SwesmithProfileError(
            "official full command paths disagree with FAIL_TO_PASS/PASS_TO_PASS"
        )
    if not source:
        return command, ()

    suffix = " " + " ".join(source)
    if not command.endswith(suffix):
        raise SwesmithProfileError(
            "official full command does not end with its declared test paths"
        )

    effective_paths = tuple(dict.fromkeys(source))
    corrections: list[str] = []
    if len(effective_paths) != len(source):
        corrections.append("deduplicate_test_paths_preserve_first_occurrence_v1")

    pytest_options: tuple[str, ...] = ()
    if image == _SCRAPY_SYBIL_IMAGE and any(
        path.endswith(".rst") for path in effective_paths
    ):
        pytest_options = ("-p", "no:doctest")
        corrections.append("scrapy_sybil_disable_builtin_doctest_v1")

    if not corrections:
        return command, ()
    command_prefix = command[: -len(suffix)].rstrip()
    effective = " ".join((*pytest_options, *effective_paths))
    return f"{command_prefix} {effective}", tuple(corrections)


def _required_text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SwesmithProfileError(f"profile/instance field {key!r} is empty")
    return value.strip()


def _required_profile_text(profile: Any, key: str) -> str:
    value = getattr(profile, key, None)
    if not isinstance(value, str) or not value.strip():
        raise SwesmithProfileError(f"official profile attribute {key!r} is empty")
    return value.strip()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SwesmithProfileError(
            "cannot attest SWE-smith source revision: "
            + completed.stderr[-500:]
        )
    value = completed.stdout.strip().lower()
    if len(value) != 40:
        raise SwesmithProfileError("SWE-smith source revision is not a full commit")
    return value
