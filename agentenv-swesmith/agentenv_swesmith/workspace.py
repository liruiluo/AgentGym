from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


WORKSPACE_CONTRACT = "swesmith_git_archive_hidden_tests_v1"
_INSTANCE_ID_RE = re.compile(r"\A[A-Za-z0-9_.-]+\Z")


class SwesmithWorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class HiddenTestFile:
    path: str
    sha256: str
    mode: int


@dataclass(frozen=True)
class SwesmithWorkspace:
    episode_root: Path
    policy_root: Path
    hidden_tests_root: Path
    mirror_root: Path
    instance_id: str
    bug_commit: str
    pristine_commit: str
    hidden_tests: tuple[HiddenTestFile, ...]
    contract: str = WORKSPACE_CONTRACT


class SwesmithWorkspaceMaterializer:
    """Export one immutable SWE-smith branch into an episode workspace."""

    def __init__(self, *, mirrors_root: Path | str, episodes_root: Path | str):
        self.mirrors_root = _require_real_directory(
            Path(mirrors_root).expanduser().resolve(),
            "SWE-smith mirrors root",
        )
        self.episodes_root = Path(episodes_root).expanduser().resolve()
        self.episodes_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_private_directory(self.episodes_root, "SWE-smith episodes root")

    def materialize(
        self,
        instance: Mapping[str, Any],
        *,
        test_paths: Sequence[str | Path],
        model_uid: int | None = None,
        model_gid: int | None = None,
    ) -> SwesmithWorkspace:
        instance_id = _required_instance_text(instance, "instance_id")
        if not _INSTANCE_ID_RE.fullmatch(instance_id):
            raise SwesmithWorkspaceError(
                f"instance_id contains unsupported characters: {instance_id!r}"
            )
        mirror_name = _mirror_name(instance)
        mirror_root = _require_real_directory(
            self.mirrors_root / mirror_name,
            f"mirror for {instance_id}",
        )
        _require_git_repository(mirror_root)
        bug_commit = _resolve_instance_commit(mirror_root, instance_id)
        pristine_commit = _git_text(
            mirror_root,
            ["rev-parse", "--verify", f"{bug_commit}~1^{{commit}}"],
            label="pristine parent commit",
        )

        episode_root = Path(
            tempfile.mkdtemp(prefix="swesmith-episode-", dir=self.episodes_root)
        )
        os.chmod(episode_root, 0o700)
        policy_root = episode_root / "workspace"
        hidden_tests_root = episode_root / "private" / "pristine-tests"
        policy_root.mkdir(mode=0o700)
        hidden_tests_root.mkdir(mode=0o700, parents=True)
        try:
            _export_git_tree(mirror_root, bug_commit, policy_root)
            if (policy_root / ".git").exists() or (policy_root / ".git").is_symlink():
                raise SwesmithWorkspaceError("policy workspace unexpectedly contains .git")
            normalized_test_paths = _normalize_test_paths(test_paths)
            hidden_tests = _capture_hidden_tests(
                mirror_root,
                pristine_commit,
                hidden_tests_root,
                normalized_test_paths,
            )
            if model_uid is not None or model_gid is not None:
                if model_uid is None or model_gid is None:
                    raise SwesmithWorkspaceError(
                        "model_uid and model_gid must be supplied together"
                    )
                _chown_policy_tree(policy_root, model_uid, model_gid)
            os.chmod(hidden_tests_root.parent, 0o700)
            os.chmod(hidden_tests_root, 0o700)
            return SwesmithWorkspace(
                episode_root=episode_root,
                policy_root=policy_root,
                hidden_tests_root=hidden_tests_root,
                mirror_root=mirror_root,
                instance_id=instance_id,
                bug_commit=bug_commit,
                pristine_commit=pristine_commit,
                hidden_tests=hidden_tests,
            )
        except Exception:
            shutil.rmtree(episode_root, ignore_errors=True)
            raise

    def close(self, workspace: SwesmithWorkspace) -> None:
        episode_root = Path(workspace.episode_root).resolve()
        if episode_root.parent != self.episodes_root:
            raise SwesmithWorkspaceError(
                "refusing to remove an episode outside the configured root"
            )
        if not episode_root.name.startswith("swesmith-episode-"):
            raise SwesmithWorkspaceError("refusing to remove an unrecognized episode path")
        shutil.rmtree(episode_root)


def restore_hidden_tests(workspace: SwesmithWorkspace) -> tuple[str, ...]:
    restored: list[str] = []
    for hidden in workspace.hidden_tests:
        source = workspace.hidden_tests_root / hidden.path
        destination = workspace.policy_root / hidden.path
        _require_contained(workspace.hidden_tests_root, source)
        _require_contained(workspace.policy_root, destination)
        if not source.is_file() or source.is_symlink():
            raise SwesmithWorkspaceError(
                f"hidden pristine test is unavailable: {hidden.path}"
            )
        if hashlib.sha256(source.read_bytes()).hexdigest() != hidden.sha256:
            raise SwesmithWorkspaceError(
                f"hidden pristine test changed before grading: {hidden.path}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise SwesmithWorkspaceError(
                f"policy test path is not a regular file: {hidden.path}"
            )
        temporary = destination.with_name(f".{destination.name}.restore-{os.getpid()}")
        try:
            shutil.copyfile(source, temporary, follow_symlinks=False)
            os.chmod(temporary, hidden.mode)
            os.replace(temporary, destination)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        restored.append(hidden.path)
    return tuple(restored)


def _mirror_name(instance: Mapping[str, Any]) -> str:
    repo = _required_instance_text(instance, "repo").rstrip("/")
    mirror_name = repo.rsplit("/", 1)[-1]
    if not _INSTANCE_ID_RE.fullmatch(mirror_name):
        raise SwesmithWorkspaceError(f"invalid mirror identity: {mirror_name!r}")
    expected = _required_instance_text(instance, "instance_id").rsplit(".", 1)[0]
    if mirror_name != expected:
        raise SwesmithWorkspaceError(
            f"instance/repo mirror mismatch: expected {expected!r}, got {mirror_name!r}"
        )
    return mirror_name


def _resolve_instance_commit(mirror_root: Path, instance_id: str) -> str:
    candidates = (
        f"refs/heads/{instance_id}",
        f"refs/remotes/origin/{instance_id}",
    )
    found: list[str] = []
    for candidate in candidates:
        completed = subprocess.run(
            _git_command(
                mirror_root,
                ["rev-parse", "--verify", f"{candidate}^{{commit}}"],
            ),
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            found.append(completed.stdout.strip())
    if not found:
        raise SwesmithWorkspaceError(
            f"instance branch is absent from mirror: {instance_id}"
        )
    if len(set(found)) != 1:
        raise SwesmithWorkspaceError(
            f"local and remote instance refs disagree: {instance_id}"
        )
    return found[0]


def _export_git_tree(mirror_root: Path, commit: str, destination: Path) -> None:
    process = subprocess.Popen(
        _git_command(mirror_root, ["archive", "--format=tar", commit]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        try:
            with process.stdout, tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    _extract_archive_member(archive, member, destination)
        except Exception:
            process.kill()
            process.wait()
            raise
        stderr = process.stderr.read()
        return_code = process.wait()
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code != 0:
        raise SwesmithWorkspaceError(
            "git archive failed: " + stderr.decode("utf-8", errors="replace")[-1000:]
        )


def _extract_archive_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> None:
    relative = _normalize_relative_path(member.name, "git archive member")
    target = destination / relative
    _require_contained(destination, target)
    _require_real_parent_chain(destination, target.parent)
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        os.chmod(target, stat.S_IMODE(member.mode) or 0o755)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if member.isfile():
        source = archive.extractfile(member)
        if source is None:
            raise SwesmithWorkspaceError(
                f"git archive member has no file content: {member.name}"
            )
        with source, target.open("xb") as handle:
            shutil.copyfileobj(source, handle)
        os.chmod(target, stat.S_IMODE(member.mode) or 0o644)
        return
    if member.issym():
        _validate_archive_symlink_target(relative, member.linkname)
        target.symlink_to(member.linkname)
        return
    raise SwesmithWorkspaceError(
        f"git archive contains unsupported entry type: {member.name}"
    )


def _capture_hidden_tests(
    mirror_root: Path,
    pristine_commit: str,
    hidden_root: Path,
    requested_paths: Sequence[str],
) -> tuple[HiddenTestFile, ...]:
    expanded: set[str] = set()
    for requested in requested_paths:
        output = _git_bytes(
            mirror_root,
            [
                "ls-tree",
                "-r",
                "-z",
                "--name-only",
                pristine_commit,
                "--",
                requested,
            ],
            label=f"hidden test path {requested}",
        )
        matches = {
            value.decode("utf-8")
            for value in output.split(b"\0")
            if value
        }
        if not matches:
            raise SwesmithWorkspaceError(
                f"declared test path is absent from pristine commit: {requested}"
            )
        expanded.update(matches)

    captured: list[HiddenTestFile] = []
    for relative in sorted(expanded):
        normalized = _normalize_relative_path(relative, "hidden test path")
        mode_text = _git_text(
            mirror_root,
            ["ls-tree", pristine_commit, "--", normalized],
            label=f"hidden test mode {normalized}",
        )
        try:
            git_mode = mode_text.split(None, 1)[0]
        except IndexError as exc:
            raise SwesmithWorkspaceError(
                f"cannot parse git mode for hidden test: {normalized}"
            ) from exc
        if git_mode not in {"100644", "100755"}:
            raise SwesmithWorkspaceError(
                f"hidden test is not a regular git blob: {normalized}"
            )
        content = _git_bytes(
            mirror_root,
            ["show", f"{pristine_commit}:{normalized}"],
            label=f"hidden test content {normalized}",
        )
        destination = hidden_root / normalized
        _require_contained(hidden_root, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        mode = 0o755 if git_mode == "100755" else 0o644
        os.chmod(destination, mode)
        captured.append(
            HiddenTestFile(
                path=normalized,
                sha256=hashlib.sha256(content).hexdigest(),
                mode=mode,
            )
        )
    if not captured:
        raise SwesmithWorkspaceError("an episode must have at least one hidden test file")
    return tuple(captured)


def _normalize_test_paths(paths: Sequence[str | Path]) -> tuple[str, ...]:
    if isinstance(paths, (str, bytes)):
        raise SwesmithWorkspaceError("test_paths must be a sequence, not text")
    normalized = tuple(
        _normalize_relative_path(str(path), "declared test path") for path in paths
    )
    if not normalized:
        raise SwesmithWorkspaceError("test_paths must not be empty")
    if len(normalized) != len(set(normalized)):
        raise SwesmithWorkspaceError("test_paths contains duplicates")
    return normalized


def _normalize_relative_path(raw: str, label: str) -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise SwesmithWorkspaceError(f"{label} must be non-empty path text")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SwesmithWorkspaceError(f"{label} must be a normalized relative path: {raw!r}")
    return str(path)


def _validate_archive_symlink_target(relative: str, raw_target: str) -> None:
    if not raw_target or "\x00" in raw_target:
        raise SwesmithWorkspaceError(
            f"git archive symlink target is invalid: {relative}"
        )
    target = PurePosixPath(raw_target)
    if target.is_absolute():
        raise SwesmithWorkspaceError(
            f"git archive contains an absolute symlink: {relative}"
        )
    resolved_parts = list(PurePosixPath(relative).parent.parts)
    for part in target.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not resolved_parts:
                raise SwesmithWorkspaceError(
                    f"git archive symlink target escapes workspace: {relative}"
                )
            resolved_parts.pop()
        else:
            resolved_parts.append(part)


def _git_text(root: Path, arguments: Sequence[str], *, label: str) -> str:
    return _git_bytes(root, arguments, label=label).decode("utf-8").strip()


def _git_command(root: Path, arguments: Sequence[str]) -> list[str]:
    resolved_root = root.resolve(strict=True)
    return [
        "git",
        "-c",
        f"safe.directory={resolved_root}",
        "-C",
        str(resolved_root),
        *arguments,
    ]


def _git_bytes(root: Path, arguments: Sequence[str], *, label: str) -> bytes:
    completed = subprocess.run(
        _git_command(root, arguments),
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise SwesmithWorkspaceError(f"git failed for {label}: {detail}")
    return completed.stdout


def _required_instance_text(instance: Mapping[str, Any], key: str) -> str:
    value = instance.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SwesmithWorkspaceError(
            f"SWE-smith instance field {key!r} must be non-empty text"
        )
    return value.strip()


def _require_git_repository(root: Path) -> None:
    output = _git_text(root, ["rev-parse", "--is-bare-repository"], label="mirror type")
    if output not in {"true", "false"}:
        raise SwesmithWorkspaceError(f"invalid git mirror: {root}")


def _require_real_directory(path: Path, label: str) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise SwesmithWorkspaceError(f"{label} must be a real directory: {path}")
    return path


def _require_private_directory(path: Path, label: str) -> None:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise SwesmithWorkspaceError(
            f"{label} must not be accessible to group or other users: {path}"
        )


def _require_contained(root: Path, path: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise SwesmithWorkspaceError(f"path escapes episode root: {path}") from exc


def _require_real_parent_chain(root: Path, parent: Path) -> None:
    _require_contained(root, parent)
    cursor = root
    for part in parent.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink() or (cursor.exists() and not cursor.is_dir()):
            raise SwesmithWorkspaceError(
                f"archive parent is not a real directory: {cursor}"
            )


def _chown_policy_tree(root: Path, uid: int, gid: int) -> None:
    if isinstance(uid, bool) or not isinstance(uid, int) or uid <= 0:
        raise SwesmithWorkspaceError("model_uid must be a positive integer")
    if isinstance(gid, bool) or not isinstance(gid, int) or gid <= 0:
        raise SwesmithWorkspaceError("model_gid must be a positive integer")
    for current_root, directories, files in os.walk(root, followlinks=False):
        current = Path(current_root)
        os.chown(current, uid, gid, follow_symlinks=False)
        for name in [*directories, *files]:
            path = current / name
            if not path.is_symlink():
                os.chown(path, uid, gid, follow_symlinks=False)
