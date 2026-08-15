from __future__ import annotations

import os
import re
import shutil
import signal
import stat
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from agentenv_swesmith.workspace import (
    SwesmithWorkspaceError,
    _chown_policy_tree,
)

from .protocol import policy_projection


WORKSPACE_CONTRACT = "swebench_verified_exact_base_archive_v1"
MAX_GIT_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_GIT_ARCHIVE_STDERR_BYTES = 64 * 1024
GIT_ARCHIVE_TIMEOUT_SECONDS = 300
_REPO_PART_RE = re.compile(r"\A[A-Za-z0-9_.-]+\Z")


class VerifiedWorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedWorkspace:
    episode_root: Path
    policy_root: Path
    private_root: Path
    mirror_root: Path
    git_dir: Path
    instance_id: str
    repo: str
    base_commit: str
    contract: str = WORKSPACE_CONTRACT


class VerifiedWorkspaceMaterializer:
    """Archive one exact local mirror commit into a private episode tree."""

    def __init__(self, *, mirrors_root: Path | str, episodes_root: Path | str):
        self.mirrors_root = require_real_directory(
            Path(mirrors_root).expanduser(), "Verified mirrors root"
        )
        episodes = Path(episodes_root).expanduser()
        created = False
        try:
            info = episodes.lstat()
        except FileNotFoundError:
            episodes.mkdir(parents=True, mode=0o700)
            created = True
            info = episodes.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise VerifiedWorkspaceError(
                "Verified episodes root must be a real directory"
            )
        if created:
            os.chmod(episodes, 0o700, follow_symlinks=False)
        self.episodes_root = require_real_directory(
            episodes, "Verified episodes root"
        )
        require_private_directory(self.episodes_root, "Verified episodes root")

    def materialize(
        self,
        instance: Mapping[str, Any],
        *,
        model_uid: int | None = None,
        model_gid: int | None = None,
    ) -> VerifiedWorkspace:
        try:
            projected = policy_projection(instance)
        except (TypeError, ValueError) as exc:
            raise VerifiedWorkspaceError(f"invalid policy instance: {exc}") from exc
        owner, name = projected["repo"].split("/")
        if not _REPO_PART_RE.fullmatch(owner) or not _REPO_PART_RE.fullmatch(name):
            raise VerifiedWorkspaceError("repo contains unsupported path characters")
        mirror_root = require_real_directory(
            self.mirrors_root / f"{owner}__{name}",
            f"mirror for {projected['instance_id']}",
        )
        resolved = git_text(
            mirror_root,
            "rev-parse",
            "--verify",
            f"{projected['base_commit']}^{{commit}}",
            label="base_commit",
        )
        if resolved != projected["base_commit"]:
            raise VerifiedWorkspaceError("base_commit did not resolve exactly")
        git_dir = Path(
            git_text(
                mirror_root,
                "rev-parse",
                "--absolute-git-dir",
                label="private Git directory",
            )
        ).resolve(strict=True)

        episode_root = Path(
            tempfile.mkdtemp(
                prefix="swebench-verified-episode-",
                dir=self.episodes_root,
            )
        )
        os.chmod(episode_root, 0o700)
        policy_root = episode_root / "workspace"
        private_root = episode_root / "private"
        policy_root.mkdir(mode=0o700)
        private_root.mkdir(mode=0o700)
        try:
            export_exact_git_tree(
                mirror_root=mirror_root,
                git_dir=git_dir,
                commit=resolved,
                destination=policy_root,
                private_root=private_root,
            )
            if (policy_root / ".git").exists() or (policy_root / ".git").is_symlink():
                raise VerifiedWorkspaceError(
                    "policy workspace unexpectedly contains Git metadata"
                )
            if model_uid is not None or model_gid is not None:
                if model_uid is None or model_gid is None:
                    raise VerifiedWorkspaceError(
                        "model_uid and model_gid must be supplied together"
                    )
                _chown_policy_tree(policy_root, model_uid, model_gid)
            return VerifiedWorkspace(
                episode_root=episode_root,
                policy_root=policy_root,
                private_root=private_root,
                mirror_root=mirror_root,
                git_dir=git_dir,
                instance_id=projected["instance_id"],
                repo=projected["repo"],
                base_commit=resolved,
            )
        except Exception as exc:
            shutil.rmtree(episode_root, ignore_errors=True)
            if isinstance(exc, VerifiedWorkspaceError):
                raise
            if isinstance(exc, SwesmithWorkspaceError):
                raise VerifiedWorkspaceError(str(exc)) from exc
            raise

    def close(self, workspace: VerifiedWorkspace) -> None:
        root = workspace.episode_root.resolve()
        if root.parent != self.episodes_root:
            raise VerifiedWorkspaceError(
                "refusing to remove an episode outside the configured root"
            )
        if not root.name.startswith("swebench-verified-episode-"):
            raise VerifiedWorkspaceError("refusing to remove an unknown episode path")
        shutil.rmtree(root)


def git_text(root: Path, *arguments: str, label: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve(strict=True)}",
            "-C",
            str(root),
            *arguments,
        ],
        capture_output=True,
        text=True,
        env=git_environment(),
    )
    if completed.returncode != 0:
        raise VerifiedWorkspaceError(f"git failed while resolving {label}")
    return completed.stdout.strip()


def export_exact_git_tree(
    *,
    mirror_root: Path,
    git_dir: Path,
    commit: str,
    destination: Path,
    private_root: Path,
) -> None:
    source_view = private_root / "source.git"
    completed = subprocess.run(
        ["git", "init", "--quiet", "--bare", str(source_view)],
        capture_output=True,
        text=True,
        env=git_environment(),
    )
    if completed.returncode != 0:
        raise VerifiedWorkspaceError("cannot initialize the private source view")
    try:
        objects = git_object_directory(mirror_root, git_dir)
        if "\n" in str(objects):
            raise VerifiedWorkspaceError("mirror object path contains a newline")
        (source_view / "objects" / "info" / "alternates").write_text(
            f"{objects}\n",
            encoding="utf-8",
        )
        (source_view / "info" / "attributes").write_text(
            "* -export-ignore -export-subst\n"
            "** -export-ignore -export-subst\n",
            encoding="utf-8",
        )
        export_git_tree(source_view, commit, destination)
    finally:
        shutil.rmtree(source_view, ignore_errors=True)


def export_git_tree(source_view: Path, commit: str, destination: Path) -> None:
    with (
        tempfile.TemporaryFile() as archive_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        process = subprocess.Popen(
            [
                "git",
                f"--git-dir={source_view}",
                "archive",
                "--format=tar",
                commit,
            ],
            cwd=source_view,
            env=git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=archive_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        deadline = time.monotonic() + GIT_ARCHIVE_TIMEOUT_SECONDS
        failure: VerifiedWorkspaceError | None = None
        while process.poll() is None:
            if os.fstat(archive_file.fileno()).st_size > MAX_GIT_ARCHIVE_BYTES:
                failure = VerifiedWorkspaceError(
                    "git archive exceeds the workspace byte limit"
                )
                break
            if (
                os.fstat(stderr_file.fileno()).st_size
                > MAX_GIT_ARCHIVE_STDERR_BYTES
            ):
                failure = VerifiedWorkspaceError(
                    "git archive stderr exceeds the bounded limit"
                )
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = VerifiedWorkspaceError("git archive timed out")
                break
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                continue
        if failure is not None:
            terminate_process_group(process)
            raise failure
        archive_size = os.fstat(archive_file.fileno()).st_size
        stderr_size = os.fstat(stderr_file.fileno()).st_size
        if archive_size > MAX_GIT_ARCHIVE_BYTES:
            raise VerifiedWorkspaceError(
                "git archive exceeds the workspace byte limit"
            )
        if stderr_size > MAX_GIT_ARCHIVE_STDERR_BYTES:
            raise VerifiedWorkspaceError(
                "git archive stderr exceeds the bounded limit"
            )
        if process.returncode != 0:
            stderr_file.seek(max(0, stderr_size - 4096))
            detail = stderr_file.read(4096).decode(
                "utf-8", errors="replace"
            )[-1000:]
            raise VerifiedWorkspaceError(f"git archive failed: {detail}")
        archive_file.seek(0)
        try:
            with tarfile.open(fileobj=archive_file, mode="r:") as archive:
                for member in archive:
                    extract_archive_member(archive, member, destination)
        except (OSError, tarfile.TarError) as exc:
            raise VerifiedWorkspaceError("git archive is invalid") from exc


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def extract_archive_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
) -> None:
    relative = normalize_archive_member(member.name)
    target = destination.joinpath(*relative.parts)
    require_real_parent_chain(destination, target.parent)
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        target.chmod(stat.S_IMODE(member.mode) or 0o755)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if member.isfile():
        source = archive.extractfile(member)
        if source is None:
            raise VerifiedWorkspaceError(
                f"git archive member has no content: {member.name}"
            )
        with source, target.open("xb") as handle:
            shutil.copyfileobj(source, handle)
        target.chmod(stat.S_IMODE(member.mode) or 0o644)
        return
    if member.issym():
        validate_symlink_target(relative, member.linkname)
        target.symlink_to(member.linkname)
        return
    raise VerifiedWorkspaceError(
        f"git archive contains unsupported entry type: {member.name}"
    )


def normalize_archive_member(raw: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise VerifiedWorkspaceError("git archive member path is invalid")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != raw.rstrip("/")
    ):
        raise VerifiedWorkspaceError(
            f"git archive member is not a normalized relative path: {raw!r}"
        )
    return path


def validate_symlink_target(member: PurePosixPath, raw_target: str) -> None:
    if (
        not isinstance(raw_target, str)
        or not raw_target
        or "\x00" in raw_target
        or PurePosixPath(raw_target).is_absolute()
    ):
        raise VerifiedWorkspaceError(
            f"git archive symlink target is invalid: {raw_target!r}"
        )
    resolved_parts = list(member.parent.parts)
    for part in raw_target.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            if not resolved_parts:
                raise VerifiedWorkspaceError(
                    f"git archive symlink escapes the workspace: {member}"
                )
            resolved_parts.pop()
            continue
        resolved_parts.append(part)


def require_real_parent_chain(root: Path, parent: Path) -> None:
    try:
        relative = parent.relative_to(root)
    except ValueError as exc:
        raise VerifiedWorkspaceError("git archive path escapes the workspace") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink() or (cursor.exists() and not cursor.is_dir()):
            raise VerifiedWorkspaceError(
                f"git archive parent is not a real directory: {cursor}"
            )


def git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return environment


def git_object_directory(mirror_root: Path, git_dir: Path) -> Path:
    completed = subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            "rev-parse",
            "--git-path",
            "objects",
        ],
        cwd=mirror_root,
        capture_output=True,
        text=True,
        env=git_environment(),
    )
    if completed.returncode != 0:
        raise VerifiedWorkspaceError("cannot resolve the mirror object directory")
    raw = completed.stdout.strip()
    if not raw:
        raise VerifiedWorkspaceError("mirror object directory is empty")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = mirror_root / candidate
    return require_real_directory(candidate, "mirror object directory")


def require_real_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise VerifiedWorkspaceError(f"{label} is unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise VerifiedWorkspaceError(f"{label} must be a real directory")
    return path.resolve(strict=True)


def require_private_directory(path: Path, label: str) -> None:
    info = path.stat()
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise VerifiedWorkspaceError(
            f"{label} must not be accessible to group or other users"
        )
