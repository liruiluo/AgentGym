from __future__ import annotations

import errno
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from agentenv_agentmemory.workspace_sandbox import (
    ExecutableFingerprint,
    ShellExecutionResult,
    ShellSandboxError,
    ShellSandboxLimits,
    _collect_bounded_output,
    _lease_ephemeral_model_uid,
    _normalize_sha256,
    _require_executable,
    assert_executable_fingerprint,
    executable_fingerprint,
    executable_sha256,
)


SWE_SHELL_SANDBOX_CONTRACT = "swesmith_linux_namespace_oci_rootfs_v1"
OCI_ROOTFS_CACHE_SCHEMA = "swesmith_oci_rootfs_cache_v1"
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOCAL_SANDBOX_SCRATCH_ROOT = Path("/tmp")
_SANDBOX_CLEANUP_TIMEOUT_SECONDS = 2.0
_SANDBOX_CLEANUP_RETRY_SECONDS = 0.05
_TRANSIENT_SANDBOX_CLEANUP_ERRNOS = {
    errno.EBUSY,
    errno.ENOENT,
    errno.ENOTEMPTY,
}


class SwesmithSandboxError(ShellSandboxError):
    pass


@dataclass(frozen=True)
class OciRootfsIdentity:
    """The immutable identity of one SWE-smith profile runtime.

    The cache is prepared out of band from an OCI image.  The launcher accepts
    only a complete directory whose metadata and image manifest agree with the
    digest selected by the dataset/profile manifest.
    """

    cache_dir: Path
    rootfs: Path
    image: str
    digest: str
    config_sha256: str
    manifest_sha256: str
    architecture: str
    operating_system: str
    working_dir: str
    rootfs_bytes: int
    rootfs_regular_files: int
    bash_sha256: str
    key_fingerprints: tuple[tuple[str, ExecutableFingerprint], ...]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "schema": OCI_ROOTFS_CACHE_SCHEMA,
            "cache_dir": str(self.cache_dir),
            "rootfs": str(self.rootfs),
            "image": self.image,
            "digest": self.digest,
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "architecture": self.architecture,
            "os": self.operating_system,
            "working_dir": self.working_dir,
            "bytes": self.rootfs_bytes,
            "regular_files": self.rootfs_regular_files,
            "bash_sha256": self.bash_sha256,
            "key_fingerprints": {
                path: fingerprint.as_metadata()
                for path, fingerprint in self.key_fingerprints
            },
        }


@dataclass(frozen=True)
class WorkspaceEntry:
    path: str
    kind: str
    mode: int
    size: int
    sha256: str | None
    link_target: str | None
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def evidence(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "mode": self.mode,
            "size": self.size,
            "sha256": self.sha256,
            "link_target": self.link_target,
        }


@dataclass(frozen=True)
class WorkspaceTreeSnapshot:
    entries: tuple[WorkspaceEntry, ...]
    regular_file_count: int
    symlink_count: int
    directory_count: int
    inode_count: int
    total_bytes: int
    tree_sha256: str

    def as_summary(self) -> dict[str, Any]:
        return {
            "schema": "swesmith_workspace_tree_snapshot_v1",
            "regular_file_count": self.regular_file_count,
            "symlink_count": self.symlink_count,
            "directory_count": self.directory_count,
            "inode_count": self.inode_count,
            "total_bytes": self.total_bytes,
            "tree_sha256": self.tree_sha256,
        }


@dataclass(frozen=True)
class WorkspaceDiff:
    added: tuple[Mapping[str, Any], ...]
    modified: tuple[Mapping[str, Any], ...]
    deleted: tuple[Mapping[str, Any], ...]
    before_tree_sha256: str
    after_tree_sha256: str

    @property
    def changed_paths(self) -> tuple[str, ...]:
        paths = {
            str(item["path"])
            for group in (self.added, self.modified, self.deleted)
            for item in group
        }
        return tuple(sorted(paths))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "swesmith_workspace_diff_v1",
            "added": [dict(item) for item in self.added],
            "modified": [dict(item) for item in self.modified],
            "deleted": [dict(item) for item in self.deleted],
            "changed_paths": list(self.changed_paths),
            "before_tree_sha256": self.before_tree_sha256,
            "after_tree_sha256": self.after_tree_sha256,
        }


@dataclass(frozen=True)
class SwesmithShellExecution:
    result: ShellExecutionResult
    workspace_before: WorkspaceTreeSnapshot
    workspace_after: WorkspaceTreeSnapshot
    workspace_diff: WorkspaceDiff


class LinuxNamespaceEpisodeSandbox:
    """Execute all commands for one SWE episode against one direct-bound tree.

    The leased UID and repository workspace live for the episode. Every command
    still gets fresh Linux namespaces and a bounded private ``/tmp``. Unlike the
    smaller WebShop sandbox, this class never copies the repository per action.
    """

    def __init__(
        self,
        *,
        limits: ShellSandboxLimits,
        rg_binary: Path,
        expected_rg_sha256: str,
        rg_sha256: str,
        rg_version: str,
        rg_fingerprint: ExecutableFingerprint,
        binaries: Mapping[str, Path],
        uid_lease_context: Any,
        model_uid: int,
        oci_rootfs_identity: OciRootfsIdentity | None = None,
    ) -> None:
        self.limits = limits
        self.rg_binary = rg_binary
        self.expected_rg_sha256 = expected_rg_sha256
        self.rg_sha256 = rg_sha256
        self.rg_version = rg_version
        self.rg_fingerprint = rg_fingerprint
        self._binaries = dict(binaries)
        self._uid_lease_context = uid_lease_context
        self._model_uid = model_uid
        self._oci_rootfs_identity = oci_rootfs_identity
        self._workspace_root: Path | None = None
        self._snapshot: WorkspaceTreeSnapshot | None = None
        self._closed = False
        self._poisoned_reason: str | None = None
        self._run_lock = threading.Lock()

    @classmethod
    def from_environment(
        cls,
        *,
        limits: ShellSandboxLimits,
        rg_binary: Path,
        expected_rg_sha256: str,
        oci_cache_root: Path,
        repo_profile_image: str,
        repo_profile_digest: str,
        lease_root: Path | None = None,
        lease_slots: int = 4096,
        run_preflight: bool = True,
    ) -> LinuxNamespaceEpisodeSandbox:
        if platform.system() != "Linux":
            raise SwesmithSandboxError(
                "the formal SWE-smith sandbox requires Linux namespaces"
            )
        if os.geteuid() != 0:
            raise SwesmithSandboxError(
                "the formal SWE-smith sandbox launcher must start as root"
            )

        def require(name: str) -> Path:
            resolved = shutil.which(name)
            if resolved is None:
                raise SwesmithSandboxError(
                    f"required SWE-smith sandbox executable is missing: {name}"
                )
            return Path(resolved).resolve()

        pinned_rg = _require_executable(rg_binary, "ripgrep")
        expected = _normalize_sha256(expected_rg_sha256, "expected ripgrep SHA256")
        actual = executable_sha256(pinned_rg)
        if actual != expected:
            raise SwesmithSandboxError(
                "ripgrep SHA256 does not match the frozen launcher contract: "
                f"expected {expected}, got {actual}"
            )
        binaries = {
            name: require(name)
            for name in (
                "bash",
                "chroot",
                "hostname",
                "mknod",
                "mount",
                "unshare",
            )
        }
        identity = load_oci_rootfs_identity(
            oci_cache_root,
            expected_image=repo_profile_image,
            expected_digest=repo_profile_digest,
            expected_owner_uid=0,
        )
        lease_kwargs: dict[str, Any] = {"slot_count": lease_slots}
        if lease_root is not None:
            lease_kwargs["lease_root"] = Path(lease_root)
        lease_context = _lease_ephemeral_model_uid(**lease_kwargs)
        model_uid = lease_context.__enter__()
        try:
            sandbox = cls(
                limits=limits,
                rg_binary=pinned_rg,
                expected_rg_sha256=expected,
                rg_sha256=actual,
                rg_version=_executable_version(pinned_rg),
                rg_fingerprint=executable_fingerprint(pinned_rg),
                binaries=binaries,
                uid_lease_context=lease_context,
                model_uid=model_uid,
                oci_rootfs_identity=identity,
            )
            if run_preflight:
                sandbox.preflight()
            return sandbox
        except BaseException:
            lease_context.__exit__(*sys.exc_info())
            raise

    @property
    def model_uid(self) -> int:
        return self._model_uid

    @property
    def model_gid(self) -> int:
        return self._model_uid

    @property
    def poisoned_reason(self) -> str | None:
        return self._poisoned_reason

    @property
    def metadata(self) -> Mapping[str, Any]:
        rootfs = (
            None
            if self._oci_rootfs_identity is None
            else self._oci_rootfs_identity.as_metadata()
        )
        return {
            "contract": SWE_SHELL_SANDBOX_CONTRACT,
            "formal_eligible": True,
            "network": "new_namespace_no_routes",
            "rootfs": "digest_pinned_oci_profile_rootfs_read_only",
            "oci_rootfs": rootfs,
            "workspace_mount": "episode_persistent_direct_rw_bind",
            "workspace_quota": "rlimit_fsize_plus_post_command_full_tree_validation",
            "tmp_mount": "bounded_tmpfs",
            "shell": "bash_no_profile_no_rc",
            "ripgrep_path": "/run/tools/rg",
            "ripgrep_sha256": self.rg_sha256,
            "ripgrep_expected_sha256": self.expected_rg_sha256,
            "ripgrep_version": self.rg_version,
            "ripgrep_startup_fingerprint": self.rg_fingerprint.as_metadata(),
            "model_identity": "exclusive_leased_high_uid_per_episode",
            "model_uid": self.model_uid,
            "no_new_privileges": True,
            "capability_bounding_set": "empty",
            "fresh_namespaces_per_command": ["mount", "network", "pid", "ipc", "uts"],
            "resource_limits": asdict(self.limits),
        }

    def __enter__(self) -> LinuxNamespaceEpisodeSandbox:
        self._require_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def preflight(self) -> None:
        self._require_open()
        with tempfile.TemporaryDirectory(prefix="swesmith-sandbox-preflight-") as raw:
            workspace = Path(raw) / "workspace"
            workspace.mkdir(mode=0o700)
            os.chown(workspace, self.model_uid, self.model_gid)
            result = self._run_namespace(
                workspace,
                command=(
                    "test \"$(command -v rg)\" = /run/tools/rg && "
                    "rg --version >/dev/null && "
                    "printf SWESMITH_OCI_ROOTFS_SANDBOX_OK > proof && "
                    "cat proof"
                ),
                workdir=".",
                timeout_ms=min(10_000, self.limits.max_timeout_ms),
            )
            proof = workspace / "proof"
            valid = (
                result.exit_code == 0
                and not result.timed_out
                and result.stdout == b"SWESMITH_OCI_ROOTFS_SANDBOX_OK"
                and not result.stderr
                and proof.is_file()
                and proof.read_bytes() == b"SWESMITH_OCI_ROOTFS_SANDBOX_OK"
            )
            if not valid:
                raise SwesmithSandboxError(
                    "SWE-smith OCI-rootfs sandbox preflight failed: "
                    f"exit={result.exit_code} timeout={result.timed_out} "
                    f"stdout={result.stdout!r} stderr={result.stderr!r}"
                )

    def attach_workspace(self, workspace_root: Path | str) -> WorkspaceTreeSnapshot:
        self._require_open()
        if self._workspace_root is not None:
            raise SwesmithSandboxError("a SWE-smith sandbox may attach one workspace only")
        root = _require_workspace_root(Path(workspace_root))
        info = os.stat(root, follow_symlinks=False)
        if info.st_uid != self.model_uid or info.st_gid != self.model_gid:
            raise SwesmithSandboxError(
                "workspace root must be owned by the episode's leased UID/GID"
            )
        snapshot = snapshot_workspace_tree(root, self.limits)
        self._workspace_root = root
        self._snapshot = snapshot
        return snapshot

    def refresh_after_host_mutation(self) -> WorkspaceDiff:
        """Validate and attest a trusted host-side mutation such as apply_patch."""

        root, before = self._attached_state()
        try:
            after = snapshot_workspace_tree(root, self.limits, previous=before)
        except BaseException as exc:
            self._poison(f"workspace validation failed after host mutation: {exc}")
            raise
        diff = diff_workspace_trees(before, after)
        self._snapshot = after
        return diff

    def run(
        self,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
    ) -> SwesmithShellExecution:
        root, expected_before = self._attached_state()
        if not isinstance(command, str) or not command or "\x00" in command:
            raise SwesmithSandboxError("shell_command must be non-empty text without NUL")
        normalized_workdir = _normalize_workdir(workdir)
        if type(timeout_ms) is not int or not 0 < timeout_ms <= self.limits.max_timeout_ms:
            raise SwesmithSandboxError(
                f"shell_command timeout must be within 1..{self.limits.max_timeout_ms} ms"
            )
        _require_resolved_workdir(root, normalized_workdir)
        if not self._run_lock.acquire(blocking=False):
            raise SwesmithSandboxError(
                "commands within one SWE-smith episode must execute sequentially"
            )
        try:
            before = snapshot_workspace_tree(root, self.limits, previous=expected_before)
            if before.tree_sha256 != expected_before.tree_sha256:
                self._poison("workspace changed outside the attested episode action path")
                raise SwesmithSandboxError(self._poisoned_reason or "workspace changed")
            result = self._run_namespace(
                root,
                command=command,
                workdir=normalized_workdir,
                timeout_ms=timeout_ms,
            )
            try:
                after = snapshot_workspace_tree(root, self.limits, previous=before)
            except BaseException as exc:
                self._poison(f"workspace validation failed after shell_command: {exc}")
                raise
            diff = diff_workspace_trees(before, after)
            self._snapshot = after
            return SwesmithShellExecution(
                result=result,
                workspace_before=before,
                workspace_after=after,
                workspace_diff=diff,
            )
        finally:
            self._run_lock.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._workspace_root = None
        self._snapshot = None
        context = self._uid_lease_context
        self._uid_lease_context = None
        if context is not None:
            context.__exit__(None, None, None)

    def _attached_state(self) -> tuple[Path, WorkspaceTreeSnapshot]:
        self._require_open()
        if self._poisoned_reason is not None:
            raise SwesmithSandboxError(
                "SWE-smith episode sandbox is poisoned: " + self._poisoned_reason
            )
        if self._workspace_root is None or self._snapshot is None:
            raise SwesmithSandboxError("SWE-smith episode sandbox has no workspace")
        return self._workspace_root, self._snapshot

    def _require_open(self) -> None:
        if self._closed:
            raise SwesmithSandboxError("SWE-smith episode sandbox is closed")

    def _poison(self, reason: str) -> None:
        if self._poisoned_reason is None:
            self._poisoned_reason = reason

    def _run_namespace(
        self,
        workspace_root: Path,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
    ) -> ShellExecutionResult:
        assert_executable_fingerprint(self.rg_binary, self.rg_fingerprint, "ripgrep")
        identity = self._oci_rootfs_identity
        if identity is None:
            raise SwesmithSandboxError(
                "formal SWE-smith execution requires a validated OCI rootfs"
            )
        _attest_oci_rootfs_identity(identity)
        parent = workspace_root.parent
        with _temporary_sandbox_directory(
            prefix=".swesmith-sandbox-root-"
        ) as rootfs, _temporary_sandbox_directory(
            prefix=".swesmith-sandbox-output-"
        ) as output:
            self._prepare_rootfs(rootfs, output)
            started = time.monotonic()
            process = subprocess.Popen(
                [
                    str(self._binaries["unshare"]),
                    "--mount",
                    "--pid",
                    "--fork",
                    "--net",
                    "--ipc",
                    "--uts",
                    str(self._binaries["bash"]),
                    "-c",
                    _DIRECT_BIND_NAMESPACE_SETUP,
                    "swesmith-sandbox",
                    str(rootfs),
                    str(identity.rootfs),
                    str(workspace_root),
                    str(output),
                    str(self.rg_binary),
                    command,
                    workdir,
                    str(self.limits.tmp_bytes),
                    str(self.limits.tmp_inodes),
                    str(self.limits.cpu_seconds),
                    str(self.limits.address_space_bytes),
                    str(self.limits.max_processes),
                    str(self.limits.max_open_files),
                    str(self.limits.max_file_bytes),
                    str(self.model_uid),
                    str(self._binaries["mount"]),
                    str(self._binaries["chroot"]),
                    str(self._binaries["hostname"]),
                    str(self._binaries["mknod"]),
                ],
                cwd=parent,
                env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            stdout, stderr, stdout_truncated, stderr_truncated, timed_out = (
                _collect_bounded_output(
                    process,
                    stdout_limit=self.limits.stdout_bytes,
                    stderr_limit=self.limits.stderr_bytes,
                    timeout_ms=timeout_ms,
                )
            )
            elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
            termination_reason: str | None = None
            if timed_out:
                termination_reason = "wall_timeout"
                exit_code = 124
            else:
                status = output / "status"
                cleanup = output / "cleanup"
                if (
                    not status.is_file()
                    or not cleanup.is_file()
                    or cleanup.read_text(encoding="ascii") != "complete\n"
                ):
                    detail = stderr.decode("utf-8", errors="replace")[-2000:]
                    raise SwesmithSandboxError(
                        "SWE-smith sandbox did not attest command cleanup: " + detail
                    )
                try:
                    exit_code = int(status.read_text(encoding="ascii").strip())
                except (OSError, ValueError) as exc:
                    raise SwesmithSandboxError(
                        "SWE-smith sandbox emitted an invalid command status"
                    ) from exc
            if stdout_truncated or stderr_truncated:
                termination_reason = termination_reason or "output_limit"
            return ShellExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                elapsed_ms=elapsed_ms,
                timed_out=timed_out,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                termination_reason=termination_reason,
                sandbox_contract=SWE_SHELL_SANDBOX_CONTRACT,
                model_uid=self.model_uid,
            )

    def _prepare_rootfs(self, rootfs: Path, output: Path) -> None:
        # The temporary directory becomes a read-only bind mount of the
        # validated profile rootfs inside the namespace.  Do not populate it
        # from the training container: that would silently change dependencies.
        os.chmod(output, 0o700)
        os.chmod(rootfs, 0o700)


@contextmanager
def _temporary_sandbox_directory(*, prefix: str) -> Iterator[Path]:
    # Namespace mountpoints and status files are control-plane scratch. Keeping
    # them off the NFS episode tree avoids .nfs placeholders when a mount or
    # file descriptor finishes closing just after the launcher exits.
    path = Path(
        tempfile.mkdtemp(prefix=prefix, dir=_LOCAL_SANDBOX_SCRATCH_ROOT)
    )
    try:
        yield path
    finally:
        _remove_temporary_sandbox_directory(path)


def _remove_temporary_sandbox_directory(
    path: Path,
    *,
    timeout_seconds: float = _SANDBOX_CLEANUP_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    last_error: OSError | None = None
    while True:
        try:
            shutil.rmtree(path)
        except OSError as exc:
            if not path.exists():
                return
            if exc.errno not in _TRANSIENT_SANDBOX_CLEANUP_ERRNOS:
                raise SwesmithSandboxError(
                    f"SWE-smith sandbox scratch cleanup failed for {path}: {exc}"
                ) from exc
            last_error = exc
        else:
            return

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            try:
                residue = sorted(entry.name for entry in path.iterdir())[:16]
            except OSError as exc:
                residue = [f"<unreadable errno={exc.errno}>"]
            raise SwesmithSandboxError(
                "SWE-smith sandbox scratch cleanup stayed busy after "
                f"{timeout_seconds:.3f}s: path={path} residue={residue} "
                f"last_error={last_error}"
            ) from last_error
        time.sleep(min(_SANDBOX_CLEANUP_RETRY_SECONDS, remaining))


def snapshot_workspace_tree(
    root: Path | str,
    limits: ShellSandboxLimits,
    *,
    previous: WorkspaceTreeSnapshot | None = None,
) -> WorkspaceTreeSnapshot:
    root_path = _require_workspace_root(Path(root))
    previous_entries = {} if previous is None else {entry.path: entry for entry in previous.entries}
    entries: list[WorkspaceEntry] = []
    regular_file_count = 0
    symlink_count = 0
    directory_count = 0
    total_bytes = 0

    for current, directory_names, file_names in os.walk(
        root_path,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        traversed_directories: list[str] = []
        for name in directory_names:
            path = current_path / name
            info = os.lstat(path)
            relative = _relative_evidence_path(root_path, path, limits)
            if stat.S_ISLNK(info.st_mode):
                entries.append(_symlink_entry(root_path, path, relative, info))
                symlink_count += 1
                continue
            if not stat.S_ISDIR(info.st_mode):
                raise SwesmithSandboxError(
                    f"workspace directory entry is not a real directory: {relative}"
                )
            _reject_privileged_mode(info.st_mode, relative)
            entries.append(_entry_from_stat(relative, "directory", info))
            directory_count += 1
            traversed_directories.append(name)
        directory_names[:] = traversed_directories

        for name in file_names:
            path = current_path / name
            info = os.lstat(path)
            relative = _relative_evidence_path(root_path, path, limits)
            if stat.S_ISLNK(info.st_mode):
                entries.append(_symlink_entry(root_path, path, relative, info))
                symlink_count += 1
                continue
            if not stat.S_ISREG(info.st_mode):
                raise SwesmithSandboxError(
                    f"workspace contains a socket, device, FIFO, or unsupported entry: {relative}"
                )
            if info.st_nlink != 1:
                raise SwesmithSandboxError(f"workspace contains a hard-linked file: {relative}")
            _reject_privileged_mode(info.st_mode, relative)
            if info.st_size > limits.max_file_bytes:
                raise SwesmithSandboxError(
                    f"workspace file exceeds the per-file byte limit: {relative}"
                )
            old = previous_entries.get(relative)
            digest = (
                old.sha256
                if old is not None and _same_regular_file_fingerprint(old, info)
                else _file_sha256(path)
            )
            entries.append(
                _entry_from_stat(
                    relative,
                    "file",
                    info,
                    size=info.st_size,
                    sha256=digest,
                )
            )
            regular_file_count += 1
            total_bytes += info.st_size

    entries.sort(key=lambda entry: entry.path)
    inode_count = 1 + len(entries)
    if regular_file_count + symlink_count > limits.max_files:
        raise SwesmithSandboxError("workspace exceeded its file-count limit")
    if directory_count > limits.max_directories:
        raise SwesmithSandboxError("workspace exceeded its directory-count limit")
    if inode_count > limits.workspace_inodes:
        raise SwesmithSandboxError("workspace exceeded its inode limit")
    if total_bytes > limits.workspace_bytes:
        raise SwesmithSandboxError("workspace exceeded its aggregate byte limit")
    payload = [entry.evidence() for entry in entries]
    tree_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return WorkspaceTreeSnapshot(
        entries=tuple(entries),
        regular_file_count=regular_file_count,
        symlink_count=symlink_count,
        directory_count=directory_count,
        inode_count=inode_count,
        total_bytes=total_bytes,
        tree_sha256=tree_sha256,
    )


def diff_workspace_trees(
    before: WorkspaceTreeSnapshot,
    after: WorkspaceTreeSnapshot,
) -> WorkspaceDiff:
    before_by_path = {entry.path: entry for entry in before.entries}
    after_by_path = {entry.path: entry for entry in after.entries}
    added = tuple(
        after_by_path[path].evidence()
        for path in sorted(after_by_path.keys() - before_by_path.keys())
    )
    deleted = tuple(
        before_by_path[path].evidence()
        for path in sorted(before_by_path.keys() - after_by_path.keys())
    )
    modified = tuple(
        {
            "path": path,
            "before": before_by_path[path].evidence(),
            "after": after_by_path[path].evidence(),
        }
        for path in sorted(before_by_path.keys() & after_by_path.keys())
        if before_by_path[path].evidence() != after_by_path[path].evidence()
    )
    return WorkspaceDiff(
        added=added,
        modified=modified,
        deleted=deleted,
        before_tree_sha256=before.tree_sha256,
        after_tree_sha256=after.tree_sha256,
    )


def _entry_from_stat(
    relative: str,
    kind: str,
    info: os.stat_result,
    *,
    size: int = 0,
    sha256: str | None = None,
    link_target: str | None = None,
) -> WorkspaceEntry:
    return WorkspaceEntry(
        path=relative,
        kind=kind,
        mode=stat.S_IMODE(info.st_mode),
        size=size,
        sha256=sha256,
        link_target=link_target,
        device=info.st_dev,
        inode=info.st_ino,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
    )


def _symlink_entry(
    root: Path,
    path: Path,
    relative: str,
    info: os.stat_result,
) -> WorkspaceEntry:
    target = os.readlink(path)
    _validate_symlink_target(relative, target)
    encoded = target.encode("utf-8")
    return _entry_from_stat(
        relative,
        "symlink",
        info,
        size=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        link_target=target,
    )


def _validate_symlink_target(relative: str, target: str) -> None:
    if not target or "\x00" in target:
        raise SwesmithSandboxError(f"workspace symlink has an invalid target: {relative}")
    target_path = PurePosixPath(target)
    if target_path.is_absolute():
        raise SwesmithSandboxError(f"workspace contains an absolute symlink: {relative}")
    parts = list(PurePosixPath(relative).parent.parts)
    for part in target_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise SwesmithSandboxError(
                    f"workspace symlink target escapes the repository: {relative}"
                )
            parts.pop()
        else:
            parts.append(part)


def _same_regular_file_fingerprint(old: WorkspaceEntry, info: os.stat_result) -> bool:
    return (
        old.kind == "file"
        and old.device == info.st_dev
        and old.inode == info.st_ino
        and old.mode == stat.S_IMODE(info.st_mode)
        and old.size == info.st_size
        and old.mtime_ns == info.st_mtime_ns
        and old.ctime_ns == info.st_ctime_ns
        and old.sha256 is not None
    )


def _relative_evidence_path(
    root: Path,
    path: Path,
    limits: ShellSandboxLimits,
) -> str:
    relative = path.relative_to(root).as_posix()
    try:
        relative.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SwesmithSandboxError("workspace paths must be valid UTF-8") from exc
    if len(relative) > limits.max_path_chars:
        raise SwesmithSandboxError(
            f"workspace path exceeds the character limit: {relative!r}"
        )
    return relative


def _reject_privileged_mode(mode: int, relative: str) -> None:
    if mode & (stat.S_ISUID | stat.S_ISGID):
        raise SwesmithSandboxError(
            f"workspace entry may not carry setuid or setgid bits: {relative}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise SwesmithSandboxError(f"cannot attest workspace file: {path}") from exc
    return digest.hexdigest()


def _require_workspace_root(path: Path) -> Path:
    expanded = path.expanduser()
    try:
        info = os.lstat(expanded)
    except OSError as exc:
        raise SwesmithSandboxError("workspace root must be a real directory") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise SwesmithSandboxError("workspace root must be a real directory")
    return expanded.resolve(strict=True)


def _normalize_workdir(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise SwesmithSandboxError("shell_command workdir must be non-empty text")
    if raw in {".", "/testbed"}:
        return "."
    relative = raw.removeprefix("/testbed/") if raw.startswith("/testbed/") else raw
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or str(path) != relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise SwesmithSandboxError(
            "shell_command workdir must be . or a normalized path inside /testbed"
        )
    return str(path)


def _require_resolved_workdir(root: Path, relative: str) -> None:
    candidate = root if relative == "." else root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise SwesmithSandboxError(
            "shell_command workdir must resolve to a directory inside the workspace"
        ) from exc
    if not resolved.is_dir():
        raise SwesmithSandboxError(
            "shell_command workdir must resolve to a workspace directory"
        )


def _executable_version(path: Path) -> str:
    completed = subprocess.run(
        [str(path), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    first_line = (completed.stdout or completed.stderr).splitlines()
    if completed.returncode != 0 or not first_line:
        raise SwesmithSandboxError(f"cannot read executable version: {path}")
    return first_line[0]


def load_oci_rootfs_identity(
    cache_root: Path | str,
    *,
    expected_image: str,
    expected_digest: str,
    expected_owner_uid: int | None = None,
) -> OciRootfsIdentity:
    """Load and validate one complete digest-pinned profile rootfs cache.

    This function intentionally performs no network fallback and never accepts
    a partially materialized cache.  A missing image must be prepared by the
    separate image-cache step before a formal episode can start.
    """

    if not isinstance(expected_image, str) or not expected_image.strip():
        raise SwesmithSandboxError("SWE-smith repo profile image must be non-empty")
    digest = _normalize_image_digest(expected_digest)
    cache_root_path = _require_secure_directory(
        Path(cache_root), "OCI rootfs cache root", expected_owner_uid
    )
    cache_dir = cache_root_path / f"sha256-{digest.removeprefix('sha256:')}"
    _require_secure_directory(cache_dir, "OCI rootfs cache directory", expected_owner_uid)
    complete = cache_dir / ".complete"
    _require_secure_regular_file(complete, "OCI rootfs completion marker", expected_owner_uid)
    if complete.read_text(encoding="ascii") != "complete\n":
        raise SwesmithSandboxError(
            "OCI rootfs cache is incomplete; refusing to run with a partial image"
        )
    metadata_path = cache_dir / "metadata.json"
    manifest_path = cache_dir / "manifest.json"
    config_path = cache_dir / "config.json"
    for path, label in (
        (metadata_path, "OCI rootfs metadata"),
        (manifest_path, "OCI rootfs manifest"),
        (config_path, "OCI rootfs config"),
    ):
        _require_secure_regular_file(path, label, expected_owner_uid)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SwesmithSandboxError("OCI rootfs cache metadata is not valid JSON") from exc
    if not isinstance(metadata, Mapping) or metadata.get("schema") != OCI_ROOTFS_CACHE_SCHEMA:
        raise SwesmithSandboxError("OCI rootfs cache has an unsupported metadata schema")
    if metadata.get("resolved_digest") != digest:
        raise SwesmithSandboxError("OCI rootfs metadata digest does not match the requested digest")
    if metadata.get("repo_profile_image") != expected_image:
        raise SwesmithSandboxError(
            "OCI rootfs metadata profile image does not match the dataset profile"
        )
    manifest_hash = _file_sha256(manifest_path)
    if manifest_hash != digest.removeprefix("sha256:"):
        raise SwesmithSandboxError(
            "OCI rootfs manifest bytes do not hash to the requested image digest"
        )
    if metadata.get("manifest_sha256") != manifest_hash:
        raise SwesmithSandboxError("OCI rootfs metadata has a stale manifest hash")
    if not isinstance(manifest.get("config"), Mapping):
        raise SwesmithSandboxError("OCI rootfs manifest has no config descriptor")
    config_digest = manifest["config"].get("digest")
    if config_digest != f"sha256:{_file_sha256(config_path)}":
        raise SwesmithSandboxError("OCI rootfs config bytes do not match the manifest descriptor")
    if metadata.get("config_sha256") != _file_sha256(config_path):
        raise SwesmithSandboxError("OCI rootfs metadata has a stale config hash")
    if config.get("architecture") != "amd64" or config.get("os") != "linux":
        raise SwesmithSandboxError("SWE-smith OCI rootfs must be a Linux amd64 image")
    config_spec = config.get("config")
    if not isinstance(config_spec, Mapping) or config_spec.get("WorkingDir") != "/testbed":
        raise SwesmithSandboxError("SWE-smith OCI rootfs must use /testbed as WorkingDir")
    rootfs = _require_secure_directory(cache_dir / "rootfs", "OCI rootfs tree", expected_owner_uid)
    if stat.S_IMODE(os.stat(rootfs, follow_symlinks=False).st_mode) & 0o555 != 0o555:
        raise SwesmithSandboxError(
            "OCI rootfs top directory must be traversable by the unprivileged policy"
        )
    for relative in ("/testbed", "/tmp", "/var/tmp", "/dev", "/proc", "/run"):
        _resolve_rootfs_directory(rootfs, relative)
    rootfs_meta = metadata.get("rootfs")
    if not isinstance(rootfs_meta, Mapping):
        raise SwesmithSandboxError("OCI rootfs metadata has no rootfs measurements")
    try:
        rootfs_bytes = int(rootfs_meta["bytes"])
        rootfs_regular_files = int(rootfs_meta["regular_files"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SwesmithSandboxError("OCI rootfs measurements are invalid") from exc
    if rootfs_bytes <= 0 or rootfs_regular_files <= 0:
        raise SwesmithSandboxError("OCI rootfs measurements must be positive")

    required_paths = (
        "/bin/bash",
        "/usr/bin/setpriv",
        "/usr/bin/prlimit",
        "/usr/bin/env",
        "/bin/sleep",
        "/usr/bin/cut",
    )
    resolved_paths: list[tuple[str, Path]] = []
    for relative in required_paths:
        resolved_paths.append((relative, _resolve_rootfs_file(rootfs, relative)))
    bash_sha256 = executable_sha256(resolved_paths[0][1])
    if bash_sha256 != rootfs_meta.get("bash_sha256"):
        raise SwesmithSandboxError("OCI rootfs /bin/bash hash does not match metadata")
    key_fingerprints = tuple(
        (relative, executable_fingerprint(path)) for relative, path in resolved_paths
    )
    return OciRootfsIdentity(
        cache_dir=cache_dir,
        rootfs=rootfs,
        image=expected_image,
        digest=digest,
        config_sha256=_file_sha256(config_path),
        manifest_sha256=manifest_hash,
        architecture="amd64",
        operating_system="linux",
        working_dir="/testbed",
        rootfs_bytes=rootfs_bytes,
        rootfs_regular_files=rootfs_regular_files,
        bash_sha256=bash_sha256,
        key_fingerprints=key_fingerprints,
    )


def _attest_oci_rootfs_identity(identity: OciRootfsIdentity) -> None:
    cache_dir = identity.cache_dir
    try:
        if (cache_dir / ".complete").read_text(encoding="ascii") != "complete\n":
            raise SwesmithSandboxError("OCI rootfs completion marker changed")
        manifest_path = cache_dir / "manifest.json"
        config_path = cache_dir / "config.json"
        if _file_sha256(manifest_path) != identity.manifest_sha256:
            raise SwesmithSandboxError("OCI rootfs manifest changed after attestation")
        if _file_sha256(config_path) != identity.config_sha256:
            raise SwesmithSandboxError("OCI rootfs config changed after attestation")
        for relative, fingerprint in identity.key_fingerprints:
            path = _resolve_rootfs_file(identity.rootfs, relative)
            assert_executable_fingerprint(path, fingerprint, f"OCI rootfs {relative}")
    except SwesmithSandboxError:
        raise
    except (OSError, ShellSandboxError) as exc:
        raise SwesmithSandboxError(
            f"OCI rootfs cannot be revalidated before execution: {exc}"
        ) from exc


def _normalize_image_digest(value: str) -> str:
    if not isinstance(value, str) or _SHA256_DIGEST_RE.fullmatch(value) is None:
        raise SwesmithSandboxError("SWE-smith OCI image digest must be sha256:<64 lowercase hex>")
    return value


def _require_secure_directory(
    path: Path,
    label: str,
    expected_owner_uid: int | None,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise SwesmithSandboxError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SwesmithSandboxError(f"{label} must be a real directory: {path}")
    if expected_owner_uid is not None and info.st_uid != expected_owner_uid:
        raise SwesmithSandboxError(f"{label} has unexpected owner: {path}")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise SwesmithSandboxError(f"{label} is writable by group or other: {path}")
    return path


def _require_secure_regular_file(
    path: Path,
    label: str,
    expected_owner_uid: int | None,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise SwesmithSandboxError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SwesmithSandboxError(f"{label} must be a regular file: {path}")
    if expected_owner_uid is not None and info.st_uid != expected_owner_uid:
        raise SwesmithSandboxError(f"{label} has unexpected owner: {path}")
    if stat.S_IMODE(info.st_mode) & 0o022:
        raise SwesmithSandboxError(f"{label} is writable by group or other: {path}")
    return path


def _resolve_rootfs_file(rootfs: Path, absolute_path: str) -> Path:
    candidate = rootfs / absolute_path.lstrip("/")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(rootfs.resolve(strict=True))
        info = os.stat(resolved, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise SwesmithSandboxError(
            f"OCI rootfs required runtime file is missing or escapes the rootfs: {absolute_path}"
        ) from exc
    if not stat.S_ISREG(info.st_mode) or not (stat.S_IMODE(info.st_mode) & 0o111):
        raise SwesmithSandboxError(
            f"OCI rootfs required runtime file is not executable: {absolute_path}"
        )
    return resolved


def _resolve_rootfs_directory(rootfs: Path, absolute_path: str) -> Path:
    candidate = rootfs / absolute_path.lstrip("/")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(rootfs.resolve(strict=True))
        info = os.stat(resolved, follow_symlinks=False)
    except (OSError, ValueError) as exc:
        raise SwesmithSandboxError(
            f"OCI rootfs required runtime directory is missing or escapes the rootfs: {absolute_path}"
        ) from exc
    if not stat.S_ISDIR(info.st_mode):
        raise SwesmithSandboxError(
            f"OCI rootfs required runtime path is not a directory: {absolute_path}"
        )
    return resolved


_DIRECT_BIND_NAMESPACE_SETUP = r"""
set -eu
mount_root=$1
oci_rootfs=$2
workspace=$3
output=$4
rg_binary=$5
model_command=$6
model_workdir=$7
tmp_bytes=$8
tmp_inodes=$9
cpu_seconds=${10}
address_space_bytes=${11}
max_processes=${12}
max_open_files=${13}
max_file_bytes=${14}
model_uid=${15}
mount_binary=${16}
chroot_binary=${17}
hostname_binary=${18}
mknod_binary=${19}

"$mount_binary" --make-rprivate /
"$mount_binary" --bind "$oci_rootfs" "$mount_root"
"$mount_binary" -o remount,bind,ro,nosuid,nodev "$mount_root"

# The profile image is immutable.  Only the episode workspace and explicitly
# bounded temporary mounts are writable inside this namespace.
"$mount_binary" -t tmpfs -o "mode=1777,nosuid,nodev,size=$tmp_bytes,nr_inodes=$tmp_inodes" tmpfs "$mount_root/tmp"
"$mount_binary" --bind "$mount_root/tmp" "$mount_root/var/tmp"
"$mount_binary" -o remount,bind,rw,nosuid,nodev "$mount_root/var/tmp"
"$mount_binary" -t tmpfs -o mode=0755,nosuid tmpfs "$mount_root/dev"
mkdir -p "$mount_root/dev/shm"
"$mount_binary" -t tmpfs -o "mode=1777,nosuid,nodev,size=$tmp_bytes,nr_inodes=$tmp_inodes" tmpfs "$mount_root/dev/shm"
"$mknod_binary" -m 666 "$mount_root/dev/null" c 1 3
"$mknod_binary" -m 666 "$mount_root/dev/zero" c 1 5
"$mknod_binary" -m 444 "$mount_root/dev/random" c 1 8
"$mknod_binary" -m 444 "$mount_root/dev/urandom" c 1 9
"$mount_binary" -t proc -o nosuid,nodev,noexec,hidepid=2 proc "$mount_root/proc"
"$mount_binary" -t tmpfs -o mode=0755,nosuid,nodev tmpfs "$mount_root/run"
mkdir -p "$mount_root/run/out" "$mount_root/run/tools"
chmod 0700 "$mount_root/run/out"
chmod 0755 "$mount_root/run/tools"
"$mount_binary" --bind "$output" "$mount_root/run/out"
"$mount_binary" -o remount,bind,rw,nosuid,nodev,noexec "$mount_root/run/out"
touch "$mount_root/run/tools/rg"
chmod 0755 "$mount_root/run/tools/rg"
"$mount_binary" --bind "$rg_binary" "$mount_root/run/tools/rg"
"$mount_binary" -o remount,bind,ro,nosuid,nodev "$mount_root/run/tools/rg"
"$mount_binary" --bind "$workspace" "$mount_root/testbed"
"$mount_binary" -o remount,bind,rw,nosuid,nodev "$mount_root/testbed"
"$hostname_binary" swesmith-sandbox

exec "$chroot_binary" "$mount_root" /bin/bash -c '
set -u
model_command=$1
model_workdir=$2
cpu_seconds=$3
address_space_bytes=$4
max_processes=$5
max_open_files=$6
max_file_bytes=$7
model_uid=$8
setpriv_binary=/usr/bin/setpriv
prlimit_binary=/usr/bin/prlimit
env_binary=/usr/bin/env
bash_binary=/bin/bash
sleep_binary=/bin/sleep
if [ "$model_workdir" = . ]; then
    model_workdir=/testbed
else
    model_workdir=/testbed/$model_workdir
fi
set +e
"$setpriv_binary" \
    --no-new-privs \
    --reuid="$model_uid" --regid="$model_uid" \
    --clear-groups --bounding-set=-all --inh-caps=-all --ambient-caps=-all \
    "$prlimit_binary" \
    --cpu="$cpu_seconds" \
    --as="$address_space_bytes" \
    --nproc="$max_processes" \
    --nofile="$max_open_files" \
    --fsize="$max_file_bytes" \
    -- \
    "$env_binary" -i \
    HOME=/testbed PATH=/run/tools:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C LC_ALL=C TMPDIR=/tmp \
    "$bash_binary" --noprofile --norc -c '\''
        umask 077
        cd -- "$1"
        exec "$3" --noprofile --norc -c "$2"
    '\'' swesmith-command "$model_workdir" "$model_command" "$bash_binary"
command_exit=$?
kill -TERM -1 2>/dev/null || true
"$sleep_binary" 0.05
kill -KILL -1 2>/dev/null || true
printf "%s\n" "$command_exit" > /run/out/status
printf "complete\n" > /run/out/cleanup
exit 0
' swesmith-inner "$model_command" "$model_workdir" "$cpu_seconds" "$address_space_bytes" "$max_processes" "$max_open_files" "$max_file_bytes" "$model_uid"
"""
