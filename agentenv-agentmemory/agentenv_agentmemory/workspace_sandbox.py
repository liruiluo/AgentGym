from __future__ import annotations

import hashlib
import fcntl
import os
import platform
import secrets
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Protocol


SHELL_SANDBOX_CONTRACT = "linux_namespace_chroot_tmpfs_v1"
_UID_LEASE_ROOT = Path("/run/agentmemorygym-workspace-sandbox-uids")
_UID_LEASE_BASE = 1_500_000_000
_UID_LEASE_SLOTS = 4096


class ShellSandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutableFingerprint:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    def as_metadata(self) -> dict[str, int]:
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True)
class ShellSandboxLimits:
    workspace_bytes: int
    workspace_inodes: int
    max_files: int
    max_directories: int
    max_file_bytes: int
    max_path_chars: int
    default_timeout_ms: int
    max_timeout_ms: int
    cpu_seconds: int
    address_space_bytes: int
    max_processes: int
    max_open_files: int
    stdout_bytes: int
    stderr_bytes: int
    tmp_bytes: int
    tmp_inodes: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in self.__dict__.values()
        ):
            raise ValueError("shell sandbox limits must be positive integers")
        if self.default_timeout_ms > self.max_timeout_ms:
            raise ValueError("default shell timeout cannot exceed the maximum")
        if self.max_file_bytes > self.workspace_bytes:
            raise ValueError("shell file limit cannot exceed workspace capacity")


@dataclass(frozen=True)
class ShellExecutionResult:
    stdout: bytes
    stderr: bytes
    exit_code: int
    elapsed_ms: int
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    termination_reason: str | None
    sandbox_contract: str
    model_uid: int | None = None


class ShellSandbox(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def run(
        self,
        workspace_root: Path,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
    ) -> ShellExecutionResult: ...


@dataclass(frozen=True)
class LinuxNamespaceShellSandbox:
    """Run shell_command in a minimal, networkless Linux namespace.

    The model command never sees the env-server root. A bounded tmpfs is copied
    in before execution and copied to a root-only staging directory afterward.
    The caller validates and atomically installs that staged workspace.
    """

    limits: ShellSandboxLimits
    rg_binary: Path
    expected_rg_sha256: str
    rg_sha256: str
    rg_version: str
    rg_fingerprint: ExecutableFingerprint
    unshare_binary: Path
    bash_binary: Path
    mount_binary: Path
    chroot_binary: Path
    setpriv_binary: Path
    prlimit_binary: Path
    cp_binary: Path
    chown_binary: Path
    hostname_binary: Path
    env_binary: Path
    mkdir_binary: Path
    mknod_binary: Path
    sleep_binary: Path
    capsh_binary: Path

    @classmethod
    def from_environment(
        cls,
        *,
        limits: ShellSandboxLimits,
        rg_binary: Path,
        expected_rg_sha256: str,
        run_preflight: bool = True,
    ) -> LinuxNamespaceShellSandbox:
        if platform.system() != "Linux":
            raise ShellSandboxError(
                "the formal shell_command sandbox requires Linux namespaces"
            )
        if os.geteuid() != 0:
            raise ShellSandboxError(
                "the formal shell_command sandbox launcher must start as root so it can "
                "construct namespaces before dropping privileges"
            )

        def require(name: str) -> Path:
            resolved = shutil.which(name)
            if resolved is None:
                raise ShellSandboxError(
                    f"required shell sandbox executable is missing: {name}"
                )
            return Path(resolved).resolve()

        pinned_rg = _require_executable(rg_binary, "ripgrep")
        expected_rg_sha256 = _normalize_sha256(
            expected_rg_sha256,
            "expected ripgrep SHA256",
        )
        actual_rg_sha256 = executable_sha256(pinned_rg)
        if actual_rg_sha256 != expected_rg_sha256:
            raise ShellSandboxError(
                "ripgrep SHA256 does not match the frozen launcher contract: "
                f"expected {expected_rg_sha256}, got {actual_rg_sha256}"
            )

        sandbox = cls(
            limits=limits,
            rg_binary=pinned_rg,
            expected_rg_sha256=expected_rg_sha256,
            rg_sha256=actual_rg_sha256,
            rg_version=_executable_version(pinned_rg),
            rg_fingerprint=executable_fingerprint(pinned_rg),
            unshare_binary=require("unshare"),
            bash_binary=require("bash"),
            mount_binary=require("mount"),
            chroot_binary=require("chroot"),
            setpriv_binary=require("setpriv"),
            prlimit_binary=require("prlimit"),
            cp_binary=require("cp"),
            chown_binary=require("chown"),
            hostname_binary=require("hostname"),
            env_binary=require("env"),
            mkdir_binary=require("mkdir"),
            mknod_binary=require("mknod"),
            sleep_binary=require("sleep"),
            capsh_binary=require("capsh"),
        )
        if run_preflight:
            sandbox.preflight()
        return sandbox

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "contract": SHELL_SANDBOX_CONTRACT,
            "formal_eligible": True,
            "network": "new_namespace_no_routes",
            "rootfs": "minimal_read_only_system_roots",
            "workspace_mount": "bounded_tmpfs_copy_in_copy_out",
            "shell": "bash_no_profile_no_rc",
            "ripgrep_path": "/tools/rg",
            "ripgrep_sha256": self.rg_sha256,
            "ripgrep_expected_sha256": self.expected_rg_sha256,
            "ripgrep_version": self.rg_version,
            "ripgrep_startup_fingerprint": self.rg_fingerprint.as_metadata(),
            "ripgrep_revalidation": "stat_fingerprint_before_each_command",
            "model_identity": "exclusive_leased_high_uid_per_command",
            "rlimit_nproc_scope": "host_uid_lease_per_concurrent_command",
            "uid_lease_slots": _UID_LEASE_SLOTS,
            "no_new_privileges": True,
            "capability_bounding_set": "empty",
            "process_namespace": True,
            "mount_namespace": True,
            "ipc_namespace": True,
            "uts_namespace": True,
            "resource_limits": asdict(self.limits),
        }

    def preflight(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentmemory-sandbox-preflight-") as raw:
            root = Path(raw) / "workspace"
            root.mkdir(mode=0o700)
            result = self.run(
                root,
                command=(
                    "test \"$(command -v rg)\" = /tools/rg && "
                    "rg --version >/dev/null && "
                    "printf AGENTMEMORY_SHELL_SANDBOX_OK"
                ),
                workdir=".",
                timeout_ms=min(10_000, self.limits.max_timeout_ms),
            )
        if (
            result.exit_code != 0
            or result.timed_out
            or result.stdout != b"AGENTMEMORY_SHELL_SANDBOX_OK"
            or result.stderr
        ):
            raise ShellSandboxError(
                "shell_command sandbox preflight failed: "
                f"exit={result.exit_code} timeout={result.timed_out} "
                f"stdout={result.stdout!r} stderr={result.stderr!r}"
            )

    def run(
        self,
        workspace_root: Path,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
    ) -> ShellExecutionResult:
        if not isinstance(command, str) or not command:
            raise ShellSandboxError("shell_command command must be a non-empty string")
        if "\x00" in command:
            raise ShellSandboxError("shell_command contains a NUL byte")
        workdir = _validate_relative_workdir(workdir)
        if type(timeout_ms) is not int or not 0 < timeout_ms <= self.limits.max_timeout_ms:
            raise ShellSandboxError(
                f"shell_command timeout must be within 1..{self.limits.max_timeout_ms} ms"
            )
        workspace_root = Path(workspace_root).resolve()
        if not workspace_root.is_dir() or workspace_root.is_symlink():
            raise ShellSandboxError("workspace root must be a real directory")
        host_workdir = workspace_root if workdir == "." else workspace_root / workdir
        if (
            not host_workdir.is_dir()
            or host_workdir.is_symlink()
            or not _is_contained(workspace_root, host_workdir)
        ):
            raise ShellSandboxError(
                "shell_command workdir must be an existing real workspace directory"
            )

        assert_executable_fingerprint(
            self.rg_binary,
            self.rg_fingerprint,
            "ripgrep",
        )

        parent = workspace_root.parent
        with _lease_ephemeral_model_uid() as model_uid, tempfile.TemporaryDirectory(
            prefix=".agentmemory-sandbox-root-",
            dir=parent,
        ) as rootfs_raw, tempfile.TemporaryDirectory(
            prefix=".agentmemory-sandbox-output-",
            dir=parent,
        ) as output_raw:
            rootfs = Path(rootfs_raw)
            output = Path(output_raw)
            self._prepare_rootfs(rootfs, output, model_uid=model_uid)
            started = time.monotonic()
            process = subprocess.Popen(
                [
                    str(self.unshare_binary),
                    "--mount",
                    "--pid",
                    "--fork",
                    "--net",
                    "--ipc",
                    "--uts",
                    str(self.bash_binary),
                    "-c",
                    _LINUX_NAMESPACE_SETUP,
                    "agentmemory-sandbox",
                    str(rootfs),
                    str(workspace_root),
                    str(output),
                    str(self.rg_binary),
                    command,
                    workdir,
                    str(self.limits.workspace_bytes),
                    str(self.limits.workspace_inodes),
                    str(self.limits.tmp_bytes),
                    str(self.limits.tmp_inodes),
                    str(self.limits.cpu_seconds),
                    str(self.limits.address_space_bytes),
                    str(self.limits.max_processes),
                    str(self.limits.max_open_files),
                    str(self.limits.max_file_bytes),
                    str(model_uid),
                    str(self.mount_binary),
                    str(self.chroot_binary),
                    str(self.cp_binary),
                    str(self.chown_binary),
                    str(self.hostname_binary),
                    str(self.mknod_binary),
                    str(self.setpriv_binary),
                    str(self.prlimit_binary),
                    str(self.env_binary),
                    str(self.bash_binary),
                    str(self.mkdir_binary),
                    str(self.sleep_binary),
                    str(self.capsh_binary),
                ],
                cwd=parent,
                env={
                    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
                    "LC_ALL": "C",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            (
                stdout,
                stderr,
                stdout_truncated,
                stderr_truncated,
                timed_out,
            ) = _collect_bounded_output(
                process,
                stdout_limit=self.limits.stdout_bytes,
                stderr_limit=self.limits.stderr_bytes,
                timeout_ms=timeout_ms,
            )
            termination_reason: str | None = None
            if timed_out:
                termination_reason = "wall_timeout"
            elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
            if stdout_truncated or stderr_truncated:
                termination_reason = termination_reason or "output_limit"

            status_path = output / "status"
            staged_workspace = output / "workspace"
            if timed_out:
                return ShellExecutionResult(
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=124,
                    elapsed_ms=elapsed_ms,
                    timed_out=True,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                    termination_reason=termination_reason,
                    sandbox_contract=SHELL_SANDBOX_CONTRACT,
                    model_uid=model_uid,
                )
            if not status_path.is_file() or not staged_workspace.is_dir():
                detail = stderr.decode("utf-8", errors="replace")[-2000:]
                raise ShellSandboxError(
                    "shell_command sandbox did not produce a committed workspace: "
                    + detail
                )
            try:
                command_exit = int(status_path.read_text(encoding="ascii").strip())
            except (OSError, ValueError) as exc:
                raise ShellSandboxError(
                    "shell_command sandbox emitted an invalid status"
                ) from exc
            _validate_staged_workspace(staged_workspace, self.limits)
            _replace_directory(workspace_root, staged_workspace)
            return ShellExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=124 if timed_out else command_exit,
                elapsed_ms=elapsed_ms,
                timed_out=timed_out,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                termination_reason=termination_reason,
                sandbox_contract=SHELL_SANDBOX_CONTRACT,
                model_uid=model_uid,
            )

    def _prepare_rootfs(
        self,
        rootfs: Path,
        output: Path,
        *,
        model_uid: int,
    ) -> None:
        os.chmod(rootfs, 0o755)
        for relative in (
            "usr",
            "etc",
            "dev",
            "proc",
            "tmp",
            "workspace",
            "run",
            "run/out",
            "tools",
        ):
            (rootfs / relative).mkdir(parents=True, exist_ok=True)
        # TemporaryDirectory and the service process may use umask 077. These
        # static roots must remain traversable after the command drops to its
        # leased unprivileged UID.
        os.chmod(rootfs / "etc", 0o755)
        os.chmod(rootfs / "tools", 0o755)
        os.chmod(rootfs / "run", 0o700)
        os.chmod(output, 0o700)
        (rootfs / "tools/rg").touch(mode=0o755)
        os.chmod(rootfs / "tools/rg", 0o755)
        (rootfs / "etc/ld.so.cache").touch(mode=0o644)
        (rootfs / "etc/passwd").write_text(
            "root:x:0:0:root:/root:/usr/bin/false\n"
            f"agent:x:{model_uid}:{model_uid}:agent:/workspace:/usr/bin/bash\n",
            encoding="ascii",
        )
        (rootfs / "etc/group").write_text(
            f"root:x:0:\nagent:x:{model_uid}:\n",
            encoding="ascii",
        )
        for name in ("bin", "sbin", "lib", "lib64"):
            source = Path("/") / name
            target = rootfs / name
            if source.is_symlink():
                target.symlink_to(os.readlink(source))
            elif source.is_dir():
                target.mkdir()


_LINUX_NAMESPACE_SETUP = r"""
set -eu
rootfs=$1
workspace=$2
output=$3
rg_binary=$4
model_command=$5
model_workdir=$6
workspace_bytes=$7
workspace_inodes=$8
tmp_bytes=$9
tmp_inodes=${10}
cpu_seconds=${11}
address_space_bytes=${12}
max_processes=${13}
max_open_files=${14}
max_file_bytes=${15}
model_uid=${16}
mount_binary=${17}
chroot_binary=${18}
cp_binary=${19}
chown_binary=${20}
hostname_binary=${21}
mknod_binary=${22}
setpriv_binary=${23}
prlimit_binary=${24}
env_binary=${25}
bash_binary=${26}
mkdir_binary=${27}
sleep_binary=${28}
capsh_binary=${29}

"$mount_binary" --make-rprivate /
for name in usr bin sbin lib lib64; do
    source=/$name
    target=$rootfs/$name
    if [ -d "$source" ] && [ ! -L "$target" ]; then
        # A non-recursive bind excludes host submounts instead of accidentally
        # exposing them with their original writable mount flags.
        "$mount_binary" --bind "$source" "$target"
        "$mount_binary" -o remount,bind,ro,nosuid,nodev "$target"
    fi
done
if [ -f /etc/ld.so.cache ]; then
    "$mount_binary" --bind /etc/ld.so.cache "$rootfs/etc/ld.so.cache"
    "$mount_binary" -o remount,bind,ro,nosuid,nodev,noexec "$rootfs/etc/ld.so.cache"
fi
"$mount_binary" --bind "$rg_binary" "$rootfs/tools/rg"
"$mount_binary" -o remount,bind,ro,nosuid,nodev "$rootfs/tools/rg"
"$mount_binary" -t tmpfs -o "mode=0700,nosuid,nodev,size=$workspace_bytes,nr_inodes=$workspace_inodes" tmpfs "$rootfs/workspace"
"$cp_binary" -a "$workspace/." "$rootfs/workspace/"
"$chown_binary" -R "$model_uid:$model_uid" "$rootfs/workspace"
"$mount_binary" -t tmpfs -o "mode=1777,nosuid,nodev,size=$tmp_bytes,nr_inodes=$tmp_inodes" tmpfs "$rootfs/tmp"
"$mount_binary" -t tmpfs -o mode=0755,nosuid tmpfs "$rootfs/dev"
"$mknod_binary" -m 666 "$rootfs/dev/null" c 1 3
"$mknod_binary" -m 666 "$rootfs/dev/zero" c 1 5
"$mknod_binary" -m 444 "$rootfs/dev/random" c 1 8
"$mknod_binary" -m 444 "$rootfs/dev/urandom" c 1 9
"$mount_binary" -t proc -o nosuid,nodev,noexec,hidepid=2 proc "$rootfs/proc"
"$mount_binary" --bind "$output" "$rootfs/run/out"
"$mount_binary" -o remount,bind,rw,nosuid,nodev,noexec "$rootfs/run/out"
"$hostname_binary" agentmemory-sandbox

exec "$chroot_binary" "$rootfs" "$bash_binary" -c '
set -u
model_command=$1
model_workdir=$2
cpu_seconds=$3
address_space_bytes=$4
max_processes=$5
max_open_files=$6
max_file_bytes=$7
model_uid=$8
setpriv_binary=$9
prlimit_binary=${10}
env_binary=${11}
bash_binary=${12}
mkdir_binary=${13}
sleep_binary=${14}
cp_binary=${15}
capsh_binary=${16}
if [ "$model_workdir" = . ]; then
    model_workdir=/workspace
else
    model_workdir=/workspace/$model_workdir
fi
set +e
"$capsh_binary" \
    --keep=1 --drop=all --groups= \
    --gid="$model_uid" --uid="$model_uid" \
    --caps= --keep=0 -- \
    -c '\''exec "$@"'\'' agentmemory-capdrop \
    "$setpriv_binary" --no-new-privs \
    "$prlimit_binary" \
    --cpu="$cpu_seconds" \
    --as="$address_space_bytes" \
    --nproc="$max_processes" \
    --nofile="$max_open_files" \
    --fsize="$max_file_bytes" \
    -- \
    "$env_binary" -i \
    HOME=/workspace PATH=/tools:/usr/bin:/bin \
    LANG=C LC_ALL=C TMPDIR=/tmp \
    "$bash_binary" --noprofile --norc -c '\''
        umask 077
        cd -- "$1"
        exec "$3" --noprofile --norc -c "$2"
    '\'' agentmemory-command "$model_workdir" "$model_command" "$bash_binary"
command_exit=$?
kill -TERM -1 2>/dev/null || true
"$sleep_binary" 0.05
kill -KILL -1 2>/dev/null || true
"$mkdir_binary" -p /run/out/workspace
"$cp_binary" -a /workspace/. /run/out/workspace/
printf "%s\n" "$command_exit" > /run/out/status
exit 0
' agentmemory-inner "$model_command" "$model_workdir" "$cpu_seconds" "$address_space_bytes" "$max_processes" "$max_open_files" "$max_file_bytes" "$model_uid" "$setpriv_binary" "$prlimit_binary" "$env_binary" "$bash_binary" "$mkdir_binary" "$sleep_binary" "$cp_binary" "$capsh_binary"
"""


def _require_executable(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ShellSandboxError(f"{label} executable is not usable: {resolved}")
    info = os.stat(resolved)
    if not stat.S_ISREG(info.st_mode):
        raise ShellSandboxError(f"{label} executable is not a regular file")
    return resolved


def _normalize_sha256(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ShellSandboxError(f"{label} must be a string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ShellSandboxError(f"{label} must be exactly 64 hexadecimal characters")
    return normalized


@contextmanager
def _lease_ephemeral_model_uid(
    lease_root: Path = _UID_LEASE_ROOT,
    *,
    slot_count: int = _UID_LEASE_SLOTS,
) -> Iterator[int]:
    """Lease one host-wide high UID for the lifetime of a command.

    RLIMIT_NPROC is accounted by real UID. A bounded flock bank prevents two
    cooperating env-server processes from accidentally sharing that accounting
    domain while their commands overlap. Lock files are retained so unlink/open
    races cannot create two independently locked inodes for the same UID.
    """

    if type(slot_count) is not int or slot_count <= 0:
        raise ShellSandboxError("UID lease slot count must be a positive integer")
    lease_root = Path(lease_root)
    lease_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root_info = os.lstat(lease_root)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or stat.S_ISLNK(root_info.st_mode)
        or root_info.st_uid != os.geteuid()
        or root_info.st_mode & 0o022
    ):
        raise ShellSandboxError(
            "UID lease root must be a private real directory owned by the launcher"
        )

    start = secrets.randbelow(slot_count)
    for offset in range(slot_count):
        slot = (start + offset) % slot_count
        uid = _UID_LEASE_BASE + slot
        path = lease_root / f"uid-{uid}.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o022
            ):
                raise ShellSandboxError(
                    "UID lease file must be a private regular file owned by the launcher"
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                yield uid
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            return
        finally:
            os.close(descriptor)
    raise ShellSandboxError(
        f"all {slot_count} shell sandbox UID lease slots are in use"
    )


def executable_fingerprint(path: Path) -> ExecutableFingerprint:
    info = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ShellSandboxError(f"executable is not a regular file: {path}")
    return ExecutableFingerprint(
        device=info.st_dev,
        inode=info.st_ino,
        mode=info.st_mode,
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
    )


def assert_executable_fingerprint(
    path: Path,
    expected: ExecutableFingerprint,
    label: str,
) -> None:
    try:
        observed = executable_fingerprint(path)
    except OSError as exc:
        raise ShellSandboxError(
            f"{label} executable cannot be revalidated before command execution"
        ) from exc
    if observed != expected:
        raise ShellSandboxError(
            f"{label} executable changed after startup; restart with a freshly attested pin"
        )


def executable_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executable_version(path: Path) -> str:
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ShellSandboxError(
            f"cannot attest executable version for {path}"
        ) from exc
    first_line = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    if not first_line:
        raise ShellSandboxError(f"empty executable version for {path}")
    return first_line


def _validate_relative_workdir(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ShellSandboxError("shell_command workdir must be a non-empty string")
    if "\x00" in value or "\\" in value or value.startswith(("/", "~")):
        raise ShellSandboxError("shell_command workdir must be workspace-relative")
    if value == ".":
        return value
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ShellSandboxError(
            "shell_command workdir contains an empty, dot, or parent component"
        )
    return path.as_posix()


def _is_contained(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is None:
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
    else:
        # The group leader may exit before descendants that inherited its
        # output pipes. Give TERM a short grace period, then reap the group.
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait(timeout=2.0)


def _collect_bounded_output(
    process: subprocess.Popen[bytes],
    *,
    stdout_limit: int,
    stderr_limit: int,
    timeout_ms: int,
) -> tuple[bytes, bytes, bool, bool, bool]:
    """Drain subprocess pipes without ever buffering unbounded model output."""

    if process.stdout is None or process.stderr is None:  # pragma: no cover - caller invariant.
        raise ShellSandboxError("shell sandbox output pipes were not created")
    selector = selectors.DefaultSelector()
    streams = {
        process.stdout.fileno(): ("stdout", process.stdout, stdout_limit),
        process.stderr.fileno(): ("stderr", process.stderr, stderr_limit),
    }
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}
    for descriptor, (name, stream, _limit) in streams.items():
        os.set_blocking(descriptor, False)
        selector.register(stream, selectors.EVENT_READ, data=name)

    deadline = time.monotonic() + timeout_ms / 1000.0
    process_finished_at: float | None = None
    timed_out = False
    try:
        while selector.get_map() or process.poll() is None:
            now = time.monotonic()
            if process.poll() is None and now >= deadline:
                timed_out = True
                _terminate_process_group(process)
                process_finished_at = time.monotonic()
            elif process.poll() is not None and process_finished_at is None:
                process_finished_at = now
                if selector.get_map():
                    _terminate_process_group(process)

            if process_finished_at is not None and now - process_finished_at > 1.0:
                # A pipe still held open after the namespace launcher exits is
                # evidence of failed descendant cleanup, not valid tool output.
                if selector.get_map():
                    raise ShellSandboxError(
                        "shell sandbox left an output pipe open after process cleanup"
                    )
                break

            wait_seconds = 0.05
            if process.poll() is None:
                wait_seconds = max(0.0, min(wait_seconds, deadline - now))
            events = selector.select(wait_seconds)
            for key, _mask in events:
                descriptor = key.fileobj.fileno()
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                name = key.data
                limit = streams[descriptor][2]
                remaining = max(0, limit - len(buffers[name]))
                if remaining:
                    buffers[name].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[name] = True
        if process.poll() is None:  # pragma: no cover - defensive fallback.
            _terminate_process_group(process)
    finally:
        for key in list(selector.get_map().values()):
            selector.unregister(key.fileobj)
            key.fileobj.close()
        selector.close()

    return (
        bytes(buffers["stdout"]),
        bytes(buffers["stderr"]),
        truncated["stdout"],
        truncated["stderr"],
        timed_out,
    )


def _replace_directory(destination: Path, source: Path) -> None:
    candidate = destination.with_name(
        destination.name + f".candidate-{os.getpid()}-{time.time_ns()}"
    )
    previous = destination.with_name(
        destination.name + f".previous-{os.getpid()}-{time.time_ns()}"
    )
    shutil.copytree(source, candidate, symlinks=True)
    os.chmod(candidate, 0o700)
    os.replace(destination, previous)
    try:
        os.replace(candidate, destination)
    except Exception:
        os.replace(previous, destination)
        raise
    else:
        shutil.rmtree(previous)
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)


def _validate_staged_workspace(
    root: Path,
    limits: ShellSandboxLimits,
) -> None:
    inode_count = 1
    file_count = 0
    directory_count = 0
    total_bytes = 0
    root = root.resolve()
    for current, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        clean_directories = []
        for name in directory_names:
            path = current_path / name
            info = os.lstat(path)
            relative = path.relative_to(root).as_posix()
            if len(relative) > limits.max_path_chars:
                raise ShellSandboxError(
                    "shell_command created a path longer than the workspace limit"
                )
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise ShellSandboxError(
                    "shell_command workspace may contain only real directories and regular files"
                )
            inode_count += 1
            directory_count += 1
            clean_directories.append(name)
        directory_names[:] = clean_directories
        for name in file_names:
            path = current_path / name
            info = os.lstat(path)
            relative = path.relative_to(root).as_posix()
            if len(relative) > limits.max_path_chars:
                raise ShellSandboxError(
                    "shell_command created a path longer than the workspace limit"
                )
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise ShellSandboxError(
                    "shell_command workspace may not contain symlinks, hard links, or special files"
                )
            if info.st_size > limits.max_file_bytes:
                raise ShellSandboxError(
                    "shell_command created a file larger than the workspace limit"
                )
            inode_count += 1
            file_count += 1
            total_bytes += info.st_size
    if file_count > limits.max_files:
        raise ShellSandboxError("shell_command workspace exceeded its file-count limit")
    if directory_count > limits.max_directories:
        raise ShellSandboxError(
            "shell_command workspace exceeded its directory-count limit"
        )
    if inode_count > limits.workspace_inodes:
        raise ShellSandboxError("shell_command workspace exceeded its inode limit")
    if total_bytes > limits.workspace_bytes:
        raise ShellSandboxError("shell_command workspace exceeded its byte limit")
