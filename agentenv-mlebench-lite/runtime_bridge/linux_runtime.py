"""Linux isolation primitives for the MLE-bench Lite runtime bridge.

This module deliberately implements only host isolation, accounting, and
lifecycle mechanics.  It does not know task packages, grading, policy arms,
or MLE/OpenMLE execution semantics.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import math
import os
import platform
import posixpath
import re
import resource
import selectors
import signal
import stat
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    from .runner import (
        BridgeError,
        BundleIdentity,
        ExecutionOutcome,
        LifecycleOutcome,
        RuntimeAttestation,
        canonical_json_bytes,
        canonical_sha256,
        open_bundle_regular_file,
        strict_json_loads,
    )
except ImportError:  # Immutable bundle execution runs runner.py as __main__.
    try:
        from __main__ import (  # type: ignore
            BridgeError,
            BundleIdentity,
            ExecutionOutcome,
            LifecycleOutcome,
            RuntimeAttestation,
            canonical_json_bytes,
            canonical_sha256,
            open_bundle_regular_file,
            strict_json_loads,
        )
    except ImportError:
        from runner import (  # type: ignore
            BridgeError,
            BundleIdentity,
            ExecutionOutcome,
            LifecycleOutcome,
            RuntimeAttestation,
            canonical_json_bytes,
            canonical_sha256,
            open_bundle_regular_file,
            strict_json_loads,
        )


CPU_LIMIT_CORES = 36
MEMORY_LIMIT_BYTES = 440_000_000_000
PIDS_LIMIT = 4096
WRITABLE_BYTES_LIMIT = 500_000_000_000
WRITABLE_INODES_LIMIT = 2_000_000
CGROUP_PERIOD_US = 100_000
CGROUP_QUOTA_US = CPU_LIMIT_CORES * CGROUP_PERIOD_US
CGROUP_CONTROLLERS = ("cpu,cpuacct", "memory", "pids", "devices")
RUNTIME_IDENTITY_SCHEMA = "mlebench_lite_linux_runtime_identity_v1"
SUPERVISOR_STATS_SCHEMA = "mlebench_lite_supervisor_stats_v1"
ROOTFS_LOCK_SCHEMA = "mlebench_lite_rootfs_tree_lock_v1"
ELF_CLOSURE_SCHEMA = "mlebench_lite_rootfs_elf_closure_v1"
RUNTIME_AUDIT_IDENTITY = "mlebench_lite_runtime_audit_v1"
RUNTIME_AUDIT_MARKER = (RUNTIME_AUDIT_IDENTITY + "\n").encode("ascii")
RUNTIME_AUDIT_RELATIVE_PATH = "lib/mlebench-lite-runtime-audit.so"
RUNTIME_AUDIT_STATE_FD = 198
RUNTIME_AUDIT_PROBE_FD = 199
RUNTIME_AUDIT_ROOTFS_FD = 200
ROOTFS_SYMLINK_EXPANSION_LIMIT = 40

CLONE_NEWNS = 0x00020000
MS_RDONLY = 1
MS_NOSUID = 2
MS_NODEV = 4
MS_BIND = 4096
MS_REMOUNT = 32
MS_REC = 16384
MS_PRIVATE = 1 << 18
MNT_DETACH = 2

_CGROUP_NAME = re.compile(r"^mlebridge-[0-9a-f]{32}-[0-9a-f]{32}$")
_EPISODE_CGROUP_NAME = re.compile(r"^mlebridge-[0-9a-f]{32}$")
_OPERATION_CGROUP_COMPONENT = re.compile(r"^[0-9a-f]{32}$")
_BOOT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def operation_cgroup_paths(
    episode_id: str, name: str
) -> dict[str, tuple[str, ...]]:
    prefix = f"mlebridge-{episode_id}-"
    if not name.startswith(prefix) or not _CGROUP_NAME.fullmatch(name):
        raise BridgeError("malformed owned cgroup name")
    component = name.removeprefix(prefix)
    if not _OPERATION_CGROUP_COMPONENT.fullmatch(component):
        raise BridgeError("malformed operation cgroup component")
    episode_memory_name = f"mlebridge-{episode_id}"
    if not _EPISODE_CGROUP_NAME.fullmatch(episode_memory_name):
        raise BridgeError("episode memory cgroup name drifted")
    memory_root = "/sys/fs/cgroup/memory"
    return {
        "cpu,cpuacct": (os.path.join("/sys/fs/cgroup/cpu,cpuacct", name),),
        "memory": (
            os.path.join(memory_root, episode_memory_name, component),
            os.path.join(memory_root, name),
        ),
        "pids": (os.path.join("/sys/fs/cgroup/pids", name),),
        "devices": (os.path.join("/sys/fs/cgroup/devices", name),),
    }


@dataclass(frozen=True)
class MountRecord:
    mount_id: int
    parent_id: int
    device: str
    root: str
    target: str
    mount_options: frozenset[str]
    optional_fields: tuple[str, ...]
    filesystem: str
    source: str
    super_options: frozenset[str]

    @property
    def read_only(self) -> bool:
        return "ro" in self.mount_options


@dataclass
class CgroupSet:
    name: str
    run_dir: str
    mounts: dict[str, str]
    children: dict[str, str]
    episode_memory_cgroup: str


@dataclass
class ExecutionWatchdog:
    pid: int
    control_fd: int

    def finish(self, *, cleanup_complete: bool) -> None:
        try:
            if cleanup_complete:
                os.write(self.control_fd, b"C")
        except OSError:
            pass
        finally:
            os.close(self.control_fd)
        while True:
            try:
                waited, status = os.waitpid(self.pid, 0)
                break
            except InterruptedError:
                continue
        if waited != self.pid or not os.WIFEXITED(status) or os.WEXITSTATUS(status):
            raise BridgeError("execution watchdog cleanup failed")


def verify_audited_runtime_mappings(
    deployment: Mapping[str, Any], bundle_root: str, bundle_fd: int
) -> dict[str, Any]:
    if bundle_root != f"/proc/self/fd/{bundle_fd}":
        raise BridgeError("audited bundle anchor drifted")
    if os.environ.get("MLE_BRIDGE_RUNTIME_AUDIT") != RUNTIME_AUDIT_IDENTITY:
        raise BridgeError("native loader audit identity is absent")
    try:
        audit_state = os.fstat(RUNTIME_AUDIT_STATE_FD)
        marker = os.pread(
            RUNTIME_AUDIT_STATE_FD, len(RUNTIME_AUDIT_MARKER), 0
        )
    except OSError as exc:
        raise BridgeError("native loader audit state is unavailable") from exc
    if not stat.S_ISREG(audit_state.st_mode) or marker != RUNTIME_AUDIT_MARKER:
        raise BridgeError("native loader audit did not initialize")

    closure_descriptor: int | None = None
    audit_descriptor: int | None = None
    try:
        closure_descriptor = open_bundle_regular_file(
            bundle_fd, "rootfs-elf-closure.json", expected_uid=os.geteuid()
        )
        audit_descriptor = open_bundle_regular_file(
            bundle_fd, RUNTIME_AUDIT_RELATIVE_PATH, expected_uid=os.geteuid()
        )
    except (BridgeError, OSError) as exc:
        if closure_descriptor is not None:
            os.close(closure_descriptor)
        raise BridgeError("audited bundle member is unavailable") from exc
    try:
        closure_payload = read_stable_regular_fd(
            closure_descriptor, 2 * 1024 * 1024
        )
        audit_metadata = os.fstat(audit_descriptor)
    finally:
        os.close(closure_descriptor)
        os.close(audit_descriptor)
    try:
        closure = strict_json_loads(closure_payload)
    except ValueError as exc:
        raise BridgeError("rootfs ELF closure is not strict JSON") from exc
    expected_fields = {
        "schema",
        "rootfs_digest",
        "loader",
        "library_paths",
        "cache_inhibited",
        "targets",
        "prebuilt_test_fixture",
    }
    expected_targets = {
        deployment["rootfs_python_path"],
        deployment["rootfs_nvidia_smi_path"],
    }
    targets = closure.get("targets") if isinstance(closure, dict) else None
    if (
        not isinstance(closure, dict)
        or set(closure) != expected_fields
        or closure.get("schema") != ELF_CLOSURE_SCHEMA
        or closure.get("rootfs_digest") != deployment["rootfs_digest"]
        or closure.get("loader") != deployment["rootfs_loader_path"]
        or closure.get("library_paths") != deployment["rootfs_library_paths"]
        or closure.get("cache_inhibited") is not True
        or closure.get("prebuilt_test_fixture") is not False
        or not isinstance(targets, list)
        or len(targets) != 2
        or closure_payload != canonical_json_bytes(closure)
    ):
        raise BridgeError("rootfs ELF closure identity drifted")
    observed_targets = set()
    for target in targets:
        if (
            not isinstance(target, dict)
            or set(target) != {"path", "resolved_objects"}
            or not isinstance(target.get("path"), str)
            or not isinstance(target.get("resolved_objects"), list)
            or not target["resolved_objects"]
            or target["resolved_objects"] != sorted(set(target["resolved_objects"]))
            or any(
                not isinstance(path, str) or not path.startswith("/")
                for path in target["resolved_objects"]
            )
        ):
            raise BridgeError("rootfs ELF closure target drifted")
        observed_targets.add(target["path"])
    if observed_targets != expected_targets:
        raise BridgeError("rootfs ELF closure target inventory drifted")

    maps_payload = read_stable_regular_file("/proc/self/maps", 4 * 1024 * 1024)
    try:
        map_lines = maps_payload.decode("utf-8", "strict").splitlines()
    except UnicodeError as exc:
        raise BridgeError("runtime mapping inventory is not UTF-8") from exc
    objects = validate_audited_runtime_map_lines(
        map_lines,
        deployment=deployment,
        audit_metadata=audit_metadata,
    )
    return {
        "schema": "mlebench_lite_audited_runtime_mappings_v1",
        "runtime_audit": RUNTIME_AUDIT_RELATIVE_PATH,
        "elf_closure_sha256": hashlib.sha256(closure_payload).hexdigest(),
        "mapped_objects_sha256": canonical_sha256(objects),
        "mapped_objects": objects,
    }


def validate_audited_runtime_map_lines(
    map_lines: list[str],
    *,
    deployment: Mapping[str, Any],
    audit_metadata: os.stat_result,
) -> list[dict[str, Any]]:
    rootfs = deployment["rootfs"]
    audit_map_identity = (
        os.major(audit_metadata.st_dev),
        os.minor(audit_metadata.st_dev),
        audit_metadata.st_ino,
    )
    mapped: dict[tuple[int, int], dict[str, Any]] = {}
    mapped_real_paths: set[str] = set()
    audit_mapping_seen = False
    for line in map_lines:
        parts = line.split(maxsplit=5)
        if len(parts) < 5:
            raise BridgeError("runtime mapping inventory is malformed")
        permissions = parts[1]
        if "x" not in permissions or len(parts) == 5:
            continue
        path = parts[5]
        if path.startswith("[") and path.endswith("]"):
            continue
        if (
            not path.startswith("/")
            or path.endswith(" (deleted)")
            or "\\" in path
            or "\x00" in path
        ):
            raise BridgeError("executable runtime mapping has an unsafe path")
        try:
            major_text, minor_text = parts[3].split(":", 1)
            map_device = (int(major_text, 16), int(minor_text, 16))
            map_inode = int(parts[4])
        except ValueError as exc:
            raise BridgeError("executable runtime mapping cannot be attested") from exc
        if (*map_device, map_inode) == audit_map_identity:
            if not stat.S_ISREG(audit_metadata.st_mode) or audit_metadata.st_nlink != 1:
                raise BridgeError("runtime audit mapping identity drifted")
            identity = (audit_metadata.st_dev, audit_metadata.st_ino)
            mapped[identity] = {
                "origin": "bundle",
                "path": RUNTIME_AUDIT_RELATIVE_PATH,
                "device_major": map_device[0],
                "device_minor": map_device[1],
                "inode": map_inode,
            }
            audit_mapping_seen = True
            continue
        real_path = os.path.realpath(path)
        if not is_path_within(real_path, rootfs):
            raise BridgeError("executable runtime mapping escaped sealed roots")
        try:
            metadata = os.stat(real_path, follow_symlinks=False)
        except OSError as exc:
            raise BridgeError("executable runtime mapping cannot be attested") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or map_device != (os.major(metadata.st_dev), os.minor(metadata.st_dev))
            or map_inode != metadata.st_ino
        ):
            raise BridgeError("executable runtime mapping identity drifted")
        identity = (metadata.st_dev, metadata.st_ino)
        mapped[identity] = {
            "origin": "rootfs",
            "path": "/" + os.path.relpath(real_path, rootfs),
            "device_major": os.major(metadata.st_dev),
            "device_minor": os.minor(metadata.st_dev),
            "inode": metadata.st_ino,
        }
        mapped_real_paths.add(real_path)
    required_paths = {
        os.path.realpath(
            os.path.join(
                rootfs, deployment["rootfs_python_path"].lstrip("/")
            )
        ),
        os.path.realpath(
            os.path.join(
                rootfs, deployment["rootfs_loader_path"].lstrip("/")
            )
        ),
    }
    if not required_paths.issubset(mapped_real_paths) or not audit_mapping_seen:
        raise BridgeError("required audited runtime mapping is absent")
    return sorted(mapped.values(), key=lambda item: (item["origin"], item["path"]))


class _LibC:
    def __init__(self) -> None:
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mount.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
        )
        self.libc.mount.restype = ctypes.c_int
        self.libc.umount2.argtypes = (ctypes.c_char_p, ctypes.c_int)
        self.libc.umount2.restype = ctypes.c_int
        self.libc.unshare.argtypes = (ctypes.c_int,)
        self.libc.unshare.restype = ctypes.c_int

    def mount(
        self,
        source: str | None,
        target: str,
        filesystem: str | None,
        flags: int,
        data: str | None = None,
    ) -> None:
        encoded_data = None if data is None else data.encode("utf-8")
        result = self.libc.mount(
            None if source is None else source.encode("utf-8"),
            target.encode("utf-8"),
            None if filesystem is None else filesystem.encode("utf-8"),
            flags,
            encoded_data,
        )
        if result != 0:
            code = ctypes.get_errno()
            raise BridgeError("mount operation failed") from OSError(
                code, os.strerror(code)
            )

    def umount(self, target: str, *, detach: bool = False) -> None:
        if self.libc.umount2(target.encode("utf-8"), MNT_DETACH if detach else 0) != 0:
            code = ctypes.get_errno()
            raise BridgeError("unmount operation failed") from OSError(
                code, os.strerror(code)
            )

    def private_mount_namespace(self) -> None:
        if self.libc.unshare(CLONE_NEWNS) != 0:
            code = ctypes.get_errno()
            raise BridgeError("cannot create private mount namespace") from OSError(
                code, os.strerror(code)
            )
        self.mount(None, "/", None, MS_REC | MS_PRIVATE)


class LinuxRuntime:
    """Concrete Linux facade consumed by :class:`runner.BridgeEngine`."""

    def __init__(
        self,
        bundle: BundleIdentity,
        *,
        operation_started: float,
        operation_deadline: float,
        runtime_mapping_identity: Mapping[str, Any],
    ) -> None:
        if operation_deadline <= operation_started:
            raise BridgeError("operation deadline is already expired")
        if bundle.bundle_fd != 197 or bundle.bundle_root != "/proc/self/fd/197":
            raise BridgeError("runtime bundle anchor drifted")
        try:
            bundle_metadata = os.fstat(bundle.bundle_fd)
        except OSError as exc:
            raise BridgeError("runtime bundle anchor is unavailable") from exc
        if not stat.S_ISDIR(bundle_metadata.st_mode):
            raise BridgeError("runtime bundle anchor is not a directory")
        self.bundle = bundle
        self.deployment = bundle.deployment
        self.bundle_identity_sha256 = canonical_sha256(
            {
                "runner_sha256": bundle.identity.runner_sha256,
                "runtime_digest": bundle.identity.runtime_digest,
            }
        )
        self.operation_started = operation_started
        self.operation_deadline = operation_deadline
        if (
            not isinstance(runtime_mapping_identity, Mapping)
            or runtime_mapping_identity.get("schema")
            != "mlebench_lite_audited_runtime_mappings_v1"
        ):
            raise BridgeError("audited runtime mapping identity drifted")
        self.runtime_mapping_identity = dict(runtime_mapping_identity)
        self._libc = _LibC()

    def _check_operation_deadline(self, stage: str) -> None:
        if time.monotonic() >= self.operation_deadline:
            raise BridgeError(f"operation deadline expired during {stage}")

    def _remaining_time(self, stage: str, *, cap: float | None = None) -> float:
        remaining = self.operation_deadline - time.monotonic()
        if remaining <= 0:
            raise BridgeError(f"operation deadline expired during {stage}")
        return remaining if cap is None else min(remaining, cap)

    def attest(
        self,
        request: Mapping[str, Any],
        state: Mapping[str, Any] | None = None,
    ) -> RuntimeAttestation:
        self._check_operation_deadline("attest validation")
        self._require_host()
        self._validate_request_roots(request)
        rootfs_anchor = self._verify_rootfs_identity()
        self._verify_cgroup_surface()
        rootfs_fd = self._open_anchored_read_only_tree(
            rootfs_anchor, label="rootfs"
        )
        try:
            gpu = self._verify_gpu_identity(rootfs_fd)
        finally:
            os.close(rootfs_fd)
        public_fd, public_digest, public_anchor = self._open_and_hash_public(
            request["public_root"]
        )
        try:
            if public_digest != request["public_tree_sha256"]:
                raise BridgeError("public tree SHA256 drifted")
            expected_identity = None if state is None else state.get("runtime_identity")
            episode_root = os.path.dirname(request["workspace_root"])
            mount_was_absent = self._find_exact_mount(episode_root) is None
            memory_parent_was_absent = not os.path.isdir(
                self._episode_memory_cgroup_path(request["episode_id"])
            )
            try:
                identity = self._attest_episode_mount(
                    request,
                    public_anchor=public_anchor,
                    gpu=gpu,
                    rootfs_anchor=rootfs_anchor,
                    expected_identity=expected_identity,
                )
            except BaseException as original_error:
                cleanup_failures: list[str] = []
                if mount_was_absent:
                    record = self._find_exact_mount(episode_root)
                    if (
                        record is not None
                        and record.filesystem == "tmpfs"
                        and record.source == "mlebridge-" + request["episode_id"]
                    ):
                        try:
                            self._libc.umount(episode_root)
                        except BridgeError:
                            cleanup_failures.append("owned_mount")
                if memory_parent_was_absent:
                    try:
                        self._remove_episode_memory_cgroup(request["episode_id"])
                    except BridgeError:
                        cleanup_failures.append("episode_memory_cgroup")
                if cleanup_failures:
                    raise BridgeError(
                        "failed attestation cleanup left owned residue: "
                        + ",".join(cleanup_failures)
                    ) from original_error
                raise
        finally:
            os.close(public_fd)
        return RuntimeAttestation(
            cpu_limit_cores=CPU_LIMIT_CORES,
            memory_limit_bytes=MEMORY_LIMIT_BYTES,
            pids_limit=PIDS_LIMIT,
            gpu_count=1,
            gpu_uuid=self.deployment["gpu"]["uuid"],
            mount_namespace=True,
            network_disabled=True,
            non_root=True,
            read_only_rootfs=True,
            runtime_identity=identity,
        )

    def execute(
        self, request: Mapping[str, Any], state: Mapping[str, Any]
    ) -> ExecutionOutcome:
        self._check_operation_deadline("execute validation")
        self._require_host()
        identity = self._require_runtime_identity(request, state, frozen=False)
        self._verify_cgroup_surface()
        public_fd = self._open_anchored_read_only_tree(
            identity["public_anchor"], label="public"
        )
        try:
            rootfs_fd = self._open_anchored_read_only_tree(
                identity["rootfs_anchor"], label="rootfs"
            )
        except BaseException:
            os.close(public_fd)
            raise
        try:
            if self._verify_gpu_identity(rootfs_fd) != identity["gpu"]:
                raise BridgeError("GPU runtime identity drifted")
        except BaseException:
            os.close(rootfs_fd)
            os.close(public_fd)
            raise

        opened: list[int] = [public_fd, rootfs_fd]
        cgroups: CgroupSet | None = None
        watchdog: ExecutionWatchdog | None = None
        process: subprocess.Popen[bytes] | None = None
        stats_read: int | None = None
        stats_write: int | None = None
        start_read: int | None = None
        start_write: int | None = None
        selector: selectors.BaseSelector | None = None
        started_ns = time.monotonic_ns()
        timeout_seconds = request["timeout_ms"] / 1000.0
        now = time.monotonic()
        execution_deadline = min(now + timeout_seconds, self.operation_deadline - 4.0)
        cleanup_deadline = self.operation_deadline
        if execution_deadline <= now:
            os.close(rootfs_fd)
            os.close(public_fd)
            raise BridgeError("operation deadline has no execution budget")
        stdout = bytearray()
        stderr = bytearray()
        stats_payload = bytearray()
        output_seen = 0
        output_limited = False
        killed_for_timeout = False
        final_stats: dict[str, int] | None = None
        before_stats: dict[str, int] | None = None
        supervisor: dict[str, Any] | None = None
        try:
            workspace_fd = self._open_episode_child(request, identity, "workspace")
            submission_fd = self._open_episode_child(request, identity, "submission")
            tmp_fd = self._open_episode_child_path(request, identity, "tmp")
            shm_fd = self._open_episode_child_path(request, identity, "shm")
            quota_fd = open_absolute_directory_nofollow(identity["episode_root"])
            opened.extend((workspace_fd, submission_fd, tmp_fd, shm_fd, quota_fd))
            memory_fd = -1
            if request["mode"] == "amg_memory":
                memory_fd = self._open_episode_child(
                    request, identity, "external-memory"
                )
                opened.append(memory_fd)

            device_fds: list[tuple[int, str]] = []
            for device in self._gpu_devices():
                descriptor = self._open_device(device)
                opened.append(descriptor)
                device_fds.append((descriptor, device["target"]))

            self._libc.private_mount_namespace()
            cgroups = self._create_cgroups(request)
            watchdog = self.start_execution_watchdog(cgroups)
            before_stats = self._cgroup_stats(cgroups)
            stats_read, stats_write = os.pipe2(os.O_CLOEXEC)
            start_read, start_write = os.pipe2(os.O_CLOEXEC)
            os.set_inheritable(stats_write, True)
            os.set_inheritable(start_read, True)
            pass_fds = [
                self.bundle.bundle_fd,
                stats_write,
                start_read,
                rootfs_fd,
                public_fd,
                workspace_fd,
                submission_fd,
                tmp_fd,
                shm_fd,
                quota_fd,
                *(descriptor for descriptor, _target in device_fds),
            ]
            if memory_fd >= 0:
                pass_fds.append(memory_fd)
            soft_nofile, hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
            del soft_nofile
            minimum_open_files = PIDS_LIMIT * 4
            if hard_nofile != resource.RLIM_INFINITY and hard_nofile < minimum_open_files:
                raise BridgeError("host open-file limit is below the MLE contract")
            max_open_files = (
                1_048_576
                if hard_nofile == resource.RLIM_INFINITY
                else int(hard_nofile)
            )
            argv = [
                self.bundle.supervisor_path,
                "--rootfs-fd",
                str(rootfs_fd),
                "--run-dir",
                cgroups.run_dir,
                "--command",
                request["command"],
                "--gpu-uuid",
                self.deployment["gpu"]["uuid"],
                "--public-fd",
                str(public_fd),
                "--workspace-fd",
                str(workspace_fd),
                "--submission-fd",
                str(submission_fd),
                "--tmp-fd",
                str(tmp_fd),
                "--shm-fd",
                str(shm_fd),
                "--quota-fd",
                str(quota_fd),
                "--stats-fd",
                str(stats_write),
                "--start-fd",
                str(start_read),
                "--host-uid",
                str(self.deployment["sandbox_host_uid"]),
                "--host-gid",
                str(self.deployment["sandbox_host_gid"]),
                "--max-processes",
                str(PIDS_LIMIT),
                "--max-open-files",
                str(max_open_files),
                "--max-file-bytes",
                str(WRITABLE_BYTES_LIMIT),
            ]
            if memory_fd >= 0:
                argv.extend(("--memory-fd", str(memory_fd)))
            for descriptor, target in device_fds:
                argv.extend(("--device", f"{descriptor}:{target}"))
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                close_fds=True,
                pass_fds=tuple(pass_fds),
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
                start_new_session=True,
            )
            os.close(stats_write)
            stats_write = None
            os.close(start_read)
            start_read = None
            self._move_pid_to_cgroups(cgroups, process.pid)
            os.write(start_write, b"G")
            os.close(start_write)
            start_write = None

            assert process.stdout is not None and process.stderr is not None
            selector = selectors.DefaultSelector()
            streams: dict[int, tuple[str, bytearray]] = {
                process.stdout.fileno(): ("stdout", stdout),
                process.stderr.fileno(): ("stderr", stderr),
                stats_read: ("stats", stats_payload),
            }
            for descriptor, pair in streams.items():
                os.set_blocking(descriptor, False)
                selector.register(descriptor, selectors.EVENT_READ, pair)
            visible_cap = request["resource_contract"]["max_visible_output_bytes"]
            while selector.get_map():
                now = time.monotonic()
                if now >= cleanup_deadline:
                    raise BridgeError("sandbox pipe-drain deadline expired")
                if now >= execution_deadline and not killed_for_timeout:
                    killed_for_timeout = process.poll() is None or bool(
                        self._cgroup_pids(cgroups)
                    )
                    self._kill_untrusted_cgroup_processes(cgroups)
                wait_until = execution_deadline if now < execution_deadline else cleanup_deadline
                wait = max(0.0, min(0.05, wait_until - now))
                for key, _mask in selector.select(wait):
                    descriptor = key.fd
                    kind, target = key.data
                    try:
                        chunk = os.read(descriptor, 65536)
                    except OSError as exc:
                        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                            continue
                        chunk = b""
                    if not chunk:
                        selector.unregister(descriptor)
                        continue
                    if kind == "stats":
                        if len(target) + len(chunk) > 1024 * 1024:
                            raise BridgeError("supervisor stats exceed byte cap")
                        target.extend(chunk)
                        continue
                    output_seen += len(chunk)
                    room = max(0, visible_cap - len(stdout) - len(stderr))
                    if room:
                        target.extend(chunk[:room])
                    if output_seen > visible_cap and not output_limited:
                        output_limited = True
                        self._kill_untrusted_cgroup_processes(cgroups)
            process.wait(timeout=max(0.1, cleanup_deadline - time.monotonic()))
            residual = self._cgroup_pids(cgroups)
            if residual:
                self._kill_cgroup(cgroups)
                residual = self._cgroup_pids(cgroups)
            if residual:
                raise BridgeError("sandbox descendants remain")
            final_stats = self._cgroup_stats(cgroups)
            if before_stats is None:
                raise BridgeError("cgroup baseline is absent")
            self._validate_cgroup_stats(before_stats, final_stats)
            supervisor = self._parse_supervisor_stats(bytes(stats_payload))
            public_after = self._open_anchored_read_only_tree(
                identity["public_anchor"], label="public"
            )
            os.close(public_after)
            writable = self._writable_usage(request, identity)
            baseline = identity["writable_baseline"]
            writable_high_water = max(
                writable["bytes"], int(supervisor["writable_bytes_high_water"])
            )
            inode_high_water = max(
                writable["inodes"], int(supervisor["writable_inodes_high_water"])
            )
            cpu_ns = max(
                0,
                final_stats["cpu_usage_ns"] - before_stats["cpu_usage_ns"],
            )
            returncode = int(supervisor["exit_code"])
            if output_limited:
                returncode = 137
            return ExecutionOutcome(
                returncode=returncode,
                stdout=bytes(stdout).decode("utf-8", "replace"),
                stderr=bytes(stderr).decode("utf-8", "replace"),
                timed_out=killed_for_timeout,
                execution_time_ms=max(
                    0, math.ceil((time.monotonic_ns() - started_ns) / 1_000_000)
                ),
                cpu_time_ms=max(0, math.ceil(cpu_ns / 1_000_000)),
                writable_bytes=max(0, writable_high_water - int(baseline["bytes"])),
                writable_inodes=max(0, inode_high_water - int(baseline["inodes"])),
                processes_started=int(supervisor["processes_started"]),
                descendant_process_count=0,
            )
        finally:
            if selector is not None:
                selector.close()
            for descriptor in (stats_write, stats_read, start_read, start_write):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            for descriptor in reversed(opened):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if process is not None and process.poll() is None:
                if cgroups is not None:
                    try:
                        self._kill_cgroup(cgroups)
                    except BridgeError:
                        pass
                try:
                    process.kill()
                    process.wait(timeout=1.0)
                except (OSError, subprocess.SubprocessError):
                    pass
            if cgroups is not None:
                cleanup_complete = False
                try:
                    self._cleanup_cgroups(cgroups)
                    cleanup_complete = True
                finally:
                    if watchdog is not None:
                        watchdog.finish(cleanup_complete=cleanup_complete)

    def freeze(
        self, request: Mapping[str, Any], state: Mapping[str, Any]
    ) -> LifecycleOutcome:
        self._check_operation_deadline("freeze validation")
        identity = self._require_runtime_identity(request, state, frozen=None)
        self._reap_episode_residue(request["episode_id"])
        record = self._require_episode_mount(identity)
        if not record.read_only:
            self._libc.mount(
                None,
                identity["episode_root"],
                None,
                MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV,
            )
        record = self._require_episode_mount(identity)
        if not record.read_only:
            raise BridgeError("episode quota mount did not freeze")
        self._writable_usage(request, identity)
        if self._episode_residue(
            request["episode_id"], allow_episode_memory_parent=True
        ):
            raise BridgeError("owned process or cgroup residue remains")
        return LifecycleOutcome(
            processes_reaped=True,
            workspace_frozen=True,
            mounts_released=False,
            descendant_process_count=0,
            mount_count=1,
            sandbox_present=True,
        )

    def teardown(
        self, request: Mapping[str, Any], state: Mapping[str, Any]
    ) -> LifecycleOutcome:
        self._check_operation_deadline("teardown validation")
        identity_value = state.get("runtime_identity")
        identity: dict[str, Any] | None
        if identity_value is None:
            if state.get("lifecycle") not in {"attesting", "torn_down"}:
                raise BridgeError("runtime identity is absent outside cleanup state")
            self._validate_request_roots(request)
            identity = None
            episode_root = os.path.dirname(request["workspace_root"])
        else:
            identity = self._state_runtime_identity(state)
            self._require_identity_matches_request(identity, request)
            episode_root = identity["episode_root"]
        self._reap_episode_residue(request["episode_id"])
        record = self._find_exact_mount(episode_root)
        if record is not None:
            if identity is None:
                self._validate_quota_mount(
                    record, request["episode_id"], frozen=None
                )
            else:
                self._require_episode_mount(identity)
            self._libc.umount(episode_root)
        if self._find_exact_mount(episode_root) is not None:
            raise BridgeError("episode quota mount remains after teardown")
        self._remove_episode_memory_cgroup(request["episode_id"])
        if self._episode_residue(request["episode_id"]):
            raise BridgeError("owned residue remains after teardown")
        self._write_tombstone(request, state, identity)
        return LifecycleOutcome(
            processes_reaped=True,
            workspace_frozen=False,
            mounts_released=True,
            descendant_process_count=0,
            mount_count=0,
            sandbox_present=False,
        )

    def reconcile(
        self,
        operation: str,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> LifecycleOutcome:
        if operation == "freeze":
            return self.freeze(request, state)
        if operation != "teardown":
            raise BridgeError("only lifecycle operations can be reconciled")
        identity_value = state.get("runtime_identity")
        identity = (
            None if identity_value is None else self._state_runtime_identity(state)
        )
        if identity is None:
            self._validate_request_roots(request)
            episode_root = os.path.dirname(request["workspace_root"])
        else:
            self._require_identity_matches_request(identity, request)
            episode_root = identity["episode_root"]
        if state.get("lifecycle") == "torn_down":
            if self._find_exact_mount(episode_root) is not None:
                raise BridgeError("torn-down episode mount reappeared")
            self._reap_episode_residue(request["episode_id"])
            if self._episode_residue(request["episode_id"]):
                raise BridgeError("torn-down episode residue reappeared")
            self._verify_tombstone(request, state, identity)
            return LifecycleOutcome(
                processes_reaped=True,
                workspace_frozen=False,
                mounts_released=True,
                descendant_process_count=0,
                mount_count=0,
                sandbox_present=False,
            )
        if self._find_exact_mount(episode_root) is not None:
            return self.teardown(request, state)
        self._reap_episode_residue(request["episode_id"])
        self._remove_episode_memory_cgroup(request["episode_id"])
        if self._episode_residue(request["episode_id"]):
            raise BridgeError("pending teardown residue is ambiguous")
        self._write_tombstone(request, state, identity)
        return LifecycleOutcome(
            processes_reaped=True,
            workspace_frozen=False,
            mounts_released=True,
            descendant_process_count=0,
            mount_count=0,
            sandbox_present=False,
        )

    def _require_host(self) -> None:
        if platform.system() != "Linux" or platform.machine() not in {
            "x86_64",
            "AMD64",
        }:
            raise BridgeError("runtime bridge requires Linux/x86_64")
        if os.geteuid() != 0:
            raise BridgeError("runtime bridge requires a privileged host launcher")

    def _validate_request_roots(self, request: Mapping[str, Any]) -> None:
        episodes_root = self.deployment["episodes_root"]
        containing_mount = find_containing_mount(episodes_root)
        if any(
            field.startswith(("shared:", "master:", "propagate_from:"))
            for field in containing_mount.optional_fields
        ):
            raise BridgeError("episode root mount propagation is not private")
        episode_root = os.path.dirname(request["workspace_root"])
        expected = os.path.join(episodes_root, request["episode_id"])
        if episode_root != expected:
            raise BridgeError("episode root is outside the deployment root")
        if request["submission_root"] != os.path.join(expected, "submission"):
            raise BridgeError("submission root crossed the episode")
        if request["workspace_root"] != os.path.join(expected, "workspace"):
            raise BridgeError("workspace root crossed the episode")
        memory = request.get("external_memory_root")
        if memory is not None and memory != os.path.join(expected, "external-memory"):
            raise BridgeError("external-memory root crossed the episode")

    def _verify_rootfs_identity(self) -> dict[str, Any]:
        return verify_rootfs_tree_lock(
            rootfs_path=self.deployment["rootfs"],
            tree_lock_path=self.deployment["rootfs_tree_lock"],
            expected_tree_sha256=self.deployment["rootfs_digest"],
            expected_lock_sha256=self.deployment["rootfs_tree_lock_sha256"],
            deadline=self.operation_deadline,
        )

    def _verify_cgroup_surface(self) -> None:
        required = {
            "cpu,cpuacct": ("cpu.cfs_period_us", "cpu.cfs_quota_us", "cpuacct.usage"),
            "memory": (
                "memory.limit_in_bytes",
                "memory.memsw.limit_in_bytes",
                "memory.swappiness",
                "memory.use_hierarchy",
                "memory.usage_in_bytes",
                "memory.max_usage_in_bytes",
                "memory.failcnt",
                "memory.oom_control",
                "memory.stat",
            ),
            "pids": ("pids.max", "pids.events", "cgroup.procs"),
            "devices": ("devices.allow", "devices.deny", "devices.list"),
        }
        for controller, files in required.items():
            root = os.path.join("/sys/fs/cgroup", controller)
            if not os.path.isdir(root) or os.path.islink(root):
                raise BridgeError("required cgroup-v1 controller is absent")
            if any(not os.path.exists(os.path.join(root, name)) for name in files):
                raise BridgeError("required cgroup-v1 control file is absent")

    def _verify_gpu_identity(self, rootfs_fd: int) -> dict[str, Any]:
        devices = self._gpu_devices()
        for device in devices:
            descriptor = self._open_device(device)
            os.close(descriptor)
        loader_fd = _open_rootfs_member(
            rootfs_fd, self.deployment["rootfs_loader_path"], directory=False
        )
        inventory_descriptor = _open_rootfs_member(
            rootfs_fd,
            self.deployment["rootfs_nvidia_smi_path"],
            directory=False,
        )
        audit_descriptor = self._open_runtime_audit_module()
        audit_probe = os.memfd_create(
            "mlebridge-runtime-audit-probe", os.MFD_CLOEXEC
        )
        os.ftruncate(audit_probe, 4096)
        reserved_rootfs_fd = False
        try:
            library_paths = []
            for member in self.deployment["rootfs_library_paths"]:
                directory = _open_rootfs_member(rootfs_fd, member, directory=True)
                os.close(directory)
                library_paths.append(
                    f"/proc/self/fd/{RUNTIME_AUDIT_ROOTFS_FD}/"
                    + member.lstrip("/")
                )
            if rootfs_fd != RUNTIME_AUDIT_ROOTFS_FD:
                try:
                    fcntl.fcntl(RUNTIME_AUDIT_ROOTFS_FD, fcntl.F_GETFD)
                except OSError as exc:
                    if exc.errno != errno.EBADF:
                        raise BridgeError(
                            "runtime audit rootfs descriptor is ambiguous"
                        ) from exc
                else:
                    raise BridgeError(
                        "runtime audit rootfs descriptor is already open"
                    )
                os.dup2(
                    rootfs_fd,
                    RUNTIME_AUDIT_ROOTFS_FD,
                    inheritable=True,
                )
                reserved_rootfs_fd = True
            else:
                os.set_inheritable(RUNTIME_AUDIT_ROOTFS_FD, True)
            loader = f"/proc/self/fd/{loader_fd}"
            inventory = f"/proc/self/fd/{inventory_descriptor}"
            audit = f"/proc/self/fd/{audit_descriptor}"
            try:
                fcntl.fcntl(RUNTIME_AUDIT_PROBE_FD, fcntl.F_GETFD)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise BridgeError("runtime audit probe descriptor is ambiguous") from exc
            else:
                raise BridgeError("runtime audit probe descriptor is already open")
            os.dup2(audit_probe, RUNTIME_AUDIT_PROBE_FD, inheritable=True)
            completed = subprocess.run(
                [
                    loader,
                    "--inhibit-cache",
                    "--audit",
                    audit,
                    "--library-path",
                    ":".join(library_paths),
                    inventory,
                    "--query-gpu=uuid,minor_number",
                    "--format=csv,noheader,nounits",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=True,
                timeout=self._remaining_time("rootfs GPU inventory", cap=15.0),
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                close_fds=True,
                pass_fds=(
                    rootfs_fd,
                    loader_fd,
                    inventory_descriptor,
                    audit_descriptor,
                    RUNTIME_AUDIT_PROBE_FD,
                    RUNTIME_AUDIT_ROOTFS_FD,
                ),
            )
            if os.pread(audit_probe, len(RUNTIME_AUDIT_MARKER), 0) != RUNTIME_AUDIT_MARKER:
                raise BridgeError("rootfs GPU inventory was not loader-audited")
        except (OSError, subprocess.SubprocessError) as exc:
            raise BridgeError("GPU inventory is unavailable") from exc
        finally:
            try:
                os.close(RUNTIME_AUDIT_PROBE_FD)
            except OSError:
                pass
            if reserved_rootfs_fd:
                os.close(RUNTIME_AUDIT_ROOTFS_FD)
            elif rootfs_fd == RUNTIME_AUDIT_ROOTFS_FD:
                os.set_inheritable(rootfs_fd, False)
            os.close(audit_probe)
            os.close(audit_descriptor)
            os.close(inventory_descriptor)
            os.close(loader_fd)
        try:
            rows = []
            for line in completed.stdout.decode("ascii", "strict").splitlines():
                uuid_value, minor_value = (part.strip() for part in line.split(",", 1))
                rows.append((uuid_value, int(minor_value)))
        except (UnicodeError, ValueError) as exc:
            raise BridgeError("GPU inventory is malformed") from exc
        selected = [row for row in rows if row[0] == self.deployment["gpu"]["uuid"]]
        primary = self.deployment["gpu"]["device"]
        if len(selected) != 1 or selected[0][1] != primary["minor"]:
            raise BridgeError("pinned GPU UUID/minor identity drifted")
        return {
            "uuid": selected[0][0],
            "inventory_tool_identity": canonical_sha256(
                {
                    "rootfs_digest": self.deployment["rootfs_digest"],
                    "path": self.deployment["rootfs_nvidia_smi_path"],
                    "loader": self.deployment["rootfs_loader_path"],
                    "library_paths": self.deployment["rootfs_library_paths"],
                    "runtime_audit": RUNTIME_AUDIT_RELATIVE_PATH,
                }
            ),
            "devices": [
                {
                    "target": item["target"],
                    "major": item["major"],
                    "minor": item["minor"],
                }
                for item in devices
            ],
        }

    def _open_runtime_audit_module(self) -> int:
        try:
            descriptor = open_bundle_regular_file(
                self.bundle.bundle_fd,
                RUNTIME_AUDIT_RELATIVE_PATH,
                expected_uid=self.bundle.expected_uid,
            )
        except (BridgeError, OSError) as exc:
            raise BridgeError("runtime audit module is unavailable") from exc
        return descriptor

    def _gpu_devices(self) -> list[dict[str, Any]]:
        return [
            self.deployment["gpu"]["device"],
            *self.deployment["gpu"]["control_devices"],
        ]

    @staticmethod
    def _open_device(device: Mapping[str, Any]) -> int:
        flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            descriptor = os.open(device["source"], flags)
        except OSError as exc:
            raise BridgeError("GPU device is unavailable") from exc
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISCHR(metadata.st_mode)
            or os.major(metadata.st_rdev) != device["major"]
            or os.minor(metadata.st_rdev) != device["minor"]
        ):
            os.close(descriptor)
            raise BridgeError("GPU device number drifted")
        return descriptor

    def _attest_episode_mount(
        self,
        request: Mapping[str, Any],
        *,
        public_anchor: Mapping[str, Any],
        gpu: Mapping[str, Any],
        rootfs_anchor: Mapping[str, Any],
        expected_identity: Any,
    ) -> dict[str, Any]:
        episode_root = os.path.dirname(request["workspace_root"])
        episode_memory_cgroup = self._ensure_episode_memory_cgroup(
            request["episode_id"]
        )
        record = self._find_exact_mount(episode_root)
        if record is None:
            if expected_identity is not None:
                raise BridgeError("attested episode mount disappeared")
            self._validate_underlying_episode_root(request)
            label = "mlebridge-" + request["episode_id"]
            options = (
                f"size={WRITABLE_BYTES_LIMIT},nr_inodes={WRITABLE_INODES_LIMIT},"
                "mode=0700"
            )
            self._libc.mount(
                label,
                episode_root,
                "tmpfs",
                MS_NOSUID | MS_NODEV,
                options,
            )
            record = self._find_exact_mount(episode_root)
        if record is None:
            raise BridgeError("episode quota mount is absent")
        self._validate_quota_mount(record, request["episode_id"], frozen=False)
        root_fd = open_absolute_directory_nofollow(episode_root)
        try:
            self._ensure_episode_directories(root_fd, request)
            root_metadata = os.fstat(root_fd)
            directories = {}
            for name in self._expected_episode_directories(request):
                descriptor = open_directory_at(root_fd, name)
                metadata = os.fstat(descriptor)
                os.close(descriptor)
                directories[name] = {"dev": metadata.st_dev, "ino": metadata.st_ino}
            usage = writable_usage_fd(root_fd)
        finally:
            os.close(root_fd)
        self._validate_writable_capacity(episode_root, usage)
        boot_id = read_stable_regular_file("/proc/sys/kernel/random/boot_id", 128).decode(
            "ascii", "strict"
        ).strip()
        if not _BOOT_ID.fullmatch(boot_id):
            raise BridgeError("host boot identity drifted")
        identity = {
            "schema": RUNTIME_IDENTITY_SCHEMA,
            "boot_id": boot_id,
            "episode_root": episode_root,
            "episode_mount_id": record.mount_id,
            "episode_mount_device": record.device,
            "episode_mount_source": record.source,
            "episode_root_stat": {"dev": root_metadata.st_dev, "ino": root_metadata.st_ino},
            "episode_memory_cgroup": episode_memory_cgroup,
            "directories": directories,
            "public_anchor": dict(public_anchor),
            "rootfs_anchor": dict(rootfs_anchor),
            "runner_runtime_mappings": dict(self.runtime_mapping_identity),
            "writable_baseline": dict(usage),
            "gpu": dict(gpu),
            "resource_envelope": {
                "cpu_limit_cores": CPU_LIMIT_CORES,
                "cpu_period_us": CGROUP_PERIOD_US,
                "cpu_quota_us": CGROUP_QUOTA_US,
                "memory_limit_bytes": MEMORY_LIMIT_BYTES,
                "memory_memsw_limit_bytes": MEMORY_LIMIT_BYTES,
                "memory_swappiness": 0,
                "pids_limit": PIDS_LIMIT,
                "writable_bytes_limit": WRITABLE_BYTES_LIMIT,
                "writable_inodes_limit": WRITABLE_INODES_LIMIT,
            },
        }
        if expected_identity is not None and identity != expected_identity:
            raise BridgeError("runtime mount identity drifted")
        return identity

    def _validate_underlying_episode_root(self, request: Mapping[str, Any]) -> None:
        episodes_fd = open_absolute_directory_nofollow(self.deployment["episodes_root"])
        try:
            root_fd = open_directory_at(episodes_fd, request["episode_id"])
        finally:
            os.close(episodes_fd)
        try:
            allowed = {"workspace", "submission"}
            if request["mode"] == "amg_memory":
                allowed.add("external-memory")
            names = {entry.name for entry in os.scandir(root_fd)}
            if names - allowed:
                raise BridgeError("underlying episode root contains unexpected data")
            for name in names:
                descriptor = open_directory_at(root_fd, name)
                try:
                    if list(os.scandir(descriptor)):
                        raise BridgeError("underlying episode directory is not empty")
                finally:
                    os.close(descriptor)
        finally:
            os.close(root_fd)

    def _ensure_episode_directories(
        self, root_fd: int, request: Mapping[str, Any]
    ) -> None:
        expected = self._expected_episode_directories(request)
        existing = {entry.name for entry in os.scandir(root_fd)}
        if existing - set(expected):
            raise BridgeError("episode quota mount contains unexpected entries")
        sandbox_uid = self.deployment["sandbox_host_uid"]
        sandbox_gid = self.deployment["sandbox_host_gid"]
        for name, (mode, uid, gid) in expected.items():
            try:
                os.mkdir(name, mode, dir_fd=root_fd)
            except FileExistsError:
                pass
            descriptor = open_directory_at(root_fd, name)
            try:
                metadata = os.fstat(descriptor)
                if metadata.st_uid != uid or metadata.st_gid != gid:
                    os.fchown(descriptor, uid, gid)
                os.fchmod(descriptor, mode)
                after = os.fstat(descriptor)
                if (
                    after.st_uid != uid
                    or after.st_gid != gid
                    or stat.S_IMODE(after.st_mode) != mode
                ):
                    raise BridgeError("episode directory identity drifted")
            finally:
                os.close(descriptor)
        del sandbox_uid, sandbox_gid

    def _expected_episode_directories(
        self, request: Mapping[str, Any]
    ) -> dict[str, tuple[int, int, int]]:
        uid = self.deployment["sandbox_host_uid"]
        gid = self.deployment["sandbox_host_gid"]
        result = {
            "workspace": (0o700, uid, gid),
            "submission": (0o700, uid, gid),
            "tmp": (0o1777, uid, gid),
            "shm": (0o1777, uid, gid),
            "runtime-temp": (0o700, os.geteuid(), os.getegid()),
        }
        if request["mode"] == "amg_memory":
            result["external-memory"] = (0o700, uid, gid)
        return result

    def _validate_quota_mount(
        self, record: MountRecord, episode_id: str, *, frozen: bool | None
    ) -> None:
        if (
            record.filesystem != "tmpfs"
            or record.source != "mlebridge-" + episode_id
            or "nosuid" not in record.mount_options
            or "nodev" not in record.mount_options
            or any(
                field.startswith(("shared:", "master:", "propagate_from:"))
                for field in record.optional_fields
            )
        ):
            raise BridgeError("episode quota mount identity drifted")
        if frozen is True and not record.read_only:
            raise BridgeError("episode quota mount is not frozen")
        if frozen is False and record.read_only:
            raise BridgeError("episode quota mount froze unexpectedly")
        nested = [
            item
            for item in read_mountinfo()
            if item.target != record.target
            and is_path_within(item.target, record.target)
        ]
        if nested:
            raise BridgeError("episode quota mount contains a subordinate mount")
        self._validate_writable_capacity(record.target, None)

    @staticmethod
    def _validate_writable_capacity(
        path: str, usage: Mapping[str, int] | None
    ) -> None:
        descriptor = open_absolute_directory_nofollow(path)
        try:
            values = os.fstatvfs(descriptor)
        finally:
            os.close(descriptor)
        capacity = values.f_blocks * values.f_frsize
        page = os.sysconf("SC_PAGE_SIZE")
        if not (WRITABLE_BYTES_LIMIT <= capacity < WRITABLE_BYTES_LIMIT + page):
            raise BridgeError("tmpfs byte capacity drifted")
        if values.f_files != WRITABLE_INODES_LIMIT:
            raise BridgeError("tmpfs inode capacity drifted")
        if usage is not None and (
            usage["bytes"] > WRITABLE_BYTES_LIMIT
            or usage["inodes"] > WRITABLE_INODES_LIMIT
        ):
            raise BridgeError("writable quota high-water exceeded")

    def _open_and_hash_public(
        self, path: str
    ) -> tuple[int, str, dict[str, Any]]:
        self._check_operation_deadline("public attestation")
        descriptor = open_absolute_directory_nofollow(path)
        metadata = os.fstat(descriptor)
        mount_before = find_containing_mount(path)
        if not mount_before.read_only:
            os.close(descriptor)
            raise BridgeError("public source is not on a read-only mount")
        if any(
            item.target != path and is_path_within(item.target, path)
            for item in read_mountinfo()
        ):
            os.close(descriptor)
            raise BridgeError("public source contains a subordinate mount")
        try:
            digest = stable_public_tree_sha256_fd(
                descriptor, deadline=self.operation_deadline
            )
            after = os.fstat(descriptor)
            if stable_stat(metadata) != stable_stat(after):
                raise BridgeError("public root changed while hashing")
            mount_after = find_containing_mount(path)
            if mount_after != mount_before or not mount_after.read_only:
                raise BridgeError("public mount changed while hashing")
            return descriptor, digest, {
                "schema": "mlebench_lite_read_only_tree_anchor_v1",
                "path": path,
                "mount_id": mount_after.mount_id,
                "mount_device": mount_after.device,
                "mount_root": mount_after.root,
                "mount_target": mount_after.target,
                "root_stat": stable_anchor_stat(after),
                "tree_sha256": digest,
            }
        except BaseException:
            os.close(descriptor)
            raise

    def _open_anchored_read_only_tree(
        self, anchor: Mapping[str, Any], *, label: str
    ) -> int:
        expected_fields = {
            "schema",
            "path",
            "mount_id",
            "mount_device",
            "mount_root",
            "mount_target",
            "root_stat",
            "tree_sha256",
        }
        if label == "rootfs":
            expected_fields.add("tree_lock_sha256")
        if (
            not isinstance(anchor, Mapping)
            or set(anchor) != expected_fields
            or anchor.get("schema") != "mlebench_lite_read_only_tree_anchor_v1"
            or not isinstance(anchor.get("path"), str)
        ):
            raise BridgeError(f"{label} read-only anchor drifted")
        descriptor = open_absolute_directory_nofollow(anchor["path"])
        try:
            metadata = os.fstat(descriptor)
            record = find_containing_mount(anchor["path"])
            if (
                not record.read_only
                or record.mount_id != anchor["mount_id"]
                or record.device != anchor["mount_device"]
                or record.root != anchor["mount_root"]
                or record.target != anchor["mount_target"]
                or stable_anchor_stat(metadata) != anchor["root_stat"]
                or any(
                    item.target != anchor["path"]
                    and is_path_within(item.target, anchor["path"])
                    for item in read_mountinfo()
                )
            ):
                raise BridgeError(f"{label} read-only anchor changed")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def _require_runtime_identity(
        self,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        frozen: bool | None,
    ) -> dict[str, Any]:
        identity = self._state_runtime_identity(state)
        self._require_identity_matches_request(identity, request)
        self._require_episode_memory_cgroup(
            request["episode_id"], identity.get("episode_memory_cgroup", {})
        )
        record = self._require_episode_mount(identity)
        self._validate_quota_mount(record, request["episode_id"], frozen=frozen)
        return identity

    def _state_runtime_identity(self, state: Mapping[str, Any]) -> dict[str, Any]:
        identity = state.get("runtime_identity")
        if not isinstance(identity, dict) or identity.get("schema") != RUNTIME_IDENTITY_SCHEMA:
            raise BridgeError("runtime identity is absent")
        if identity.get("runner_runtime_mappings") != self.runtime_mapping_identity:
            raise BridgeError("runner audited runtime mappings drifted")
        return dict(identity)

    def _require_identity_matches_request(
        self, identity: Mapping[str, Any], request: Mapping[str, Any]
    ) -> None:
        if (
            identity.get("episode_root") != os.path.dirname(request["workspace_root"])
            or identity.get("public_anchor", {}).get("tree_sha256")
            != request["public_tree_sha256"]
            or identity.get("public_anchor", {}).get("path")
            != request["public_root"]
            or identity.get("rootfs_anchor", {}).get("path")
            != self.deployment["rootfs"]
            or identity.get("gpu", {}).get("uuid") != self.deployment["gpu"]["uuid"]
            or identity.get("resource_envelope", {}).get("cpu_limit_cores")
            != CPU_LIMIT_CORES
            or identity.get("resource_envelope", {}).get("memory_limit_bytes")
            != MEMORY_LIMIT_BYTES
            or identity.get("resource_envelope", {}).get("pids_limit") != PIDS_LIMIT
            or identity.get("episode_memory_cgroup")
            != self._episode_memory_cgroup_identity(request["episode_id"])
        ):
            raise BridgeError("runtime identity does not match the request")

    def _require_episode_mount(self, identity: Mapping[str, Any]) -> MountRecord:
        record = self._find_exact_mount(identity["episode_root"])
        if (
            record is None
            or record.mount_id != identity["episode_mount_id"]
            or record.device != identity["episode_mount_device"]
            or record.source != identity["episode_mount_source"]
        ):
            raise BridgeError("episode mount identity drifted")
        descriptor = open_absolute_directory_nofollow(identity["episode_root"])
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if {
            "dev": metadata.st_dev,
            "ino": metadata.st_ino,
        } != identity["episode_root_stat"]:
            raise BridgeError("episode mount inode drifted")
        return record

    @staticmethod
    def _find_exact_mount(path: str) -> MountRecord | None:
        matches = [item for item in read_mountinfo() if item.target == path]
        if len(matches) > 1:
            raise BridgeError("episode root has stacked mounts")
        return None if not matches else matches[0]

    def _open_episode_child(
        self, request: Mapping[str, Any], identity: Mapping[str, Any], name: str
    ) -> int:
        mapping = {
            "workspace": request["workspace_root"],
            "submission": request["submission_root"],
            "external-memory": request.get("external_memory_root"),
        }
        path = mapping[name]
        if not isinstance(path, str):
            raise BridgeError("requested episode child is absent")
        return self._open_episode_child_path(request, identity, name)

    def _open_episode_child_path(
        self, request: Mapping[str, Any], identity: Mapping[str, Any], name: str
    ) -> int:
        del request
        root_fd = open_absolute_directory_nofollow(identity["episode_root"])
        try:
            descriptor = open_directory_at(root_fd, name)
        finally:
            os.close(root_fd)
        metadata = os.fstat(descriptor)
        expected = identity["directories"].get(name)
        if expected != {"dev": metadata.st_dev, "ino": metadata.st_ino}:
            os.close(descriptor)
            raise BridgeError("episode child inode drifted")
        return descriptor

    def _writable_usage(
        self, request: Mapping[str, Any], identity: Mapping[str, Any]
    ) -> dict[str, int]:
        del request
        descriptor = open_absolute_directory_nofollow(identity["episode_root"])
        try:
            usage = writable_usage_fd(descriptor)
        finally:
            os.close(descriptor)
        self._validate_writable_capacity(identity["episode_root"], usage)
        return usage

    @staticmethod
    def _episode_memory_cgroup_name(episode_id: str) -> str:
        name = f"mlebridge-{episode_id}"
        if not _EPISODE_CGROUP_NAME.fullmatch(name):
            raise BridgeError("episode memory cgroup name drifted")
        return name

    def _episode_memory_cgroup_path(self, episode_id: str) -> str:
        return os.path.join(
            "/sys/fs/cgroup/memory",
            self._episode_memory_cgroup_name(episode_id),
        )

    def _episode_memory_cgroup_identity(self, episode_id: str) -> dict[str, Any]:
        return {
            "schema": "mlebench_lite_episode_memory_cgroup_v1",
            "name": self._episode_memory_cgroup_name(episode_id),
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "memory_memsw_limit_bytes": MEMORY_LIMIT_BYTES,
            "memory_swappiness": 0,
            "memory_use_hierarchy": 1,
        }

    @staticmethod
    def _cgroup_child_directories(path: str) -> list[str]:
        try:
            return sorted(
                entry.name
                for entry in os.scandir(path)
                if entry.is_dir(follow_symlinks=False)
            )
        except OSError as exc:
            raise BridgeError("episode memory cgroup is unreadable") from exc

    def _require_episode_memory_cgroup(
        self, episode_id: str, identity: Mapping[str, Any]
    ) -> str:
        expected = self._episode_memory_cgroup_identity(episode_id)
        if identity != expected:
            raise BridgeError("episode memory cgroup identity drifted")
        path = self._episode_memory_cgroup_path(episode_id)
        if not os.path.isdir(path) or os.path.islink(path):
            raise BridgeError("episode memory cgroup is absent")
        expected_values = {
            "memory.limit_in_bytes": MEMORY_LIMIT_BYTES,
            "memory.memsw.limit_in_bytes": MEMORY_LIMIT_BYTES,
            "memory.swappiness": 0,
            "memory.use_hierarchy": 1,
        }
        required_files = {
            *expected_values,
            "memory.usage_in_bytes",
            "memory.max_usage_in_bytes",
            "memory.failcnt",
            "memory.oom_control",
            "memory.stat",
            "cgroup.procs",
            "tasks",
        }
        if any(not os.path.exists(os.path.join(path, name)) for name in required_files):
            raise BridgeError("episode memory cgroup control file is absent")
        for name, value in expected_values.items():
            if read_int(os.path.join(path, name)) != value:
                raise BridgeError("episode memory cgroup value drifted")
        memory_stat = read_key_values(os.path.join(path, "memory.stat"))
        if (
            memory_stat.get("hierarchical_memory_limit") != MEMORY_LIMIT_BYTES
            or memory_stat.get("hierarchical_memsw_limit") != MEMORY_LIMIT_BYTES
        ):
            raise BridgeError("episode memory hierarchy limit drifted")
        if "oom_kill" not in read_key_values(
            os.path.join(path, "memory.oom_control")
        ):
            raise BridgeError("episode memory OOM counter is absent")
        try:
            with open(os.path.join(path, "cgroup.procs"), encoding="ascii") as handle:
                if any(line.strip() for line in handle):
                    raise BridgeError("episode memory cgroup contains a process")
        except (OSError, UnicodeError) as exc:
            raise BridgeError("episode memory cgroup membership is unreadable") from exc
        return path

    def _ensure_episode_memory_cgroup(self, episode_id: str) -> dict[str, Any]:
        identity = self._episode_memory_cgroup_identity(episode_id)
        path = self._episode_memory_cgroup_path(episode_id)
        created = False
        try:
            os.mkdir(path, 0o700)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise BridgeError("episode memory cgroup cannot be created") from exc
        try:
            if created:
                write_control(path, "memory.use_hierarchy", 1)
                write_control(path, "memory.limit_in_bytes", MEMORY_LIMIT_BYTES)
                write_control(
                    path, "memory.memsw.limit_in_bytes", MEMORY_LIMIT_BYTES
                )
                write_control(path, "memory.swappiness", 0)
            self._require_episode_memory_cgroup(episode_id, identity)
            if self._cgroup_child_directories(path):
                raise BridgeError("episode memory cgroup contains stale operations")
            return identity
        except BaseException:
            if created:
                try:
                    os.rmdir(path)
                except OSError:
                    pass
            raise

    def _remove_episode_memory_cgroup(self, episode_id: str) -> None:
        path = self._episode_memory_cgroup_path(episode_id)
        if not os.path.exists(path):
            return
        identity = self._episode_memory_cgroup_identity(episode_id)
        self._require_episode_memory_cgroup(episode_id, identity)
        if self._cgroup_child_directories(path):
            raise BridgeError("episode memory cgroup still has operation children")
        for _ in range(50):
            try:
                os.rmdir(path)
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                if exc.errno not in (errno.EBUSY, errno.ENOTEMPTY):
                    raise BridgeError(
                        "episode memory cgroup cannot be removed"
                    ) from exc
                time.sleep(0.02)
        raise BridgeError("episode memory cgroup cannot be removed")

    def _create_cgroups(self, request: Mapping[str, Any]) -> CgroupSet:
        operation_hex = request["operation_id"].replace("-", "")
        name = f"mlebridge-{request['episode_id']}-{operation_hex}"
        if not _CGROUP_NAME.fullmatch(name):
            raise BridgeError("owned cgroup name drifted")
        runs_root = os.path.join(self.deployment["state_root"], "runs")
        ensure_owned_directory(runs_root, mode=0o700, parents=True)
        run_dir = os.path.join(runs_root, name)
        try:
            os.mkdir(run_dir, 0o700)
        except OSError as exc:
            raise BridgeError("owned run directory already exists") from exc
        cgroup_root = os.path.join(run_dir, "cgroup")
        os.mkdir(cgroup_root, 0o700)
        episode_memory_cgroup = self._require_episode_memory_cgroup(
            request["episode_id"],
            self._episode_memory_cgroup_identity(request["episode_id"]),
        )
        result = CgroupSet(
            name=name,
            run_dir=run_dir,
            mounts={},
            children={},
            episode_memory_cgroup=episode_memory_cgroup,
        )
        try:
            for controller in CGROUP_CONTROLLERS:
                source = os.path.join("/sys/fs/cgroup", controller)
                destination = os.path.join(cgroup_root, controller.replace(",", "_"))
                os.mkdir(destination, 0o700)
                self._libc.mount(source, destination, None, MS_BIND | MS_REC)
                self._libc.mount(None, destination, None, MS_BIND | MS_REMOUNT)
                result.mounts[controller] = destination
                if controller == "memory":
                    operation_component = request["operation_id"].replace("-", "")
                    if not _OPERATION_CGROUP_COMPONENT.fullmatch(operation_component):
                        raise BridgeError("operation memory cgroup name drifted")
                    mounted_episode_memory = os.path.join(
                        destination,
                        self._episode_memory_cgroup_name(request["episode_id"]),
                    )
                    if not os.path.isdir(mounted_episode_memory):
                        raise BridgeError("episode memory hierarchy disappeared")
                    child = os.path.join(
                        mounted_episode_memory, operation_component
                    )
                else:
                    child = os.path.join(destination, name)
                os.mkdir(child, 0o700)
                result.children[controller] = child
            write_control(result.children["cpu,cpuacct"], "cpu.cfs_period_us", CGROUP_PERIOD_US)
            write_control(result.children["cpu,cpuacct"], "cpu.cfs_quota_us", CGROUP_QUOTA_US)
            write_control(result.children["memory"], "memory.limit_in_bytes", MEMORY_LIMIT_BYTES)
            write_control(
                result.children["memory"],
                "memory.memsw.limit_in_bytes",
                MEMORY_LIMIT_BYTES,
            )
            write_control(result.children["memory"], "memory.swappiness", 0)
            write_control(result.children["pids"], "pids.max", PIDS_LIMIT)
            write_control(result.children["devices"], "devices.deny", "a")
            allowed = {(1, 3), (1, 5), (1, 8), (1, 9)}
            allowed.update((item["major"], item["minor"]) for item in self._gpu_devices())
            for major, minor in sorted(allowed):
                write_control(
                    result.children["devices"],
                    "devices.allow",
                    f"c {major}:{minor} rwm",
                )
            self._verify_cgroup_values(result, allowed)
            return result
        except BaseException:
            self._cleanup_cgroups(result)
            raise

    def start_execution_watchdog(self, cgroups: CgroupSet) -> ExecutionWatchdog:
        if not hasattr(os, "pidfd_open"):
            raise BridgeError("race-safe pidfd execution watchdog is unavailable")
        runner_pid = os.getpid()
        pidfd = os.pidfd_open(runner_pid, 0)
        control_read, control_write = os.pipe2(os.O_CLOEXEC)
        try:
            watchdog_pid = os.fork()
        except BaseException:
            os.close(pidfd)
            os.close(control_read)
            os.close(control_write)
            raise
        if watchdog_pid == 0:
            try:
                os.close(control_write)
                keep = {0, 1, 2, control_read, pidfd}
                try:
                    descriptors = [int(name) for name in os.listdir("/proc/self/fd")]
                except (OSError, ValueError):
                    descriptors = []
                for descriptor in descriptors:
                    if descriptor not in keep:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                poller = selectors.DefaultSelector()
                poller.register(control_read, selectors.EVENT_READ, "control")
                poller.register(pidfd, selectors.EVENT_READ, "runner")
                events = poller.select()
                cancelled = False
                for key, _mask in events:
                    if key.data == "control":
                        cancelled = os.read(control_read, 1) == b"C"
                        break
                poller.close()
                if not cancelled:
                    _watchdog_cleanup_cgroups(cgroups)
                os._exit(0)
            except BaseException:  # noqa: BLE001 - watchdog must fail via _exit.
                os._exit(125)
        os.close(pidfd)
        os.close(control_read)
        return ExecutionWatchdog(pid=watchdog_pid, control_fd=control_write)

    def _verify_cgroup_values(
        self, cgroups: CgroupSet, allowed_devices: set[tuple[int, int]]
    ) -> None:
        expected = {
            ("cpu,cpuacct", "cpu.cfs_period_us"): CGROUP_PERIOD_US,
            ("cpu,cpuacct", "cpu.cfs_quota_us"): CGROUP_QUOTA_US,
            ("memory", "memory.limit_in_bytes"): MEMORY_LIMIT_BYTES,
            ("memory", "memory.memsw.limit_in_bytes"): MEMORY_LIMIT_BYTES,
            ("memory", "memory.swappiness"): 0,
            ("pids", "pids.max"): PIDS_LIMIT,
        }
        for (controller, name), value in expected.items():
            if read_int(os.path.join(cgroups.children[controller], name)) != value:
                raise BridgeError("cgroup resource value drifted")
        operation_memory_stat = read_key_values(
            os.path.join(cgroups.children["memory"], "memory.stat")
        )
        if (
            read_int(
                os.path.join(cgroups.children["memory"], "memory.use_hierarchy")
            )
            != 1
            or operation_memory_stat.get("hierarchical_memory_limit")
            != MEMORY_LIMIT_BYTES
            or operation_memory_stat.get("hierarchical_memsw_limit")
            != MEMORY_LIMIT_BYTES
        ):
            raise BridgeError("operation memory hierarchy value drifted")
        for name, value in {
            "memory.limit_in_bytes": MEMORY_LIMIT_BYTES,
            "memory.memsw.limit_in_bytes": MEMORY_LIMIT_BYTES,
            "memory.swappiness": 0,
            "memory.use_hierarchy": 1,
        }.items():
            if read_int(os.path.join(cgroups.episode_memory_cgroup, name)) != value:
                raise BridgeError("episode memory hierarchy value drifted")
        path = os.path.join(cgroups.children["devices"], "devices.list")
        try:
            with open(path, encoding="ascii") as handle:
                lines = {line.strip() for line in handle if line.strip()}
        except (OSError, UnicodeError) as exc:
            raise BridgeError("devices cgroup allowlist is unreadable") from exc
        expected_lines = {f"c {major}:{minor} rwm" for major, minor in allowed_devices}
        if lines != expected_lines:
            raise BridgeError("devices cgroup allowlist drifted")

    @staticmethod
    def _move_pid_to_cgroups(cgroups: CgroupSet, pid: int) -> None:
        for controller in CGROUP_CONTROLLERS:
            write_control(cgroups.children[controller], "tasks", pid)

    def _cgroup_pids(self, cgroups: CgroupSet) -> list[int]:
        path = os.path.join(cgroups.children["pids"], "cgroup.procs")
        try:
            with open(path, encoding="ascii") as handle:
                values = [int(line.strip()) for line in handle if line.strip()]
        except (OSError, UnicodeError, ValueError) as exc:
            raise BridgeError("cgroup membership is unreadable") from exc
        return sorted({value for value in values if value > 1})

    def _trusted_supervisor_pids(self, cgroups: CgroupSet) -> set[int]:
        try:
            supervisor_fd = open_bundle_regular_file(
                self.bundle.bundle_fd,
                "bin/mlebench-lite-sandbox-supervisor",
                expected_uid=self.bundle.expected_uid,
            )
            expected = os.fstat(supervisor_fd)
            os.close(supervisor_fd)
        except (BridgeError, OSError) as exc:
            raise BridgeError("supervisor identity is unavailable") from exc
        result = set()
        for pid in self._cgroup_pids(cgroups):
            try:
                actual = os.stat(f"/proc/{pid}/exe")
            except OSError:
                continue
            if actual.st_dev == expected.st_dev and actual.st_ino == expected.st_ino:
                result.add(pid)
        return result

    def _kill_untrusted_cgroup_processes(self, cgroups: CgroupSet) -> None:
        trusted = self._trusted_supervisor_pids(cgroups)
        self._kill_cgroup(cgroups, exclude=trusted)

    def _kill_cgroup(
        self, cgroups: CgroupSet, *, exclude: set[int] | None = None
    ) -> None:
        excluded = set() if exclude is None else set(exclude)
        for _ in range(50):
            pids = [pid for pid in self._cgroup_pids(cgroups) if pid not in excluded]
            if not pids:
                return
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    raise BridgeError("cannot reap cgroup process") from exc
            time.sleep(0.02)
        if [pid for pid in self._cgroup_pids(cgroups) if pid not in excluded]:
            raise BridgeError("cgroup processes could not be reaped")

    @staticmethod
    def _cgroup_stats(cgroups: CgroupSet) -> dict[str, int]:
        memory = cgroups.children["memory"]
        episode_memory = cgroups.episode_memory_cgroup
        pids = cgroups.children["pids"]
        oom = read_key_values(os.path.join(memory, "memory.oom_control"))
        episode_oom = read_key_values(
            os.path.join(episode_memory, "memory.oom_control")
        )
        pid_events = read_key_values(os.path.join(pids, "pids.events"))
        if (
            "oom_kill" not in oom
            or "oom_kill" not in episode_oom
            or "max" not in pid_events
        ):
            raise BridgeError("required cgroup event counter is absent")
        return {
            "cpu_usage_ns": read_int(
                os.path.join(cgroups.children["cpu,cpuacct"], "cpuacct.usage")
            ),
            "memory_peak_bytes": read_int(
                os.path.join(memory, "memory.max_usage_in_bytes")
            ),
            "memory_failcnt": read_int(os.path.join(memory, "memory.failcnt")),
            "oom_kill_count": oom["oom_kill"],
            "episode_memory_usage_bytes": read_int(
                os.path.join(episode_memory, "memory.usage_in_bytes")
            ),
            "episode_memory_peak_bytes": read_int(
                os.path.join(episode_memory, "memory.max_usage_in_bytes")
            ),
            "episode_memory_failcnt": read_int(
                os.path.join(episode_memory, "memory.failcnt")
            ),
            "episode_oom_kill_count": episode_oom["oom_kill"],
            "pids_max_events": pid_events["max"],
        }

    @staticmethod
    def _validate_cgroup_stats(
        before: Mapping[str, int], final: Mapping[str, int]
    ) -> None:
        fields = {
            "cpu_usage_ns",
            "memory_peak_bytes",
            "memory_failcnt",
            "oom_kill_count",
            "episode_memory_usage_bytes",
            "episode_memory_peak_bytes",
            "episode_memory_failcnt",
            "episode_oom_kill_count",
            "pids_max_events",
        }
        if set(before) != fields or set(final) != fields:
            raise BridgeError("cgroup statistics shape drifted")
        if any(
            type(values[field]) is not int or values[field] < 0
            for values in (before, final)
            for field in fields
        ):
            raise BridgeError("cgroup statistics value drifted")
        monotonic = {
            "cpu_usage_ns",
            "memory_peak_bytes",
            "episode_memory_peak_bytes",
            "memory_failcnt",
            "oom_kill_count",
            "episode_memory_failcnt",
            "episode_oom_kill_count",
            "pids_max_events",
        }
        if any(final[field] < before[field] for field in monotonic):
            raise BridgeError("cgroup statistics moved backwards")
        for values in (before, final):
            if (
                values["memory_peak_bytes"] > MEMORY_LIMIT_BYTES
                or values["episode_memory_usage_bytes"] > MEMORY_LIMIT_BYTES
                or values["episode_memory_peak_bytes"] > MEMORY_LIMIT_BYTES
                or values["episode_memory_peak_bytes"]
                < values["memory_peak_bytes"]
            ):
                raise BridgeError("cgroup memory accounting exceeded its limit")
        event_counters = {
            "memory_failcnt",
            "oom_kill_count",
            "episode_memory_failcnt",
            "episode_oom_kill_count",
            "pids_max_events",
        }
        if any(final[field] != before[field] for field in event_counters):
            raise BridgeError("cgroup resource event counter increased")

    def _cleanup_cgroups(self, cgroups: CgroupSet) -> None:
        failures = []
        if "pids" in cgroups.children:
            try:
                self._kill_cgroup(cgroups)
            except BridgeError:
                failures.append("live_processes")
        for controller in reversed(CGROUP_CONTROLLERS):
            child = cgroups.children.get(controller)
            if child is not None:
                removed = False
                for _ in range(50):
                    try:
                        os.rmdir(child)
                        removed = True
                        break
                    except FileNotFoundError:
                        removed = True
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EBUSY, errno.ENOTEMPTY):
                            break
                        time.sleep(0.02)
                if not removed:
                    failures.append("cgroup:" + controller)
            mountpoint = cgroups.mounts.get(controller)
            if mountpoint is not None:
                try:
                    self._libc.umount(mountpoint)
                except BridgeError:
                    failures.append("mount:" + controller)
        try:
            remove_owned_tree(cgroups.run_dir)
        except BridgeError:
            failures.append("run_dir")
        if failures:
            raise BridgeError("cgroup cleanup failed")

    @staticmethod
    def _parse_supervisor_stats(payload: bytes) -> dict[str, Any]:
        try:
            value = strict_json_loads(payload)
        except ValueError as exc:
            raise BridgeError("supervisor stats are not strict JSON") from exc
        expected = {
            "schema",
            "exit_code",
            "security_violation",
            "background_process",
            "file_limit",
            "processes_started",
            "process_peak",
            "bytes_read",
            "bytes_written",
            "writable_bytes_high_water",
            "writable_inodes_high_water",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise BridgeError("supervisor stats shape drifted")
        if value["schema"] != SUPERVISOR_STATS_SCHEMA:
            raise BridgeError("supervisor stats schema drifted")
        if any(
            type(value[name]) is not int or value[name] < 0
            for name in (
                "processes_started",
                "process_peak",
                "bytes_read",
                "bytes_written",
                "writable_bytes_high_water",
                "writable_inodes_high_water",
            )
        ):
            raise BridgeError("supervisor numeric stats drifted")
        if type(value["exit_code"]) is not int or any(
            type(value[name]) is not bool
            for name in ("security_violation", "background_process", "file_limit")
        ):
            raise BridgeError("supervisor status fields drifted")
        return value

    @staticmethod
    def _cgroup_processes(path: str) -> list[int]:
        membership = os.path.join(path, "cgroup.procs")
        if not os.path.exists(membership):
            return []
        try:
            with open(membership, encoding="ascii") as handle:
                return sorted(
                    {
                        int(line.strip())
                        for line in handle
                        if line.strip() and int(line.strip()) > 1
                    }
                )
        except (OSError, UnicodeError, ValueError) as exc:
            raise BridgeError("owned cgroup membership is unreadable") from exc

    def _operation_cgroup_paths(
        self, episode_id: str, name: str
    ) -> dict[str, tuple[str, ...]]:
        return operation_cgroup_paths(episode_id, name)

    def _reap_episode_residue(self, episode_id: str) -> None:
        names = self._owned_cgroup_names(episode_id)
        for name in sorted(names):
            paths = self._operation_cgroup_paths(episode_id, name)
            for _ in range(50):
                pids = sorted(
                    {
                        pid
                        for controller_paths in paths.values()
                        for path in controller_paths
                        for pid in self._cgroup_processes(path)
                    }
                )
                if not pids:
                    break
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError as exc:
                        raise BridgeError(
                            "owned cgroup process cannot be reaped"
                        ) from exc
                time.sleep(0.02)
            else:
                raise BridgeError("owned cgroup processes could not be reaped")
            for controller in reversed(CGROUP_CONTROLLERS):
                for path in paths[controller]:
                    if not os.path.exists(path):
                        continue
                    try:
                        os.rmdir(path)
                    except OSError as exc:
                        raise BridgeError(
                            "owned cgroup residue cannot be removed"
                        ) from exc
            run_dir = os.path.join(self.deployment["state_root"], "runs", name)
            if os.path.exists(run_dir):
                remove_owned_tree(run_dir)

    def _owned_cgroup_names(self, episode_id: str) -> set[str]:
        prefix = f"mlebridge-{episode_id}-"
        result: set[str] = set()
        for controller in CGROUP_CONTROLLERS:
            root = os.path.join("/sys/fs/cgroup", controller)
            try:
                names = os.listdir(root)
            except OSError as exc:
                raise BridgeError("cgroup root is unreadable") from exc
            for name in names:
                if not name.startswith(prefix):
                    continue
                if not _CGROUP_NAME.fullmatch(name):
                    raise BridgeError("malformed owned cgroup name")
                result.add(name)

        episode_memory = self._episode_memory_cgroup_path(episode_id)
        if os.path.exists(episode_memory):
            if not os.path.isdir(episode_memory) or os.path.islink(episode_memory):
                raise BridgeError("episode memory cgroup identity drifted")
            for component in self._cgroup_child_directories(episode_memory):
                if not _OPERATION_CGROUP_COMPONENT.fullmatch(component):
                    raise BridgeError("malformed operation memory cgroup")
                result.add(prefix + component)

        runs_root = os.path.join(self.deployment["state_root"], "runs")
        if os.path.isdir(runs_root):
            try:
                names = os.listdir(runs_root)
            except OSError as exc:
                raise BridgeError("owned run root is unreadable") from exc
            for name in names:
                if not name.startswith(prefix):
                    continue
                if not _CGROUP_NAME.fullmatch(name):
                    raise BridgeError("malformed owned run directory")
                result.add(name)
        return result

    def _episode_residue(
        self, episode_id: str, *, allow_episode_memory_parent: bool = False
    ) -> bool:
        if self._owned_cgroup_names(episode_id):
            return True
        episode_memory = self._episode_memory_cgroup_path(episode_id)
        if not os.path.exists(episode_memory):
            return allow_episode_memory_parent
        if not allow_episode_memory_parent:
            return True
        self._require_episode_memory_cgroup(
            episode_id, self._episode_memory_cgroup_identity(episode_id)
        )
        return bool(self._cgroup_child_directories(episode_memory))

    def _write_tombstone(
        self,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
        identity: Mapping[str, Any] | None,
    ) -> None:
        root = os.path.join(self.deployment["state_root"], "tombstones")
        ensure_owned_directory(root, mode=0o700, parents=True)
        path = os.path.join(root, request["episode_id"] + ".json")
        payload = canonical_json_bytes(self._tombstone_value(request, state, identity))
        if os.path.exists(path):
            if read_stable_regular_file(path, 1024 * 1024) != payload:
                raise BridgeError("bundle-bound teardown tombstone drifted")
            self._verify_tombstone(request, state, identity)
            return
        atomic_write_owned(path, payload, mode=0o600)

    def _tombstone_value(
        self,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
        identity: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "schema": "mlebench_lite_runtime_tombstone_v2",
            "episode_id": request["episode_id"],
            "base_sha256": state.get("base_sha256"),
            "bundle_identity_sha256": self.bundle_identity_sha256,
            "runtime_identity_sha256": (
                None if identity is None else canonical_sha256(identity)
            ),
            "mount_attestation_sha256": state.get("mount_attestation_sha256"),
            "mounts_released": True,
            "residue_zero": True,
        }

    def _verify_tombstone(
        self,
        request: Mapping[str, Any],
        state: Mapping[str, Any],
        identity: Mapping[str, Any] | None,
    ) -> None:
        path = os.path.join(
            self.deployment["state_root"],
            "tombstones",
            request["episode_id"] + ".json",
        )
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise BridgeError("bundle-bound teardown tombstone is absent") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise BridgeError("bundle-bound teardown tombstone is unsafe")
        payload = read_stable_regular_file(path, 1024 * 1024)
        try:
            value = strict_json_loads(payload)
        except ValueError as exc:
            raise BridgeError("bundle-bound teardown tombstone is malformed") from exc
        if (
            payload != canonical_json_bytes(value)
            or value != self._tombstone_value(request, state, identity)
        ):
            raise BridgeError("bundle-bound teardown tombstone identity drifted")


def read_mountinfo() -> list[MountRecord]:
    try:
        with open("/proc/self/mountinfo", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except (OSError, UnicodeError) as exc:
        raise BridgeError("mountinfo is unavailable") from exc
    result = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            result.append(
                MountRecord(
                    mount_id=int(fields[0]),
                    parent_id=int(fields[1]),
                    device=fields[2],
                    root=decode_mount_path(fields[3]),
                    target=decode_mount_path(fields[4]),
                    mount_options=frozenset(fields[5].split(",")),
                    optional_fields=tuple(fields[6:separator]),
                    filesystem=fields[separator + 1],
                    source=decode_mount_path(fields[separator + 2]),
                    super_options=frozenset(fields[separator + 3].split(",")),
                )
            )
        except (ValueError, IndexError) as exc:
            raise BridgeError("mountinfo is malformed") from exc
    return result


def _watchdog_cleanup_cgroups(cgroups: CgroupSet) -> None:
    pids_path = os.path.join(cgroups.children["pids"], "cgroup.procs")
    for _ in range(100):
        try:
            with open(pids_path, encoding="ascii") as handle:
                pids = [int(line.strip()) for line in handle if line.strip()]
        except FileNotFoundError:
            pids = []
        except (OSError, UnicodeError, ValueError) as exc:
            raise BridgeError("watchdog cannot read cgroup membership") from exc
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(0.02)
    else:
        raise BridgeError("watchdog could not reap execution cgroup")
    for controller in reversed(CGROUP_CONTROLLERS):
        child = cgroups.children.get(controller)
        if child is not None:
            for _ in range(100):
                try:
                    os.rmdir(child)
                    break
                except FileNotFoundError:
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EBUSY, errno.ENOTEMPTY):
                        raise BridgeError("watchdog cannot remove cgroup") from exc
                    time.sleep(0.02)
            else:
                raise BridgeError("watchdog cgroup removal timed out")
        mountpoint = cgroups.mounts.get(controller)
        if mountpoint is not None and os.path.exists(mountpoint):
            _LibC().umount(mountpoint)
    if os.path.exists(cgroups.run_dir):
        remove_owned_tree(cgroups.run_dir)


def decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def is_path_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def find_containing_mount(path: str) -> MountRecord:
    candidates = [item for item in read_mountinfo() if is_path_within(path, item.target)]
    if not candidates:
        raise BridgeError("containing mount is absent")
    return max(candidates, key=lambda item: len(item.target))


def open_absolute_directory_nofollow(path: str) -> int:
    if not isinstance(path, str) or not path.startswith("/") or os.path.normpath(path) != path:
        raise BridgeError("directory path is not canonical")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open("/", flags)
    try:
        for component in (item for item in path.split("/") if item):
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise BridgeError("directory path contains an unsafe component") from exc


def open_directory_at(parent_fd: int, name: str) -> int:
    if not name or "/" in name or name in {".", ".."}:
        raise BridgeError("directory name is unsafe")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise BridgeError("episode directory is unavailable") from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise BridgeError("episode entry is not a directory")
    return descriptor


def _open_rootfs_member(root_fd: int, member: str, *, directory: bool) -> int:
    components = [component for component in member.split("/") if component]
    if not member.startswith("/") or not components:
        raise BridgeError("rootfs member path is unsafe")
    current = os.dup(root_fd)
    try:
        for index, component in enumerate(components):
            if component in {".", ".."}:
                raise BridgeError("rootfs member path is unsafe")
            final = index == len(components) - 1
            flags = os.O_RDONLY
            if not final or directory:
                flags |= os.O_DIRECTORY
            opened = os.open(
                component,
                flags | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            os.close(current)
            current = opened
        metadata = os.fstat(current)
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected_type(metadata.st_mode)
            or metadata.st_uid != 0
            or (not directory and metadata.st_nlink != 1)
            or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise BridgeError("rootfs member identity drifted")
        return current
    except BaseException:
        os.close(current)
        raise


def verify_rootfs_tree_lock(
    *,
    rootfs_path: str,
    tree_lock_path: str,
    expected_tree_sha256: str,
    expected_lock_sha256: str,
    deadline: float | None = None,
) -> dict[str, Any]:
    _require_before_deadline(deadline, "rootfs tree verification")
    descriptor = open_absolute_directory_nofollow(rootfs_path)
    try:
        root_metadata = os.fstat(descriptor)
        mount_before = (
            find_containing_mount(rootfs_path)
            if platform.system() == "Linux"
            else MountRecord(
                mount_id=0,
                parent_id=0,
                device=str(root_metadata.st_dev),
                root="/",
                target="/",
                mount_options=frozenset(("ro",)),
                optional_fields=(),
                filesystem="synthetic",
                source="synthetic",
                super_options=frozenset(("ro",)),
            )
        )
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (platform.system() == "Linux" and root_metadata.st_uid != 0)
            or not rootfs_mount_is_read_only(rootfs_path)
        ):
            raise BridgeError("rootfs host mount is not sealed read-only")
        if rootfs_has_subordinate_mount(rootfs_path):
            raise BridgeError("rootfs contains a subordinate mount")
        validate_rootfs_mountpoints(descriptor)
        actual_inventory = stable_rootfs_tree_inventory_fd(
            descriptor, deadline=deadline
        )
    finally:
        os.close(descriptor)
    _require_before_deadline(deadline, "rootfs tree-lock verification")
    payload = read_stable_regular_file(tree_lock_path, maximum=128 * 1024 * 1024)
    lock_sha256 = hashlib.sha256(payload).hexdigest()
    if lock_sha256 != expected_lock_sha256:
        raise BridgeError("rootfs tree-lock SHA256 drifted")
    try:
        lock = strict_json_loads(payload)
    except ValueError as exc:
        raise BridgeError("rootfs tree lock is not strict JSON") from exc
    if (
        not isinstance(lock, dict)
        or set(lock) != {"schema", "rootfs_digest", "files"}
        or lock.get("schema") != ROOTFS_LOCK_SCHEMA
        or lock.get("rootfs_digest") != expected_tree_sha256
        or not isinstance(lock.get("files"), list)
        or not lock["files"]
        or payload != canonical_json_bytes(lock)
        or canonical_sha256(lock["files"]) != lock["rootfs_digest"]
        or lock["files"] != actual_inventory
    ):
        raise BridgeError("rootfs tree-lock identity drifted")
    mount_after = (
        find_containing_mount(rootfs_path)
        if platform.system() == "Linux"
        else mount_before
    )
    if mount_after != mount_before or not mount_after.read_only:
        raise BridgeError("rootfs mount changed while attesting")
    return {
        "schema": "mlebench_lite_read_only_tree_anchor_v1",
        "path": rootfs_path,
        "mount_id": mount_after.mount_id,
        "mount_device": mount_after.device,
        "mount_root": mount_after.root,
        "mount_target": mount_after.target,
        "root_stat": stable_anchor_stat(root_metadata),
        "tree_sha256": lock["rootfs_digest"],
        "tree_lock_sha256": lock_sha256,
    }


def stable_rootfs_tree_inventory(path: str) -> list[dict[str, Any]]:
    if not rootfs_mount_is_read_only(path):
        raise BridgeError("rootfs host mount is not sealed read-only")
    if rootfs_has_subordinate_mount(path):
        raise BridgeError("rootfs contains a subordinate mount")
    descriptor = open_absolute_directory_nofollow(path)
    try:
        validate_rootfs_mountpoints(descriptor)
        return stable_rootfs_tree_inventory_fd(descriptor)
    finally:
        os.close(descriptor)


def rootfs_has_subordinate_mount(path: str) -> bool:
    if platform.system() != "Linux":
        return False
    return any(
        item.target != path and is_path_within(item.target, path)
        for item in read_mountinfo()
    )


def rootfs_mount_is_read_only(path: str) -> bool:
    if platform.system() != "Linux":
        return True
    return find_containing_mount(path).read_only


def validate_rootfs_mountpoints(root_fd: int) -> None:
    for forbidden in ("host", "private"):
        try:
            os.stat(forbidden, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise BridgeError("rootfs denied path cannot be inspected") from exc
        raise BridgeError("rootfs contains a denied host path")
    for relative in (
        ".oldroot",
        "dev",
        "dev/shm",
        "home",
        "home/data",
        "home/submission",
        "home/workspace",
        "proc",
        "run",
        "run/amg_memory",
        "sys",
        "tmp",
    ):
        current = os.dup(root_fd)
        try:
            for component in relative.split("/"):
                try:
                    next_descriptor = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=current,
                    )
                except FileNotFoundError:
                    break
                except OSError as exc:
                    raise BridgeError("rootfs mount target is not a real directory") from exc
                os.close(current)
                current = next_descriptor
        finally:
            os.close(current)


def stable_rootfs_tree_inventory_fd(
    root_fd: int, *, deadline: float | None = None
) -> list[dict[str, Any]]:
    _require_before_deadline(deadline, "rootfs inventory")
    root_before = os.fstat(root_fd)
    if not stat.S_ISDIR(root_before.st_mode):
        raise BridgeError("rootfs root is not a directory")
    inventory: list[dict[str, Any]] = []
    symlinks: dict[str, str] = {}
    seen_directories: set[tuple[int, int]] = set()

    def visit(directory_fd: int, prefix: str) -> None:
        _require_before_deadline(deadline, "rootfs inventory")
        before = os.fstat(directory_fd)
        identity = (before.st_dev, before.st_ino)
        if identity in seen_directories:
            raise BridgeError("rootfs tree contains a directory alias")
        seen_directories.add(identity)
        try:
            entries = sorted(os.scandir(directory_fd), key=lambda item: item.name)
        except OSError as exc:
            raise BridgeError("rootfs directory cannot be scanned") from exc
        for entry in entries:
            _require_before_deadline(deadline, "rootfs inventory")
            name = entry.name
            if not name or "/" in name or "\x00" in name or any(
                0xD800 <= ord(character) <= 0xDFFF for character in name
            ):
                raise BridgeError("rootfs filename is unsafe")
            relative = name if not prefix else prefix + "/" + name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise BridgeError("rootfs entry cannot be stated") from exc
            common = {
                "path": relative,
                "mode": stat.S_IMODE(metadata.st_mode),
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
            }
            if stat.S_ISDIR(metadata.st_mode):
                child = open_directory_at(directory_fd, name)
                try:
                    opened = os.fstat(child)
                    if stable_stat(metadata) != stable_stat(opened):
                        raise BridgeError("rootfs directory changed while opening")
                    inventory.append({**common, "type": "directory"})
                    visit(child, relative)
                finally:
                    os.close(child)
                continue
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.readlink(name, dir_fd=directory_fd)
                    after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise BridgeError("rootfs symlink cannot be read") from exc
                if (
                    not target
                    or "\x00" in target
                    or any(0xD800 <= ord(character) <= 0xDFFF for character in target)
                    or stable_stat(metadata) != stable_stat(after)
                ):
                    raise BridgeError("rootfs symlink identity drifted")
                symlinks[relative] = target
                inventory.append({**common, "type": "symlink", "target": target})
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise BridgeError("rootfs tree contains a special file")
            try:
                file_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise BridgeError("rootfs file cannot be opened") from exc
            try:
                opened = os.fstat(file_fd)
                if stable_stat(metadata) != stable_stat(opened):
                    raise BridgeError("rootfs file changed while opening")
                digest = hashlib.sha256()
                while True:
                    _require_before_deadline(deadline, "rootfs file hashing")
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(file_fd)
                if stable_stat(opened) != stable_stat(after):
                    raise BridgeError("rootfs file changed while hashing")
                inventory.append(
                    {
                        **common,
                        "type": "regular",
                        "size": opened.st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
            finally:
                os.close(file_fd)
        after = os.fstat(directory_fd)
        if stable_stat(before) != stable_stat(after):
            raise BridgeError("rootfs directory changed while hashing")

    visit(root_fd, "")
    if not inventory:
        raise BridgeError("rootfs tree is empty")
    inventory.sort(key=lambda item: item["path"])
    _require_contained_rootfs_symlinks(symlinks)
    root_after = os.fstat(root_fd)
    if stable_stat(root_before) != stable_stat(root_after):
        raise BridgeError("rootfs root changed while hashing")
    return inventory


def _require_contained_rootfs_symlinks(symlinks: Mapping[str, str]) -> None:
    for relative, target in symlinks.items():
        if target.startswith("/"):
            raise BridgeError("rootfs symlink target escapes the locked tree")
        components = [
            *posixpath.dirname(relative).split("/"),
            *target.split("/"),
        ]
        resolved: list[str] = []
        index = 0
        expansions = 0
        while index < len(components):
            component = components[index]
            index += 1
            if component in ("", "."):
                continue
            if component == "..":
                if not resolved:
                    raise BridgeError(
                        "rootfs symlink target escapes the locked tree"
                    )
                resolved.pop()
                continue
            candidate = "/".join((*resolved, component))
            nested_target = symlinks.get(candidate)
            if nested_target is None:
                resolved.append(component)
                continue
            if nested_target.startswith("/"):
                raise BridgeError("rootfs symlink target escapes the locked tree")
            expansions += 1
            if expansions > ROOTFS_SYMLINK_EXPANSION_LIMIT:
                raise BridgeError("rootfs symlink expansion is cyclic or excessive")
            components[index:index] = nested_target.split("/")


def stable_public_tree_sha256(path: str) -> str:
    descriptor = open_absolute_directory_nofollow(path)
    try:
        return stable_public_tree_sha256_fd(descriptor)
    finally:
        os.close(descriptor)


def stable_public_tree_sha256_fd(
    root_fd: int, *, deadline: float | None = None
) -> str:
    _require_before_deadline(deadline, "public inventory")
    root_before = os.fstat(root_fd)
    if not stat.S_ISDIR(root_before.st_mode):
        raise BridgeError("public root is not a directory")
    inventory: list[dict[str, Any]] = []
    seen_directories: set[tuple[int, int]] = set()

    def visit(directory_fd: int, prefix: str) -> None:
        _require_before_deadline(deadline, "public inventory")
        before = os.fstat(directory_fd)
        identity = (before.st_dev, before.st_ino)
        if identity in seen_directories:
            raise BridgeError("public tree contains a directory alias")
        seen_directories.add(identity)
        try:
            entries = sorted(os.scandir(directory_fd), key=lambda item: item.name)
        except OSError as exc:
            raise BridgeError("public directory cannot be scanned") from exc
        for entry in entries:
            _require_before_deadline(deadline, "public inventory")
            name = entry.name
            if not name or "/" in name or "\x00" in name or any(
                0xD800 <= ord(character) <= 0xDFFF for character in name
            ):
                raise BridgeError("public filename is unsafe")
            relative = name if not prefix else prefix + "/" + name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise BridgeError("public entry cannot be stated") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise BridgeError("public tree contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                child = open_directory_at(directory_fd, name)
                try:
                    child_metadata = os.fstat(child)
                    if stable_stat(metadata) != stable_stat(child_metadata):
                        raise BridgeError("public directory changed while opening")
                    visit(child, relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise BridgeError("public tree contains a special or linked file")
            try:
                file_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise BridgeError("public file cannot be opened") from exc
            try:
                opened = os.fstat(file_fd)
                if stable_stat(metadata) != stable_stat(opened):
                    raise BridgeError("public file changed while opening")
                digest = hashlib.sha256()
                while True:
                    _require_before_deadline(deadline, "public file hashing")
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                after = os.fstat(file_fd)
                if stable_stat(opened) != stable_stat(after):
                    raise BridgeError("public file changed while hashing")
                inventory.append(
                    {"path": relative, "size": opened.st_size, "sha256": digest.hexdigest()}
                )
            finally:
                os.close(file_fd)
        after = os.fstat(directory_fd)
        if stable_stat(before) != stable_stat(after):
            raise BridgeError("public directory changed while hashing")

    visit(root_fd, "")
    if not inventory:
        raise BridgeError("public tree is empty")
    inventory.sort(key=lambda item: item["path"])
    root_after = os.fstat(root_fd)
    if stable_stat(root_before) != stable_stat(root_after):
        raise BridgeError("public root changed while hashing")
    return canonical_sha256(inventory)


def stable_stat(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def stable_anchor_stat(value: os.stat_result) -> dict[str, int]:
    return {
        "dev": value.st_dev,
        "ino": value.st_ino,
        "mode": value.st_mode,
        "nlink": value.st_nlink,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "size": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def _require_before_deadline(deadline: float | None, stage: str) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise BridgeError(f"operation deadline expired during {stage}")


def writable_usage_fd(descriptor: int) -> dict[str, int]:
    values = os.fstatvfs(descriptor)
    return {
        "bytes": max(0, (values.f_blocks - values.f_bfree) * values.f_frsize),
        "inodes": max(0, values.f_files - values.f_ffree),
    }


def read_stable_regular_file(path: str, maximum: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise BridgeError("regular file is unavailable") from exc
    try:
        return read_stable_regular_fd(descriptor, maximum)
    finally:
        os.close(descriptor)


def read_stable_regular_fd(descriptor: int, maximum: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
        raise BridgeError("regular file identity drifted")
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = bytearray()
    while len(payload) <= maximum:
        chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
    if len(payload) > maximum:
        raise BridgeError("regular file exceeds byte cap")
    after = os.fstat(descriptor)
    if stable_stat(before) != stable_stat(after):
        raise BridgeError("regular file changed while reading")
    return bytes(payload)


def write_control(directory: str, name: str, value: int | str) -> None:
    path = os.path.join(directory, name)
    try:
        with open(path, "w", encoding="ascii") as handle:
            handle.write(str(value))
    except OSError as exc:
        raise BridgeError("cgroup control write failed") from exc


def read_int(path: str) -> int:
    try:
        with open(path, encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, UnicodeError, ValueError) as exc:
        raise BridgeError("cgroup counter is unreadable") from exc


def read_key_values(path: str) -> dict[str, int]:
    result = {}
    try:
        with open(path, encoding="ascii") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    raise BridgeError("cgroup event counter is malformed")
                result[parts[0]] = int(parts[1])
    except (OSError, UnicodeError, ValueError) as exc:
        raise BridgeError("cgroup event counter is unreadable") from exc
    return result


def ensure_owned_directory(path: str, *, mode: int, parents: bool) -> None:
    try:
        if parents:
            os.makedirs(path, mode=mode, exist_ok=True)
        else:
            os.mkdir(path, mode)
        os.chmod(path, mode, follow_symlinks=False)
        metadata = os.lstat(path)
    except OSError as exc:
        raise BridgeError("owned directory is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != mode
    ):
        raise BridgeError("owned directory identity drifted")


def atomic_write_owned(path: str, payload: bytes, *, mode: int) -> None:
    directory = os.path.dirname(path)
    name = os.path.basename(path)
    temporary = ".tmp-" + str(os.getpid()) + "-" + hashlib.sha256(payload).hexdigest()[:16]
    directory_fd = open_absolute_directory_nofollow(directory)
    descriptor = None
    try:
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                mode,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            os.unlink(temporary, dir_fd=directory_fd)
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                mode,
                dir_fd=directory_fd,
            )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise BridgeError("atomic file write failed")
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        raise BridgeError("atomic file write failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except OSError:
            pass
        os.close(directory_fd)


def remove_owned_tree(path: str) -> None:
    if not os.path.exists(path):
        return
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BridgeError("owned cleanup root is unsafe")
    for current, directories, files in os.walk(path, topdown=False, followlinks=False):
        for name in files:
            target = os.path.join(current, name)
            entry = os.lstat(target)
            if stat.S_ISDIR(entry.st_mode):
                raise BridgeError("owned cleanup entry drifted")
            os.unlink(target)
        for name in directories:
            target = os.path.join(current, name)
            entry = os.lstat(target)
            if stat.S_ISLNK(entry.st_mode) or not stat.S_ISDIR(entry.st_mode):
                raise BridgeError("owned cleanup directory drifted")
            os.rmdir(target)
    os.rmdir(path)
