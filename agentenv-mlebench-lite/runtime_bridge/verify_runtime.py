#!/usr/bin/python3
"""Reproducible static and explicitly opted-in live runtime verification."""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import platform
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from .linux_runtime import (
        operation_cgroup_paths,
        read_mountinfo,
        stable_public_tree_sha256,
    )
    from .runner import (
        BridgeError,
        canonical_json_bytes,
        canonical_sha256,
        load_bundle_identity,
        strict_json_loads,
        validate_deployment,
    )
except ImportError:
    from linux_runtime import (  # type: ignore
        operation_cgroup_paths,
        read_mountinfo,
        stable_public_tree_sha256,
    )
    from runner import (  # type: ignore
        BridgeError,
        canonical_json_bytes,
        canonical_sha256,
        load_bundle_identity,
        strict_json_loads,
        validate_deployment,
    )


RESOURCE_CONTRACT = {
    "schema": "mlebench_lite_resource_contract_v2",
    "max_actions": 30,
    "max_submission_bytes": 100_000_000,
    "max_shell_timeout_ms": 3_600_000,
    "max_visible_output_bytes": 65_536,
    "episode_timeout_ms": 86_400_000,
    "max_total_execution_ms": 72_000_000,
    "cpu_limit_cores": 36,
    "memory_limit_bytes": 440_000_000_000,
    "pids_limit": 4096,
    "writable_bytes_limit": 500_000_000_000,
    "writable_inodes_limit": 2_000_000,
    "gpu_count": 1,
    "max_step_response_ms": 86_430_000,
    "submission_path": "/home/submission/submission.csv",
    "network_disabled": True,
    "read_only_public_data": True,
    "process_scope": "episode_cgroup_descendants",
    "cgroup_required": True,
    "isolated_process_group_required": True,
}
LIVE_MODES = ("native", "amg_compaction_only", "amg_memory")


def static_certificate(*, run_tests: bool = True) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    project_root = source_root.parent
    deployment_payload = (source_root / "deployment.example.json").read_bytes()
    deployment = validate_deployment(strict_json_loads(deployment_payload))
    if canonical_json_bytes(deployment) != deployment_payload.rstrip(b"\n"):
        raise BridgeError("deployment example is not canonical JSON")
    files = {}
    for name in (
        "runner.py",
        "runner_launcher.c",
        "runtime_audit.c",
        "linux_runtime.py",
        "sandbox_supervisor.c",
        "build_bundle.py",
        "verify_runtime.py",
        "deployment.example.json",
    ):
        payload = (source_root / name).read_bytes()
        files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    test_result: dict[str, Any]
    if run_tests:
        command = [
            sys.executable,
            "-m",
            "unittest",
            "-q",
            "tests.test_runtime_bridge_protocol",
            "tests.test_runtime_bridge_linux_source",
        ]
        completed = subprocess.run(
            command,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
            check=False,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(project_root),
            },
        )
        if completed.returncode != 0:
            raise BridgeError("static runtime tests failed")
        test_result = {
            "command": command,
            "returncode": completed.returncode,
            "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        }
    else:
        test_result = {"skipped": True}
    return {
        "schema": "mlebench_lite_runtime_prehost_certificate_v1",
        "status": "prehost_pass" if run_tests else "prehost_pending",
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "source_files": files,
        "resource_contract": RESOURCE_CONTRACT,
        "resource_contract_sha256": canonical_sha256(RESOURCE_CONTRACT),
        "openmle_v7_provenance": deployment["openmle_v7_provenance"],
        "targeted_tests": test_result,
        "actual_host_admission": "pending",
        "pending_external_gates": [
            "coordinator-authorized Linux/B200 runtime admission",
            "Kaggle access and 22-task checksum manifest",
            "one-task three-arm gate",
            "official host-only grader",
        ],
    }


def seal_public_fixture(public: Path) -> None:
    mount = shutil.which("mount")
    if mount is None:
        raise BridgeError("live verifier mount utility is unavailable")
    try:
        subprocess.run(
            [mount, "--bind", str(public), str(public)],
            check=True,
            capture_output=True,
            timeout=15,
        )
        subprocess.run(
            [
                mount,
                "-o",
                "remount,bind,ro,nosuid,nodev,noexec",
                str(public),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        try:
            unseal_public_fixture(public)
        except BridgeError:
            pass
        raise BridgeError("cannot create synthetic read-only public mount") from exc
    matches = [item for item in read_mountinfo() if item.target == str(public)]
    if len(matches) != 1 or not matches[0].read_only:
        raise BridgeError("synthetic public mount is not sealed read-only")


def unseal_public_fixture(public: Path) -> None:
    matches = [item for item in read_mountinfo() if item.target == str(public)]
    if not matches:
        return
    umount = shutil.which("umount")
    if umount is None:
        raise BridgeError("live verifier unmount utility is unavailable")
    try:
        subprocess.run(
            [umount, str(public)],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BridgeError("cannot release synthetic public mount") from exc


def live_certificate(*, bundle_path: Path, live_root: Path, gpu_uuid: str) -> dict[str, Any]:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise BridgeError("live verification requires Linux/x86_64")
    if os.geteuid() != 0:
        raise BridgeError("live verification requires root")
    live_root = live_root.resolve(strict=True)
    metadata = live_root.lstat()
    if (
        live_root.parent == Path("/")
        or not live_root.name.startswith("mlebridge-live-")
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != 0
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or any(live_root.iterdir())
    ):
        raise BridgeError("live root must be a dedicated empty root-owned directory")
    bundle = load_bundle_identity(bundle_path, expected_uid=0)
    if bundle.deployment["gpu"]["uuid"] != gpu_uuid:
        raise BridgeError("live GPU UUID does not match the bundle")
    for configured in (
        Path(bundle.deployment["state_root"]),
        Path(bundle.deployment["episodes_root"]),
    ):
        configured_metadata = configured.lstat()
        if (
            not stat.S_ISDIR(configured_metadata.st_mode)
            or configured_metadata.st_uid != 0
            or stat.S_IMODE(configured_metadata.st_mode) != 0o700
        ):
            raise BridgeError("configured runtime root is unsafe")

    synthetic = live_root / ("session-" + uuid.uuid4().hex)
    public = synthetic / "public"
    private = synthetic / "private"
    host_only = synthetic / "host-only"
    public.mkdir(parents=True, mode=0o700)
    private.mkdir(mode=0o700)
    host_only.mkdir(mode=0o700)
    (public / "train.csv").write_text("x,y\n1,0\n", encoding="utf-8")
    (private / "secret.txt").write_text("never-visible", encoding="utf-8")
    (host_only / "sentinel.txt").write_text("host-only", encoding="utf-8")
    public_digest = stable_public_tree_sha256(str(public))
    results: dict[str, Any] = {}
    identities = set()
    public_sealed = False
    try:
        seal_public_fixture(public)
        public_sealed = True
        for mode in LIVE_MODES:
            episode_id = uuid.uuid4().hex
            episode = Path(bundle.deployment["episodes_root"]) / episode_id
            workspace = episode / "workspace"
            submission = episode / "submission"
            workspace.mkdir(parents=True, mode=0o700)
            submission.mkdir(mode=0o700)
            request = {
                "schema": "mlebench_lite_sandbox_request_v3",
                "episode_id": episode_id,
                "competition_id": "synthetic-runtime-admission",
                "mode": mode,
                "resource_contract": RESOURCE_CONTRACT,
                "resource_contract_sha256": canonical_sha256(RESOURCE_CONTRACT),
                "public_root": str(public),
                "public_tree_sha256": public_digest,
                "workspace_root": str(workspace),
                "submission_root": str(submission),
            }
            if mode == "amg_memory":
                memory = episode / "external-memory"
                memory.mkdir(mode=0o700)
                request["external_memory_root"] = str(memory)
            cleanup_required = True
            try:
                attestation = invoke(bundle_path, "attest", request)
                identities.add(
                    (
                        attestation["runner_sha256"],
                        attestation["runtime_digest"],
                        attestation["resource_contract_sha256"],
                    )
                )
                mode_result: dict[str, Any] = {"attestation": attestation}
                mode_result.update(
                    run_live_adversarial_sequence(
                        bundle_path,
                        request,
                        gpu_uuid,
                        private_root=private,
                        host_only_root=host_only,
                    )
                )
                cleanup_required = False
                results[mode] = mode_result
            finally:
                if cleanup_required:
                    cleanup_live_episode(bundle_path, request)
                if episode.exists():
                    if any(item.target == str(episode) for item in read_mountinfo()):
                        raise BridgeError("live episode mount survived cleanup")
                    shutil.rmtree(episode)
        if len(identities) != 1:
            raise BridgeError("three arms do not share one runtime identity")
    finally:
        if public_sealed:
            unseal_public_fixture(public)
        if synthetic.exists():
            shutil.rmtree(synthetic)
    if any(
        item.target.startswith(str(Path(bundle.deployment["episodes_root"])))
        for item in read_mountinfo()
    ):
        raise BridgeError("episode mount residue remains after live verification")
    return {
        "schema": "mlebench_lite_runtime_live_admission_v1",
        "status": "pass",
        "gpu_uuid": gpu_uuid,
        "runtime_identity_count": len(identities),
        "arms": results,
        "actual_host_admission": "pass",
        "synthetic_only": True,
    }


def run_live_adversarial_sequence(
    bundle_path: Path,
    request: dict[str, Any],
    gpu_uuid: str,
    *,
    private_root: Path,
    host_only_root: Path,
) -> dict[str, Any]:
    def execute(command: str, timeout_ms: int = 20_000) -> dict[str, Any]:
        return invoke(
            bundle_path,
            "execute",
            {
                **request,
                "operation_id": str(uuid.uuid4()),
                "command": command,
                "timeout_ms": timeout_ms,
            },
            timeout=max(30.0, timeout_ms / 1000.0 + 5.0),
        )

    memory_enabled = request["mode"] == "amg_memory"
    write_command = (
        "printf persistent > /home/workspace/marker && "
        "printf 'id,prediction\\n1,0\\n' > /home/submission/submission.csv"
    )
    if memory_enabled:
        write_command += " && printf memory > /run/amg_memory/note"
    write = execute(write_command)
    if write["returncode"] != 0:
        raise BridgeError("live persistent write failed")
    read_command = "test \"$(cat /home/workspace/marker)\" = persistent && "
    if memory_enabled:
        read_command += "test \"$(cat /run/amg_memory/note)\" = memory && "
    else:
        read_command += "test ! -e /run/amg_memory && "
    read_command += (
        "test ! -e /private && test ! -e /host && test ! -e "
        + shlex.quote(str(private_root))
        + " && test ! -e "
        + shlex.quote(str(host_only_root))
    )
    read = execute(read_command)
    if read["returncode"] != 0:
        raise BridgeError("live persistence or private-path isolation failed")
    public_write = execute("printf bad > /home/data/forbidden")
    if public_write["returncode"] == 0:
        raise BridgeError("public mount accepted a write")
    symlink_escape = execute(
        "ln -s "
        + shlex.quote(str(host_only_root))
        + " /home/workspace/escape && test ! -e /home/workspace/escape/sentinel.txt"
    )
    if symlink_escape["returncode"] != 0:
        raise BridgeError("sandbox root symlink control failed")

    network = {}
    families = {
        "inet": int(socket.AF_INET),
        "inet6": int(socket.AF_INET6),
        "packet": int(getattr(socket, "AF_PACKET", 17)),
        "netlink": int(getattr(socket, "AF_NETLINK", 16)),
        "vsock": int(getattr(socket, "AF_VSOCK", 40)),
    }
    for name, family in families.items():
        result = execute(
            "exec python3 -c 'import socket; socket.socket("
            + str(family)
            + ", socket.SOCK_STREAM)'"
        )
        if result["returncode"] != 128 + signal_number("SIGSYS"):
            raise BridgeError("non-UNIX socket family was not trapped")
        network[name] = result["returncode"]
    x32_socket = execute(
        "exec python3 -c 'import ctypes; ctypes.CDLL(None).syscall(0x40000000 + 41, 2, 1, 0)'"
    )
    if x32_socket["returncode"] != 128 + signal_number("SIGSYS"):
        raise BridgeError("x32 socket syscall was not trapped")
    network["x32_inet"] = x32_socket["returncode"]
    unix_socket = execute(
        "exec python3 -c 'import socket; a,b=socket.socketpair(socket.AF_UNIX); a.close(); b.close()'"
    )
    if unix_socket["returncode"] != 0:
        raise BridgeError("AF_UNIX socketpair was unexpectedly denied")

    gpu = execute(
        "CUDA_VISIBLE_DEVICES=999 NVIDIA_VISIBLE_DEVICES=all "
        "nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits"
    )
    visible = [line.strip() for line in gpu["stdout"].splitlines() if line.strip()]
    if gpu["returncode"] != 0 or visible != [gpu_uuid]:
        raise BridgeError("NVML-visible GPU identity drifted")
    cuda = execute(
        "CUDA_VISIBLE_DEVICES=0 python3 -c 'import torch; "
        "print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'"
    )
    if cuda["returncode"] != 0 or cuda["stdout"].splitlines()[:1] != ["1"]:
        raise BridgeError("CUDA-visible GPU count drifted")
    capacity = execute(
        "python3 -c 'import os; value=os.statvfs(\"/home/workspace\"); "
        "print(value.f_blocks * value.f_frsize); print(value.f_files)'"
    )
    capacity_lines = capacity["stdout"].splitlines()
    try:
        capacity_bytes, capacity_inodes = map(int, capacity_lines)
    except (TypeError, ValueError) as exc:
        raise BridgeError("writable capacity output drifted") from exc
    if (
        capacity["returncode"] != 0
        or not (
            500_000_000_000
            <= capacity_bytes
            < 500_000_000_000 + os.sysconf("SC_PAGE_SIZE")
        )
        or capacity_inodes != 2_000_000
    ):
        raise BridgeError("writable byte/inode capacity drifted")
    writable_before_splice = capacity["receipt"]["resource_cumulative"][
        "writable_bytes"
    ]
    concurrent_close_splice_high_water = execute(
        "python3 -c 'import errno,os,threading,time; "
        "fd=os.open(\"/home/workspace\", "
        "os.O_RDWR|os.O_CLOEXEC|os.O_TMPFILE, 0o600); "
        "r,w=os.pipe(); ready=threading.Event(); outcome=[]\n"
        "def consume():\n"
        " ready.set()\n"
        " try: outcome.append(os.splice(r,fd,4096))\n"
        " except OSError as exc: outcome.append(-exc.errno)\n"
        "t=threading.Thread(target=consume); t.start(); "
        "ready.wait(1); time.sleep(0.05); os.close(fd); "
        "os.write(w,b\"x\"*4096); os.close(w); t.join(2); "
        "assert not t.is_alive() and len(outcome)==1; "
        "print(outcome[0]); os.close(r)'",
        timeout_ms=10_000,
    )
    try:
        splice_result = int(concurrent_close_splice_high_water["stdout"].strip())
    except (TypeError, ValueError) as exc:
        raise BridgeError("concurrent close/splice result drifted") from exc
    splice_cumulative = concurrent_close_splice_high_water["receipt"][
        "resource_cumulative"
    ]
    if concurrent_close_splice_high_water["returncode"] != 0:
        raise BridgeError("concurrent close/splice adversary did not terminate")
    if splice_result > 0 and (
        splice_cumulative["writable_bytes"]
        < writable_before_splice + splice_result
    ):
        raise BridgeError("concurrent close/splice writable high-water was missed")
    if splice_result <= 0 and splice_result not in {-errno.EBADF, -errno.EINTR}:
        raise BridgeError("concurrent close/splice cancellation drifted")
    create_then_delete_high_water = execute(
        "python3 -c 'import os; "
        "p=\"/home/workspace/high-water.bin\"; "
        "f=open(p,\"wb\"); f.write(b\"x\"*(32*1024*1024)); "
        "f.flush(); os.fsync(f.fileno()); f.close(); "
        "d=\"/home/workspace/high-water-inodes\"; os.mkdir(d); "
        "[open(d+\"/\"+str(i),\"wb\").close() for i in range(256)]; "
        "[os.unlink(d+\"/\"+str(i)) for i in range(256)]; "
        "os.rmdir(d); os.unlink(p)'"
    )
    high_water = create_then_delete_high_water["receipt"]["resource_cumulative"]
    if (
        create_then_delete_high_water["returncode"] != 0
        or high_water["writable_bytes"] < 32 * 1024 * 1024
        or high_water["writable_inodes"] < 256
    ):
        raise BridgeError("create-then-delete writable high-water was not captured")
    exec_tmpfile_high_water = execute(
        "python3 -c 'import os; "
        "fd=os.open(\"/home/workspace\", "
        "os.O_RDWR|os.O_CLOEXEC|os.O_TMPFILE, 0o600); "
        "block=b\"x\"*(1024*1024); "
        "[os.write(fd, block) for _ in range(48)]; os.fsync(fd); "
        "os.execlp(\"python3\", \"python3\", \"-c\", \"pass\")'"
    )
    exec_high_water = exec_tmpfile_high_water["receipt"]["resource_cumulative"]
    if (
        exec_tmpfile_high_water["returncode"] != 0
        or exec_high_water["writable_bytes"] < 48 * 1024 * 1024
    ):
        raise BridgeError("CLOEXEC anonymous tmpfile high-water was not captured")
    mmap_tmpfile_high_water = execute(
        "python3 -c 'import mmap,os; "
        "fd=os.open(\"/home/workspace\", "
        "os.O_RDWR|os.O_CLOEXEC|os.O_TMPFILE, 0o600); "
        "size=64*1024*1024; os.ftruncate(fd,size); "
        "mapping=mmap.mmap(fd,size,flags=mmap.MAP_SHARED,"
        "prot=mmap.PROT_READ|mmap.PROT_WRITE); os.close(fd); "
        "block=b\"x\"*(1024*1024); "
        "[mapping.write(block) for _ in range(64)]; "
        "mapping.flush(); mapping.close()'"
    )
    mmap_high_water = mmap_tmpfile_high_water["receipt"]["resource_cumulative"]
    if (
        mmap_tmpfile_high_water["returncode"] != 0
        or mmap_high_water["writable_bytes"] < 64 * 1024 * 1024
    ):
        raise BridgeError("mmap anonymous tmpfile high-water was not captured")
    threaded_execve = execute(
        "exec python3 -c 'import os,threading,time; ready=threading.Event()\n"
        "def linger():\n"
        " ready.set()\n"
        " while True: time.sleep(1)\n"
        "def replace():\n"
        " assert ready.wait(1)\n"
        " os.execvp(\"python3\", [\"python3\", \"-c\", "
        "\"print(\\\"threaded-exec-ok\\\")\"])\n"
        "threading.Thread(target=linger,daemon=True).start(); "
        "worker=threading.Thread(target=replace); worker.start(); worker.join()'",
        timeout_ms=10_000,
    )
    if (
        threaded_execve["returncode"] != 0
        or threaded_execve["stdout"].strip() != "threaded-exec-ok"
    ):
        raise BridgeError("nonleader threaded execve state drifted")
    threaded_failed_execve = execute(
        "exec python3 -c 'import os,threading,time; ready=threading.Event()\n"
        "def linger():\n"
        " ready.set()\n"
        " while True: time.sleep(1)\n"
        "def replace():\n"
        " assert ready.wait(1)\n"
        " try: os.execv(\"/definitely-missing-mlebridge\", [\"missing\"])\n"
        " except FileNotFoundError: print(\"threaded-failed-exec-ok\")\n"
        "threading.Thread(target=linger,daemon=True).start(); "
        "worker=threading.Thread(target=replace); worker.start(); worker.join()'",
        timeout_ms=10_000,
    )
    if (
        threaded_failed_execve["returncode"] != 0
        or threaded_failed_execve["stdout"].strip()
        != "threaded-failed-exec-ok"
    ):
        raise BridgeError("failed nonleader threaded execve state drifted")
    threaded_exit_group = execute(
        "exec python3 -c 'import ctypes,threading,time; "
        "ready=threading.Event(); libc=ctypes.CDLL(None)\n"
        "def linger():\n"
        " ready.set()\n"
        " while True: time.sleep(1)\n"
        "def terminate():\n"
        " assert ready.wait(1)\n"
        " libc.syscall(231, 23)\n"
        " raise RuntimeError(\"exit_group returned\")\n"
        "threading.Thread(target=linger,daemon=True).start(); "
        "worker=threading.Thread(target=terminate); worker.start(); worker.join()'",
        timeout_ms=10_000,
    )
    if threaded_exit_group["returncode"] != 23:
        raise BridgeError("nonleader threaded exit_group state drifted")
    timeout = execute("sleep 60 & wait", timeout_ms=100)
    if timeout["timed_out"] is not True:
        raise BridgeError("timeout/reap gate did not fire")
    cascade = kill_runner_cascade(bundle_path, request)
    freeze = invoke(
        bundle_path,
        "freeze",
        {**request, "operation_id": str(uuid.uuid4())},
    )
    submission = Path(request["submission_root"]) / "submission.csv"
    if submission.read_text(encoding="utf-8") != "id,prediction\n1,0\n":
        raise BridgeError("host submission staging readback drifted")
    teardown = invoke(
        bundle_path,
        "teardown",
        {**request, "operation_id": str(uuid.uuid4())},
    )
    return {
        "persistent_write": write["receipt"],
        "persistent_read": read["receipt"],
        "public_write_returncode": public_write["returncode"],
        "network_denials": network,
        "unix_socket_returncode": unix_socket["returncode"],
        "gpu_stdout_sha256": hashlib.sha256(gpu["stdout"].encode()).hexdigest(),
        "cuda_stdout_sha256": hashlib.sha256(cuda["stdout"].encode()).hexdigest(),
        "writable_capacity": {
            "bytes": capacity_bytes,
            "inodes": capacity_inodes,
        },
        "concurrent_close_splice_high_water": {
            "splice_result": splice_result,
            "resource_cumulative": splice_cumulative,
        },
        "timeout": timeout["receipt"],
        "create_then_delete_high_water": high_water,
        "exec_tmpfile_high_water": exec_high_water,
        "mmap_tmpfile_high_water": mmap_high_water,
        "threaded_execve": threaded_execve["receipt"],
        "threaded_failed_execve": threaded_failed_execve["receipt"],
        "threaded_exit_group": threaded_exit_group["receipt"],
        "kill_runner_cascade": cascade,
        "freeze": freeze,
        "teardown": teardown,
    }


def kill_runner_cascade(
    bundle_path: Path, request: dict[str, Any]
) -> dict[str, Any]:
    bundle = load_bundle_identity(bundle_path, expected_uid=0)
    operation_id = str(uuid.uuid4())
    operation_hex = operation_id.replace("-", "")
    name = f"mlebridge-{request['episode_id']}-{operation_hex}"
    execution_request = {
        **request,
        "operation_id": operation_id,
        "command": "sleep 60 & wait",
        "timeout_ms": 60_000,
    }
    process = subprocess.Popen(
        [
            str(bundle_path),
            "--expected-runtime-digest",
            bundle.identity.runtime_digest,
            "execute",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        start_new_session=True,
    )
    assert process.stdin is not None
    process.stdin.write(canonical_json_bytes(execution_request))
    process.stdin.close()
    process.stdin = None
    runs_root = Path(bundle.deployment["state_root"]) / "runs"
    path_candidates = operation_cgroup_paths(request["episode_id"], name)
    cgroup_paths = {
        controller: Path(paths[0])
        for controller, paths in path_candidates.items()
    }
    residue_paths = [
        Path(path) for paths in path_candidates.values() for path in paths
    ]
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BridgeError("runner exited before death-cascade injection")
        if all(path.is_dir() for path in cgroup_paths.values()) and (
            runs_root / name
        ).is_dir():
            break
        time.sleep(0.02)
    else:
        process.kill()
        process.wait(timeout=5)
        raise BridgeError("runner death-cascade cgroup did not become observable")
    pids = []
    try:
        pids = [
            int(line)
            for line in (cgroup_paths["pids"] / "cgroup.procs").read_text(
                encoding="ascii"
            ).splitlines()
            if line
        ]
    except (OSError, UnicodeError, ValueError) as exc:
        process.kill()
        process.wait(timeout=5)
        raise BridgeError("runner death-cascade membership is unreadable") from exc
    process.kill()
    process.wait(timeout=10)
    stdout = b"" if process.stdout is None else process.stdout.read()
    stderr = b"" if process.stderr is None else process.stderr.read()
    residue_deadline = time.monotonic() + 15.0
    while time.monotonic() < residue_deadline:
        if not any(path.exists() for path in residue_paths) and not (
            runs_root / name
        ).exists():
            break
        time.sleep(0.05)
    else:
        raise BridgeError("runner death watchdog left owned execution residue")
    if process.returncode != -signal.SIGKILL:
        raise BridgeError("runner death-cascade signal drifted")
    return {
        "runner_returncode": process.returncode,
        "captured_cgroup_processes": len(pids),
        "owned_cgroup_residue": 0,
        "owned_run_directory_residue": 0,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
    }


def signal_number(name: str) -> int:
    return int(getattr(signal, name))


def cleanup_live_episode(bundle_path: Path, request: dict[str, Any]) -> None:
    teardown_request = {**request, "operation_id": str(uuid.uuid4())}
    try:
        invoke(bundle_path, "teardown", teardown_request)
    except BridgeError:
        try:
            invoke(bundle_path, "attest", request)
            invoke(
                bundle_path,
                "teardown",
                {**request, "operation_id": str(uuid.uuid4())},
            )
        except BridgeError:
            episode_root = str(Path(request["workspace_root"]).parent)
            if any(item.target == episode_root for item in read_mountinfo()):
                raise BridgeError("live verifier could not release an episode mount")


def invoke(
    runner: Path,
    operation: str,
    request: dict[str, Any],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    bundle = load_bundle_identity(runner, expected_uid=0)
    completed = subprocess.run(
        [
            str(runner),
            "--expected-runtime-digest",
            bundle.identity.runtime_digest,
            operation,
        ],
        input=canonical_json_bytes(request),
        stdin=None,
        capture_output=True,
        timeout=timeout,
        check=False,
        close_fds=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"},
        start_new_session=True,
    )
    if completed.returncode != 0 or completed.stderr:
        raise BridgeError("live runner invocation failed")
    try:
        value = strict_json_loads(completed.stdout)
    except ValueError as exc:
        raise BridgeError("live runner response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise BridgeError("live runner response is not an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--live-root", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--allow-live-destructive", action="store_true")
    arguments = parser.parse_args(argv)
    live_requested = any(
        value is not None
        for value in (arguments.bundle, arguments.live_root, arguments.gpu_uuid)
    )
    live_requested = live_requested or arguments.allow_live_destructive
    if live_requested:
        if not (
            arguments.bundle
            and arguments.live_root
            and arguments.gpu_uuid
            and arguments.allow_live_destructive
        ):
            raise BridgeError("live verification requires every explicit live flag")
        certificate = live_certificate(
            bundle_path=arguments.bundle,
            live_root=arguments.live_root,
            gpu_uuid=arguments.gpu_uuid,
        )
    else:
        certificate = static_certificate(run_tests=True)
    payload = canonical_json_bytes(certificate)
    if arguments.output is not None:
        arguments.output.write_bytes(payload + b"\n")
    sys.stdout.buffer.write(payload + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError:
        raise SystemExit(2)
