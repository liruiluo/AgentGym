#!/usr/bin/python3
"""Build one immutable MLE-bench Lite runtime bridge bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from .linux_runtime import ROOTFS_LOCK_SCHEMA, stable_rootfs_tree_inventory
    from .runner import (
        OPENMLE_V7_ARTIFACT_LOCK_SHA256,
        OPENMLE_V7_SUPERVISOR_SHA256,
        BridgeError,
        canonical_json_bytes,
        canonical_sha256,
        strict_json_loads,
        validate_deployment,
    )
except ImportError:  # Installed/source direct execution.
    from linux_runtime import (  # type: ignore
        ROOTFS_LOCK_SCHEMA,
        stable_rootfs_tree_inventory,
    )
    from runner import (  # type: ignore
        OPENMLE_V7_ARTIFACT_LOCK_SHA256,
        OPENMLE_V7_SUPERVISOR_SHA256,
        BridgeError,
        canonical_json_bytes,
        canonical_sha256,
        strict_json_loads,
        validate_deployment,
    )


COMPILE_FLAGS = (
    "-std=c11",
    "-O2",
    "-pipe",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wformat",
    "-Wformat-security",
    "-Werror=format-security",
    "-fPIE",
    "-pie",
    "-fstack-protector-strong",
    "-D_FORTIFY_SOURCE=2",
    "-fno-ident",
    "-fno-record-gcc-switches",
    "-Wl,--build-id=none",
    "-Wl,-z,relro,-z,now,-z,noexecstack",
)

AUDIT_COMPILE_FLAGS = (
    "-std=c11",
    "-O2",
    "-pipe",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-fPIC",
    "-shared",
    "-fno-builtin",
    "-fno-ident",
    "-fno-record-gcc-switches",
    "-nostdlib",
    "-Wl,--build-id=none",
    "-Wl,-z,defs,-z,relro,-z,now,-z,noexecstack",
)

ELF_CLOSURE_SCHEMA = "mlebench_lite_rootfs_elf_closure_v1"
RUNTIME_AUDIT_RELATIVE_PATH = "lib/mlebench-lite-runtime-audit.so"


def build_rootfs_tree_lock(rootfs: Path) -> dict[str, Any]:
    files = stable_rootfs_tree_inventory(str(Path(rootfs)))
    return {
        "schema": ROOTFS_LOCK_SCHEMA,
        "rootfs_digest": canonical_sha256(files),
        "files": files,
    }


def write_rootfs_tree_lock(rootfs: Path, output: Path) -> dict[str, Any]:
    output = Path(output)
    lock = build_rootfs_tree_lock(rootfs)
    payload = canonical_json_bytes(lock)
    descriptor: int | None = None
    created = False
    complete = False
    try:
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o444,
        )
        created = True
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise BridgeError("rootfs tree-lock write failed")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        complete = True
    except OSError as exc:
        raise BridgeError("rootfs tree-lock output is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created and not complete:
            try:
                output.unlink()
            except FileNotFoundError:
                pass
    return {
        "schema": "mlebench_lite_rootfs_tree_lock_receipt_v1",
        "output": str(output.resolve()),
        "rootfs_digest": lock["rootfs_digest"],
        "rootfs_tree_lock_sha256": hashlib.sha256(payload).hexdigest(),
        "file_count": len(lock["files"]),
    }


def build_bundle(
    *,
    source_root: Path,
    deployment_path: Path,
    output_root: Path,
    compiler: str = "cc",
    supervisor_binary: Path | None = None,
) -> dict[str, Any]:
    source_root = Path(source_root)
    deployment_path = Path(deployment_path)
    output_root = Path(output_root)
    if output_root.exists():
        raise BridgeError("bundle output already exists")
    deployment_payload = _read_bounded(deployment_path, 1024 * 1024)
    try:
        deployment_value = strict_json_loads(deployment_payload)
    except ValueError as exc:
        raise BridgeError("deployment input is not strict JSON") from exc
    deployment = validate_deployment(deployment_value)
    canonical_deployment = canonical_json_bytes(deployment)
    if deployment_payload not in {
        canonical_deployment,
        canonical_deployment + b"\n",
    }:
        raise BridgeError("deployment input is not canonical JSON")
    required_sources = {
        "runner.py": source_root / "runner.py",
        "runner_launcher.c": source_root / "runner_launcher.c",
        "runtime_audit.c": source_root / "runtime_audit.c",
        "linux_runtime.py": source_root / "linux_runtime.py",
        "sandbox_supervisor.c": source_root / "sandbox_supervisor.c",
    }
    for path in required_sources.values():
        if not path.is_file() or path.is_symlink():
            raise BridgeError("runtime bridge source is unavailable")
    output_root.mkdir(mode=0o700)
    bin_root = output_root / "bin"
    bin_root.mkdir(mode=0o700)
    lib_root = output_root / "lib"
    lib_root.mkdir(mode=0o700)
    try:
        if supervisor_binary is None:
            supervisor_payload, build_metadata = _compile_reproducibly(
                required_sources["sandbox_supervisor.c"], compiler, "supervisor"
            )
        else:
            supervisor_payload = _read_bounded(
                Path(supervisor_binary), 128 * 1024 * 1024
            )
            build_metadata = {
                "schema": "mlebench_lite_supervisor_build_v1",
                "platform": platform.platform(),
                "compiler": None,
                "compiler_version": None,
                "flags": [],
                "independent_builds": 0,
                "byte_identical": False,
                "prebuilt_test_fixture": True,
            }
        if platform.system() == "Linux" and platform.machine() in {
            "x86_64",
            "AMD64",
        }:
            audit_payload, audit_metadata = _compile_reproducibly(
                required_sources["runtime_audit.c"],
                compiler,
                "runtime_audit",
                base_flags=AUDIT_COMPILE_FLAGS,
                extra_flags=_audit_compile_flags(deployment),
            )
            launcher_flags = _launcher_compile_flags(
                deployment,
                runner_source_sha256=_sha256(required_sources["runner.py"]),
                linux_runtime_sha256=_sha256(
                    required_sources["linux_runtime.py"]
                ),
                runtime_audit_sha256=hashlib.sha256(audit_payload).hexdigest(),
            )
            launcher_payload, launcher_metadata = _compile_reproducibly(
                required_sources["runner_launcher.c"],
                compiler,
                "runner_launcher",
                extra_flags=launcher_flags,
            )
        elif supervisor_binary is not None:
            # Bundle unit tests on non-Linux hosts use a non-executable fixture.
            # A production receipt can never take this branch.
            audit_payload = supervisor_payload
            audit_metadata = {
                "schema": "mlebench_lite_runtime_audit_build_v1",
                "platform": platform.platform(),
                "compiler": None,
                "compiler_version": None,
                "flags": [],
                "independent_builds": 0,
                "byte_identical": False,
                "prebuilt_test_fixture": True,
            }
            launcher_flags = _launcher_compile_flags(
                deployment,
                runner_source_sha256=_sha256(required_sources["runner.py"]),
                linux_runtime_sha256=_sha256(
                    required_sources["linux_runtime.py"]
                ),
                runtime_audit_sha256=hashlib.sha256(audit_payload).hexdigest(),
            )
            launcher_payload = supervisor_payload
            launcher_metadata = {
                "schema": "mlebench_lite_runner_launcher_build_v1",
                "platform": platform.platform(),
                "compiler": None,
                "compiler_version": None,
                "flags": launcher_flags,
                "independent_builds": 0,
                "byte_identical": False,
                "prebuilt_test_fixture": True,
            }
        else:
            raise BridgeError("production runner launcher build requires Linux/x86_64")
        if _rootfs_elf_targets_exist(deployment):
            elf_closure = verify_elf_dependency_closure(deployment)
        elif supervisor_binary is not None:
            elf_closure = {
                "schema": ELF_CLOSURE_SCHEMA,
                "rootfs_digest": deployment["rootfs_digest"],
                "loader": deployment["rootfs_loader_path"],
                "library_paths": deployment["rootfs_library_paths"],
                "cache_inhibited": True,
                "targets": [],
                "prebuilt_test_fixture": True,
            }
        else:
            raise BridgeError("sealed rootfs ELF targets are unavailable")
        copies = {
            "runner.py": required_sources["runner.py"],
            "runner_launcher.c": required_sources["runner_launcher.c"],
            "runtime_audit.c": required_sources["runtime_audit.c"],
            "linux_runtime.py": required_sources["linux_runtime.py"],
            "sandbox_supervisor.c": required_sources["sandbox_supervisor.c"],
        }
        for relative, source in copies.items():
            target = output_root / Path(relative)
            shutil.copyfile(source, target)
        (output_root / "sandbox-runner").write_bytes(launcher_payload)
        (output_root / "bin" / "mlebench-lite-sandbox-supervisor").write_bytes(
            supervisor_payload
        )
        (output_root / Path(RUNTIME_AUDIT_RELATIVE_PATH)).write_bytes(audit_payload)
        (output_root / "deployment.json").write_bytes(canonical_json_bytes(deployment))
        (output_root / "rootfs-elf-closure.json").write_bytes(
            canonical_json_bytes(elf_closure)
        )
        provenance = {
            "schema": "mlebench_lite_runtime_build_provenance_v2",
            "supervisor_build": build_metadata,
            "runner_launcher_build": launcher_metadata,
            "runtime_audit_build": audit_metadata,
            "source_sha256": {
                name: _sha256(path) for name, path in sorted(required_sources.items())
            },
            "runner_launcher_sha256": hashlib.sha256(launcher_payload).hexdigest(),
            "runtime_audit_sha256": hashlib.sha256(audit_payload).hexdigest(),
            "supervisor_sha256": hashlib.sha256(supervisor_payload).hexdigest(),
            "openmle_v7_provenance": {
                "artifact_lock_sha256": OPENMLE_V7_ARTIFACT_LOCK_SHA256,
                "supervisor_sha256": OPENMLE_V7_SUPERVISOR_SHA256,
            },
        }
        (output_root / "build-provenance.json").write_bytes(
            canonical_json_bytes(provenance)
        )
        executable = {
            "sandbox-runner",
            "bin/mlebench-lite-sandbox-supervisor",
        }
        members = []
        for relative in sorted(
            (
                "sandbox-runner",
                "runner.py",
                "runner_launcher.c",
                "runtime_audit.c",
                "linux_runtime.py",
                "sandbox_supervisor.c",
                "bin/mlebench-lite-sandbox-supervisor",
                RUNTIME_AUDIT_RELATIVE_PATH,
                "deployment.json",
                "rootfs-elf-closure.json",
                "build-provenance.json",
            )
        ):
            path = output_root / Path(relative)
            path.chmod(0o555 if relative in executable else 0o444)
            members.append({"path": relative, "sha256": _sha256(path)})
        lock = {
            "schema": "mlebench_lite_runtime_artifact_lock_v1",
            "files": members,
            "deployment_schema": "mlebench_lite_runtime_bridge_deployment_v1",
            "openmle_v7_provenance": {
                "artifact_lock_sha256": OPENMLE_V7_ARTIFACT_LOCK_SHA256,
                "supervisor_sha256": OPENMLE_V7_SUPERVISOR_SHA256,
            },
        }
        lock_path = output_root / "artifact-lock.json"
        lock_path.write_bytes(canonical_json_bytes(lock))
        lock_path.chmod(0o444)
        bin_root.chmod(0o555)
        lib_root.chmod(0o555)
        output_root.chmod(0o555)
        return {
            "schema": "mlebench_lite_runtime_bundle_receipt_v1",
            "bundle_root": str(output_root.resolve()),
            "runner_path": str((output_root / "sandbox-runner").resolve()),
            "runner_sha256": next(
                item["sha256"] for item in members if item["path"] == "sandbox-runner"
            ),
            "runtime_digest": canonical_sha256(lock),
            "supervisor_sha256": _sha256(
                output_root / "bin" / "mlebench-lite-sandbox-supervisor"
            ),
            "prebuilt_test_fixture": supervisor_binary is not None,
        }
    except BaseException:
        if output_root.exists():
            for current, directories, files in os.walk(
                output_root, topdown=False, followlinks=False
            ):
                os.chmod(current, 0o700)
                for name in files:
                    path = Path(current) / name
                    if not path.is_symlink():
                        path.chmod(0o600)
                    path.unlink()
                for name in directories:
                    path = Path(current) / name
                    path.chmod(0o700)
                    path.rmdir()
            output_root.rmdir()
        raise


def _compile_reproducibly(
    source: Path,
    compiler: str,
    label: str,
    *,
    base_flags: tuple[str, ...] = COMPILE_FLAGS,
    extra_flags: list[str] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise BridgeError("production supervisor build requires Linux/x86_64")
    try:
        version = subprocess.run(
            [compiler, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            timeout=10,
        ).stdout.decode("utf-8", "strict").splitlines()[0]
    except (OSError, subprocess.SubprocessError, UnicodeError, IndexError) as exc:
        raise BridgeError("compiler identity is unavailable") from exc
    with tempfile.TemporaryDirectory(prefix="mlebridge-build-") as temporary_name:
        temporary = Path(temporary_name)
        outputs = []
        for index in (1, 2):
            output = temporary / f"{label}-{index}"
            command = [
                compiler,
                *base_flags,
                *(extra_flags or []),
                f"-ffile-prefix-map={source.parent}=.",
                "-o",
                str(output),
                str(source),
            ]
            try:
                subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                    timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise BridgeError(f"{label} compilation failed") from exc
            outputs.append(output.read_bytes())
    if outputs[0] != outputs[1]:
        raise BridgeError(f"independent {label} builds differ")
    return outputs[0], {
        "schema": f"mlebench_lite_{label}_build_v1",
        "platform": "linux/amd64",
        "compiler": compiler,
        "compiler_version": version,
        "flags": [*base_flags, *(extra_flags or [])],
        "independent_builds": 2,
        "byte_identical": True,
        "prebuilt_test_fixture": False,
    }


def _launcher_compile_flags(
    deployment: dict[str, Any],
    *,
    runner_source_sha256: str,
    linux_runtime_sha256: str,
    runtime_audit_sha256: str,
) -> list[str]:
    rootfs = deployment["rootfs"]

    def host_path(relative: str) -> str:
        return os.path.join(rootfs, relative.lstrip("/"))

    library_path = ":".join(
        host_path(value) for value in deployment["rootfs_library_paths"]
    )
    values = {
        "MLE_ROOTFS_LOADER_PATH": host_path(deployment["rootfs_loader_path"]),
        "MLE_ROOTFS_PYTHON_PATH": host_path(deployment["rootfs_python_path"]),
        "MLE_ROOTFS_PYTHON_HOME": host_path(deployment["rootfs_python_home"]),
        "MLE_ROOTFS_LIBRARY_PATH": library_path,
        "MLE_RUNNER_SOURCE_SHA256": runner_source_sha256,
        "MLE_LINUX_RUNTIME_SHA256": linux_runtime_sha256,
        "MLE_RUNTIME_AUDIT_SHA256": runtime_audit_sha256,
    }
    result = []
    for name, value in values.items():
        if any(character in value for character in ('"', "\\", "\n", "\r")):
            raise BridgeError("launcher trusted path is not C-literal safe")
        result.append(f'-D{name}="{value}"')
    return result


def _audit_compile_flags(deployment: dict[str, Any]) -> list[str]:
    rootfs = deployment["rootfs"]
    if any(character in rootfs for character in ('"', "\\", "\n", "\r")):
        raise BridgeError("runtime audit rootfs path is not C-literal safe")
    return [
        f'-DMLE_ALLOWED_ROOTFS_PREFIX="{rootfs}"',
        '-DMLE_ALLOWED_ROOTFS_FD_PREFIX="/proc/self/fd/200"',
    ]


def _rootfs_elf_targets_exist(deployment: dict[str, Any]) -> bool:
    rootfs = deployment["rootfs"]
    members = (
        deployment["rootfs_loader_path"],
        deployment["rootfs_python_path"],
        deployment["rootfs_nvidia_smi_path"],
        *deployment["rootfs_library_paths"],
    )
    return all(os.path.exists(os.path.join(rootfs, value.lstrip("/"))) for value in members)


def parse_elf_dependency_list(payload: bytes, *, rootfs: str) -> list[str]:
    try:
        lines = payload.decode("ascii", "strict").splitlines()
    except UnicodeError as exc:
        raise BridgeError("ELF dependency list is not ASCII") from exc
    resolved: set[str] = set()
    prefix = rootfs.rstrip("/") + "/"
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if "=> not found" in line:
            raise BridgeError("sealed rootfs ELF dependency is missing")
        candidate: str | None = None
        if "=>" in line:
            right = line.split("=>", 1)[1].strip()
            if right.startswith("/"):
                candidate = right.split(" ", 1)[0]
        elif line.startswith("/"):
            candidate = line.split(" ", 1)[0]
        elif line.startswith("linux-vdso.so.1 "):
            continue
        else:
            raise BridgeError("ELF dependency list is malformed")
        if candidate is None or not candidate.startswith(prefix):
            raise BridgeError("ELF dependency resolved outside sealed rootfs")
        relative = "/" + candidate[len(prefix) :]
        if relative == "/" or "//" in relative or "/../" in relative:
            raise BridgeError("ELF dependency path is not canonical")
        resolved.add(relative)
    if not resolved:
        raise BridgeError("ELF dependency closure is empty")
    return sorted(resolved)


def verify_elf_dependency_closure(deployment: dict[str, Any]) -> dict[str, Any]:
    rootfs = deployment["rootfs"]

    def host_path(member: str) -> str:
        return os.path.join(rootfs, member.lstrip("/"))

    loader = host_path(deployment["rootfs_loader_path"])
    library_path = ":".join(
        host_path(member) for member in deployment["rootfs_library_paths"]
    )
    targets = []
    for member in (
        deployment["rootfs_python_path"],
        deployment["rootfs_nvidia_smi_path"],
    ):
        try:
            completed = subprocess.run(
                [
                    loader,
                    "--inhibit-cache",
                    "--library-path",
                    library_path,
                    "--list",
                    host_path(member),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BridgeError("sealed rootfs ELF dependency listing failed") from exc
        objects = parse_elf_dependency_list(completed.stdout, rootfs=rootfs)
        targets.append({"path": member, "resolved_objects": objects})
    return {
        "schema": ELF_CLOSURE_SCHEMA,
        "rootfs_digest": deployment["rootfs_digest"],
        "loader": deployment["rootfs_loader_path"],
        "library_paths": deployment["rootfs_library_paths"],
        "cache_inhibited": True,
        "targets": targets,
        "prebuilt_test_fixture": False,
    }


def _read_bounded(path: Path, maximum: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise BridgeError("bounded file is unavailable")
        payload = path.read_bytes()
    except OSError as exc:
        raise BridgeError("bounded file is unavailable") from exc
    if len(payload) > maximum:
        raise BridgeError("bounded file exceeds byte cap")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--rootfs-lock-source", type=Path)
    parser.add_argument("--rootfs-lock-output", type=Path)
    arguments = parser.parse_args(argv)
    lock_mode = (
        arguments.rootfs_lock_source is not None
        or arguments.rootfs_lock_output is not None
    )
    if lock_mode:
        if (
            arguments.rootfs_lock_source is None
            or arguments.rootfs_lock_output is None
            or arguments.deployment is not None
            or arguments.output is not None
            or arguments.compiler != "cc"
        ):
            raise BridgeError("rootfs tree-lock mode arguments drifted")
        receipt = write_rootfs_tree_lock(
            arguments.rootfs_lock_source, arguments.rootfs_lock_output
        )
    else:
        if arguments.deployment is None or arguments.output is None:
            raise BridgeError("bundle build requires deployment and output paths")
        receipt = build_bundle(
            source_root=Path(__file__).resolve().parent,
            deployment_path=arguments.deployment,
            output_root=arguments.output,
            compiler=arguments.compiler,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError:
        raise SystemExit(2)
