from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from runtime_bridge.build_bundle import (
    build_bundle,
    build_rootfs_tree_lock,
    parse_elf_dependency_list,
    validate_deployment,
)
from runtime_bridge.linux_runtime import (
    LinuxRuntime,
    MountRecord,
    operation_cgroup_paths,
    stable_anchor_stat,
    stable_public_tree_sha256,
    validate_audited_runtime_map_lines,
    verify_rootfs_tree_lock,
)
from runtime_bridge.runner import (
    BridgeError,
    canonical_json_bytes,
    canonical_sha256,
    load_bundle_identity,
)
from runtime_bridge.verify_runtime import RESOURCE_CONTRACT, static_certificate
from runtime_bridge.verify_runtime import main as verify_main


def locked_entrypoint_namespace_fixture(source: Path, root: Path) -> int:
    rootfs = root / "entrypoint-rootfs"
    rootfs.mkdir()
    mounted = False
    try:
        subprocess.run(["mount", "--bind", "/", str(rootfs)], check=True)
        mounted = True
        subprocess.run(
            ["mount", "-o", "remount,bind,ro", str(rootfs)], check=True
        )
    except (OSError, subprocess.CalledProcessError):
        if mounted:
            subprocess.run(["umount", str(rootfs)], check=False)
        return 77
    try:
        return run_locked_entrypoint_fixture(source, root, rootfs)
    finally:
        subprocess.run(["umount", str(rootfs)], check=True)


def run_locked_entrypoint_fixture(source: Path, root: Path, rootfs: Path) -> int:
    python_path = Path(sys.executable).resolve()
    python_home = Path(sys.base_prefix).resolve()
    loader_candidates = set()
    library_directories = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or "x" not in parts[1] or not parts[5].startswith("/"):
            continue
        mapped = Path(parts[5]).resolve()
        if "ld-linux" in mapped.name or mapped.name.startswith("ld-"):
            loader_candidates.add(mapped)
        library_directories.add(mapped.parent)
    if len(loader_candidates) != 1:
        return 78
    loader_path = loader_candidates.pop()
    true_path = Path(shutil.which("true") or "/usr/bin/true").resolve()
    library_directories.update(
        (loader_path.parent, python_path.parent, true_path.parent)
    )
    library_paths = sorted(
        {str(path) for path in library_directories if path.is_dir()}
    )
    if not library_paths or len(library_paths) > 16:
        return 78

    published = root / "entrypoint-published"
    admitted = root / "entrypoint-admitted"
    replacement_supervisor = (
        published / "bundle" / "bin" / "mlebench-lite-sandbox-supervisor"
    )
    published.mkdir()
    entrypoint_source = root / "entrypoint-source"
    entrypoint_source.mkdir()
    for name in (
        "runner.py",
        "runner_launcher.c",
        "runtime_audit.c",
        "sandbox_supervisor.c",
    ):
        shutil.copyfile(source / name, entrypoint_source / name)
    (entrypoint_source / "linux_runtime.py").write_text(
        "import os\n"
        "from __main__ import RuntimeAttestation\n"
        "def verify_audited_runtime_mappings(deployment, bundle_root, bundle_fd):\n"
        "    if bundle_root != '/proc/self/fd/197' or bundle_fd != 197:\n"
        "        raise RuntimeError('bundle anchor absent')\n"
        "    marker = b'mlebench_lite_runtime_audit_v1\\n'\n"
        "    if os.environ.get('MLE_BRIDGE_RUNTIME_AUDIT') != "
        "'mlebench_lite_runtime_audit_v1':\n"
        "        raise RuntimeError('audit identity absent')\n"
        "    if os.pread(198, len(marker), 0) != marker:\n"
        "        raise RuntimeError('audit marker absent')\n"
        f"    os.rename({str(published)!r}, {str(admitted)!r})\n"
        f"    os.makedirs({str(published / 'bundle' / 'bin')!r})\n"
        f"    with open({str(replacement_supervisor)!r}, 'wb') "
        "as handle:\n"
        "        handle.write(b'replacement')\n"
        "    return {'schema': 'mlebench_lite_audited_runtime_mappings_v1'}\n"
        "class LinuxRuntime:\n"
        "    def __init__(self, bundle, **kwargs):\n"
        "        self.bundle = bundle\n"
        "    def attest(self, request, state=None):\n"
        "        with open(self.bundle.supervisor_path, 'rb') as handle:\n"
        "            if handle.read() != b'synthetic-linux-elf-fixture':\n"
        "                raise RuntimeError('anchored supervisor drifted')\n"
        "        return RuntimeAttestation(\n"
        "            cpu_limit_cores=36, memory_limit_bytes=440000000000,\n"
        "            pids_limit=4096, gpu_count=1,\n"
        "            gpu_uuid='GPU-00000000-0000-0000-0000-000000000001',\n"
        "            mount_namespace=True, network_disabled=True, non_root=True,\n"
        "            read_only_rootfs=True,\n"
        "            runtime_identity={'schema': 'synthetic_linux_identity_v1'},\n"
        "        )\n",
        encoding="utf-8",
    )
    host_member = lambda path: "/" + str(path).lstrip("/")
    deployment = {
        "schema": "mlebench_lite_runtime_bridge_deployment_v1",
        "rootfs": str(rootfs),
        "rootfs_digest": "1" * 64,
        "rootfs_tree_lock": str(root / "unused-rootfs-tree-lock.json"),
        "rootfs_tree_lock_sha256": "2" * 64,
        "state_root": str(root / "entrypoint-state"),
        "episodes_root": str(root / "entrypoint-episodes"),
        "sandbox_host_uid": max(1, os.geteuid()),
        "sandbox_host_gid": max(1, os.getegid()),
        "rootfs_loader_path": host_member(loader_path),
        "rootfs_python_path": host_member(python_path),
        "rootfs_python_home": host_member(python_home),
        "rootfs_library_paths": [host_member(Path(path)) for path in library_paths],
        "rootfs_nvidia_smi_path": host_member(true_path),
        "gpu": {
            "uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "device": {
                "source": "/dev/nvidia7",
                "target": "/dev/nvidia0",
                "major": 195,
                "minor": 7,
            },
            "control_devices": [
                {
                    "source": "/dev/nvidiactl",
                    "target": "/dev/nvidiactl",
                    "major": 195,
                    "minor": 255,
                },
                {
                    "source": "/dev/nvidia-uvm",
                    "target": "/dev/nvidia-uvm",
                    "major": 511,
                    "minor": 0,
                },
            ],
        },
        "openmle_v7_provenance": {
            "artifact_lock_sha256": "f04f269d39f66c025d70620f41016fb3a555fb175b9feb6c8977fed6debae1f6",
            "supervisor_sha256": "25a93be7ec835df83c2100bede5743c66dee18246cd32aa44ffaa67f8c625032",
        },
    }
    state_root = Path(deployment["state_root"])
    episodes_root = Path(deployment["episodes_root"])
    state_root.mkdir(mode=0o700)
    episodes_root.mkdir(mode=0o700)
    deployment_path = root / "entrypoint-deployment.json"
    deployment_path.write_bytes(canonical_json_bytes(deployment) + b"\n")
    fake_supervisor = root / "entrypoint-supervisor"
    fake_supervisor.write_bytes(b"synthetic-linux-elf-fixture")
    fake_supervisor.chmod(0o500)
    bundle = published / "bundle"
    receipt = build_bundle(
        source_root=entrypoint_source,
        deployment_path=deployment_path,
        output_root=bundle,
        supervisor_binary=fake_supervisor,
    )

    episode_id = "a" * 32
    episode = episodes_root / episode_id
    public = root / "entrypoint-public"
    workspace = episode / "workspace"
    submission = episode / "submission"
    public.mkdir()
    workspace.mkdir(parents=True)
    submission.mkdir()
    request = {
        "schema": "mlebench_lite_sandbox_request_v3",
        "episode_id": episode_id,
        "competition_id": "synthetic-entrypoint",
        "mode": "native",
        "resource_contract": RESOURCE_CONTRACT,
        "resource_contract_sha256": canonical_sha256(RESOURCE_CONTRACT),
        "public_root": str(public),
        "public_tree_sha256": "5" * 64,
        "workspace_root": str(workspace),
        "submission_root": str(submission),
    }
    completed = subprocess.run(
        [
            str(bundle / "sandbox-runner"),
            "--expected-runtime-digest",
            receipt["runtime_digest"],
            "attest",
        ],
        input=canonical_json_bytes(request),
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        sys.stderr.buffer.write(completed.stderr)
        return completed.returncode or 1
    response = json.loads(completed.stdout)
    return 0 if response["schema"] == "mlebench_lite_sandbox_attestation_v3" else 1


class RuntimeBridgeLinuxSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".mle-bridge-bundle-", dir=Path.cwd()
        )
        self.root = Path(self.temporary.name)
        self.source = Path(__file__).resolve().parents[1] / "runtime_bridge"
        self.deployment = {
            "schema": "mlebench_lite_runtime_bridge_deployment_v1",
            "rootfs": "/opt/mlebench-lite/rootfs",
            "rootfs_digest": "1" * 64,
            "rootfs_tree_lock": "/opt/mlebench-lite/rootfs-tree-lock.json",
            "rootfs_tree_lock_sha256": "2" * 64,
            "state_root": str(self.root / "state"),
            "episodes_root": str(self.root / "episodes"),
            "sandbox_host_uid": max(1, os.geteuid()),
            "sandbox_host_gid": max(1, os.getegid()),
            "rootfs_loader_path": "/lib64/ld-linux-x86-64.so.2",
            "rootfs_python_path": "/usr/bin/python3.11",
            "rootfs_python_home": "/usr",
            "rootfs_library_paths": [
                "/lib/x86_64-linux-gnu",
                "/usr/lib/x86_64-linux-gnu",
                "/lib64",
                "/usr/lib64",
            ],
            "rootfs_nvidia_smi_path": "/usr/bin/nvidia-smi",
            "gpu": {
                "uuid": "GPU-00000000-0000-0000-0000-000000000001",
                "device": {
                    "source": "/dev/nvidia7",
                    "target": "/dev/nvidia0",
                    "major": 195,
                    "minor": 7,
                },
                "control_devices": [
                    {
                        "source": "/dev/nvidiactl",
                        "target": "/dev/nvidiactl",
                        "major": 195,
                        "minor": 255,
                    },
                    {
                        "source": "/dev/nvidia-uvm",
                        "target": "/dev/nvidia-uvm",
                        "major": 511,
                        "minor": 0,
                    },
                ],
            },
            "openmle_v7_provenance": {
                "artifact_lock_sha256": "f04f269d39f66c025d70620f41016fb3a555fb175b9feb6c8977fed6debae1f6",
                "supervisor_sha256": "25a93be7ec835df83c2100bede5743c66dee18246cd32aa44ffaa67f8c625032",
            },
        }
        self.deployment_path = self.root / "deployment-input.json"
        self.deployment_path.write_text(
            json.dumps(self.deployment, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        self.fake_supervisor = self.root / "supervisor"
        self.fake_supervisor.write_bytes(b"synthetic-linux-elf-fixture")
        self.fake_supervisor.chmod(0o500)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_deployment_schema_requires_exact_one_gpu_and_safe_paths(self) -> None:
        self.assertEqual(validate_deployment(self.deployment), self.deployment)
        variants = []
        extra = copy.deepcopy(self.deployment)
        extra["unexpected"] = True
        variants.append(extra)
        no_gpu = copy.deepcopy(self.deployment)
        no_gpu["gpu"] = []
        variants.append(no_gpu)
        duplicate_target = copy.deepcopy(self.deployment)
        duplicate_target["gpu"]["control_devices"][0]["target"] = "/dev/nvidia0"
        variants.append(duplicate_target)
        relative = copy.deepcopy(self.deployment)
        relative["state_root"] = "relative/state"
        variants.append(relative)
        weak_provenance = copy.deepcopy(self.deployment)
        weak_provenance["openmle_v7_provenance"]["artifact_lock_sha256"] = "0" * 64
        variants.append(weak_provenance)
        root_identity = copy.deepcopy(self.deployment)
        root_identity["sandbox_host_uid"] = 0
        variants.append(root_identity)
        unsafe_python = copy.deepcopy(self.deployment)
        unsafe_python["rootfs_python_path"] = "/usr/bin/python 3"
        variants.append(unsafe_python)
        overlapping_rootfs = copy.deepcopy(self.deployment)
        overlapping_rootfs["rootfs"] = overlapping_rootfs["state_root"] + "/rootfs"
        variants.append(overlapping_rootfs)
        host_rootfs = copy.deepcopy(self.deployment)
        host_rootfs["rootfs"] = "/"
        variants.append(host_rootfs)
        second_compute = copy.deepcopy(self.deployment)
        second_compute["gpu"]["control_devices"].append(
            {
                "source": "/dev/nvidia8",
                "target": "/dev/nvidia1",
                "major": 195,
                "minor": 8,
            }
        )
        variants.append(second_compute)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(BridgeError):
                validate_deployment(value)

    def test_bundle_lock_binds_every_member_and_rejects_mutation_or_extra_file(self) -> None:
        output = self.root / "bundle"
        receipt = build_bundle(
            source_root=self.source,
            deployment_path=self.deployment_path,
            output_root=output,
            supervisor_binary=self.fake_supervisor,
        )
        identity = load_bundle_identity(output / "sandbox-runner", expected_uid=os.geteuid())
        self.assertEqual(identity.identity.runner_sha256, receipt["runner_sha256"])
        self.assertEqual(identity.identity.runtime_digest, receipt["runtime_digest"])
        self.assertEqual(identity.deployment, self.deployment)
        self.assertFalse((output / "sandbox-runner").read_bytes().startswith(b"#!"))
        self.assertTrue((output / "runner.py").is_file())
        self.assertTrue((output / "runner_launcher.c").is_file())
        (output / "linux_runtime.py").chmod(0o644)
        (output / "linux_runtime.py").write_text("tampered", encoding="utf-8")
        (output / "linux_runtime.py").chmod(0o444)
        with self.assertRaises(BridgeError):
            load_bundle_identity(output / "sandbox-runner", expected_uid=os.geteuid())

        second = self.root / "bundle-extra"
        build_bundle(
            source_root=self.source,
            deployment_path=self.deployment_path,
            output_root=second,
            supervisor_binary=self.fake_supervisor,
        )
        second.chmod(0o755)
        (second / "unlocked.txt").write_text("not locked", encoding="utf-8")
        (second / "unlocked.txt").chmod(0o444)
        second.chmod(0o555)
        with self.assertRaises(BridgeError):
            load_bundle_identity(second / "sandbox-runner", expected_uid=os.geteuid())

    def test_launcher_selection_binds_the_configured_runtime_digest_before_python(self) -> None:
        launcher = (self.source / "runner_launcher.c").read_text(encoding="utf-8")
        builder = (self.source / "build_bundle.py").read_text(encoding="utf-8")
        adapter = (
            self.source.parent / "agentenv_mlebench_lite" / "executor.py"
        ).read_text(encoding="utf-8")
        for required in (
            "--expected-runtime-digest",
            "verify_artifact_lock_digest",
            "MLE_RUNNER_SOURCE_SHA256",
            "MLE_LINUX_RUNTIME_SHA256",
            "MLE_RUNTIME_AUDIT_SHA256",
        ):
            with self.subTest(required=required):
                self.assertIn(required, launcher)
        self.assertIn("expected_runtime_digest", adapter)
        self.assertIn('"--expected-runtime-digest"', adapter)
        self.assertIn("MLE_RUNNER_SOURCE_SHA256", builder)
        self.assertIn("MLE_LINUX_RUNTIME_SHA256", builder)
        self.assertIn("MLE_RUNTIME_AUDIT_SHA256", builder)

    def test_bundle_identity_survives_ancestor_rename_and_path_replacement(self) -> None:
        published = self.root / "published"
        published.mkdir()
        output = published / "bundle"
        receipt = build_bundle(
            source_root=self.source,
            deployment_path=self.deployment_path,
            output_root=output,
            supervisor_binary=self.fake_supervisor,
        )
        bundle_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            admitted = self.root / "admitted"
            published.rename(admitted)
            published.mkdir()
            replacement = published / "bundle"
            replacement.mkdir()
            (replacement / "sandbox-runner").write_bytes(b"replacement")

            identity = load_bundle_identity(
                f"/proc/self/fd/{bundle_fd}/sandbox-runner",
                expected_uid=os.geteuid(),
                bundle_fd=bundle_fd,
            )
            self.assertEqual(identity.identity.runner_sha256, receipt["runner_sha256"])
            self.assertEqual(identity.identity.runtime_digest, receipt["runtime_digest"])
            self.assertEqual(identity.bundle_root, f"/proc/self/fd/{bundle_fd}")
        finally:
            os.close(bundle_fd)

    def test_bundle_build_refuses_existing_output_and_noncanonical_deployment(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(BridgeError):
            build_bundle(
                source_root=self.source,
                deployment_path=self.deployment_path,
                output_root=output,
                supervisor_binary=self.fake_supervisor,
            )

    def test_bundle_build_accepts_one_canonical_text_newline(self) -> None:
        deployment = self.root / "deployment-newline.json"
        deployment.write_bytes(canonical_json_bytes(self.deployment) + b"\n")
        receipt = build_bundle(
            source_root=self.source,
            deployment_path=deployment,
            output_root=self.root / "bundle-newline",
            supervisor_binary=self.fake_supervisor,
        )
        self.assertEqual(receipt["schema"], "mlebench_lite_runtime_bundle_receipt_v1")
        noncanonical = self.root / "noncanonical.json"
        noncanonical.write_text(json.dumps(self.deployment, indent=2), encoding="utf-8")
        with self.assertRaises(BridgeError):
            build_bundle(
                source_root=self.source,
                deployment_path=noncanonical,
                output_root=self.root / "bad-bundle",
                supervisor_binary=self.fake_supervisor,
            )

    def test_supervisor_source_excludes_openmle_task_and_grader_semantics(self) -> None:
        source = (self.source / "sandbox_supervisor.c").read_text(encoding="utf-8")
        for forbidden in (
            "fit_hook",
            "managed_runtime",
            "allowed_utility",
            "private_runner",
            "private_worker",
            "grader",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        for required in (
            "CLONE_NEWUSER",
            "CLONE_NEWCGROUP",
            "CLONE_NEWNET",
            "CLONE_NEWPID",
            "PR_SET_NO_NEW_PRIVS",
            "SECCOMP_MODE_FILTER",
            "PTRACE_O_TRACEFORK",
            "/home/data",
            "/home/workspace",
            "/home/submission",
            "/run/amg_memory",
            "--rootfs-fd",
            "CUDA_VISIBLE_DEVICES",
            "AF_UNIX",
            "SECCOMP_RET_TRAP",
            "__X32_SYSCALL_BIT",
            "io_uring_setup",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_linux_runtime_source_freezes_exact_resource_and_device_values(self) -> None:
        source = (self.source / "linux_runtime.py").read_text(encoding="utf-8")
        for required in (
            "CPU_LIMIT_CORES = 36",
            "MEMORY_LIMIT_BYTES = 440_000_000_000",
            "PIDS_LIMIT = 4096",
            "WRITABLE_BYTES_LIMIT = 500_000_000_000",
            "WRITABLE_INODES_LIMIT = 2_000_000",
            '"memory.memsw.limit_in_bytes"',
            '"memory.swappiness"',
            '"devices.deny", "a"',
            '"devices.allow"',
            "stable_public_tree_sha256_fd",
            "MS_BIND | MS_REMOUNT | MS_RDONLY",
            "runtime_identity",
            "verify_rootfs_tree_lock",
            "rootfs_mount_is_read_only",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        for forbidden in (
            "OPENMLE_TEST_",
            "MAX_PROCS = 256",
            "RLIMIT_NPROC, 64",
            'CUDA_VISIBLE_DEVICES", ""',
            "fit_hook",
            "private_worker",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_stable_public_hash_matches_dataset_inventory_and_rejects_aliases(self) -> None:
        public = self.root / "public"
        nested = public / "nested"
        nested.mkdir(parents=True)
        (public / "a.txt").write_bytes(b"alpha")
        (nested / "b.bin").write_bytes(b"beta\x00")
        inventory = [
            {
                "path": "a.txt",
                "size": 5,
                "sha256": hashlib.sha256(b"alpha").hexdigest(),
            },
            {
                "path": "nested/b.bin",
                "size": 5,
                "sha256": hashlib.sha256(b"beta\x00").hexdigest(),
            },
        ]
        self.assertEqual(stable_public_tree_sha256(str(public)), canonical_sha256(inventory))

        prefix_directory = public / "prefix"
        prefix_directory.mkdir()
        (prefix_directory / "child").write_bytes(b"child")
        (public / "prefix!").write_bytes(b"sibling")
        reordered_inventory = [
            *inventory,
            {
                "path": "prefix!",
                "size": 7,
                "sha256": hashlib.sha256(b"sibling").hexdigest(),
            },
            {
                "path": "prefix/child",
                "size": 5,
                "sha256": hashlib.sha256(b"child").hexdigest(),
            },
        ]
        reordered_inventory.sort(key=lambda item: item["path"])
        self.assertEqual(
            stable_public_tree_sha256(str(public)),
            canonical_sha256(reordered_inventory),
        )

        (public / "unsafe-link").symlink_to("a.txt")
        with self.assertRaises(BridgeError):
            stable_public_tree_sha256(str(public))
        (public / "unsafe-link").unlink()
        os.link(public / "a.txt", public / "hardlink")
        with self.assertRaises(BridgeError):
            stable_public_tree_sha256(str(public))

    def test_rootfs_lock_verifies_the_actual_tree_and_detects_mutation(self) -> None:
        rootfs = self.root / "rootfs"
        (rootfs / "usr" / "bin").mkdir(parents=True)
        executable = rootfs / "usr" / "bin" / "python3"
        executable.write_bytes(b"synthetic-python")
        executable.chmod(0o755)
        (rootfs / "bin").symlink_to("usr/bin", target_is_directory=True)
        if platform.system() == "Linux":
            with self.assertRaises(BridgeError):
                build_rootfs_tree_lock(rootfs)

        with mock.patch(
            "runtime_bridge.linux_runtime.rootfs_mount_is_read_only",
            return_value=True,
        ):
            lock = build_rootfs_tree_lock(rootfs)
            lock_path = self.root / "rootfs-tree-lock.json"
            payload = canonical_json_bytes(lock)
            lock_path.write_bytes(payload)
            identity = verify_rootfs_tree_lock(
                rootfs_path=str(rootfs),
                tree_lock_path=str(lock_path),
                expected_tree_sha256=lock["rootfs_digest"],
                expected_lock_sha256=hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(identity["tree_sha256"], lock["rootfs_digest"])

            executable.write_bytes(b"mutated-python")
            with self.assertRaises(BridgeError):
                verify_rootfs_tree_lock(
                    rootfs_path=str(rootfs),
                    tree_lock_path=str(lock_path),
                    expected_tree_sha256=lock["rootfs_digest"],
                    expected_lock_sha256=hashlib.sha256(payload).hexdigest(),
                )

            unsafe_rootfs = self.root / "unsafe-rootfs"
            unsafe_rootfs.mkdir()
            (unsafe_rootfs / "home").symlink_to("/tmp", target_is_directory=True)
            with self.assertRaises(BridgeError):
                build_rootfs_tree_lock(unsafe_rootfs)

            escaping_rootfs = self.root / "escaping-rootfs"
            (escaping_rootfs / "usr").mkdir(parents=True)
            (escaping_rootfs / "usr" / "escape").symlink_to(
                "../../outside", target_is_directory=True
            )
            with self.assertRaises(BridgeError):
                build_rootfs_tree_lock(escaping_rootfs)

            chained_rootfs = self.root / "chained-escaping-rootfs"
            (chained_rootfs / "sub").mkdir(parents=True)
            (chained_rootfs / "sub" / "a").symlink_to(
                "..", target_is_directory=True
            )
            (chained_rootfs / "sub" / "x").symlink_to(
                "a/../..", target_is_directory=True
            )
            with self.assertRaises(BridgeError):
                build_rootfs_tree_lock(chained_rootfs)

            contained_rootfs = self.root / "contained-rootfs"
            (contained_rootfs / "usr" / "bin").mkdir(parents=True)
            (contained_rootfs / "usr" / "bin" / "python3").write_bytes(b"python")
            (contained_rootfs / "bin").symlink_to("usr/bin", target_is_directory=True)
            contained_lock = build_rootfs_tree_lock(contained_rootfs)
            contained_link = next(
                item for item in contained_lock["files"] if item["path"] == "bin"
            )
            self.assertEqual(contained_link["type"], "symlink")
            self.assertEqual(contained_link["target"], "usr/bin")

    def test_rootfs_symlink_resolution_rejects_cycles_and_allows_chains(self) -> None:
        cyclic_rootfs = self.root / "cyclic-rootfs"
        cyclic_rootfs.mkdir()
        (cyclic_rootfs / "a").symlink_to("b")
        (cyclic_rootfs / "b").symlink_to("a")

        contained_rootfs = self.root / "chained-contained-rootfs"
        (contained_rootfs / "usr" / "lib").mkdir(parents=True)
        (contained_rootfs / "usr" / "lib" / "runtime.so").write_bytes(b"runtime")
        (contained_rootfs / "lib").symlink_to("usr/lib", target_is_directory=True)
        (contained_rootfs / "runtime").symlink_to(
            "lib/runtime.so", target_is_directory=False
        )

        with mock.patch(
            "runtime_bridge.linux_runtime.rootfs_mount_is_read_only",
            return_value=True,
        ):
            with self.assertRaises(BridgeError):
                build_rootfs_tree_lock(cyclic_rootfs)
            lock = build_rootfs_tree_lock(contained_rootfs)

        targets = {
            item["path"]: item["target"]
            for item in lock["files"]
            if item["type"] == "symlink"
        }
        self.assertEqual(
            targets,
            {"lib": "usr/lib", "runtime": "lib/runtime.so"},
        )

    def test_static_certificate_cannot_claim_pass_when_tests_are_skipped(self) -> None:
        certificate = static_certificate(run_tests=False)
        self.assertEqual(certificate["status"], "prehost_pending")
        self.assertEqual(certificate["actual_host_admission"], "pending")

    def test_read_only_tree_anchor_uses_bounded_mount_and_inode_identity(self) -> None:
        tree = self.root / "anchored-public"
        tree.mkdir()
        metadata = tree.stat()
        record = MountRecord(
            mount_id=71,
            parent_id=1,
            device="0:71",
            root="/",
            target=str(tree),
            mount_options=frozenset(("ro",)),
            optional_fields=(),
            filesystem="ext4",
            source="/dev/readonly",
            super_options=frozenset(("ro",)),
        )
        anchor = {
            "schema": "mlebench_lite_read_only_tree_anchor_v1",
            "path": str(tree),
            "mount_id": record.mount_id,
            "mount_device": record.device,
            "mount_root": record.root,
            "mount_target": record.target,
            "root_stat": stable_anchor_stat(metadata),
            "tree_sha256": "7" * 64,
        }
        runtime = object.__new__(LinuxRuntime)
        with mock.patch(
            "runtime_bridge.linux_runtime.find_containing_mount",
            return_value=record,
        ), mock.patch(
            "runtime_bridge.linux_runtime.read_mountinfo", return_value=[record]
        ):
            descriptor = runtime._open_anchored_read_only_tree(
                anchor, label="public"
            )
            os.close(descriptor)
            changed = copy.deepcopy(anchor)
            changed["root_stat"]["ino"] += 1
            with self.assertRaises(BridgeError):
                runtime._open_anchored_read_only_tree(changed, label="public")

    def test_operation_deadline_expires_before_runtime_side_effects(self) -> None:
        runtime = object.__new__(LinuxRuntime)
        runtime.operation_deadline = time.monotonic() - 1.0
        with self.assertRaises(BridgeError):
            runtime._check_operation_deadline("focused regression")

    def test_supervisor_stats_require_kernel_writable_high_water(self) -> None:
        value = {
            "schema": "mlebench_lite_supervisor_stats_v1",
            "exit_code": 0,
            "security_violation": False,
            "background_process": False,
            "file_limit": False,
            "processes_started": 3,
            "process_peak": 2,
            "bytes_read": 4,
            "bytes_written": 5,
            "writable_bytes_high_water": 33_554_432,
            "writable_inodes_high_water": 256,
        }
        parsed = LinuxRuntime._parse_supervisor_stats(canonical_json_bytes(value))
        self.assertEqual(parsed["writable_bytes_high_water"], 33_554_432)
        del value["writable_inodes_high_water"]
        with self.assertRaises(BridgeError):
            LinuxRuntime._parse_supervisor_stats(canonical_json_bytes(value))

    def test_cgroup_resource_counter_deltas_fail_closed(self) -> None:
        baseline = {
            "cpu_usage_ns": 100,
            "memory_peak_bytes": 1024,
            "memory_failcnt": 0,
            "oom_kill_count": 0,
            "episode_memory_usage_bytes": 2048,
            "episode_memory_peak_bytes": 4096,
            "episode_memory_failcnt": 0,
            "episode_oom_kill_count": 0,
            "pids_max_events": 0,
        }
        final = dict(baseline)
        final["cpu_usage_ns"] += 1
        final["memory_peak_bytes"] += 1
        final["episode_memory_usage_bytes"] += 1
        final["episode_memory_peak_bytes"] += 1
        LinuxRuntime._validate_cgroup_stats(baseline, final)

        for counter in (
            "memory_failcnt",
            "oom_kill_count",
            "episode_memory_failcnt",
            "episode_oom_kill_count",
            "pids_max_events",
        ):
            with self.subTest(counter=counter):
                drifted = dict(final)
                drifted[counter] += 1
                with self.assertRaises(BridgeError):
                    LinuxRuntime._validate_cgroup_stats(baseline, drifted)

        for peak in ("memory_peak_bytes", "episode_memory_peak_bytes"):
            with self.subTest(peak=peak):
                drifted = dict(final)
                drifted[peak] = RESOURCE_CONTRACT["memory_limit_bytes"] + 1
                with self.assertRaises(BridgeError):
                    LinuxRuntime._validate_cgroup_stats(baseline, drifted)

    def test_death_cascade_uses_production_hierarchical_memory_path(self) -> None:
        verifier = (self.source / "verify_runtime.py").read_text(encoding="utf-8")
        start = verifier.index("def kill_runner_cascade(")
        cascade = verifier[start : verifier.index("def signal_number(", start)]
        self.assertIn("operation_cgroup_paths(", cascade)
        self.assertNotIn(
            'Path("/sys/fs/cgroup") / controller / name', cascade
        )
        episode_id = "a" * 32
        operation_id = "b" * 32
        name = f"mlebridge-{episode_id}-{operation_id}"
        paths = operation_cgroup_paths(episode_id, name)
        self.assertEqual(
            paths["memory"][0],
            f"/sys/fs/cgroup/memory/mlebridge-{episode_id}/{operation_id}",
        )

    def test_runtime_tombstone_is_bound_to_current_bundle_identity(self) -> None:
        state_root = self.root / "runtime-state"
        state_root.mkdir(mode=0o700)
        runtime = object.__new__(LinuxRuntime)
        runtime.deployment = {"state_root": str(state_root)}
        runtime.bundle_identity_sha256 = "8" * 64
        request = {"episode_id": "a" * 32}
        state = {
            "base_sha256": "9" * 64,
            "mount_attestation_sha256": "b" * 64,
        }
        runtime._write_tombstone(request, state, None)
        runtime._verify_tombstone(request, state, None)
        runtime.bundle_identity_sha256 = "c" * 64
        with self.assertRaises(BridgeError):
            runtime._verify_tombstone(request, state, None)

    def test_live_verification_requires_every_explicit_gate(self) -> None:
        variants = (
            ["--allow-live-destructive"],
            ["--bundle", "/tmp/missing-bundle"],
            [
                "--bundle",
                "/tmp/missing-bundle",
                "--live-root",
                "/tmp/mlebridge-live-missing",
                "--gpu-uuid",
                "GPU-00000000-0000-0000-0000-000000000001",
            ],
        )
        for arguments in variants:
            with self.subTest(arguments=arguments), self.assertRaises(BridgeError):
                verify_main(arguments)

    def test_runner_success_exit_is_not_caught_by_its_failure_guard(self) -> None:
        source = (self.source / "runner.py").read_text(encoding="utf-8")
        self.assertIn("exit_code = main()", source)
        self.assertIn("raise SystemExit(exit_code)", source)
        self.assertNotIn("try:\n        raise SystemExit(main())", source)

    def test_reviewed_runtime_boundaries_are_present_in_source(self) -> None:
        linux = (self.source / "linux_runtime.py").read_text(encoding="utf-8")
        supervisor = (self.source / "sandbox_supervisor.c").read_text(
            encoding="utf-8"
        )
        verifier = (self.source / "verify_runtime.py").read_text(encoding="utf-8")
        runner = (self.source / "runner.py").read_text(encoding="utf-8")
        adapter = (
            self.source.parent / "agentenv_mlebench_lite" / "executor.py"
        ).read_text(encoding="utf-8")
        launcher_path = self.source / "runner_launcher.c"

        self.assertTrue(launcher_path.is_file())
        launcher = launcher_path.read_text(encoding="utf-8")
        for required in (
            "PR_SET_PDEATHSIG",
            "MleBridgeLauncherIdentity",
            "rootfs_loader_path",
            "rootfs_python_path",
            "MLE_BUNDLE_ROOT_FD 197",
            "openat",
        ):
            with self.subTest(launcher=required):
                self.assertIn(required, launcher)
        self.assertNotIn("#!", launcher)
        self.assertIn('"/proc/self/fd/%d"', launcher)
        self.assertIn('openat(directory, "runner.py"', launcher)
        self.assertIn("*runner_descriptor = source", launcher)
        self.assertNotIn("os.path.realpath(bundle_root)", runner)
        self.assertNotIn("host_python_path", runner)
        self.assertNotIn("host_python_sha256", runner)
        self.assertNotIn('self.deployment["nvidia_smi_path"]', linux)
        self.assertIn('self.deployment["rootfs_nvidia_smi_path"]', linux)

        for required in (
            "bundle_identity_sha256",
            "mlebench_lite_bridge_state_v3",
        ):
            with self.subTest(runner=required):
                self.assertIn(required, runner)
        for required in (
            "rootfs_anchor",
            "public_anchor",
            "operation_deadline",
            "start_execution_watchdog",
            "writable_bytes_high_water",
            "writable_inodes_high_water",
        ):
            with self.subTest(linux=required):
                self.assertIn(required, linux)
        for required in (
            "PR_SET_PDEATHSIG",
            "PTRACE_SEIZE",
            "PTRACE_INTERRUPT",
            "PTRACE_O_TRACESECCOMP",
            "PTRACE_EVENT_SECCOMP",
            "quiesce_tracees",
            "writable_bytes_high_water",
            "writable_inodes_high_water",
            "validate_public_mount_topology",
        ):
            with self.subTest(supervisor=required):
                self.assertIn(required, supervisor)
        self.assertNotIn("MS_BIND | MS_REC", supervisor)
        self.assertIn("max_step_response_ms", adapter)
        self.assertIn("create_then_delete_high_water", verifier)
        self.assertIn("exec_tmpfile_high_water", verifier)
        self.assertIn("mmap_tmpfile_high_water", verifier)
        self.assertIn("concurrent_close_splice_high_water", verifier)
        self.assertIn("kill_runner_cascade", verifier)
        self.assertIn("for mode in LIVE_MODES", verifier)

    def test_episode_memory_limit_is_hierarchical_and_persists_until_teardown(self) -> None:
        linux = (self.source / "linux_runtime.py").read_text(encoding="utf-8")
        for required in (
            "_ensure_episode_memory_cgroup",
            "episode_memory_cgroup",
            '"memory.use_hierarchy"',
            "_remove_episode_memory_cgroup",
            "allow_episode_memory_parent",
        ):
            with self.subTest(required=required):
                self.assertIn(required, linux)
        create_start = linux.index("def _create_cgroups")
        create_end = linux.index("def start_execution_watchdog", create_start)
        create = linux[create_start:create_end]
        self.assertIn('result.children["memory"]', create)
        self.assertIn("episode_memory_cgroup", create)
        self.assertIn("mounted_episode_memory", create)
        freeze = linux[linux.index("def freeze(") : linux.index("def teardown(")]
        self.assertIn("allow_episode_memory_parent=True", freeze)
        teardown = linux[
            linux.index("def teardown(") : linux.index("def reconcile(")
        ]
        self.assertLess(
            teardown.index("self._libc.umount(episode_root)"),
            teardown.index("self._remove_episode_memory_cgroup"),
        )

    def test_gpu_loader_audit_accepts_only_the_identity_checked_rootfs_fd(self) -> None:
        linux = (self.source / "linux_runtime.py").read_text(encoding="utf-8")
        audit = (self.source / "runtime_audit.c").read_text(encoding="utf-8")
        self.assertIn("RUNTIME_AUDIT_ROOTFS_FD = 200", linux)
        self.assertIn("RUNTIME_AUDIT_ROOTFS_FD", linux)
        self.assertIn("MLE_ALLOWED_ROOTFS_FD_PREFIX", audit)
        self.assertIn("path_is_below_rootfs_fd", audit)
        self.assertIn("path_has_safe_suffix", audit)

    @unittest.skipUnless(
        platform.system() == "Linux"
        and platform.machine() in {"x86_64", "AMD64"}
        and shutil.which("gcc"),
        "runtime audit policy harness requires Linux/x86_64 and GCC",
    )
    def test_runtime_audit_policy_rejects_lexical_escape(self) -> None:
        probe = self.root / "runtime-audit-policy-probe"
        subprocess.run(
            [
                str(shutil.which("gcc")),
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DMLE_RUNTIME_AUDIT_POLICY_TEST",
                '-DMLE_ALLOWED_ROOTFS_PREFIX="/sealed/root"',
                '-DMLE_ALLOWED_ROOTFS_FD_PREFIX="/proc/self/fd/200"',
                str(self.source / "runtime_audit.c"),
                "-o",
                str(probe),
            ],
            check=True,
        )
        accepted = (
            ("rootfs", "/sealed/root/lib64/libc.so.6"),
            ("fd", "/proc/self/fd/200/lib64/libc.so.6"),
        )
        rejected = (
            ("rootfs", "/sealed/root"),
            ("rootfs", "/sealed/root/../outside/libevil.so"),
            ("rootfs", "/sealed/root/lib64//libc.so.6"),
            ("fd", "/proc/self/fd/201/lib64/libc.so.6"),
            ("fd", "/proc/self/fd/200/../outside/libevil.so"),
            ("fd", "/proc/self/fd/200/lib64/./libc.so.6"),
        )
        for kind, path in accepted:
            with self.subTest(kind=kind, path=path):
                completed = subprocess.run([probe, kind, path], check=False)
                self.assertEqual(completed.returncode, 0)
        for kind, path in rejected:
            with self.subTest(kind=kind, path=path):
                completed = subprocess.run([probe, kind, path], check=False)
                self.assertEqual(completed.returncode, 1)

    def test_group_destructive_mutations_drain_thread_exits(self) -> None:
        supervisor = (self.source / "sandbox_supervisor.c").read_text(
            encoding="utf-8"
        )
        verifier = (self.source / "verify_runtime.py").read_text(encoding="utf-8")
        for required in (
            "group_destructive_mutation",
            "handle_group_exit_stop",
            "remap_exec_owner",
        ):
            with self.subTest(required=required):
                self.assertIn(required, supervisor)
        self.assertIn("threaded_execve", verifier)
        self.assertIn("threaded_failed_execve", verifier)
        self.assertIn("threaded_exit_group", verifier)

    def test_non_memory_supervisor_launch_omits_memory_fd_argument(self) -> None:
        linux = (self.source / "linux_runtime.py").read_text(encoding="utf-8")
        self.assertIn("if memory_fd >= 0:\n                argv.extend", linux)
        self.assertNotIn(
            '"--memory-fd",\n                str(memory_fd),',
            linux,
        )

    def test_native_launcher_denies_cwd_import_precedence(self) -> None:
        launcher = (self.source / "runner_launcher.c").read_text(encoding="utf-8")
        safe_path = launcher.index('"-P"')
        bootstrap = launcher.index('"-c"')
        self.assertLess(safe_path, bootstrap)

    def test_native_launcher_requires_owner_immutable_bundle_members(self) -> None:
        launcher = (self.source / "runner_launcher.c").read_text(encoding="utf-8")
        self.assertIn(
            "value.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)", launcher
        )
        self.assertIn(
            "metadata.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)", launcher
        )

    def test_execve_is_sampled_before_cloexec_tmpfile_release(self) -> None:
        supervisor = (self.source / "sandbox_supervisor.c").read_text(
            encoding="utf-8"
        )
        self.assertIn("TRACE_MUTATION_SYSCALL(execve)", supervisor)
        self.assertIn("TRACE_MUTATION_SYSCALL(execveat)", supervisor)
        self.assertIn("TRACE_MUTATION_SYSCALL(dup2)", supervisor)
        self.assertIn("TRACE_MUTATION_SYSCALL(dup3)", supervisor)
        self.assertIn("TRACE_MUTATION_SYSCALL(mmap)", supervisor)
        self.assertIn("TRACE_MUTATION_SYSCALL(munmap)", supervisor)
        self.assertIn("TRACE_MUTATION_SYSCALL(mremap)", supervisor)
        self.assertIn("TRACE_MUTATION_SYSCALL(madvise)", supervisor)
        self.assertIn("TRACE_MUTATION_SYSCALL(process_madvise)", supervisor)
        self.assertIn("complete_exec_mutation", supervisor)

    def test_sealed_runtime_rejects_glibc_host_library_fallback(self) -> None:
        launcher = (self.source / "runner_launcher.c").read_text(encoding="utf-8")
        linux = (self.source / "linux_runtime.py").read_text(encoding="utf-8")
        builder = (self.source / "build_bundle.py").read_text(encoding="utf-8")
        audit_path = self.source / "runtime_audit.c"

        self.assertTrue(audit_path.is_file())
        audit = audit_path.read_text(encoding="utf-8")
        for required in (
            "la_version",
            "la_objopen",
            "MLE_ALLOWED_ROOTFS_PREFIX",
            "mlebench_lite_runtime_audit_v1",
        ):
            with self.subTest(audit=required):
                self.assertIn(required, audit)
        for source in (launcher, linux):
            with self.subTest(source=source[:80]):
                self.assertIn('"--inhibit-cache"', source)
                self.assertIn('"--audit"', source)
        for required in (
            "runtime_audit.c",
            "lib/mlebench-lite-runtime-audit.so",
            "verify_elf_dependency_closure",
        ):
            with self.subTest(builder=required):
                self.assertIn(required, builder)
        self.assertIn("verify_audited_runtime_mappings", linux)
        self.assertIn("verify_audited_runtime_mappings", launcher)

    def test_elf_closure_parser_rejects_default_host_resolution(self) -> None:
        rootfs = "/sealed/mle-rootfs"
        accepted = (
            "linux-vdso.so.1 (0x00007fff00000000)\n"
            "libc.so.6 => /sealed/mle-rootfs/usr/lib64/libc.so.6 "
            "(0x00007f0000000000)\n"
            "/lib64/ld-linux-x86-64.so.2 => "
            "/sealed/mle-rootfs/usr/lib64/ld-linux-x86-64.so.2 "
            "(0x00007f0000100000)\n"
        )
        self.assertEqual(
            parse_elf_dependency_list(accepted.encode("ascii"), rootfs=rootfs),
            [
                "/usr/lib64/ld-linux-x86-64.so.2",
                "/usr/lib64/libc.so.6",
            ],
        )
        fallback = accepted.replace(
            "/sealed/mle-rootfs/usr/lib64/libc.so.6", "/lib64/libc.so.6"
        )
        with self.assertRaises(BridgeError):
            parse_elf_dependency_list(fallback.encode("ascii"), rootfs=rootfs)

    def test_mapped_runtime_inventory_rejects_executable_outside_sealed_roots(
        self,
    ) -> None:
        rootfs = self.root / "mapping-rootfs"
        bundle = self.root / "mapping-bundle"
        python = rootfs / "usr/bin/python3.11"
        loader = rootfs / "usr/lib/ld-linux-x86-64.so.2"
        audit = bundle / "lib/mlebench-lite-runtime-audit.so"
        outsider = self.root / "host-libc.so.6"
        for path in (python, loader, audit, outsider):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.name.encode("ascii"))
            path.chmod(0o555)

        def mapping_line(path: Path) -> str:
            metadata = path.stat()
            return (
                "00000000-00001000 r-xp 00000000 "
                f"{os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x} "
                f"{metadata.st_ino} {path}"
            )

        deployment = {
            "rootfs": str(rootfs),
            "rootfs_python_path": "/usr/bin/python3.11",
            "rootfs_loader_path": "/usr/lib/ld-linux-x86-64.so.2",
        }
        valid_lines = [mapping_line(path) for path in (python, loader, audit)]
        objects = validate_audited_runtime_map_lines(
            valid_lines, deployment=deployment, audit_metadata=audit.stat()
        )
        self.assertEqual(len(objects), 3)
        with self.assertRaises(BridgeError):
            validate_audited_runtime_map_lines(
                [*valid_lines, mapping_line(outsider)],
                deployment=deployment,
                audit_metadata=audit.stat(),
            )

    @unittest.skipUnless(
        platform.system() == "Linux" and Path("/proc/self/exe").exists(),
        "locked runner entrypoint requires Linux procfs",
    )
    def test_locked_runner_entrypoint_returns_zero_for_a_valid_response(self) -> None:
        if sys.version_info < (3, 11):
            self.skipTest("locked runner entrypoint requires Python 3.11+")
        if os.geteuid() != 0:
            self.skipTest("locked runner rootfs fixture requires root ownership")
        for executable in ("cc", "mount", "umount", "unshare"):
            if shutil.which(executable) is None:
                self.skipTest(f"locked runner fixture requires {executable}")
        namespace_prefixes = (
            ["unshare", "--mount", "--propagation", "private"],
            [
                "unshare",
                "--user",
                "--map-root-user",
                "--mount",
                "--propagation",
                "private",
            ],
        )
        selected = None
        for prefix in namespace_prefixes:
            probe = subprocess.run(
                [*prefix, "true"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == 0:
                selected = prefix
                break
        if selected is None:
            self.skipTest("CAP_SYS_ADMIN/private mount namespace is unavailable")
        expression = (
            "from pathlib import Path; "
            "from tests.test_runtime_bridge_linux_source import "
            "locked_entrypoint_namespace_fixture; "
            f"raise SystemExit(locked_entrypoint_namespace_fixture(Path({str(self.source)!r}), "
            f"Path({str(self.root)!r})))"
        )
        completed = subprocess.run(
            [*selected, sys.executable, "-c", expression],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=120,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            },
        )
        if completed.returncode in {77, 78}:
            self.skipTest("private read-only host-runtime rootfs could not be derived")
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())


if __name__ == "__main__":
    unittest.main()
