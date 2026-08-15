from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import requests

from agentenv_agentmemory.workspace_sandbox import (
    ExecutableFingerprint,
    ShellExecutionResult,
    ShellSandboxLimits,
)
from agentenv_swebench_verified.dataset import (
    DATASET_MANIFEST_SCHEMA,
    VerifiedDataset,
)
from agentenv_swebench_verified.environment import VerifiedEpisodeManager
from agentenv_swebench_verified.exporter import PredictionStore, SolutionPatchExporter
from agentenv_swebench_verified.protocol import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    EVALUATION_MAX_POLICY_TURNS,
    FrozenDatasetPins,
)
from agentenv_swebench_verified.sandbox import (
    VerifiedLinuxNamespaceEpisodeSandbox,
)
from agentenv_swebench_verified.server import create_http_server
from agentenv_swebench_verified.testspec import VerifiedTestSpecBinding
from agentenv_swebench_verified.workspace import VerifiedWorkspaceMaterializer


RUN_CAPABILITY = "r" * 43


def sandbox_limits() -> ShellSandboxLimits:
    return ShellSandboxLimits(
        workspace_bytes=4 * 1024 * 1024,
        workspace_inodes=4096,
        max_files=3072,
        max_directories=1024,
        max_file_bytes=1024 * 1024,
        max_path_chars=512,
        default_timeout_ms=10_000,
        max_timeout_ms=30_000,
        cpu_seconds=10,
        address_space_bytes=1024 * 1024 * 1024,
        max_processes=32,
        max_open_files=128,
        stdout_bytes=4096,
        stderr_bytes=4096,
        tmp_bytes=1024 * 1024,
        tmp_inodes=512,
    )


class Lease:
    def __exit__(self, *_exc: object) -> None:
        return None


class LocalSandbox(VerifiedLinuxNamespaceEpisodeSandbox):
    def __init__(self) -> None:
        super().__init__(
            limits=sandbox_limits(),
            rg_binary=Path("/unused/rg"),
            expected_rg_sha256="0" * 64,
            rg_sha256="0" * 64,
            rg_version="test",
            rg_fingerprint=ExecutableFingerprint(0, 0, 0, 0, 0, 0),
            binaries={},
            uid_lease_context=Lease(),
            model_uid=(1000 if os.getuid() == 0 else os.getuid()),
        )

    @property
    def model_gid(self) -> int:
        return 1000 if os.getgid() == 0 else os.getgid()

    def _run_namespace(
        self,
        workspace_root: Path,
        *,
        command: str,
        workdir: str,
        timeout_ms: int,
    ) -> ShellExecutionResult:
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=workspace_root / workdir,
            capture_output=True,
            timeout=timeout_ms / 1000,
        )
        return ShellExecutionResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            exit_code=completed.returncode,
            elapsed_ms=1,
            timed_out=False,
            stdout_truncated=False,
            stderr_truncated=False,
            termination_reason=None,
            sandbox_contract="test",
            model_uid=self.model_uid,
        )


class Resolver:
    def __init__(self) -> None:
        self.private_rows: list[dict[str, object]] = []

    def resolve(self, instance):
        self.private_rows.append(dict(instance))
        return VerifiedTestSpecBinding(
            instance_id=instance["instance_id"],
            repo=instance["repo"],
            base_commit=instance["base_commit"],
            instance_image_key="swebench/sweb.eval.x86_64.owner_1776_repo-1:latest",
            platform="linux/x86_64",
            namespace="swebench",
            source_revision="7" * 40,
            source_tag="v4.1.0",
        )


class VerifiedEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.mirrors = self.root / "mirrors"
        self.mirrors.mkdir(mode=0o700)
        self.mirror = self.mirrors / "owner__repo"
        self.mirror.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.write_mirror("src/value.py", "value = 'bug'\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "base")
        self.base_commit = self.git("rev-parse", "HEAD").strip()
        row = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": self.base_commit,
            "problem_statement": "Fix the public value.",
            "version": "1.0",
            "patch": "SECRET_GOLD_PATCH",
            "test_patch": "SECRET_TEST_PATCH",
            "FAIL_TO_PASS": "[\"SECRET_F2P\"]",
            "PASS_TO_PASS": "[\"SECRET_P2P\"]",
            "hints_text": "SECRET_HINT",
            "eval_script": "SECRET_EVAL_SCRIPT",
            "log_parser": "SECRET_PARSER",
        }
        payload = (
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode()
        jsonl = self.root / "dataset.jsonl"
        jsonl.write_bytes(payload)
        id_ledger = b"owner__repo-1\n"
        pins = FrozenDatasetPins(
            repository=DATASET_REPOSITORY,
            revision=DATASET_REVISION,
            split="test",
            row_count=1,
            canonical_jsonl_sha256=hashlib.sha256(payload).hexdigest(),
            id_ledger_sha256=hashlib.sha256(id_ledger).hexdigest(),
        )
        manifest = {
            "schema_version": DATASET_MANIFEST_SCHEMA,
            "dataset": {
                "repository": pins.repository,
                "revision": pins.revision,
                "split": pins.split,
            },
            "canonical_jsonl": {
                "path": jsonl.name,
                "sha256": pins.canonical_jsonl_sha256,
                "rows": pins.row_count,
                "id_ledger_sha256": pins.id_ledger_sha256,
            },
        }
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.dataset = VerifiedDataset(manifest_path, pins=pins)
        self.resolver = Resolver()
        self.materializer = VerifiedWorkspaceMaterializer(
            mirrors_root=self.mirrors,
            episodes_root=self.root / "episodes",
        )
        self.store = PredictionStore(
            self.root / "predictions",
            instance_ids=self.dataset.instance_ids,
        )
        self.manager = VerifiedEpisodeManager(
            dataset=self.dataset,
            materializer=self.materializer,
            testspec_resolver=self.resolver,
            sandbox_factory=lambda _record, _binding: LocalSandbox(),
            exporter=SolutionPatchExporter(),
            prediction_store=self.store,
            max_native_actions=EVALUATION_MAX_POLICY_TURNS,
            max_observation_bytes=512,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.mirror), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def write_mirror(self, relative: str, content: str) -> None:
        path = self.mirror / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_shell_patch_and_terminal_export_without_internal_grading(self) -> None:
        slot = self.manager.create(
            arm="native",
            run_id="native-run",
            run_capability=RUN_CAPABILITY,
        )
        reset = self.manager.reset(slot, 0)
        public_reset = json.dumps(reset.as_dict())
        self.assertIn("Fix the public value", reset.observation)
        for secret in (
            "SECRET_GOLD_PATCH",
            "SECRET_TEST_PATCH",
            "SECRET_F2P",
            "SECRET_P2P",
            "SECRET_HINT",
            "SECRET_EVAL_SCRIPT",
            "SECRET_PARSER",
        ):
            self.assertNotIn(secret, public_reset)
        self.assertEqual(self.resolver.private_rows[0]["patch"], "SECRET_GOLD_PATCH")

        written = self.manager.step(
            slot,
            'shell_command {"command":"printf persistent > state.txt"}',
        )
        self.assertFalse(written.done)
        read = self.manager.step(
            slot,
            'shell_command {"command":"cat state.txt"}',
        )
        self.assertIn("persistent", read.observation)
        patched = self.manager.step(
            slot,
            "apply_patch\n"
            "*** Begin Patch\n"
            "*** Update File: src/value.py\n"
            "@@\n"
            "-value = 'bug'\n"
            "+value = 'fixed'\n"
            "*** End Patch",
        )
        self.assertIn("apply_patch succeeded", patched.observation)
        terminal = self.manager.step(slot, "Implemented and tested the fix.")

        self.assertTrue(terminal.done)
        self.assertEqual(terminal.reward, 0.0)
        self.assertTrue(terminal.info["external_grading_required"])
        self.assertNotIn("episode_success", terminal.info)
        prediction = self.manager.prediction(slot)
        self.assertEqual(prediction["instance_id"], "owner__repo-1")
        self.assertIn("src/value.py", prediction["model_patch"])
        self.assertIn("state.txt", prediction["model_patch"])
        self.assertNotIn("SECRET", json.dumps(prediction))
        self.assertEqual(
            self.store.read(arm="native", run_id="native-run", data_idx=0),
            prediction,
        )
        self.manager.close(slot)
        self.assertEqual(list((self.root / "episodes").iterdir()), [])

    def test_both_arms_use_identical_non_memory_dispatch(self) -> None:
        slots = {
            arm: self.manager.create(
                arm=arm,
                run_id=f"{arm}-dispatch",
                run_capability=RUN_CAPABILITY,
            )
            for arm in ("native", "amg_memory")
        }
        for slot in slots.values():
            self.manager.reset(slot, 0)
        action = 'shell_command {"command":"printf same > common.txt"}'
        results = {arm: self.manager.step(slot, action) for arm, slot in slots.items()}

        self.assertEqual(
            results["native"].observation,
            results["amg_memory"].observation,
        )
        for key in ("action_kind", "actor_credit", "action_progress"):
            self.assertEqual(
                results["native"].info[key],
                results["amg_memory"].info[key],
            )
        for slot in slots.values():
            self.manager.close(slot)

    def test_parser_errors_remain_non_terminal_and_do_not_dispatch(self) -> None:
        slot = self.manager.create(
            arm="native",
            run_id="parser-run",
            run_capability=RUN_CAPABILITY,
        )
        self.manager.reset(slot, 0)

        result = self.manager.step(slot, "shell_command pwd")

        self.assertFalse(result.done)
        self.assertIn("Invalid action syntax", result.observation)
        self.assertEqual(result.info["action_kind"], "parser_error")
        self.assertEqual(
            result.info["actor_credit"],
            {
                "schema": "task_neutral_actor_credit_v1",
                "positive_eligible": False,
                "basis": "parser_rejected",
            },
        )
        self.manager.close(slot)

    def test_memory_artifacts_persist_and_reset_clean(self) -> None:
        slot = self.manager.create(
            arm="amg_memory",
            run_id="memory-first",
            run_capability=RUN_CAPABILITY,
        )
        self.manager.reset(slot, 0)
        self.manager.step(
            slot,
            'shell_command {"command":"mkdir -p .agent_memory && '
            'printf clue > .agent_memory/notes"}',
        )
        read = self.manager.step(
            slot,
            'shell_command {"command":"cat .agent_memory/notes"}',
        )
        self.assertIn("clue", read.observation)
        self.manager.step(slot, "Submit memory-only workspace.")
        self.assertEqual(self.manager.prediction(slot)["model_patch"], "")
        self.manager.close(slot)

        fresh = self.manager.create(
            arm="amg_memory",
            run_id="memory-fresh",
            run_capability=RUN_CAPABILITY,
        )
        self.manager.reset(fresh, 0)
        absent = self.manager.step(
            fresh,
            'shell_command {"command":"test ! -e .agent_memory && printf clean"}',
        )
        self.assertIn("clean", absent.observation)
        self.manager.close(fresh)

    def test_shell_observation_is_bounded_and_horizon_exports(self) -> None:
        slot = self.manager.create(
            arm="native",
            run_id="horizon-run",
            run_capability=RUN_CAPABILITY,
        )
        reset = self.manager.reset(slot, 0)
        self.assertLessEqual(len(reset.observation.encode()), 512)
        result = self.manager.step(
            slot,
            'shell_command {"command":"python3 -c \'print(\\"x\\" * 10000)\'"}',
        )
        self.assertIn("truncated", result.observation)
        self.assertLessEqual(len(result.observation.encode()), 512)
        horizon = self.manager.finalize_horizon(slot)
        self.assertTrue(horizon.done)
        self.assertEqual(horizon.reward, 0.0)
        self.assertEqual(self.manager.prediction(slot)["model_patch"], "")
        self.manager.close(slot)

    def test_apply_patch_observation_obeys_the_same_whole_message_cap(self) -> None:
        slot = self.manager.create(
            arm="native",
            run_id="patch-bound-run",
            run_capability=RUN_CAPABILITY,
        )
        self.manager.reset(slot, 0)
        patch_lines = ["apply_patch", "*** Begin Patch"]
        for index in range(40):
            patch_lines.extend(
                [
                    f"*** Add File: generated/very_long_file_name_{index:03d}.txt",
                    "+created",
                ]
            )
        patch_lines.append("*** End Patch")

        result = self.manager.step(slot, "\n".join(patch_lines))

        self.assertIn("observation truncated", result.observation)
        self.assertLessEqual(len(result.observation.encode()), 512)
        self.manager.close(slot)

    def test_metadata_freezes_external_grading_and_unified_budget(self) -> None:
        metadata = self.manager.metadata()
        self.assertEqual(metadata["evaluation_max_policy_turns"], 250)
        self.assertEqual(metadata["task_count"], 1)
        self.assertEqual(metadata["full_benchmark_task_count"], 500)
        self.assertEqual(metadata["reward_contract"], "external_official_grading_only")
        self.assertEqual(metadata["supported_arms"], ["native", "amg_memory"])
        self.assertEqual(
            metadata["policy_visible_fields"],
            ["instance_id", "repo", "base_commit", "problem_statement"],
        )
        serialized = json.dumps(metadata)
        self.assertNotIn("SECRET", serialized)

    def test_close_tombstone_rejects_a_reset_with_a_stale_slot_reference(self) -> None:
        slot_id = self.manager.create(
            arm="native",
            run_id="close-race",
            run_capability=RUN_CAPABILITY,
        )
        original_slot = self.manager._slot
        stale_obtained = threading.Event()
        release_reset = threading.Event()
        errors: list[BaseException] = []

        def hooked_slot(requested_id: int):
            slot = original_slot(requested_id)
            if threading.current_thread().name == "stale-reset":
                stale_obtained.set()
                release_reset.wait(timeout=5)
            return slot

        def reset_stale_slot() -> None:
            try:
                self.manager.reset(slot_id, 0)
            except BaseException as exc:
                errors.append(exc)

        self.manager._slot = hooked_slot
        thread = threading.Thread(target=reset_stale_slot, name="stale-reset")
        thread.start()
        self.assertTrue(stale_obtained.wait(timeout=5))
        self.manager.close(slot_id)
        release_reset.set()
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertIn("closed", str(errors[0]))
        self.assertEqual(list((self.root / "episodes").iterdir()), [])

    def test_http_foreign_capability_cannot_read_or_mutate_another_slot(self) -> None:
        server = create_http_server(self.manager, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        base = f"http://{host}:{port}"
        created_slots = []
        victim_run_headers = {"Authorization": f"Bearer {'v' * 43}"}
        foreign_run_headers = {"Authorization": f"Bearer {'f' * 43}"}
        try:
            victim = requests.post(
                f"{base}/create",
                json={"arm": "native", "run_id": "victim-run"},
                headers=victim_run_headers,
                timeout=5,
            ).json()
            created_slots.append(victim)
            foreign = requests.post(
                f"{base}/create",
                json={"arm": "native", "run_id": "foreign-run"},
                headers=foreign_run_headers,
                timeout=5,
            ).json()
            created_slots.append(foreign)
            self.assertNotEqual(victim["capability"], foreign["capability"])
            self.assertGreaterEqual(len(victim["capability"]), 32)

            reset = requests.post(
                f"{base}/reset",
                json={
                    "id": victim["id"],
                    "data_idx": 0,
                },
                headers={
                    "Authorization": f"Bearer {victim['capability']}"
                },
                timeout=5,
            )
            self.assertEqual(reset.status_code, 200)
            private = requests.post(
                f"{base}/step",
                json={
                    "id": victim["id"],
                    "action": (
                        'shell_command {"command":"printf PRIVATE_NOTE > note '
                        '&& cat note"}'
                    ),
                },
                headers={
                    "Authorization": f"Bearer {victim['capability']}"
                },
                timeout=5,
            )
            self.assertIn("PRIVATE_NOTE", private.json()["observation"])

            read = requests.get(
                f"{base}/observation",
                params={"id": victim["id"]},
                headers={
                    "Authorization": f"Bearer {foreign['capability']}"
                },
                timeout=5,
            )
            self.assertEqual(read.status_code, 403)
            self.assertNotIn("PRIVATE_NOTE", read.text)
            mutate = requests.post(
                f"{base}/step",
                json={
                    "id": victim["id"],
                    "action": 'shell_command {"command":"touch attacked"}',
                },
                headers={
                    "Authorization": f"Bearer {foreign['capability']}"
                },
                timeout=5,
            )
            self.assertEqual(mutate.status_code, 403)

            intact = requests.post(
                f"{base}/step",
                json={
                    "id": victim["id"],
                    "action": (
                        'shell_command {"command":"test ! -e attacked '
                        '&& printf intact"}'
                    ),
                },
                headers={
                    "Authorization": f"Bearer {victim['capability']}"
                },
                timeout=5,
            )
            self.assertEqual(intact.status_code, 200)
            self.assertIn("intact", intact.json()["observation"])
        finally:
            for slot in created_slots:
                requests.post(
                    f"{base}/close",
                    json={"id": slot["id"]},
                    headers={
                        "Authorization": f"Bearer {slot['capability']}"
                    },
                    timeout=5,
                )
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_same_run_requires_the_original_run_capability(self) -> None:
        server = create_http_server(self.manager, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        owner_headers = {"Authorization": f"Bearer {'a' * 43}"}
        foreign_headers = {"Authorization": f"Bearer {'b' * 43}"}
        created_slots = []
        try:
            victim = requests.post(
                f"{base}/create",
                json={"arm": "native", "run_id": "owned-run"},
                headers=owner_headers,
                timeout=5,
            )
            self.assertEqual(victim.status_code, 200)
            created_slots.append(victim.json())
            parallel = requests.post(
                f"{base}/create",
                json={"arm": "native", "run_id": "owned-run"},
                headers=owner_headers,
                timeout=5,
            )
            self.assertEqual(parallel.status_code, 200)
            created_slots.append(parallel.json())

            foreign = requests.post(
                f"{base}/create",
                json={"arm": "native", "run_id": "owned-run"},
                headers=foreign_headers,
                timeout=5,
            )
            self.assertEqual(foreign.status_code, 403)
        finally:
            for slot in created_slots:
                requests.post(
                    f"{base}/close",
                    json={"id": slot["id"]},
                    headers={
                        "Authorization": f"Bearer {slot['capability']}"
                    },
                    timeout=5,
                )
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_run_capability_survives_server_restart(self) -> None:
        owner_headers = {"Authorization": f"Bearer {'a' * 43}"}
        foreign_headers = {"Authorization": f"Bearer {'b' * 43}"}
        server = create_http_server(self.manager, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            claimed = requests.post(
                f"{base}/create",
                json={"arm": "native", "run_id": "restart-owned-run"},
                headers=owner_headers,
                timeout=5,
            )
            self.assertEqual(claimed.status_code, 200)
            slot = claimed.json()
            closed = requests.post(
                f"{base}/close",
                json={"id": slot["id"]},
                headers={"Authorization": f"Bearer {slot['capability']}"},
                timeout=5,
            )
            self.assertEqual(closed.status_code, 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        restarted_manager = VerifiedEpisodeManager(
            dataset=self.dataset,
            materializer=self.materializer,
            testspec_resolver=self.resolver,
            sandbox_factory=lambda _record, _binding: LocalSandbox(),
            exporter=SolutionPatchExporter(),
            prediction_store=self.store,
            max_native_actions=EVALUATION_MAX_POLICY_TURNS,
            max_observation_bytes=512,
        )
        restarted_server = create_http_server(
            restarted_manager,
            host="127.0.0.1",
            port=0,
        )
        restarted_thread = threading.Thread(
            target=restarted_server.serve_forever,
            daemon=True,
        )
        restarted_thread.start()
        restarted_base = f"http://127.0.0.1:{restarted_server.server_port}"
        owner_slot = None
        try:
            foreign = requests.post(
                f"{restarted_base}/create",
                json={"arm": "native", "run_id": "restart-owned-run"},
                headers=foreign_headers,
                timeout=5,
            )
            self.assertEqual(foreign.status_code, 403)

            owner = requests.post(
                f"{restarted_base}/create",
                json={"arm": "native", "run_id": "restart-owned-run"},
                headers=owner_headers,
                timeout=5,
            )
            self.assertEqual(owner.status_code, 200)
            owner_slot = owner.json()
        finally:
            if owner_slot is not None:
                requests.post(
                    f"{restarted_base}/close",
                    json={"id": owner_slot["id"]},
                    headers={
                        "Authorization": f"Bearer {owner_slot['capability']}"
                    },
                    timeout=5,
                )
            restarted_server.shutdown()
            restarted_server.server_close()
            restarted_thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
