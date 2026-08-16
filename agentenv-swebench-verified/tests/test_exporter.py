from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_swebench_verified.exporter import (
    GitOutputLimitError,
    PatchExportError,
    PredictionStore,
    PredictionStoreError,
    RunCapabilityMismatch,
    SolutionPatchExporter,
    run_git,
)
from agentenv_swebench_verified.protocol import MODEL_LABELS
from agentenv_swebench_verified.workspace import VerifiedWorkspaceMaterializer


class ExporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        mirrors = self.root / "mirrors"
        mirrors.mkdir(mode=0o700)
        self.mirror = mirrors / "owner__repo"
        self.mirror.mkdir()
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.write_mirror("src/value.py", "value = 'base'\n")
        self.write_mirror("obsolete.txt", "remove\n")
        self.write_mirror("script.sh", "#!/bin/sh\nprintf ready\n")
        self.write_mirror(".gitignore", "src/ignored_solution.py\n")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "base")
        self.base_commit = self.git("rev-parse", "HEAD").strip()
        self.row = {
            "instance_id": "owner__repo-1",
            "repo": "owner/repo",
            "base_commit": self.base_commit,
            "problem_statement": "Repair it",
        }
        self.materializer = VerifiedWorkspaceMaterializer(
            mirrors_root=mirrors,
            episodes_root=self.root / "episodes",
        )
        self.exporter = SolutionPatchExporter()

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

    def row_with_base_gitlink(self) -> dict[str, str]:
        self.git(
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{self.base_commit},vendor/submodule",
        )
        self.git("commit", "-q", "-m", "add base gitlink")
        row = dict(self.row)
        row["base_commit"] = self.git("rev-parse", "HEAD").strip()
        return row

    def test_exports_solution_files_without_runtime_artifacts(self) -> None:
        object_inventory_before = self.git("count-objects", "-v")
        workspace = self.materializer.materialize(
            self.row,
            model_uid=(1000 if os.getuid() == 0 else os.getuid()),
            model_gid=(1000 if os.getgid() == 0 else os.getgid()),
        )
        try:
            (workspace.policy_root / "src/value.py").write_text(
                "value = 'fixed'\n", encoding="utf-8"
            )
            (workspace.policy_root / "obsolete.txt").unlink()
            (workspace.policy_root / "src/new_file.py").write_text(
                "created = True\n", encoding="utf-8"
            )
            (workspace.policy_root / "src/new_binary.bin").write_bytes(
                b"\x00\x01solution\xff"
            )
            (workspace.policy_root / "src/ignored_solution.py").write_text(
                "required = True\n",
                encoding="utf-8",
            )
            (workspace.policy_root / "script.sh").chmod(0o755)
            ordinary_note = workspace.policy_root / ".agent_memory" / "debugging.md"
            ordinary_note.parent.mkdir(parents=True)
            ordinary_note.write_text("ordinary repository state", encoding="utf-8")
            artifacts = {
                ".agent_logs/tool.log": "private log",
                ".agent_receipts/action.json": "private receipt",
                ".agent_telemetry/trace.json": "private telemetry",
            }
            for relative, content in artifacts.items():
                path = workspace.policy_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            rows = {
                arm: self.exporter.prediction_row(workspace, arm=arm)
                for arm in ("native", "amg_compaction_only", "amg_memory")
            }
            self.assertFalse((workspace.private_root / "export.git").exists())
        finally:
            self.materializer.close(workspace)
        object_inventory_after = self.git("count-objects", "-v")

        for arm, row in rows.items():
            self.assertEqual(
                set(row), {"instance_id", "model_name_or_path", "model_patch"}
            )
            self.assertEqual(row["instance_id"], "owner__repo-1")
            self.assertEqual(row["model_name_or_path"], MODEL_LABELS[arm])
        patches = {row["model_patch"] for row in rows.values()}
        self.assertEqual(len(patches), 1)
        patch = patches.pop()
        self.assertIn("diff --git a/src/value.py b/src/value.py", patch)
        self.assertIn("diff --git a/obsolete.txt b/obsolete.txt", patch)
        self.assertIn("diff --git a/src/new_file.py b/src/new_file.py", patch)
        self.assertIn("diff --git a/src/new_binary.bin b/src/new_binary.bin", patch)
        self.assertIn(
            "diff --git a/src/ignored_solution.py b/src/ignored_solution.py",
            patch,
        )
        self.assertIn("GIT binary patch", patch)
        self.assertIn("old mode 100644", patch)
        self.assertIn("new mode 100755", patch)
        self.assertIn(".agent_memory/debugging.md", patch)
        self.assertIn("ordinary repository state", patch)
        self.assertEqual(object_inventory_after, object_inventory_before)
        for forbidden in (
            ".agent_logs",
            ".agent_receipts",
            ".agent_telemetry",
            "private log",
            "private receipt",
            "private telemetry",
        ):
            self.assertNotIn(forbidden, patch)

    def test_nested_git_metadata_yields_an_explicit_empty_prediction(self) -> None:
        workspace = self.materializer.materialize(self.row)
        try:
            (workspace.policy_root / "nested" / ".git").mkdir(parents=True)
            (workspace.policy_root / "nested" / "solution.py").write_text(
                "fixed = True\n",
                encoding="utf-8",
            )

            row = self.exporter.prediction_row(workspace, arm="native")
        finally:
            self.materializer.close(workspace)

        self.assertEqual(row["model_patch"], "")

    def test_base_gitlinks_are_not_exported_as_deletions(self) -> None:
        workspace = self.materializer.materialize(self.row_with_base_gitlink())
        try:
            placeholder = workspace.policy_root / "vendor" / "submodule"
            self.assertTrue(placeholder.is_dir())
            self.assertEqual(list(placeholder.iterdir()), [])
            patch = self.exporter.export(workspace)
        finally:
            self.materializer.close(workspace)

        self.assertEqual(patch, "")

    def test_base_gitlink_replacements_yield_an_empty_prediction(self) -> None:
        workspace = self.materializer.materialize(self.row_with_base_gitlink())
        try:
            replacement = workspace.policy_root / "vendor" / "submodule"
            replacement.mkdir(parents=True, exist_ok=True)
            (replacement / "replacement.py").write_text(
                "replacement = True\n",
                encoding="utf-8",
            )

            prediction = self.exporter.prediction_row(workspace, arm="native")
        finally:
            self.materializer.close(workspace)

        self.assertEqual(prediction["model_patch"], "")

    def test_base_gitlink_deletion_yields_an_empty_prediction(self) -> None:
        workspace = self.materializer.materialize(self.row_with_base_gitlink())
        try:
            (workspace.policy_root / "vendor" / "submodule").rmdir()
            (workspace.policy_root / "src/value.py").write_text(
                "value = 'otherwise valid fix'\n",
                encoding="utf-8",
            )

            prediction = self.exporter.prediction_row(workspace, arm="native")
        finally:
            self.materializer.close(workspace)

        self.assertEqual(prediction["model_patch"], "")

    def test_empty_patch_still_produces_one_exact_prediction_row(self) -> None:
        workspace = self.materializer.materialize(self.row)
        try:
            row = self.exporter.prediction_row(workspace, arm="amg_memory")
        finally:
            self.materializer.close(workspace)
        self.assertEqual(
            row,
            {
                "instance_id": "owner__repo-1",
                "model_name_or_path": MODEL_LABELS["amg_memory"],
                "model_patch": "",
            },
        )

    def test_store_scopes_duplicates_and_assembles_in_order(self) -> None:
        store = PredictionStore(
            self.root / "predictions",
            instance_ids=("task-0", "task-1"),
        )
        native_rows = [
            {
                "instance_id": f"task-{index}",
                "model_name_or_path": MODEL_LABELS["native"],
                "model_patch": "" if index == 0 else "patch-1",
            }
            for index in range(2)
        ]
        amg_rows = [
            {
                "instance_id": f"task-{index}",
                "model_name_or_path": MODEL_LABELS["amg_memory"],
                "model_patch": f"amg-{index}",
            }
            for index in range(2)
        ]

        store.write(
            arm="native",
            run_id="paired-0815",
            data_idx=1,
            row=native_rows[1],
        )
        store.write(
            arm="native",
            run_id="paired-0815",
            data_idx=0,
            row=native_rows[0],
        )
        store.write(
            arm="amg_memory",
            run_id="paired-0815",
            data_idx=0,
            row=amg_rows[0],
        )
        with self.assertRaisesRegex(PredictionStoreError, "duplicate"):
            store.write(
                arm="native",
                run_id="paired-0815",
                data_idx=0,
                row=native_rows[0],
            )
        with self.assertRaisesRegex(PredictionStoreError, "incomplete"):
            store.assemble(arm="amg_memory", run_id="paired-0815")
        store.write(
            arm="amg_memory",
            run_id="paired-0815",
            data_idx=1,
            row=amg_rows[1],
        )

        native_path = store.assemble(arm="native", run_id="paired-0815")
        amg_path = store.assemble(arm="amg_memory", run_id="paired-0815")

        native = [json.loads(line) for line in native_path.read_text().splitlines()]
        amg = [json.loads(line) for line in amg_path.read_text().splitlines()]
        self.assertEqual(native, native_rows)
        self.assertEqual(amg, amg_rows)
        self.assertNotEqual(native_path, amg_path)

        foreign = store.rows_root("native", "paired-0815") / "9999.json"
        foreign.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(PredictionStoreError, "unexpected"):
            store.assemble(arm="native", run_id="paired-0815")

    def test_store_rejects_foreign_ids_and_unpinned_model_labels(self) -> None:
        store = PredictionStore(self.root / "predictions", instance_ids=("task-0",))
        with self.assertRaisesRegex(PredictionStoreError, "instance_id"):
            store.write(
                arm="native",
                run_id="run",
                data_idx=0,
                row={
                    "instance_id": "foreign",
                    "model_name_or_path": MODEL_LABELS["native"],
                    "model_patch": "",
                },
            )
        with self.assertRaisesRegex(PredictionStoreError, "model_name"):
            store.write(
                arm="native",
                run_id="run",
                data_idx=0,
                row={
                    "instance_id": "task-0",
                    "model_name_or_path": "unfrozen-label",
                    "model_patch": "",
                },
            )
        for run_id in (".", ".."):
            with self.subTest(run_id=run_id):
                with self.assertRaisesRegex(PredictionStoreError, "run_id"):
                    store.write(
                        arm="native",
                        run_id=run_id,
                        data_idx=0,
                        row={
                            "instance_id": "task-0",
                            "model_name_or_path": MODEL_LABELS["native"],
                            "model_patch": "",
                        },
                    )

    def test_store_run_claim_is_persistent_private_and_digest_only(self) -> None:
        root = self.root / "predictions"
        store = PredictionStore(root, instance_ids=("task-0",))
        capability = "owner-secret-capability"
        digest = hashlib.sha256(capability.encode("ascii")).digest()

        claim = store.claim_run(
            arm="native",
            run_id="persistent-run",
            capability_digest=digest,
        )

        self.assertEqual(
            claim.read_bytes(),
            hashlib.sha256(capability.encode("ascii")).hexdigest().encode()
            + b"\n",
        )
        self.assertNotIn(capability.encode("ascii"), claim.read_bytes())
        self.assertEqual(claim.stat().st_mode & 0o777, 0o600)
        restarted_store = PredictionStore(root, instance_ids=("task-0",))
        self.assertEqual(
            restarted_store.claim_run(
                arm="native",
                run_id="persistent-run",
                capability_digest=digest,
            ),
            claim,
        )
        with self.assertRaises(RunCapabilityMismatch):
            restarted_store.claim_run(
                arm="native",
                run_id="persistent-run",
                capability_digest=hashlib.sha256(b"foreign").digest(),
            )

    def test_store_run_claim_is_atomic_across_store_instances(self) -> None:
        root = self.root / "predictions"
        stores = [
            PredictionStore(root, instance_ids=("task-0",))
            for _ in range(2)
        ]
        barrier = threading.Barrier(2)
        outcomes: list[str] = []

        def claim(index: int) -> None:
            barrier.wait(timeout=5)
            try:
                stores[index].claim_run(
                    arm="amg_memory",
                    run_id="atomic-run",
                    capability_digest=hashlib.sha256(
                        f"capability-{index}".encode("ascii")
                    ).digest(),
                )
            except RunCapabilityMismatch:
                outcomes.append("mismatch")
            else:
                outcomes.append("claimed")

        threads = [threading.Thread(target=claim, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertCountEqual(outcomes, ["claimed", "mismatch"])

    def test_store_run_claim_rejects_noncanonical_existing_content(self) -> None:
        store = PredictionStore(
            self.root / "predictions",
            instance_ids=("task-0",),
        )
        digest = hashlib.sha256(b"owner").digest()
        claim = store.claim_run(
            arm="native",
            run_id="tampered-run",
            capability_digest=digest,
        )
        claim.write_text("not-a-canonical-digest\n", encoding="ascii")

        with self.assertRaisesRegex(PredictionStoreError, "non-canonical"):
            store.claim_run(
                arm="native",
                run_id="tampered-run",
                capability_digest=digest,
            )

    def test_store_run_claim_rejects_unsafe_existing_nodes_without_blocking(
        self,
    ) -> None:
        store = PredictionStore(
            self.root / "predictions",
            instance_ids=("task-0",),
        )
        digest = hashlib.sha256(b"owner").digest()

        def replace_claim(run_id: str) -> Path:
            claim = store.claim_run(
                arm="native",
                run_id=run_id,
                capability_digest=digest,
            )
            claim.unlink()
            return claim

        fifo_claim = replace_claim("fifo-run")
        os.mkfifo(fifo_claim, mode=0o600)
        outcomes: list[BaseException | None] = []

        def claim_fifo() -> None:
            try:
                store.claim_run(
                    arm="native",
                    run_id="fifo-run",
                    capability_digest=digest,
                )
            except BaseException as exc:  # Captured for assertion in the test thread.
                outcomes.append(exc)
            else:
                outcomes.append(None)

        thread = threading.Thread(target=claim_fifo, daemon=True)
        thread.start()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive(), "claim reader blocked on a FIFO")
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], PredictionStoreError)

        symlink_claim = replace_claim("symlink-run")
        symlink_target = self.root / "external-claim"
        symlink_target.write_bytes(digest.hex().encode("ascii") + b"\n")
        symlink_target.chmod(0o600)
        symlink_claim.symlink_to(symlink_target)
        with self.assertRaises(PredictionStoreError):
            store.claim_run(
                arm="native",
                run_id="symlink-run",
                capability_digest=digest,
            )
        self.assertEqual(
            symlink_target.read_bytes(),
            digest.hex().encode("ascii") + b"\n",
        )

        public_claim = replace_claim("public-run")
        public_claim.write_bytes(digest.hex().encode("ascii") + b"\n")
        public_claim.chmod(0o640)
        with self.assertRaisesRegex(PredictionStoreError, "regular private file"):
            store.claim_run(
                arm="native",
                run_id="public-run",
                capability_digest=digest,
            )

        hardlink_claim = replace_claim("hardlink-run")
        hardlink_source = self.root / "hardlinked-claim"
        hardlink_source.write_bytes(digest.hex().encode("ascii") + b"\n")
        hardlink_source.chmod(0o600)
        os.link(hardlink_source, hardlink_claim)
        with self.assertRaisesRegex(PredictionStoreError, "regular private file"):
            store.claim_run(
                arm="native",
                run_id="hardlink-run",
                capability_digest=digest,
            )

    def test_git_runner_bounds_stderr(self) -> None:
        command = [
            "/bin/sh",
            "-c",
            "i=0; while [ $i -lt 1024 ]; do printf x >&2; i=$((i+1)); done",
            "bounded-stderr",
            f"--work-tree={self.mirror}",
        ]

        with self.assertRaisesRegex(GitOutputLimitError, "stderr"):
            run_git(
                command,
                os.environ,
                "exercise the stderr cap",
                stderr_limit=128,
            )

    def test_git_output_limit_still_yields_an_empty_prediction(self) -> None:
        workspace = self.materializer.materialize(self.row)
        real_run_git = run_git

        def overflow_while_staging(command, environment, label, **kwargs):
            if label == "stage solution workspace":
                raise GitOutputLimitError("model-controlled git stderr overflow")
            return real_run_git(command, environment, label, **kwargs)

        try:
            with patch(
                "agentenv_swebench_verified.exporter.run_git",
                side_effect=overflow_while_staging,
            ):
                prediction = self.exporter.prediction_row(
                    workspace,
                    arm="native",
                )
        finally:
            self.materializer.close(workspace)

        self.assertEqual(prediction["model_patch"], "")

    def test_final_diff_failure_still_yields_an_empty_prediction(self) -> None:
        workspace = self.materializer.materialize(self.row)
        real_run_git = run_git

        def fail_final_diff(command, environment, label, **kwargs):
            if label == "export solution diff":
                raise PatchExportError("model-derived diff timed out")
            return real_run_git(command, environment, label, **kwargs)

        try:
            with patch(
                "agentenv_swebench_verified.exporter.run_git",
                side_effect=fail_final_diff,
            ):
                prediction = self.exporter.prediction_row(
                    workspace,
                    arm="native",
                )
        finally:
            self.materializer.close(workspace)

        self.assertEqual(prediction["model_patch"], "")

    def test_model_attributes_staging_failure_yields_an_empty_prediction(self) -> None:
        workspace = self.materializer.materialize(self.row)
        try:
            (workspace.policy_root / ".gitattributes").write_text(
                "src/value.py working-tree-encoding=definitely-invalid\n",
                encoding="utf-8",
            )
            (workspace.policy_root / "src/value.py").write_text(
                "value = 'candidate fix'\n",
                encoding="utf-8",
            )

            prediction = self.exporter.prediction_row(workspace, arm="native")
        finally:
            self.materializer.close(workspace)

        self.assertEqual(prediction["model_patch"], "")

    def test_exporter_ignores_inherited_git_object_redirects(self) -> None:
        workspace = self.materializer.materialize(self.row)
        try:
            (workspace.policy_root / "src/value.py").write_text(
                "value = 'fixed despite host git env'\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"GIT_OBJECT_DIRECTORY": str(self.mirror / ".git" / "objects")},
            ):
                prediction = self.exporter.prediction_row(
                    workspace,
                    arm="native",
                )
        finally:
            self.materializer.close(workspace)

        self.assertIn("value = 'fixed despite host git env'", prediction["model_patch"])

    def test_store_rejects_preseeded_descendant_symlinks(self) -> None:
        store = PredictionStore(
            self.root / "predictions",
            instance_ids=("task-0",),
        )
        external = self.root / "external"
        external.mkdir(mode=0o755)
        (store.root / "native").symlink_to(
            external,
            target_is_directory=True,
        )
        row = {
            "instance_id": "task-0",
            "model_name_or_path": MODEL_LABELS["native"],
            "model_patch": "",
        }

        with self.assertRaisesRegex(PredictionStoreError, "real private directory"):
            store.write(
                arm="native",
                run_id="symlink-run",
                data_idx=0,
                row=row,
            )

        self.assertEqual(list(external.iterdir()), [])
        self.assertEqual(external.stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()
