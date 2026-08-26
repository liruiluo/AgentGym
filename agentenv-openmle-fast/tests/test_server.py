from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentenv_openmle_fast.dataset import OpenMLEFastDataset
from agentenv_openmle_fast.environment import OpenMLEFastEpisodeManager
from agentenv_openmle_fast.executor import (
    LocalCPUExecutionBackend,
    OpenMLEFastExecutor,
    OpenMLEFastResourceLimits,
)
from agentenv_openmle_fast.materializer import OpenMLEFastWorkspaceMaterializer
from agentenv_openmle_fast.server import create_app
from fastapi.testclient import TestClient
from tests.support import FakeWorkspaceMountBackend, RELEASE_REVISION, create_fixture


class _InvalidGrader:
    def grade(self, **_kwargs):
        raise AssertionError("server test does not submit")


class OpenMLEFastServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="openmle-server-test-")
        self.root = Path(self.temporary.name)
        self.fixture = create_fixture(self.root)
        dataset = OpenMLEFastDataset(
            manifest_path=Path(self.fixture["manifest"]),
            package_root=Path(self.fixture["package_root"]),
            archive_root=Path(self.fixture["archive_root"]),
            expected_manifest_sha256=str(self.fixture["manifest_sha256"]),
            expected_release_revision=RELEASE_REVISION,
            expected_role="gate_only",
        )
        limits = OpenMLEFastResourceLimits.frozen_v1()
        manager = OpenMLEFastEpisodeManager(
            dataset=dataset,
            materializer=OpenMLEFastWorkspaceMaterializer(
                Path(self.fixture["episodes_root"]),
                runner_workspace_parent=Path(self.fixture["episodes_root"]),
                workspace_bytes=2 * 1024**3,
                max_files=100_000,
                mount_backend=FakeWorkspaceMountBackend(),
            ),
            executor_factory=lambda: OpenMLEFastExecutor(
                limits=limits,
                backend=LocalCPUExecutionBackend(limits),
            ),
            grader_client=_InvalidGrader(),
            limits=limits,
            runtime_metadata={
                "runtime_source": {
                    "outer_commit": "1" * 40,
                    "inner_commit": "2" * 40,
                },
                "executor_runtime_digest": "sha256:" + "3" * 64,
            },
        )
        self.manager = manager
        self.client = TestClient(create_app(manager))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_public_route_surface_and_no_detail_or_docs(self) -> None:
        paths = {
            route.path for route in self.client.app.routes if hasattr(route, "path")
        }
        self.assertEqual(
            paths,
            {
                "/",
                "/metadata",
                "/create",
                "/reset",
                "/step",
                "/observation",
                "/horizon",
                "/close",
            },
        )
        self.assertEqual(self.client.get("/detail").status_code, 404)
        self.assertEqual(self.client.get("/docs").status_code, 404)
        metadata = self.client.get("/metadata").json()
        self.assertEqual(metadata["schema"], "openmle_fast_public_metadata_v1")
        self.assertEqual(
            metadata["task_manifest_sha256"], self.fixture["manifest_sha256"]
        )
        self.assertEqual(metadata["panel_id"], "openmle-fast-unit-gate")
        self.assertEqual(metadata["active_slot_count"], 0)
        self.assertEqual(metadata["active_environment_count"], 0)
        self.assertEqual(metadata["active_workspace_count"], 0)
        self.assertEqual(metadata["limits"]["max_policy_actions"], 30)
        for key in (
            "action",
            "observation",
            "horizon",
            "workspace",
            "executor",
            "grader_boundary",
            "cleanup",
        ):
            self.assertIn(key, metadata["contracts"])
        serialized = str(metadata).lower()
        self.assertNotIn(str(self.root).lower(), serialized)
        self.assertNotIn("credential", serialized)
        self.assertNotIn("openmle_private_canary", serialized)

    def test_create_reset_step_observe_and_close(self) -> None:
        slot = self.client.post("/create", json={}).json()["id"]
        reset = self.client.post("/reset", json={"id": slot, "data_idx": 0})
        self.assertEqual(reset.status_code, 200)
        step = self.client.post(
            "/step", json={"id": slot, "action": "malformed"}
        ).json()
        self.assertEqual(step["reward"], -0.01)
        self.assertFalse(step["done"])
        self.assertFalse(step["truncated"])
        self.assertEqual(step["info"]["source_family"], "TEST:tiny-regression")
        self.assertEqual(
            step["info"]["task_manifest_sha256"], self.fixture["manifest_sha256"]
        )
        observed = self.client.get("/observation", params={"id": slot})
        self.assertEqual(observed.json(), step["observation"])
        self.assertEqual(self.client.post("/close", json={"id": slot}).status_code, 200)

    def test_bool_index_and_extra_fields_are_rejected(self) -> None:
        slot = self.client.post("/create", json={}).json()["id"]
        response = self.client.post(
            "/reset", json={"id": slot, "data_idx": True, "extra": 1}
        )
        self.assertEqual(response.status_code, 422)

    def test_lifespan_reconciles_and_closes_owned_workspaces(self) -> None:
        workspace = self.manager.materializer.materialize(self.manager.dataset[0])
        orphan = workspace.episode_root
        with TestClient(create_app(self.manager)) as client:
            self.assertFalse(orphan.exists())
            slot = client.post("/create", json={}).json()["id"]
            client.post("/reset", json={"id": slot, "data_idx": 0})
            self.assertTrue(any(Path(self.fixture["episodes_root"]).iterdir()))
        self.assertFalse(any(Path(self.fixture["episodes_root"]).iterdir()))


if __name__ == "__main__":
    unittest.main()
