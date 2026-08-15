from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from agentenv_mlebench_lite.dataset import load_lite_dataset
from agentenv_mlebench_lite.environment import MLEBenchLiteEpisodeManager
from agentenv_mlebench_lite.executor import SandboxExecutor
from agentenv_mlebench_lite.identity import UPSTREAM_COMMIT, load_official_lite_identity
from agentenv_mlebench_lite.server import create_app
from agentenv_mlebench_lite.workspace import WorkspaceManager
from fastapi.testclient import TestClient

from tests.support import (
    FAKE_RUNNER_SHA256,
    FAKE_RUNTIME_DIGEST,
    RecordingFormalBackend,
    write_fixture,
)


class MLEBenchLiteServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mlebench-lite-server-")
        self.root = Path(self.temporary.name)
        fixture = write_fixture(self.root)
        self.fixture = fixture
        identity = load_official_lite_identity(
            fixture["upstream_root"],
            commit_resolver=lambda _root: UPSTREAM_COMMIT,
        )
        dataset = load_lite_dataset(
            identity=identity,
            manifest_path=fixture["manifest_path"],
            expected_manifest_sha256=fixture["manifest_sha256"],
            data_root=fixture["data_root"],
        )
        self.dataset = dataset

        def executor_factory():
            return SandboxExecutor(
                RecordingFormalBackend(),
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            )

        manager = MLEBenchLiteEpisodeManager(
            dataset=dataset,
            workspace_manager=WorkspaceManager(fixture["episodes_root"]),
            executor_factory=executor_factory,
            runner_sha256=FAKE_RUNNER_SHA256,
            runtime_digest=FAKE_RUNTIME_DIGEST,
            max_actions=3,
        )
        self.manager = manager
        self.app = create_app(manager)
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.manager.close_all()
        self.temporary.cleanup()

    def create_slot(self, mode="native"):
        created = self.client.post("/create", json={"mode": mode})
        self.assertEqual(created.status_code, 200)
        value = created.json()
        self.assertEqual(set(value), {"id", "capability_token"})
        self.assertRegex(value["capability_token"], r"^[0-9a-f]{64}$")
        return value["id"], value["capability_token"]

    def reset_slot(self, slot, token, data_idx=0):
        return self.client.post(
            "/reset",
            json={"id": slot, "capability_token": token, "data_idx": data_idx},
        )

    def test_metadata_and_request_bodies_are_strict(self) -> None:
        metadata = self.client.get("/metadata")
        self.assertEqual(metadata.status_code, 200)
        value = metadata.json()
        self.assertEqual(value["schema"], "mlebench_lite_metadata_v2")
        self.assertEqual(value["task_count"], 22)
        self.assertEqual(value["runner_sha256"], FAKE_RUNNER_SHA256)
        self.assertEqual(value["runtime_digest"], FAKE_RUNTIME_DIGEST)
        self.assertEqual(
            value["modes"],
            ["native", "amg_compaction_only", "amg_memory"],
        )
        self.assertEqual(value["resource_contract"]["max_actions"], 3)
        self.assertEqual(
            value["resource_contract"]["max_step_response_ms"],
            value["resource_contract"]["episode_timeout_ms"] + 30_000,
        )

        self.assertEqual(
            self.client.post("/create", json={"mode": "invalid"}).status_code,
            422,
        )
        compact_only, token = self.create_slot("amg_compaction_only")
        self.assertEqual(self.reset_slot(compact_only, token).status_code, 200)
        self.assertEqual(
            self.client.post(
                "/create", json={"mode": "native", "extra": True}
            ).status_code,
            422,
        )

    def test_mode_is_frozen_unknown_slots_and_stale_sequences_are_rejected(
        self,
    ) -> None:
        slot, token = self.create_slot("native")
        self.assertEqual(
            self.client.post(
                "/reset",
                json={
                    "id": slot,
                    "capability_token": token,
                    "data_idx": 0,
                    "mode": "amg_memory",
                },
            ).status_code,
            422,
        )
        reset = self.reset_slot(slot, token)
        self.assertEqual(reset.status_code, 200)

        rejected_control = self.client.post(
            "/step",
            json={
                "id": slot,
                "capability_token": token,
                "action_id": str(uuid.uuid4()),
                "action": "handoff",
                "control": "compaction",
                "expected_action_count": 0,
            },
        )
        self.assertEqual(rejected_control.status_code, 200)
        self.assertEqual(
            rejected_control.json()["observation"], "Action is unavailable."
        )
        self.assertEqual(rejected_control.json()["info"]["counters"]["action_count"], 1)

        stale = self.client.post(
            "/step",
            json={
                "id": slot,
                "capability_token": token,
                "action_id": str(uuid.uuid4()),
                "action": "submit",
                "expected_action_count": 0,
            },
        )
        self.assertEqual(stale.status_code, 409)
        missing = self.client.post(
            "/reset",
            json={"id": 999, "capability_token": token, "data_idx": 0},
        )
        self.assertEqual(missing.status_code, 404)

        wrong_capability = self.client.post(
            "/reset",
            json={"id": slot, "capability_token": "0" * 64, "data_idx": 0},
        )
        self.assertEqual(wrong_capability.status_code, 404)

    def test_action_id_replays_exact_result_and_rejects_payload_change(self) -> None:
        slot, token = self.create_slot("native")
        self.assertEqual(self.reset_slot(slot, token).status_code, 200)
        action_id = str(uuid.uuid4())
        payload = {
            "id": slot,
            "capability_token": token,
            "action_id": action_id,
            "action": 'inspect {"path":"/home/data/train.csv"}',
            "expected_action_count": 0,
        }
        first = self.client.post("/step", json=payload)
        replay = self.client.post("/step", json=payload)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(first.json()["info"]["counters"]["action_count"], 1)

        changed = self.client.post("/step", json={**payload, "action": "submit"})
        self.assertEqual(changed.status_code, 409)

    def test_close_endpoint_and_app_lifespan_reap_active_slots(self) -> None:
        slot, token = self.create_slot("amg_memory")
        self.assertEqual(
            self.reset_slot(slot, token).status_code,
            200,
        )
        workspace = self.manager._testing_workspace(slot)
        closed = self.client.post(
            "/close", json={"id": slot, "capability_token": token}
        )
        self.assertEqual(closed.status_code, 200)
        self.assertFalse(workspace.episode_root.exists())

        with TestClient(self.app) as lifespan_client:
            created = lifespan_client.post("/create", json={"mode": "native"}).json()
            second = created["id"]
            lifespan_client.post(
                "/reset",
                json={
                    "id": second,
                    "capability_token": created["capability_token"],
                    "data_idx": 1,
                },
            )
            second_workspace = self.manager._testing_workspace(second)
        self.assertFalse(second_workspace.episode_root.exists())
        with self.assertRaises(KeyError):
            self.manager.reset(second, 1)

    def test_close_cleanup_failure_returns_503_and_retries_same_teardown(self) -> None:
        class RetryTeardownBackend(RecordingFormalBackend):
            def __init__(self):
                super().__init__()
                self.operation_ids: list[str] = []

            def teardown(self, *, workspace, operation_id):
                self.operation_ids.append(operation_id)
                if len(self.operation_ids) == 1:
                    raise OSError("injected teardown transport failure")
                return super().teardown(
                    workspace=workspace,
                    operation_id=operation_id,
                )

        backend = RetryTeardownBackend()
        manager = MLEBenchLiteEpisodeManager(
            dataset=self.dataset,
            workspace_manager=WorkspaceManager(
                self.root / "retry-close-episodes",
                self.root / "retry-close-handoffs",
            ),
            executor_factory=lambda: SandboxExecutor(
                backend,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            ),
            runner_sha256=FAKE_RUNNER_SHA256,
            runtime_digest=FAKE_RUNTIME_DIGEST,
        )
        client = TestClient(create_app(manager), raise_server_exceptions=False)
        created = client.post("/create", json={"mode": "native"}).json()
        self.assertEqual(
            client.post(
                "/reset",
                json={
                    "id": created["id"],
                    "capability_token": created["capability_token"],
                    "data_idx": 0,
                },
            ).status_code,
            200,
        )
        workspace = manager._testing_workspace(created["id"])
        payload = {
            "id": created["id"],
            "capability_token": created["capability_token"],
        }
        first = client.post("/close", json=payload)
        self.assertEqual(first.status_code, 503)
        self.assertTrue(workspace.episode_root.exists())
        second = client.post("/close", json=payload)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(len(set(backend.operation_ids)), 1)
        self.assertFalse(workspace.episode_root.exists())
        self.assertEqual(list(manager.workspace_manager.handoff_root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
