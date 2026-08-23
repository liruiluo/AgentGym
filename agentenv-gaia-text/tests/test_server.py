from __future__ import annotations

import json
from pathlib import Path

from agentenv_gaia_text.backend import FixtureBackend
from agentenv_gaia_text.contracts import EvaluationArm, ProtocolContract
from agentenv_gaia_text.dataset import GaiaTextDataset
from agentenv_gaia_text.server import create_app
from agentenv_gaia_text.submission import SubmissionStore
from agentenv_gaia_text.wrapper import GaiaTextEpisodeManager
from fastapi.testclient import TestClient
from support import protocol_kwargs, write_runtime_fixture


def test_http_service_exposes_only_public_lifecycle_routes(tmp_path: Path) -> None:
    runtime = write_runtime_fixture(tmp_path)
    dataset = GaiaTextDataset.load(
        runtime.manifest,
        runtime.questions,
        expected_questions_sha256=runtime.questions_sha256,
        contract=ProtocolContract(**protocol_kwargs(runtime.rows)),
    )
    manager = GaiaTextEpisodeManager(
        dataset,
        FixtureBackend.load(runtime.backend, runtime.backend_sha256),
        SubmissionStore(dataset.task_ids, runtime.predictions),
        arm=EvaluationArm.NATIVE,
    )
    app = create_app(manager)
    routes = {route.path for route in app.routes}
    assert routes == {
        "/",
        "/metadata",
        "/create",
        "/reset",
        "/step",
        "/horizon",
        "/close",
        "/abort",
    }
    assert not any(
        token in route.casefold()
        for route in routes
        for token in ("detail", "gold", "scorer", "answer")
    )

    with TestClient(app) as client:
        metadata = client.get("/metadata")
        assert metadata.status_code == 200
        assert metadata.json()["arm"] == "native"
        created = client.post("/create", json={}).json()
        assert created["info"]["status"] == "unbound"
        reset = client.post("/reset", json={"id": created["id"], "data_idx": 0})
        assert reset.status_code == 200
        assert set(json.loads(reset.json()["observation"])) == {
            "task_id",
            "level",
            "question",
        }
        search = client.post(
            "/step",
            json={
                "id": created["id"],
                "action": '<tool_call>{"name":"search","arguments":{"query":"alpha"}}</tool_call>',
            },
        )
        assert search.status_code == 200
        terminal = client.post("/horizon", json={"id": created["id"]})
        assert terminal.status_code == 200
        assert terminal.json()["done"] is True
        assert client.post("/close", json={"id": created["id"]}).json() is True
        assert client.get("/detail", params={"id": created["id"]}).status_code == 404

        unfinished = client.post("/create", json={}).json()
        assert (
            client.post(
                "/reset", json={"id": unfinished["id"], "data_idx": 0}
            ).status_code
            == 200
        )
        assert client.post("/abort", json={"id": unfinished["id"]}).json() is True
        replay = client.post("/create", json={}).json()
        assert (
            client.post("/reset", json={"id": replay["id"], "data_idx": 0}).status_code
            == 200
        )
        assert client.post("/abort", json={"id": replay["id"]}).json() is True

        coerced = client.post("/reset", json={"id": True, "data_idx": "0"})
        assert coerced.status_code == 422


def test_http_errors_fail_closed_without_internal_paths(tmp_path: Path) -> None:
    runtime = write_runtime_fixture(tmp_path)
    dataset = GaiaTextDataset.load(
        runtime.manifest,
        runtime.questions,
        expected_questions_sha256=runtime.questions_sha256,
        contract=ProtocolContract(**protocol_kwargs(runtime.rows)),
    )
    app = create_app(
        GaiaTextEpisodeManager(
            dataset,
            FixtureBackend.load(runtime.backend, runtime.backend_sha256),
            SubmissionStore(dataset.task_ids, runtime.predictions),
            arm=EvaluationArm.NATIVE,
        )
    )
    with TestClient(app) as client:
        response = client.post("/step", json={"id": 999, "action": "x"})
        assert response.status_code == 404
        assert str(tmp_path) not in response.text
