from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from fastapi import HTTPException
    from agentenv_agentmemory.model import (
        WorkspaceExportRequestBody,
        WorkspaceInterventionRequestBody,
    )
except ModuleNotFoundError:  # Mac's lightweight unit-test environment.
    HTTPException = None
    WorkspaceExportRequestBody = None
    WorkspaceInterventionRequestBody = None


SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "agentenv_agentmemory"
    / "server.py"
)


def load_server_module(fake_server):
    module_name = "agentenv_agentmemory._workspace_intervention_endpoint_test"
    spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch(
        "agentenv_agentmemory.runtime.server_factory.build_server",
        return_value=fake_server,
    ):
        spec.loader.exec_module(module)
    return module


@unittest.skipIf(HTTPException is None, "fastapi is not installed")
class WorkspaceInterventionEndpointTest(unittest.TestCase):
    def test_endpoint_forwards_authenticated_control_request(self) -> None:
        control = Mock(return_value={"id": 7, "reward": 0.0, "done": False})
        module = load_server_module(
            type("FakeServer", (), {"workspace_intervention": control})()
        )
        body = WorkspaceInterventionRequestBody(
            id=7,
            arm="swapped",
            source_env_id=9,
        )
        result = module.workspace_intervention(body, token="secret-token")
        self.assertEqual(result["id"], 7)
        control.assert_called_once_with(
            7,
            arm="swapped",
            source_env_id=9,
            token="secret-token",
        )
        route_paths = {route.path for route in module.app.routes}
        self.assertIn("/workspace-intervention", route_paths)

    def test_export_endpoint_uses_the_same_private_control_token(self) -> None:
        export = Mock(
            return_value={
                "schema": "agentmemory_workspace_authenticated_export_v1",
                "id": 7,
                "workspace_state": {},
            }
        )
        module = load_server_module(
            type("FakeServer", (), {"workspace_export": export})()
        )
        result = module.workspace_export(
            WorkspaceExportRequestBody(id=7),
            token="secret-token",
        )
        self.assertEqual(result["id"], 7)
        export.assert_called_once_with(7, token="secret-token")
        route_paths = {route.path for route in module.app.routes}
        self.assertIn("/workspace-export", route_paths)

    def test_endpoint_maps_auth_and_contract_failures(self) -> None:
        cases = (
            (PermissionError("bad token"), 403),
            (ValueError("bad source"), 400),
            (RuntimeError("bad boundary"), 400),
            (KeyError("missing env"), 400),
        )
        for error, expected_status in cases:
            with self.subTest(error=type(error).__name__):
                control = Mock(side_effect=error)
                module = load_server_module(
                    type("FakeServer", (), {"workspace_intervention": control})()
                )
                body = WorkspaceInterventionRequestBody(id=1, arm="blank")
                with self.assertRaises(HTTPException) as caught:
                    module.workspace_intervention(body, token="secret-token")
                self.assertEqual(caught.exception.status_code, expected_status)

    def test_endpoint_is_unavailable_without_control_surface(self) -> None:
        module = load_server_module(object())
        body = WorkspaceInterventionRequestBody(id=1, arm="blank")
        with self.assertRaises(HTTPException) as caught:
            module.workspace_intervention(body, token="secret-token")
        self.assertEqual(caught.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
