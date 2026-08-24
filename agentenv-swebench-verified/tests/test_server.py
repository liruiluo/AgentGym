from __future__ import annotations

import threading
import unittest

import requests

from agentenv_swebench_verified.environment import EpisodeStep
from agentenv_swebench_verified.server import create_http_server


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.fail_reset = False
        self.slot_capability = "test-slot-capability-token"
        self.run_capability = "r" * 43

    def metadata(self):
        return {"schema": "test", "task_count": 500}

    def create(self, *, arm, run_id, run_capability):
        self.calls.append(("create", arm, run_id, run_capability))
        if run_capability != self.run_capability:
            raise PermissionError("run authorization failed")
        return 7

    def capability(self, slot_id):
        self.calls.append(("capability", slot_id))
        return self.slot_capability

    def authorize(self, slot_id, capability, *, arm=None, run_id=None):
        self.calls.append(("authorize", slot_id, arm, run_id))
        if slot_id != 7 or capability != self.slot_capability:
            raise PermissionError("slot authorization failed")

    def reset(self, slot_id, data_idx):
        if self.fail_reset:
            raise RuntimeError("SECRET_PRIVATE_RESET_DETAIL")
        self.calls.append(("reset", slot_id, data_idx))
        return EpisodeStep("reset", 0.0, False, {"schema": "test"})

    def step(self, slot_id, action):
        self.calls.append(("step", slot_id, action))
        return EpisodeStep("stepped", 0.0, False, {"schema": "test"})

    def observation(self, slot_id):
        self.calls.append(("observation", slot_id))
        return "observed"

    def finalize_horizon(self, slot_id):
        self.calls.append(("horizon", slot_id))
        return EpisodeStep("exported", 0.0, True, {"schema": "test"})

    def prediction(self, slot_id):
        self.calls.append(("prediction", slot_id))
        return {
            "instance_id": "task-0",
            "model_name_or_path": "qwen35-4b-native",
            "model_patch": "",
        }

    def record_no_submission(self, slot_id):
        self.calls.append(("no-submission", slot_id))
        return {
            "instance_id": "task-0",
            "model_name_or_path": "qwen35-4b-native",
            "model_patch": "",
        }

    def assemble_predictions(self, *, arm, run_id):
        self.calls.append(("assemble", arm, run_id))
        return {
            "assembled": True,
            "arm": arm,
            "run_id": run_id,
            "row_count": 500,
            "sha256": "a" * 64,
        }

    def close(self, slot_id):
        self.calls.append(("close", slot_id))
        return {"closed": True, "id": slot_id}


class VerifiedHTTPServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = FakeManager()
        self.server = create_http_server(self.manager, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_full_policy_and_prediction_control_surface(self) -> None:
        self.assertEqual(
            requests.get(f"{self.base}/", timeout=5).json(),
            {"status": "ok"},
        )
        self.assertEqual(
            requests.get(f"{self.base}/metadata", timeout=5).json()["task_count"],
            500,
        )
        created = requests.post(
            f"{self.base}/create",
            json={"arm": "native", "run_id": "paired-0815"},
            headers={"Authorization": f"Bearer {self.manager.run_capability}"},
            timeout=5,
        )
        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["id"], 7)
        capability = created.json()["capability"]
        self.assertEqual(capability, self.manager.slot_capability)
        reset = requests.post(
            f"{self.base}/reset",
            json={"id": 7, "data_idx": 3},
            headers={"Authorization": f"Bearer {capability}"},
            timeout=5,
        )
        self.assertEqual(reset.json()["observation"], "reset")
        step = requests.post(
            f"{self.base}/step",
            json={
                "id": 7,
                "action": 'shell_command {"command":"pwd"}',
            },
            headers={"Authorization": f"Bearer {capability}"},
            timeout=5,
        )
        self.assertEqual(step.json()["observation"], "stepped")
        self.assertEqual(
            requests.get(
                f"{self.base}/observation",
                params={"id": 7},
                headers={"Authorization": f"Bearer {capability}"},
                timeout=5,
            ).json(),
            {"observation": "observed"},
        )
        horizon = requests.post(
            f"{self.base}/horizon",
            json={"id": 7},
            headers={"Authorization": f"Bearer {capability}"},
            timeout=5,
        ).json()
        self.assertTrue(horizon["done"])
        no_submission = requests.post(
            f"{self.base}/no-submission",
            json={"id": 7},
            headers={"Authorization": f"Bearer {capability}"},
            timeout=5,
        ).json()
        self.assertEqual(no_submission["model_patch"], "")
        prediction = requests.get(
            f"{self.base}/prediction",
            params={"id": 7},
            headers={"Authorization": f"Bearer {capability}"},
            timeout=5,
        ).json()
        self.assertEqual(prediction["model_patch"], "")
        assembled = requests.post(
            f"{self.base}/predictions/assemble",
            json={
                "id": 7,
                "arm": "native",
                "run_id": "paired-0815",
            },
            headers={"Authorization": f"Bearer {capability}"},
            timeout=5,
        ).json()
        self.assertEqual(assembled["row_count"], 500)
        closed = requests.post(
            f"{self.base}/close",
            json={"id": 7},
            headers={"Authorization": f"Bearer {capability}"},
            timeout=5,
        ).json()
        self.assertTrue(closed["closed"])
        self.assertIn(
            ("create", "native", "paired-0815", self.manager.run_capability),
            self.manager.calls,
        )
        self.assertIn(("reset", 7, 3), self.manager.calls)
        self.assertIn(
            ("step", 7, 'shell_command {"command":"pwd"}'),
            self.manager.calls,
        )

    def test_rejects_unknown_fields_routes_and_private_detail(self) -> None:
        response = requests.post(
            f"{self.base}/create",
            json={"arm": "native", "run_id": "run", "private": "SECRET"},
            headers={"Authorization": f"Bearer {self.manager.run_capability}"},
            timeout=5,
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("SECRET", response.text)
        self.assertEqual(
            requests.get(
                f"{self.base}/detail",
                params={"id": 7},
                timeout=5,
            ).status_code,
            404,
        )
        self.assertEqual(
            requests.get(f"{self.base}/unknown", timeout=5).status_code,
            404,
        )

    def test_every_slot_endpoint_requires_a_capability(self) -> None:
        requests_without_capability = (
            (
                "POST",
                "create",
                {"json": {"arm": "native", "run_id": "paired-0815"}},
            ),
            ("GET", "observation", {"params": {"id": 7}}),
            ("GET", "prediction", {"params": {"id": 7}}),
            ("POST", "reset", {"json": {"id": 7, "data_idx": 0}}),
            ("POST", "step", {"json": {"id": 7, "action": "final"}}),
            ("POST", "horizon", {"json": {"id": 7}}),
            ("POST", "no-submission", {"json": {"id": 7}}),
            (
                "POST",
                "predictions/assemble",
                {
                    "json": {
                        "id": 7,
                        "arm": "native",
                        "run_id": "paired-0815",
                    }
                },
            ),
            ("POST", "close", {"json": {"id": 7}}),
        )
        for method, path, kwargs in requests_without_capability:
            with self.subTest(method=method, path=path):
                response = requests.request(
                    method,
                    f"{self.base}/{path}",
                    timeout=5,
                    **kwargs,
                )
                self.assertEqual(response.status_code, 403)

    def test_fail_closed_errors_do_not_echo_private_runtime_details(self) -> None:
        self.manager.fail_reset = True
        response = requests.post(
            f"{self.base}/reset",
            json={
                "id": 7,
                "data_idx": 0,
            },
            headers={
                "Authorization": f"Bearer {self.manager.slot_capability}"
            },
            timeout=5,
        )
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("SECRET_PRIVATE_RESET_DETAIL", response.text)
        self.assertIn("request failed closed", response.text)


if __name__ == "__main__":
    unittest.main()
