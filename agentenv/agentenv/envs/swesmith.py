from __future__ import annotations

from typing import Any, Mapping

import requests

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class SwesmithEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": (
                    "You are a coding agent working on one persistent repository. "
                    "Inspect, edit, and test the workspace until the issue is fixed. "
                    "Use the exact shell_command or apply_patch grammar shown in the "
                    "task observation. A normal text response submits the workspace."
                ),
            }
        ),
        ConversationMessage(
            {"from": "gpt", "loss": False, "value": "Understood."}
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int | None = None,
        *args,
        timeout: int = 900,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base.rstrip("/")
        self.timeout = timeout
        metadata = self._request("GET", "metadata")
        task_count = int(metadata["task_count"])
        if data_len is not None and int(data_len) > task_count:
            raise ValueError(
                f"SWE-smith data_len {data_len} exceeds server task_count {task_count}"
            )
        self.data_len = task_count if data_len is None else int(data_len)
        created = self._request("POST", "create", json={})
        self.env_id = int(created["id"])
        self.info = created
        self.metadata = metadata

    def __len__(self) -> int:
        return self.data_len

    def observe(self) -> str:
        return str(self.info["observation"])

    def step(self, action: str) -> StepOutput:
        response = self._request(
            "POST",
            "step",
            json={"id": self.env_id, "action": action},
        )
        self.info = response
        return StepOutput(
            state=str(response["observation"]),
            reward=float(response["reward"]),
            done=bool(response["done"]),
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._request(
            "POST",
            "reset",
            json={"id": self.env_id, "data_idx": idx},
        )
        self.info = response
        return response

    def detail(self, *, private_token: str | None = None) -> dict[str, Any]:
        headers = (
            {} if private_token is None else {"X-SWESMITH-Detail-Token": private_token}
        )
        return self._request(
            "GET", "detail", params={"id": self.env_id}, headers=headers
        )

    def finalize_horizon(self) -> StepOutput:
        response = self._request("POST", "horizon", json={"id": self.env_id})
        self.info = response
        return StepOutput(
            state=str(response["observation"]),
            reward=float(response["reward"]),
            done=bool(response["done"]),
        )

    def close(self) -> dict[str, Any]:
        return self._request("POST", "close", json={"id": self.env_id})

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self.env_server_base}/{path}",
            timeout=self.timeout,
            **kwargs,
        )
        if response.status_code != 200:
            raise requests.RequestException(
                f"SWE-smith {method} /{path} failed: "
                f"status={response.status_code} body={response.text[-1000:]}"
            )
        value = response.json()
        if not isinstance(value, dict):
            raise requests.RequestException(
                f"SWE-smith {method} /{path} returned a non-object response"
            )
        return value


class SwesmithTask(BaseTask):
    env_client_cls = SwesmithEnvClient
    env_name = "SWE-smith"

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
