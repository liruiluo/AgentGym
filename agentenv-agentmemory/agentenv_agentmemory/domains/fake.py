from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..runtime.domain import DomainContract, DomainTransition


_ADVANCE_RE = re.compile(r"\AADVANCE\s+(\{.*\})\Z", flags=re.DOTALL)


@dataclass
class FakeTwoPhaseFactory:
    task_count: int = 2
    domain_id: str = "fake_two_phase"
    surface: str = "agentmemory_fake_two_phase_v3"
    contract: DomainContract = DomainContract(
        contract_id="fake_two_phase_v1",
        system_prompt=(
            "Operate a deterministic two-phase test environment. Use exactly one "
            "ADVANCE JSON action or one AgentMemoryGym memory action."
        ),
        native_action_descriptions=('ADVANCE {"value": "..."}',),
        max_steps=4,
    )

    def create(self, env_uid: str):
        return FakeTwoPhaseDriver(env_uid=env_uid, contract=self.contract)

    def metadata(self) -> dict[str, Any]:
        return {"source": "agentmemory_test_fixture"}


class FakeTwoPhaseDriver:
    domain_id = "fake_two_phase"
    surface = "agentmemory_fake_two_phase_v3"

    def __init__(self, *, env_uid: str, contract: DomainContract) -> None:
        self.env_uid = env_uid
        self.contract = contract
        self.data_idx = 0
        self.phase_index = 0
        self.closed = False

    def reset(self, data_idx: int) -> DomainTransition:
        self.data_idx = int(data_idx)
        self.phase_index = 0
        self.closed = False
        return self._transition("Phase 1 requires ADVANCE.")

    def step(self, action: str, env_step: int) -> DomainTransition:
        match = _ADVANCE_RE.fullmatch(action.strip())
        if match is None:
            return self._transition(
                "Invalid fake-domain action.",
                reward=-0.01,
                action_execution={
                    "op": "INVALID",
                    "status": "invalid",
                    "step": env_step,
                },
                reward_components=(
                    {
                        "name": "invalid_action",
                        "value": -0.01,
                        "op": "INVALID",
                        "step": env_step,
                    },
                ),
            )
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict) or set(payload) != {"value"}:
            return self.step("INVALID", env_step)
        self.phase_index += 1
        done = self.phase_index == 2
        reward = 1.0
        return self._transition(
            "Episode complete." if done else "Phase 2 requires ADVANCE.",
            reward=reward,
            done=done,
            status="success" if done else "active",
            episode_success=done,
            action_execution={
                "op": "ADVANCE",
                "status": "executed",
                "step": env_step,
                "value": payload["value"],
            },
            tool_ops=(
                {
                    "op": "ADVANCE",
                    "step": env_step,
                    "value": payload["value"],
                },
            ),
            reward_components=(
                {
                    "name": "phase_advance",
                    "value": reward,
                    "op": "ADVANCE",
                    "step": env_step,
                },
            ),
            domain_evidence={"data_idx": self.data_idx},
        )

    def close(self) -> None:
        self.closed = True

    def _transition(
        self,
        observation: str,
        *,
        reward: float = 0.0,
        done: bool = False,
        status: str = "active",
        episode_success: bool = False,
        action_execution=None,
        tool_ops=(),
        reward_components=(),
        domain_evidence=None,
    ) -> DomainTransition:
        return DomainTransition(
            observation=(
                f"Fake task {self.data_idx}. Progress: {self.phase_index}/2. "
                f"{observation}"
            ),
            reward=reward,
            done=done,
            status=status,
            phase_index=self.phase_index,
            phase_count=2,
            episode_success=episode_success,
            action_execution=action_execution or {},
            tool_ops=tool_ops,
            reward_components=reward_components,
            domain_evidence=domain_evidence or {"data_idx": self.data_idx},
        )
