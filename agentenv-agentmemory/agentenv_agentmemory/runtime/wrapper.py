from __future__ import annotations

import re
import threading
from copy import deepcopy
from typing import Any

from .domain import DomainContract, DomainDriver, DomainFactory, DomainTransition
from .memory import MemoryActionError, MemoryRewardPolicy, MemoryToolRuntime


_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_ACTION_ENVELOPE_RE = re.compile(r"(?m)^[ \t]*Action:[ \t]*(?:\r?\n[ \t]*)?")


class MemoryAugmentedDriver:
    """Adds policy memory to a domain without changing domain outcome semantics."""

    def __init__(
        self,
        driver: DomainDriver,
        *,
        reward_policy: MemoryRewardPolicy | None = None,
        invalid_action_penalty: float = 0.0,
    ) -> None:
        self.driver = driver
        self.domain_id = driver.domain_id
        self.surface = driver.surface
        self.contract = driver.contract
        self.memory = MemoryToolRuntime(reward_policy or MemoryRewardPolicy())
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.env_step = 0
        self.transition: DomainTransition | None = None

    def reset(self, data_idx: int) -> DomainTransition:
        self.env_step = 0
        self.memory.reset_episode()
        self.transition = self.driver.reset(data_idx)
        return self._decorate(self.transition)

    def step(self, raw_policy_output: str, env_step: int | None = None) -> DomainTransition:
        if self.transition is None:
            raise RuntimeError("driver must be reset before step")
        if self.transition.done:
            return self.transition
        self.env_step = int(env_step) if env_step is not None else self.env_step + 1
        submitted_action = extract_submitted_action(raw_policy_output)
        before_phase = self.transition.phase_index
        try:
            memory_result = self.memory.apply(
                submitted_action,
                env_step=self.env_step,
                phase_index=before_phase,
            )
        except MemoryActionError as exc:
            transition = self._invalid_transition(
                raw_policy_output=raw_policy_output,
                submitted_action=submitted_action,
                message=str(exc),
            )
            transition = self._enforce_max_steps(transition)
            self.transition = transition
            return self._decorate(transition, prefix=f"Invalid action: {exc}")

        if memory_result is not None:
            execution = {
                "raw_policy_output": raw_policy_output,
                "submitted_action": submitted_action,
                "op": memory_result.op,
                "status": "executed",
                "step": self.env_step,
            }
            transition = DomainTransition(
                observation=self.transition.observation,
                reward=memory_result.reward,
                done=False,
                status=self.transition.status,
                phase_index=before_phase,
                phase_count=self.transition.phase_count,
                episode_success=False,
                action_execution=execution,
                tool_ops=(memory_result.tool_op,),
                reward_components=memory_result.reward_components,
                domain_evidence=self._inherited_domain_evidence(
                    memory_state_diff=memory_result.state_diff,
                ),
            )
            transition = self._enforce_max_steps(transition)
            self.transition = transition
            return self._decorate(transition, prefix=memory_result.message)

        transition = self.driver.step(submitted_action, self.env_step)
        execution = dict(transition.action_execution)
        execution.update(
            {
                "raw_policy_output": raw_policy_output,
                "submitted_action": submitted_action,
                "step": self.env_step,
            }
        )
        components = [dict(item) for item in transition.reward_components]
        reward = float(transition.reward)
        action_status = str(execution.get("status", "")).lower()
        action_op = str(execution.get("op", "")).upper()
        if action_status == "invalid" or action_op == "INVALID":
            reward, components = self._apply_native_invalid_penalty(
                reward,
                components,
            )
        valid_zero_reward = (
            not transition.done
            and reward == 0.0
            and action_status not in {"invalid", "error", "rejected"}
        )
        if valid_zero_reward:
            op = str(execution.get("op", "DOMAIN_ACTION"))
            shaping_reward, shaping = self.memory.shape_valid_zero_reward_action(
                submitted_action,
                op=op,
                env_step=self.env_step,
                phase_index=before_phase,
            )
            reward += shaping_reward
            if shaping is not None:
                components.append(shaping)
        phase_advanced = transition.phase_index > before_phase
        if phase_advanced:
            self.memory.advance_phase()
        else:
            self.memory.append_trace(submitted_action, transition.observation)
        transition = DomainTransition(
            observation=transition.observation,
            reward=reward,
            done=transition.done,
            status=transition.status,
            phase_index=transition.phase_index,
            phase_count=transition.phase_count,
            episode_success=transition.episode_success,
            action_execution=execution,
            tool_ops=transition.tool_ops,
            reward_components=tuple(components),
            domain_evidence={
                **transition.domain_evidence,
                "phase_index_before": before_phase,
                "phase_advanced": phase_advanced,
            },
            sample_excluded=transition.sample_excluded,
        )
        transition = self._enforce_max_steps(transition)
        self.transition = transition
        return self._decorate(transition)

    def close(self) -> None:
        self.driver.close()

    def _invalid_transition(
        self,
        *,
        raw_policy_output: str,
        submitted_action: str,
        message: str,
    ) -> DomainTransition:
        assert self.transition is not None
        component = {
            "name": "invalid_action",
            "value": self.invalid_action_penalty,
            "op": "INVALID",
            "step": self.env_step,
            "error": message,
        }
        return DomainTransition(
            observation=self.transition.observation,
            reward=self.invalid_action_penalty,
            done=False,
            status=self.transition.status,
            phase_index=self.transition.phase_index,
            phase_count=self.transition.phase_count,
            episode_success=False,
            action_execution={
                "raw_policy_output": raw_policy_output,
                "submitted_action": submitted_action,
                "op": "INVALID",
                "status": "invalid",
                "step": self.env_step,
            },
            reward_components=(component,),
            domain_evidence=self._inherited_domain_evidence(
                memory_action_error=message,
            ),
        )

    def _inherited_domain_evidence(
        self,
        *,
        memory_state_diff: dict[str, list[Any]] | None = None,
        memory_action_error: str | None = None,
    ) -> dict[str, Any]:
        assert self.transition is not None
        evidence = deepcopy(self.transition.domain_evidence)
        evidence.pop("memory_state_diff", None)
        evidence.pop("memory_action_error", None)
        evidence.pop("transition_event", None)
        evidence.pop("paper_evaluation", None)
        if "active_round_index" in evidence:
            evidence["round_index"] = evidence["active_round_index"]
        evidence.update(
            {
                "phase_index_before": self.transition.phase_index,
                "phase_advanced": False,
                "memory_state_diff": memory_state_diff
                or {"added": [], "updated": [], "deleted": []},
            }
        )
        if memory_action_error is not None:
            evidence["memory_action_error"] = memory_action_error
        return evidence

    def _apply_native_invalid_penalty(
        self,
        reward: float,
        components: list[dict[str, Any]],
    ) -> tuple[float, list[dict[str, Any]]]:
        if self.invalid_action_penalty == 0.0:
            return reward, components

        replaced = []
        inserted = False
        for component in components:
            is_invalid = (
                str(component.get("name", "")) == "invalid_action"
                or str(component.get("op", "")).upper() == "INVALID"
            )
            if not is_invalid:
                replaced.append(component)
                continue
            if inserted:
                continue
            canonical = dict(component)
            canonical.update(
                {
                    "name": "invalid_action",
                    "value": self.invalid_action_penalty,
                    "op": "INVALID",
                    "step": self.env_step,
                }
            )
            replaced.append(canonical)
            inserted = True
        if not inserted:
            replaced.append(
                {
                    "name": "invalid_action",
                    "value": self.invalid_action_penalty,
                    "op": "INVALID",
                    "step": self.env_step,
                }
            )
        return sum(float(item.get("value", 0.0)) for item in replaced), replaced

    def _enforce_max_steps(self, transition: DomainTransition) -> DomainTransition:
        if transition.done or self.env_step < self.contract.max_steps:
            return transition

        execution = deepcopy(transition.action_execution)
        execution.update(
            {
                "max_steps_exhausted": True,
                "max_steps_limit": self.contract.max_steps,
            }
        )
        evidence = deepcopy(transition.domain_evidence)
        evidence.update(
            {
                "max_steps_exhausted": True,
                "max_steps_limit": self.contract.max_steps,
            }
        )
        return DomainTransition(
            observation=(
                transition.observation.rstrip()
                + "\n\nMaximum episode action count reached."
            ),
            reward=transition.reward,
            done=True,
            status="max_steps_exhausted",
            phase_index=transition.phase_index,
            phase_count=transition.phase_count,
            episode_success=False,
            action_execution=execution,
            tool_ops=tuple(deepcopy(item) for item in transition.tool_ops),
            reward_components=tuple(
                deepcopy(item) for item in transition.reward_components
            ),
            domain_evidence=evidence,
            sample_excluded=False,
        )

    def _decorate(
        self,
        transition: DomainTransition,
        *,
        prefix: str | None = None,
    ) -> DomainTransition:
        sections = []
        if prefix:
            sections.append(prefix.strip())
        sections.extend(
            [
                transition.observation.strip(),
                self.memory.render_context(),
            ]
        )
        evidence = dict(transition.domain_evidence)
        evidence.update(
            {
                "memory_inventory_count": len(self.memory.long_term_memory),
                "memory_state_diff": evidence.get(
                    "memory_state_diff",
                    {"added": [], "updated": [], "deleted": []},
                ),
                "session_trace": list(self.memory.session_trace),
            }
        )
        return DomainTransition(
            observation="\n\n".join(section for section in sections if section),
            reward=transition.reward,
            done=transition.done,
            status=transition.status,
            phase_index=transition.phase_index,
            phase_count=transition.phase_count,
            episode_success=transition.episode_success,
            action_execution=deepcopy(transition.action_execution),
            tool_ops=tuple(deepcopy(item) for item in transition.tool_ops),
            reward_components=tuple(
                deepcopy(item) for item in transition.reward_components
            ),
            domain_evidence=evidence,
            sample_excluded=transition.sample_excluded,
        )


class DomainEnvWrapper:
    """Thread-safe HTTP-facing owner for v3 domain drivers."""

    def __init__(
        self,
        factory: DomainFactory,
        *,
        reward_policy: MemoryRewardPolicy | None = None,
        invalid_action_penalty: float = 0.0,
    ) -> None:
        if factory.task_count < 1:
            raise ValueError("domain factory must expose at least one task")
        self.factory = factory
        self.reward_policy = reward_policy or MemoryRewardPolicy()
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.max_id = 0
        self.envs: dict[int, MemoryAugmentedDriver] = {}
        self.info: dict[int, dict[str, Any]] = {}
        self.env_locks: dict[int, threading.RLock] = {}
        self.lock = threading.RLock()

    def create(self) -> dict[str, Any]:
        with self.lock:
            env_id = self.max_id
            self.max_id += 1
            env = MemoryAugmentedDriver(
                self.factory.create(f"env{env_id}"),
                reward_policy=self.reward_policy,
                invalid_action_penalty=self.invalid_action_penalty,
            )
            transition = env.reset(env_id % self.factory.task_count)
            payload = self._payload(env_id, transition)
            self.envs[env_id] = env
            self.info[env_id] = payload
            self.env_locks[env_id] = threading.RLock()
            return payload

    def step(self, env_id: int, action: str) -> dict[str, Any]:
        env = self.require_env(env_id)
        with self.require_lock(env_id):
            transition = env.step(action)
            payload = self._payload(env_id, transition, include_id=False)
            self.info[env_id] = payload
            return payload

    def reset(self, env_id: int, data_idx: int = 0) -> dict[str, Any]:
        env = self.require_env(env_id)
        with self.require_lock(env_id):
            transition = env.reset(data_idx)
            payload = self._payload(env_id, transition)
            self.info[env_id] = payload
            return payload

    def observation(self, env_id: int) -> str:
        self.require_env(env_id)
        with self.require_lock(env_id):
            return self.info[env_id]["observation"]

    def detail(self, env_id: int) -> dict[str, Any]:
        self.require_env(env_id)
        with self.require_lock(env_id):
            return deepcopy(self.info[env_id])

    def close(self, env_id: int) -> bool:
        env = self.require_env(env_id)
        with self.require_lock(env_id):
            env.close()
        with self.lock:
            del self.envs[env_id]
            del self.info[env_id]
            del self.env_locks[env_id]
        return True

    def metadata(self) -> dict[str, Any]:
        contract = self.factory.contract
        metadata = dict(self.factory.metadata())
        metadata.update(
            {
                "formal_schema_version": "agentmemory_formal_step_v3",
                "domain_id": self.factory.domain_id,
                "surface": self.factory.surface,
                "task_count": self.factory.task_count,
                "contract_id": contract.contract_id,
                "contract_sha256": contract.sha256,
                "system_prompt": contract.canonical_system_prompt,
                "system_prompt_sha256": contract.system_prompt_sha256,
                "native_action_descriptions": list(
                    contract.native_action_descriptions
                ),
                "max_steps": contract.max_steps,
                "memory_reward_policy": {
                    "first_add": self.reward_policy.first_add,
                    "first_later_phase_retrieve": (
                        self.reward_policy.first_later_phase_retrieve
                    ),
                    "exact_repeat": self.reward_policy.exact_repeat,
                    "invalid_action": self.invalid_action_penalty,
                },
                "reward_overlay": (
                    "agentmemory_policy_memory_shaping_v1"
                    if any(
                        value != 0.0
                        for value in (
                            self.reward_policy.first_add,
                            self.reward_policy.first_later_phase_retrieve,
                            self.reward_policy.exact_repeat,
                            self.invalid_action_penalty,
                        )
                    )
                    else "none"
                ),
            }
        )
        return metadata

    def require_env(self, env_id: int) -> MemoryAugmentedDriver:
        try:
            return self.envs[env_id]
        except KeyError as exc:
            raise KeyError(f"Unknown environment id {env_id}") from exc

    def require_lock(self, env_id: int) -> threading.RLock:
        try:
            return self.env_locks[env_id]
        except KeyError as exc:
            raise KeyError(f"Unknown environment id {env_id}") from exc

    def _payload(
        self,
        env_id: int,
        transition: DomainTransition,
        *,
        include_id: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "observation": transition.observation,
            "reward": float(transition.reward),
            "done": bool(transition.done),
            "info": transition.to_info(
                domain_id=self.factory.domain_id,
                surface=self.factory.surface,
                contract=self.factory.contract,
            ),
        }
        if include_id:
            payload["id"] = env_id
        return payload


def extract_submitted_action(raw_policy_output: str) -> str:
    if not isinstance(raw_policy_output, str):
        return repr(raw_policy_output)
    text = raw_policy_output.strip()
    matches = list(_THINK_CLOSE_RE.finditer(text))
    if matches:
        text = text[matches[-1].end() :].strip()
    action_envelopes = list(_ACTION_ENVELOPE_RE.finditer(text))
    if action_envelopes:
        text = text[action_envelopes[0].end() :].strip()
    return text
