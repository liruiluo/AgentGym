from __future__ import annotations

import os
from pathlib import Path

from ..domains.travel import TRAVEL_SURFACE, TravelPlannerFactory
from ..env_wrapper import AgentMemoryWrapper, NATIVE_SURFACE
from .memory import MemoryRewardPolicy
from .wrapper import DomainEnvWrapper


def build_server():
    surface = os.environ.get("AGENTMEMORY_SURFACE")
    if surface == NATIVE_SURFACE:
        return AgentMemoryWrapper()
    if surface == TRAVEL_SURFACE:
        factory = TravelPlannerFactory(
            tasks_path=_required_path("AGENTMEMORY_TRAVEL_TASKS_PATH"),
            memoryarena_root=_required_path("MEMORYARENA_ROOT"),
            database_path=_required_path("MEMORYARENA_TRAVEL_DATABASE_PATH"),
            expected_memoryarena_commit=_required_env("MEMORYARENA_BASE_COMMIT"),
        )
        return DomainEnvWrapper(
            factory,
            reward_policy=MemoryRewardPolicy(
                first_add=_env_float("AGENTMEMORY_FIRST_ADD_REWARD", 0.0),
                first_later_phase_retrieve=_env_float(
                    "AGENTMEMORY_FIRST_LATER_RETRIEVE_REWARD",
                    0.0,
                ),
                exact_repeat=_env_float("AGENTMEMORY_EXACT_REPEAT_REWARD", 0.0),
            ),
            invalid_action_penalty=_env_float(
                "AGENTMEMORY_INVALID_ACTION_REWARD",
                0.0,
            ),
        )
    raise RuntimeError(
        "AGENTMEMORY_SURFACE must select a registered formal surface; "
        f"observed {surface!r}"
    )


def _required_path(key: str) -> Path:
    value = _required_env(key)
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Required path does not exist for {key}: {path}")
    return path


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {key}")
    return value


def _env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be a floating-point number") from exc
