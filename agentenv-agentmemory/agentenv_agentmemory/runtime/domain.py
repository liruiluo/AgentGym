from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


FORMAL_SCHEMA_V3 = "agentmemory_formal_step_v3"

MEMORY_ACTION_DESCRIPTIONS = (
    'ADD {"key": "...", "value": "..."}',
    'UPDATE {"memory_id": "mem_0000", "value": "..."}',
    'DELETE {"memory_id": "mem_0000"}',
    'RETRIEVE {"query": "...", "top_k": 3}',
    'SUMMARY {"text": "...", "source_ids": ["S0", "C0"]}',
    'FILTER {"keep_ids": ["C0"], "scope": "active"}',
)


def _empty_mapping() -> dict[str, Any]:
    return {}


@dataclass(frozen=True)
class DomainContract:
    """Model-visible contract for one task domain."""

    contract_id: str
    system_prompt: str
    native_action_descriptions: tuple[str, ...]
    max_steps: int

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("contract_id must not be empty")
        if not self.system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if not self.native_action_descriptions:
            raise ValueError("at least one native action is required")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

    @property
    def sha256(self) -> str:
        return contract_digest(self)

    @property
    def canonical_system_prompt(self) -> str:
        return render_system_prompt(self)

    @property
    def system_prompt_sha256(self) -> str:
        return hashlib.sha256(
            self.canonical_system_prompt.encode("utf-8")
        ).hexdigest()


def render_system_prompt(contract: DomainContract) -> str:
    """Render the complete model-facing contract exposed by v3 metadata."""

    return "\n\n".join(
        [
            contract.system_prompt.strip(),
            "Native domain action forms:\n"
            + "\n".join(
                f"- {item.strip()}" for item in contract.native_action_descriptions
            ),
            "Policy memory action forms:\n"
            + "\n".join(f"- {item}" for item in MEMORY_ACTION_DESCRIPTIONS),
            (
                "Cross-phase memory lifecycle:\n"
                "- ADD writes policy-authored text to long-term memory for the "
                "current episode.\n"
                "- A native phase advance clears the current phase's short-term/page "
                "trace and active retrieved or summarized S*/C* context. Long-term "
                "memory is retained, but it is not automatically visible in the next "
                "phase.\n"
                "- RETRIEVE queries text previously written with ADD and exposes "
                "matching long-term memories in active context for the current phase."
            ),
            (
                "Reply in exactly this format:\n\n"
                "Thought:\nbrief reasoning\n\nAction:\n"
                "<exactly one native domain action or uppercase memory action>"
            ),
        ]
    )


def contract_digest(contract: DomainContract) -> str:
    payload = json.dumps(
        asdict(contract),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DomainTransition:
    """One authoritative transition returned by a domain driver."""

    observation: str
    reward: float
    done: bool
    status: str
    phase_index: int
    phase_count: int | None
    episode_success: bool
    action_execution: dict[str, Any] = field(default_factory=_empty_mapping)
    tool_ops: tuple[dict[str, Any], ...] = ()
    reward_components: tuple[dict[str, Any], ...] = ()
    domain_evidence: dict[str, Any] = field(default_factory=_empty_mapping)
    sample_excluded: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.observation, str):
            raise TypeError("observation must be a string")
        if isinstance(self.reward, bool) or not isinstance(self.reward, (int, float)):
            raise TypeError("reward must be numeric")
        if not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if self.phase_index < 0:
            raise ValueError("phase_index must be non-negative")
        if self.phase_count is not None:
            if self.phase_count < 1:
                raise ValueError("phase_count must be positive")
            if self.phase_index > self.phase_count:
                raise ValueError("phase_index cannot exceed phase_count")
        component_sum = sum(float(item.get("value", 0.0)) for item in self.reward_components)
        if self.reward_components and not math.isclose(
            component_sum,
            float(self.reward),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "reward_components must sum to reward: "
                f"components={component_sum} reward={self.reward}"
            )
        if self.episode_success and not self.done:
            raise ValueError("episode_success requires done=True")
        if self.sample_excluded and not self.done:
            raise ValueError("sample_excluded requires done=True")

    def to_info(
        self,
        *,
        domain_id: str,
        surface: str,
        contract: DomainContract,
    ) -> dict[str, Any]:
        progress_score = None
        if self.phase_count:
            progress_score = self.phase_index / self.phase_count
        return {
            "formal_schema_version": FORMAL_SCHEMA_V3,
            "domain_id": domain_id,
            "surface": surface,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.sha256,
            "status": self.status,
            "phase_index": self.phase_index,
            "phase_count": self.phase_count,
            "progress_score": progress_score,
            "episode_success": self.episode_success,
            "action_execution": dict(self.action_execution),
            "tool_ops": [dict(item) for item in self.tool_ops],
            "reward_components": [dict(item) for item in self.reward_components],
            "domain_evidence": dict(self.domain_evidence),
            "sample_excluded": self.sample_excluded,
        }


@runtime_checkable
class DomainDriver(Protocol):
    domain_id: str
    surface: str
    contract: DomainContract

    def reset(self, data_idx: int) -> DomainTransition:
        ...

    def step(self, action: str, env_step: int) -> DomainTransition:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class DomainFactory(Protocol):
    domain_id: str
    surface: str
    task_count: int
    contract: DomainContract

    def create(self, env_uid: str) -> DomainDriver:
        ...

    def metadata(self) -> dict[str, Any]:
        ...
