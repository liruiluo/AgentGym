from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..latent_preference.schema import canonical_sha256, require_id, require_sha256
from ..recency_override.schema import RecencyOverrideTask


SPLITS = ("train", "dev", "test")
MAPPING_BRANCHES = ("token_a", "token_b")
DIRECTORY_BRANCHES = ("identity", "swapped")
BRANCH_COORDINATES = (
    ("token_a", "identity"),
    ("token_a", "swapped"),
    ("token_b", "identity"),
    ("token_b", "swapped"),
)
TASK_SCHEMA = "agentmemory_compositional_recall_task_v1"
ORBIT_SCHEMA = "agentmemory_compositional_recall_factorial_orbit_v1"
PROOF_SCHEMA = "agentmemory_compositional_recall_proof_v1"


class CompositionalRecallDataError(ValueError):
    """Raised when a two-hop memory task is not machine-verifiable."""


@dataclass(frozen=True)
class CanonicalMemoryFact:
    role: str
    key: str
    value: str
    query: str

    def __post_init__(self) -> None:
        if self.role not in {"customer_to_profile", "profile_directory"}:
            raise CompositionalRecallDataError("invalid canonical memory role.")
        for field_name in ("key", "value", "query"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise CompositionalRecallDataError(
                    f"canonical memory {field_name} must be non-empty text."
                )

    def as_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "key": self.key,
            "value": self.value,
            "query": self.query,
        }


@dataclass(frozen=True)
class CompositionalRecallTask:
    task_id: str
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    split: str
    branch_kind: str
    mapping_branch: str
    directory_branch: str
    source_task: RecencyOverrideTask
    profile_tokens: tuple[str, str]
    active_profile_token: str
    profile_directory: tuple[tuple[str, str], tuple[str, str]]
    preferred_attribute_value: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    canonical_memories: tuple[CanonicalMemoryFact, CanonicalMemoryFact]
    generator_version: str
    generator_seed: int
    product_pool_sha256: str

    def __post_init__(self) -> None:
        for field_name in ("task_id", "orbit_id"):
            require_id(getattr(self, field_name), field=field_name)
        if (
            isinstance(self.orbit_index, bool)
            or not isinstance(self.orbit_index, int)
            or self.orbit_index < 0
        ):
            raise CompositionalRecallDataError("invalid orbit_index.")
        if (
            isinstance(self.semantic_epoch, bool)
            or not isinstance(self.semantic_epoch, int)
            or self.semantic_epoch < 0
        ):
            raise CompositionalRecallDataError("invalid semantic_epoch.")
        if self.split not in SPLITS or self.source_task.split != self.split:
            raise CompositionalRecallDataError("invalid or inconsistent split.")
        coordinate = (self.mapping_branch, self.directory_branch)
        if coordinate not in BRANCH_COORDINATES:
            raise CompositionalRecallDataError("invalid factorial coordinate.")
        expected_branch = f"active_{self.mapping_branch}.directory_{self.directory_branch}"
        if self.branch_kind != expected_branch:
            raise CompositionalRecallDataError("branch kind disagrees with coordinate.")
        if self.source_task.branch_kind != "stay":
            raise CompositionalRecallDataError(
                "compositional tasks must reuse the recency stay substrate."
            )
        if len(self.profile_tokens) != 2 or len(set(self.profile_tokens)) != 2:
            raise CompositionalRecallDataError("two distinct profile tokens are required.")
        for token in self.profile_tokens:
            require_id(token, field="profile token")
        expected_active = self.profile_tokens[
            0 if self.mapping_branch == "token_a" else 1
        ]
        if self.active_profile_token != expected_active:
            raise CompositionalRecallDataError("active profile token mismatch.")
        if tuple(token for token, _ in self.profile_directory) != self.profile_tokens:
            raise CompositionalRecallDataError("profile directory token order mismatch.")
        directory_values = tuple(value for _, value in self.profile_directory)
        if len(set(directory_values)) != 2:
            raise CompositionalRecallDataError(
                "profile directory must be a two-value permutation."
            )
        directory = dict(self.profile_directory)
        if directory[self.active_profile_token] != self.preferred_attribute_value:
            raise CompositionalRecallDataError(
                "preferred value disagrees with active profile directory lookup."
            )
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise CompositionalRecallDataError("task requires six shopping phases.")
        if len(self.canonical_memories) != 2 or tuple(
            item.role for item in self.canonical_memories
        ) != ("customer_to_profile", "profile_directory"):
            raise CompositionalRecallDataError(
                "canonical memories must be mapping then directory."
            )
        if len({item.key for item in self.canonical_memories}) != 1:
            raise CompositionalRecallDataError(
                "canonical memory keys must be identical to prevent a key shortcut."
            )
        if isinstance(self.generator_seed, bool) or not isinstance(
            self.generator_seed, int
        ):
            raise CompositionalRecallDataError("generator_seed must be an integer.")
        if not isinstance(self.generator_version, str) or not self.generator_version:
            raise CompositionalRecallDataError("generator_version must be non-empty.")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
        if self.source_task.product_pool_sha256 != self.product_pool_sha256:
            raise CompositionalRecallDataError("source task product pool mismatch.")

    @property
    def budget_cents(self) -> int:
        return self.source_task.budget_cents

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": TASK_SCHEMA,
            "task_id": self.task_id,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "split": self.split,
            "branch_kind": self.branch_kind,
            "mapping_branch": self.mapping_branch,
            "directory_branch": self.directory_branch,
            "source_task": self.source_task.as_dict(include_targets=True),
            "profile_tokens": list(self.profile_tokens),
            "active_profile_token": self.active_profile_token,
            "profile_directory": [list(item) for item in self.profile_directory],
            "preferred_attribute_value": self.preferred_attribute_value,
            "questions": list(self.questions),
            "target_asins": list(self.target_asins),
            "canonical_memories": [
                item.as_dict() for item in self.canonical_memories
            ],
            "generator_version": self.generator_version,
            "generator_seed": self.generator_seed,
            "product_pool_sha256": self.product_pool_sha256,
        }


@dataclass(frozen=True)
class CompositionalRecallOrbit:
    orbit_id: str
    orbit_index: int
    semantic_epoch: int
    source_recency_orbit_id: str
    profile_tokens: tuple[str, str]
    tasks: tuple[
        CompositionalRecallTask,
        CompositionalRecallTask,
        CompositionalRecallTask,
        CompositionalRecallTask,
    ]

    def __post_init__(self) -> None:
        require_id(self.orbit_id, field="orbit_id")
        require_id(self.source_recency_orbit_id, field="source_recency_orbit_id")
        if tuple(
            (task.mapping_branch, task.directory_branch) for task in self.tasks
        ) != BRANCH_COORDINATES:
            raise CompositionalRecallDataError(
                "orbit tasks must follow the canonical factorial order."
            )
        for task in self.tasks:
            if (
                task.orbit_id,
                task.orbit_index,
                task.semantic_epoch,
                task.profile_tokens,
                task.source_task.orbit_id,
            ) != (
                self.orbit_id,
                self.orbit_index,
                self.semantic_epoch,
                self.profile_tokens,
                self.source_recency_orbit_id,
            ):
                raise CompositionalRecallDataError(
                    "task identity disagrees with its orbit."
                )

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ORBIT_SCHEMA,
            "orbit_id": self.orbit_id,
            "orbit_index": self.orbit_index,
            "semantic_epoch": self.semantic_epoch,
            "source_recency_orbit_id": self.source_recency_orbit_id,
            "profile_tokens": list(self.profile_tokens),
            "tasks": [task.as_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class CompositionalRecallBundle:
    task_id: str
    questions: tuple[str, ...]
    target_asins: tuple[str, ...]
    budget_cents: int
    split: str
    orbit_id: str
    branch_kind: str
    proof_sha256: str
    generator_version: str
    product_pool_sha256: str

    def __post_init__(self) -> None:
        if len(self.questions) != 6 or len(self.target_asins) != 6:
            raise CompositionalRecallDataError("bundle requires six shopping phases.")
        if self.split not in SPLITS:
            raise CompositionalRecallDataError("invalid bundle split.")
        require_sha256(self.proof_sha256, field="proof_sha256")
        require_sha256(self.product_pool_sha256, field="product_pool_sha256")
