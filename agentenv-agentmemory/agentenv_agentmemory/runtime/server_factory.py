from __future__ import annotations

import glob
import os
from pathlib import Path
from urllib.parse import urlparse

from ..domains import (
    BROWSECOMP_BM25_INTEGRATION_SURFACE,
    BROWSECOMP_SURFACES,
    FORMAL_REASONING_PAPER_EVAL_SURFACES,
    FORMAL_REASONING_SURFACES_BY_MODE,
    TRAVEL_SURFACES,
    BrowseCompPlusFactory,
    FormalReasoningFactory,
    TravelPlannerFactory,
)
from ..env_wrapper import AgentMemoryWrapper, NATIVE_SURFACE
from ..filesystem_webshop_env import PROCEDURAL_FILESYSTEM_SURFACE
from ..filesystem_wrapper import ProceduralFilesystemAgentMemoryWrapper
from ..compositional_recall_webshop_env import (
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
    COMPOSITIONAL_RECALL_SURFACE,
)
from ..compositional_recall_wrapper import (
    CompositionalRecallAgentMemoryWrapper,
    CompositionalRecallFilesystemAgentMemoryWrapper,
)
from ..distractor_robustness_webshop_env import DISTRACTOR_ROBUSTNESS_SURFACE
from ..distractor_robustness_wrapper import DistractorRobustnessAgentMemoryWrapper
from ..intent_clarification_webshop_env import INTENT_CLARIFICATION_SURFACE
from ..intent_clarification_wrapper import IntentClarificationAgentMemoryWrapper
from ..latent_preference_webshop_env import LATENT_PREFERENCE_SURFACE
from ..latent_preference_wrapper import LatentPreferenceAgentMemoryWrapper
from ..procedural_webshop_env import PROCEDURAL_SURFACE
from ..procedural_wrapper import ProceduralAgentMemoryWrapper
from ..recency_override_webshop_env import (
    RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
    RECENCY_OVERRIDE_SURFACE,
)
from ..recency_override_wrapper import (
    RecencyOverrideAgentMemoryWrapper,
    RecencyOverrideFilesystemAgentMemoryWrapper,
)
from ..selective_memory_use_webshop_env import SELECTIVE_MEMORY_USE_SURFACE
from ..selective_memory_use_wrapper import SelectiveMemoryUseAgentMemoryWrapper
from ..negative_constraint_webshop_env import (
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
    NEGATIVE_CONSTRAINT_SURFACE,
)
from ..negative_constraint_wrapper import (
    NegativeConstraintAgentMemoryWrapper,
    NegativeConstraintFilesystemAgentMemoryWrapper,
)
from ..domains.browsecomp import (
    BROWSECOMP_BM25_INTEGRATION_BACKEND,
    BROWSECOMP_DENSE_BACKEND,
    BROWSECOMP_FROZEN_EMBEDDING_MODEL,
    BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
    BROWSECOMP_OPENROUTER_ENDPOINT,
)
from ..domains.formal_reasoning import FROZEN_MEMORYARENA_COMMIT
from ..domains.memoryarena_dataset import attest_frozen_memoryarena_dataset
from ..domains.travel import TRAVEL_FROZEN_MEMORYARENA_COMMIT
from .memory import MemoryRewardPolicy
from .registry import DomainRegistry
from .wrapper import DomainEnvWrapper


def build_server():
    surface = _required_env("AGENTMEMORY_SURFACE")
    if surface == NATIVE_SURFACE:
        return AgentMemoryWrapper()
    if surface == PROCEDURAL_SURFACE:
        return ProceduralAgentMemoryWrapper()
    if surface == PROCEDURAL_FILESYSTEM_SURFACE:
        return ProceduralFilesystemAgentMemoryWrapper()
    if surface == LATENT_PREFERENCE_SURFACE:
        return LatentPreferenceAgentMemoryWrapper()
    if surface == RECENCY_OVERRIDE_SURFACE:
        return RecencyOverrideAgentMemoryWrapper()
    if surface == RECENCY_OVERRIDE_FILESYSTEM_SURFACE:
        return RecencyOverrideFilesystemAgentMemoryWrapper()
    if surface == DISTRACTOR_ROBUSTNESS_SURFACE:
        return DistractorRobustnessAgentMemoryWrapper()
    if surface == COMPOSITIONAL_RECALL_SURFACE:
        return CompositionalRecallAgentMemoryWrapper()
    if surface == COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE:
        return CompositionalRecallFilesystemAgentMemoryWrapper()
    if surface == INTENT_CLARIFICATION_SURFACE:
        return IntentClarificationAgentMemoryWrapper()
    if surface == SELECTIVE_MEMORY_USE_SURFACE:
        return SelectiveMemoryUseAgentMemoryWrapper()
    if surface == NEGATIVE_CONSTRAINT_SURFACE:
        return NegativeConstraintAgentMemoryWrapper()
    if surface == NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE:
        return NegativeConstraintFilesystemAgentMemoryWrapper()

    factory = build_domain_registry().build(surface)
    first_add = _env_float("AGENTMEMORY_FIRST_ADD_REWARD", 0.0)
    first_later_retrieve = _env_float(
        "AGENTMEMORY_FIRST_LATER_RETRIEVE_REWARD",
        0.0,
    )
    exact_repeat = _env_float("AGENTMEMORY_EXACT_REPEAT_REWARD", 0.0)
    invalid_action = _env_float("AGENTMEMORY_INVALID_ACTION_REWARD", 0.0)
    if surface == BROWSECOMP_SURFACES["paper_eval"] and any(
        value != 0.0
        for value in (first_add, first_later_retrieve, exact_repeat, invalid_action)
    ):
        raise RuntimeError(
            "Progressive Search paper_eval is evaluation-only and refuses reward overlays"
        )
    if surface == TRAVEL_SURFACES["paper_eval"] and any(
        value != 0.0
        for value in (first_add, first_later_retrieve, exact_repeat, invalid_action)
    ):
        raise RuntimeError(
            "Travel paper_eval refuses reward overlays so its canonical paper "
            "ledger cannot be mixed with shaped rollout rewards"
        )
    if surface in FORMAL_REASONING_PAPER_EVAL_SURFACES.values() and any(
        value != 0.0
        for value in (first_add, first_later_retrieve, exact_repeat, invalid_action)
    ):
        raise RuntimeError(
            "Formal Reasoning paper_eval refuses reward overlays so its canonical "
            "paper ledger cannot be mixed with shaped rollout rewards"
        )
    return DomainEnvWrapper(
        factory,
        reward_policy=MemoryRewardPolicy(
            first_add=first_add,
            first_later_phase_retrieve=first_later_retrieve,
            exact_repeat=exact_repeat,
        ),
        invalid_action_penalty=invalid_action,
    )


def build_domain_registry() -> DomainRegistry:
    registry = DomainRegistry()
    for contract_mode, surface in TRAVEL_SURFACES.items():
        registry.register(
            surface,
            lambda contract_mode=contract_mode: _build_travel_factory(contract_mode),
        )
    for contract_mode, surfaces in FORMAL_REASONING_SURFACES_BY_MODE.items():
        for domain, surface in surfaces.items():
            registry.register(
                surface,
                lambda domain=domain, contract_mode=contract_mode: (
                    _build_formal_reasoning_factory(domain, contract_mode)
                ),
            )
    for contract_mode, surface in BROWSECOMP_SURFACES.items():
        registry.register(
            surface,
            lambda contract_mode=contract_mode: _build_browsecomp_factory(
                contract_mode
            ),
        )
    registry.register(
        BROWSECOMP_BM25_INTEGRATION_SURFACE,
        lambda: _build_browsecomp_factory(
            "failfast",
            search_backend=BROWSECOMP_BM25_INTEGRATION_BACKEND,
        ),
    )
    return registry


def _build_travel_factory(contract_mode: str) -> TravelPlannerFactory:
    tasks_path = _required_file("AGENTMEMORY_TRAVEL_TASKS_PATH")
    base_commit = _required_env("MEMORYARENA_BASE_COMMIT")
    if base_commit != TRAVEL_FROZEN_MEMORYARENA_COMMIT:
        raise RuntimeError(
            "MEMORYARENA_BASE_COMMIT must match the frozen Travel commit "
            f"{TRAVEL_FROZEN_MEMORYARENA_COMMIT}"
        )
    return TravelPlannerFactory(
        contract_mode=contract_mode,
        tasks_path=tasks_path,
        dataset_provenance=attest_frozen_memoryarena_dataset(
            tasks_path,
            config="group_travel_planner",
        ),
        memoryarena_root=_required_directory("MEMORYARENA_ROOT"),
        database_path=_required_directory("MEMORYARENA_TRAVEL_DATABASE_PATH"),
        expected_memoryarena_commit=TRAVEL_FROZEN_MEMORYARENA_COMMIT,
    )


def _build_formal_reasoning_factory(
    domain: str,
    contract_mode: str,
) -> FormalReasoningFactory:
    tasks_path = _required_file("AGENTMEMORY_FORMAL_REASONING_TASKS_PATH")
    dataset_config = f"formal_reasoning_{domain}"
    base_commit = _required_env("MEMORYARENA_BASE_COMMIT")
    if base_commit != FROZEN_MEMORYARENA_COMMIT:
        raise RuntimeError(
            "MEMORYARENA_BASE_COMMIT must match the frozen formal-reasoning "
            f"commit {FROZEN_MEMORYARENA_COMMIT}"
        )
    return FormalReasoningFactory(
        domain=domain,
        contract_mode=contract_mode,
        tasks_path=tasks_path,
        dataset_provenance=attest_frozen_memoryarena_dataset(
            tasks_path,
            config=dataset_config,
        ),
        memoryarena_root=_required_directory("MEMORYARENA_ROOT"),
        judge_config={
            "backend": "openai",
            "model_name": _required_env("AGENTMEMORY_FORMAL_REASONING_JUDGE_MODEL"),
            "base_url": _required_http_url(
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_BASE_URL"
            ),
            "temperature": _env_float(
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_TEMPERATURE",
                1.0,
            ),
            "max_tokens": _env_int(
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_MAX_TOKENS",
                4096,
            ),
        },
        expected_memoryarena_commit=FROZEN_MEMORYARENA_COMMIT,
    )


def _build_browsecomp_factory(
    contract_mode: str,
    *,
    search_backend: str = BROWSECOMP_DENSE_BACKEND,
) -> BrowseCompPlusFactory:
    if search_backend == BROWSECOMP_BM25_INTEGRATION_BACKEND:
        tasks_path = _required_file("AGENTMEMORY_BROWSECOMP_TASKS_PATH")
        judge_model = _required_env("AGENTMEMORY_BROWSECOMP_JUDGE_MODEL")
        base_commit = _required_env("MEMORYARENA_BASE_COMMIT")
        if base_commit != BROWSECOMP_FROZEN_MEMORYARENA_COMMIT:
            raise RuntimeError(
                "MEMORYARENA_BASE_COMMIT must match the frozen Progressive Search "
                f"commit {BROWSECOMP_FROZEN_MEMORYARENA_COMMIT}"
            )
        _required_env("OPENAI_API_KEY")
        openai_base_url = _required_http_url("OPENAI_BASE_URL")
        return BrowseCompPlusFactory(
            contract_mode="failfast",
            tasks_path=tasks_path,
            dataset_provenance=attest_frozen_memoryarena_dataset(
                tasks_path,
                config="progressive_search",
            ),
            memoryarena_root=_required_directory("MEMORYARENA_ROOT"),
            search_backend=BROWSECOMP_BM25_INTEGRATION_BACKEND,
            bm25_index_path=_required_directory(
                "MEMORYARENA_BROWSECOMP_BM25_INDEX_PATH"
            ),
            judge_config={
                "backend": "openai_responses",
                "model_name": judge_model,
                "base_url": openai_base_url,
                "max_tokens": _env_int(
                    "AGENTMEMORY_BROWSECOMP_JUDGE_MAX_TOKENS",
                    8000,
                ),
            },
            expected_memoryarena_commit=BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
        )
    if search_backend != BROWSECOMP_DENSE_BACKEND:
        raise RuntimeError(f"Unsupported BrowseComp search backend: {search_backend}")
    provider = _required_env("AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER")
    if provider not in {"openai", "openrouter"}:
        raise RuntimeError(
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER must be openai or openrouter"
        )
    if contract_mode == "paper_eval" and provider != "openai":
        raise RuntimeError(
            "Progressive Search paper_eval requires the OpenAI embedding provider"
        )
    _required_env("OPENAI_API_KEY")
    openai_base_url = _required_http_url("OPENAI_BASE_URL")
    if provider == "openrouter":
        _required_env("OPENROUTER_API_KEY")
    embedding_endpoint = (
        openai_base_url if provider == "openai" else BROWSECOMP_OPENROUTER_ENDPOINT
    )

    # Resolve all cheap, user-facing inputs before hashing the large frozen
    # FAISS/corpus assets.  This keeps missing dataset/config errors precise
    # and avoids doing expensive attestation for an unusable launch.
    tasks_path = _required_file("AGENTMEMORY_BROWSECOMP_TASKS_PATH")
    judge_model = _required_env("AGENTMEMORY_BROWSECOMP_JUDGE_MODEL")
    base_commit = _required_env("MEMORYARENA_BASE_COMMIT")
    if base_commit != BROWSECOMP_FROZEN_MEMORYARENA_COMMIT:
        raise RuntimeError(
            "MEMORYARENA_BASE_COMMIT must match the frozen Progressive Search "
            f"commit {BROWSECOMP_FROZEN_MEMORYARENA_COMMIT}"
        )

    index_path = _required_index_pattern("MEMORYARENA_BROWSECOMP_INDEX_PATH")
    corpus_path = _required_file("MEMORYARENA_BROWSECOMP_CORPUS_PATH")
    corpus_manifest_path = _required_file("MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST")
    embedding_model = _required_env("AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL")
    if embedding_model != BROWSECOMP_FROZEN_EMBEDDING_MODEL:
        raise RuntimeError(
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL must match the frozen "
            f"model {BROWSECOMP_FROZEN_EMBEDDING_MODEL}"
        )
    return BrowseCompPlusFactory(
        contract_mode=contract_mode,
        tasks_path=tasks_path,
        dataset_provenance=attest_frozen_memoryarena_dataset(
            tasks_path,
            config="progressive_search",
        ),
        memoryarena_root=_required_directory("MEMORYARENA_ROOT"),
        index_path=index_path,
        corpus_path=corpus_path,
        corpus_manifest_path=corpus_manifest_path,
        embedding_model=embedding_model,
        provider=provider,
        embedding_endpoint=embedding_endpoint,
        judge_config={
            "backend": "openai_responses",
            "model_name": judge_model,
            "base_url": openai_base_url,
            "max_tokens": _env_int(
                "AGENTMEMORY_BROWSECOMP_JUDGE_MAX_TOKENS",
                8000,
            ),
        },
        expected_memoryarena_commit=BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
    )


def _required_file(key: str) -> Path:
    path = _required_path(key)
    if not path.is_file():
        raise RuntimeError(f"Required file does not exist for {key}: {path}")
    return path


def _required_directory(key: str) -> Path:
    path = _required_path(key)
    if not path.is_dir():
        raise RuntimeError(f"Required directory does not exist for {key}: {path}")
    return path


def _required_path(key: str) -> Path:
    return Path(_required_env(key)).expanduser().resolve()


def _required_index_pattern(key: str) -> str:
    raw_pattern = _required_env(key)
    path = Path(raw_pattern).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    pattern = str(path)
    if not pattern.endswith(".index"):
        raise RuntimeError(f"{key} must select .index files: {pattern}")
    index_paths = [Path(item) for item in sorted(glob.glob(pattern))]
    if not index_paths or any(not item.is_file() for item in index_paths):
        raise RuntimeError(
            f"Required FAISS index pattern has no files for {key}: {pattern}"
        )
    missing_id_maps = [
        index_path.with_name(index_path.stem + "_id_map.json")
        for index_path in index_paths
        if not index_path.with_name(index_path.stem + "_id_map.json").is_file()
    ]
    if missing_id_maps:
        rendered = ", ".join(str(item) for item in missing_id_maps)
        raise RuntimeError(f"BrowseComp FAISS indexes lack id maps: {rendered}")
    return pattern


def _required_http_url(key: str) -> str:
    value = _required_env(key)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"{key} must be an absolute HTTP(S) URL")
    return value.rstrip("/")


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value or not value.strip():
        raise RuntimeError(f"Required environment variable is missing: {key}")
    return value.strip()


def _env_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be a floating-point number") from exc


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer") from exc
    if parsed < 1:
        raise RuntimeError(f"{key} must be positive")
    return parsed
