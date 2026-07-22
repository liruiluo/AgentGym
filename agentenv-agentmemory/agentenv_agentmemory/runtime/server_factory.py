from __future__ import annotations

import glob
import os
from pathlib import Path
from urllib.parse import urlparse

from ..domains import (
    BROWSECOMP_SURFACE,
    FORMAL_REASONING_SURFACES,
    TRAVEL_SURFACE,
    BrowseCompPlusFactory,
    FormalReasoningFactory,
    TravelPlannerFactory,
)
from ..env_wrapper import AgentMemoryWrapper, NATIVE_SURFACE
from ..domains.browsecomp import attest_browsecomp_search_assets
from .memory import MemoryRewardPolicy
from .registry import DomainRegistry
from .wrapper import DomainEnvWrapper


def build_server():
    surface = _required_env("AGENTMEMORY_SURFACE")
    if surface == NATIVE_SURFACE:
        return AgentMemoryWrapper()

    factory = build_domain_registry().build(surface)
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


def build_domain_registry() -> DomainRegistry:
    registry = DomainRegistry()
    registry.register(TRAVEL_SURFACE, _build_travel_factory)
    for domain, surface in FORMAL_REASONING_SURFACES.items():
        registry.register(
            surface,
            lambda domain=domain: _build_formal_reasoning_factory(domain),
        )
    registry.register(BROWSECOMP_SURFACE, _build_browsecomp_factory)
    return registry


def _build_travel_factory() -> TravelPlannerFactory:
    return TravelPlannerFactory(
        tasks_path=_required_file("AGENTMEMORY_TRAVEL_TASKS_PATH"),
        memoryarena_root=_required_directory("MEMORYARENA_ROOT"),
        database_path=_required_directory("MEMORYARENA_TRAVEL_DATABASE_PATH"),
        expected_memoryarena_commit=_required_env("MEMORYARENA_BASE_COMMIT"),
    )


def _build_formal_reasoning_factory(domain: str) -> FormalReasoningFactory:
    return FormalReasoningFactory(
        domain=domain,
        tasks_path=_required_file("AGENTMEMORY_FORMAL_REASONING_TASKS_PATH"),
        memoryarena_root=_required_directory("MEMORYARENA_ROOT"),
        judge_config={
            "backend": "openai",
            "model_name": _required_env(
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_MODEL"
            ),
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
        expected_memoryarena_commit=_required_env("MEMORYARENA_BASE_COMMIT"),
    )


def _build_browsecomp_factory() -> BrowseCompPlusFactory:
    provider = _required_env("AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER")
    if provider not in {"openai", "openrouter"}:
        raise RuntimeError(
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER must be openai or openrouter"
        )
    _required_env("OPENAI_API_KEY")
    _required_http_url("OPENAI_BASE_URL")
    if provider == "openrouter":
        _required_env("OPENROUTER_API_KEY")

    # Resolve all cheap, user-facing inputs before hashing the large frozen
    # FAISS/corpus assets.  This keeps missing dataset/config errors precise
    # and avoids doing expensive attestation for an unusable launch.
    ground_truth_path = _required_file(
        "AGENTMEMORY_BROWSECOMP_GROUND_TRUTH_PATH"
    )
    decomposition_path = _required_file(
        "AGENTMEMORY_BROWSECOMP_DECOMPOSITION_PATH"
    )
    judge_model = _required_env("AGENTMEMORY_BROWSECOMP_JUDGE_MODEL")

    index_path = _required_index_pattern("MEMORYARENA_BROWSECOMP_INDEX_PATH")
    corpus_path = _required_file("MEMORYARENA_BROWSECOMP_CORPUS_PATH")
    corpus_manifest_path = _required_file(
        "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST"
    )
    embedding_model = _required_env(
        "AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL"
    )
    search_asset_provenance = attest_browsecomp_search_assets(
        index_pattern=index_path,
        corpus_path=corpus_path,
        corpus_manifest_path=corpus_manifest_path,
        embedding_model=embedding_model,
    )

    return BrowseCompPlusFactory(
        ground_truth_path=ground_truth_path,
        decomposition_path=decomposition_path,
        memoryarena_root=_required_directory("MEMORYARENA_ROOT"),
        index_path=index_path,
        corpus_path=corpus_path,
        embedding_model=embedding_model,
        provider=provider,
        judge_model=judge_model,
        search_asset_provenance=search_asset_provenance,
        expected_memoryarena_commit=_required_env("MEMORYARENA_BASE_COMMIT"),
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
        raise RuntimeError(f"Required FAISS index pattern has no files for {key}: {pattern}")
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
