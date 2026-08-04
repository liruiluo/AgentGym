from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Any

from .annotation_gate import (
    ANNOTATION_GATE_MODES,
    AnnotationGateDecision,
    validate_annotation_gate_manifest,
)
from .memoryarena_dataset import (
    EXPECTED_DOMAIN_DATA_SHA256,
    MemoryArenaDataset,
    load_memoryarena_dataset,
)
from .memoryarena_webshop_env import (
    ACTION_LISTING_MODES,
    LTM_INVENTORY_KEY_MAX_CHARS,
    LTM_INVENTORY_MODES,
    LTM_TRANSITION_NOTICE_MODES,
    MemoryArenaWebShopEnv,
)
from .native_webshop_backend import MemoryArenaNativeWebShopBackend
from .reward_hierarchy import (
    FIRST_VALID_ADD_BONUS,
    FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
    build_memoryarena_reward_contract,
)


NATIVE_SURFACE = "memoryarena_webshop_native_v1"
LATENT_PREFERENCE_PROMPT_MODE = "latent_preference_sop"
SELECTIVE_MEMORY_PROMPT_MODE = "selective_memory_sop"
NATURAL_FILESYSTEM_PROMPT_MODE = "natural_filesystem"
MEMORY_PROMPT_MODES = (
    "legacy",
    "neutral",
    "neutral_horizon",
    "neutral_horizon_responsibility",
    LATENT_PREFERENCE_PROMPT_MODE,
    SELECTIVE_MEMORY_PROMPT_MODE,
    NATURAL_FILESYSTEM_PROMPT_MODE,
)
FORBIDDEN_SURROGATE_ENV = {
    "AGENTMEMORY_CATALOG_INDEX_PATH",
    "AGENTMEMORY_SEARCH_TIMEOUT_MS",
}


class AgentMemoryWrapper:
    """AgentGym HTTP wrapper for the original MemoryArena WebShop surface."""

    def __init__(self) -> None:
        surface = os.environ.get("AGENTMEMORY_SURFACE")
        if surface != NATIVE_SURFACE:
            raise RuntimeError(
                f"AGENTMEMORY_SURFACE must be explicitly set to {NATIVE_SURFACE!r}; "
                "the legacy SQLite surface is offline engineering-audit code only."
            )
        forbidden = sorted(key for key in FORBIDDEN_SURROGATE_ENV if os.environ.get(key))
        if forbidden:
            raise RuntimeError(
                "Native MemoryArena launch refuses legacy SQLite variables: "
                + ", ".join(forbidden)
            )

        self.reward_contract = build_memoryarena_reward_contract(
            first_valid_add_reward=_env_nonnegative_float(
                "AGENTMEMORY_FIRST_VALID_ADD_REWARD",
                FIRST_VALID_ADD_BONUS,
            ),
            first_valid_later_session_retrieve_reward=_env_nonnegative_float(
                "AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD",
                FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
            ),
        )
        self.ltm_inventory_mode = _env_choice(
            "AGENTMEMORY_LTM_INVENTORY_MODE",
            default="hidden",
            choices=LTM_INVENTORY_MODES,
        )
        self.ltm_transition_notice_mode = _env_choice(
            "AGENTMEMORY_LTM_TRANSITION_NOTICE_MODE",
            default="none",
            choices=LTM_TRANSITION_NOTICE_MODES,
        )
        self.action_listing_mode = _env_choice(
            "AGENTMEMORY_ACTION_LISTING_MODE",
            default="separate",
            choices=ACTION_LISTING_MODES,
        )
        self.memory_prompt_mode = _env_choice(
            "AGENTMEMORY_MEMORY_PROMPT_MODE",
            default="legacy",
            choices=MEMORY_PROMPT_MODES,
        )
        if self.memory_prompt_mode in {
            LATENT_PREFERENCE_PROMPT_MODE,
            SELECTIVE_MEMORY_PROMPT_MODE,
        }:
            raise RuntimeError(
                "The native MemoryArena WebShop surface cannot use the "
                "specialized procedural memory prompt."
            )

        domain_data_path = _required_path("MEMORYARENA_WEBSHOP_DOMAIN_DATA_PATH")
        domain_data_sha256 = _sha256_file(domain_data_path)
        if domain_data_sha256 != EXPECTED_DOMAIN_DATA_SHA256:
            raise RuntimeError(
                "MemoryArena domain_data.json SHA256 mismatch: "
                f"expected {EXPECTED_DOMAIN_DATA_SHA256}, observed {domain_data_sha256}."
            )
        self.backend = MemoryArenaNativeWebShopBackend(
            memoryarena_root=_required_path("MEMORYARENA_ROOT"),
            items_file=_required_path("MEMORYARENA_WEBSHOP_ITEMS_FILE"),
            attributes_file=_required_path("MEMORYARENA_WEBSHOP_ATTR_FILE"),
            search_root=_required_path("MEMORYARENA_WEBSHOP_SEARCH_ROOT"),
            java_home=_required_path("MEMORYARENA_WEBSHOP_JAVA_HOME"),
            expected_memoryarena_commit=_required_env("MEMORYARENA_BASE_COMMIT"),
            price_seed=_env_int("AGENTMEMORY_WEBSHOP_PRICE_SEED", 233),
        )
        self.dataset = load_memoryarena_dataset(
            _required_path("AGENTMEMORY_MEMORYARENA_RAW_PATH"),
            frozen_product_asins=self.backend.product_asins(),
            domain_data_sha256=domain_data_sha256,
        )
        split_tasks = self._select_tasks(self.dataset)
        self.annotation_gate = self._validate_annotation_gate()
        allowed_task_ids = set(self.annotation_gate.allowed_task_ids)
        self.tasks = tuple(task for task in split_tasks if task.task_id in allowed_task_ids)
        if not self.tasks:
            raise RuntimeError("Annotation gate clears zero tasks in the selected split.")
        self.max_id = 0
        self.envs: dict[int, MemoryArenaWebShopEnv] = {}
        self.info: dict[int, dict[str, Any]] = {}
        self.env_locks: dict[int, threading.RLock] = {}
        self.lock = threading.RLock()

    def create(self) -> dict[str, Any]:
        with self.lock:
            env_id = self.max_id
            self.max_id += 1
            env = MemoryArenaWebShopEnv(
                bundles=self.tasks,
                backend=self.backend,
                env_uid=f"env{env_id}",
                first_valid_add_reward=float(
                    self.reward_contract["first_valid_add_reward"]
                ),
                first_valid_later_session_retrieve_reward=float(
                    self.reward_contract[
                        "first_valid_later_session_retrieve_reward"
                    ]
                ),
                ltm_inventory_mode=self.ltm_inventory_mode,
                ltm_transition_notice_mode=self.ltm_transition_notice_mode,
                action_listing_mode=self.action_listing_mode,
            )
            observation, info = env.reset(data_idx=env_id)
            payload = {
                "id": env_id,
                "observation": observation,
                "reward": 0.0,
                "done": False,
                "info": info,
            }
            self.envs[env_id] = env
            self.info[env_id] = payload
            self.env_locks[env_id] = threading.RLock()
            return payload

    def step(self, env_id: int, action: str) -> dict[str, Any]:
        env = self.require_env(env_id)
        with self.require_lock(env_id):
            observation, reward, done, _, info = env.step(action)
            payload = {
                "observation": observation,
                "reward": reward,
                "done": done,
                "info": info,
            }
            self.info[env_id] = payload
            return payload

    def reset(self, env_id: int, data_idx: int = 0) -> dict[str, Any]:
        env = self.require_env(env_id)
        with self.require_lock(env_id):
            observation, info = env.reset(data_idx=data_idx)
            payload = {
                "id": env_id,
                "observation": observation,
                "reward": 0.0,
                "done": False,
                "info": info,
            }
            self.info[env_id] = payload
            return payload

    def observation(self, env_id: int) -> str:
        self.require_env(env_id)
        with self.require_lock(env_id):
            return self.info[env_id]["observation"]

    def detail(self, env_id: int) -> dict[str, Any]:
        self.require_env(env_id)
        with self.require_lock(env_id):
            return self.info[env_id]

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
        provenance = self.dataset.provenance
        dataset_provenance = provenance.as_manifest()
        return {
            "surface": NATIVE_SURFACE,
            "task_count": len(self.tasks),
            "splits": sorted({task.split for task in self.tasks}),
            "dataset_sha256": provenance.raw_dataset_sha256,
            "dataset_provenance": dataset_provenance,
            "raw_dataset_sha256": provenance.raw_dataset_sha256,
            "split_manifest_sha256": provenance.split_manifest_sha256,
            "memoryarena_commit": provenance.memoryarena_commit,
            "domain_data_sha256": provenance.domain_data_sha256,
            "annotation_gate_mode": self.annotation_gate.mode,
            "annotation_gate_sha256": self.annotation_gate.manifest_sha256,
            "annotation_gate_allowed_task_ids_sha256": self.annotation_gate.allowed_task_ids_sha256,
            "annotation_gate_allowed_task_count": len(self.annotation_gate.allowed_task_ids),
            "reward_contract": dict(self.reward_contract),
            "ltm_inventory_mode": self.ltm_inventory_mode,
            "ltm_transition_notice_mode": self.ltm_transition_notice_mode,
            "action_listing_mode": self.action_listing_mode,
            "memory_prompt_mode": self.memory_prompt_mode,
            "active_environment_count": len(getattr(self, "envs", {})),
            "ltm_inventory_key_max_chars": LTM_INVENTORY_KEY_MAX_CHARS,
            "ltm_inventory_key_format": "ascii_identifier",
            "backend": self.backend.metadata(),
        }

    def require_env(self, env_id: int) -> MemoryArenaWebShopEnv:
        try:
            return self.envs[env_id]
        except KeyError as exc:
            raise KeyError(f"Unknown environment id {env_id}.") from exc

    def require_lock(self, env_id: int) -> threading.RLock:
        try:
            return self.env_locks[env_id]
        except KeyError as exc:
            raise KeyError(f"Unknown environment id {env_id}.") from exc

    @staticmethod
    def _select_tasks(dataset: MemoryArenaDataset):
        split = os.environ.get("AGENTMEMORY_SPLIT", "train")
        if split == "all":
            tasks = dataset.bundles
        else:
            tasks = dataset.for_split(split)
        if not tasks:
            raise RuntimeError(f"No MemoryArena tasks selected for split {split!r}.")
        return tuple(tasks)

    def _validate_annotation_gate(self) -> AnnotationGateDecision:
        mode = _required_env("AGENTMEMORY_ANNOTATION_GATE_MODE")
        if mode not in ANNOTATION_GATE_MODES:
            raise RuntimeError(
                "AGENTMEMORY_ANNOTATION_GATE_MODE must be one of: "
                + ", ".join(ANNOTATION_GATE_MODES)
                + "."
            )
        manifest_path = _required_path("AGENTMEMORY_ANNOTATION_GATE_MANIFEST")
        manifest_sha256 = _required_env("AGENTMEMORY_ANNOTATION_GATE_MANIFEST_SHA256")
        run_id = _required_env("AGENTMEMORY_RUN_ID")
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            selected_task_ids = payload["allowed_task_ids"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Cannot read annotation gate task IDs: {manifest_path}") from exc

        decision = validate_annotation_gate_manifest(
            manifest_path,
            expected_mode=mode,
            expected_run_id=run_id,
            expected_manifest_sha256=manifest_sha256,
            selected_task_ids=selected_task_ids,
            raw_dataset_path=_required_path("AGENTMEMORY_MEMORYARENA_RAW_PATH"),
            domain_data_path=_required_path("MEMORYARENA_WEBSHOP_DOMAIN_DATA_PATH"),
            items_shuffle_path=_required_path("MEMORYARENA_WEBSHOP_ITEMS_FILE"),
            items_ins_v2_path=_required_path("MEMORYARENA_WEBSHOP_ATTR_FILE"),
            lucene_index_manifest_path=_required_path("MEMORYARENA_LUCENE_INDEX_MANIFEST"),
            lucene_index_root=_required_path("MEMORYARENA_WEBSHOP_SEARCH_ROOT") / "indexes-full",
            audit_summary_path=_required_path("AGENTMEMORY_ANNOTATION_AUDIT_SUMMARY"),
            audit_chains_path=_required_path("AGENTMEMORY_ANNOTATION_AUDIT_CHAINS"),
            manual_evidence_path=_required_path("AGENTMEMORY_ANNOTATION_MANUAL_EVIDENCE"),
            memoryarena_repo_path=_required_path("MEMORYARENA_ROOT"),
            memoryarena_base_commit=_required_env("MEMORYARENA_BASE_COMMIT"),
            price_seed=_env_int("AGENTMEMORY_WEBSHOP_PRICE_SEED", 233),
        )
        backend_metadata = self.backend.metadata()
        if decision.price_table_sha256 != backend_metadata["price_table_sha256"]:
            raise RuntimeError("Annotation gate and native backend price-table hashes disagree.")
        if decision.price_table_row_count != backend_metadata["product_count"]:
            raise RuntimeError("Annotation gate and native backend price-table row counts disagree.")
        return decision


def _required_path(key: str) -> Path:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {key}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"Required path does not exist for {key}: {path}")
    return path


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {key}")
    return value


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer.") from exc


def _env_nonnegative_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    try:
        parsed = float(default if value is None else value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{key} must be a finite, non-negative number.") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise RuntimeError(f"{key} must be a finite, non-negative number.")
    return parsed


def _env_choice(key: str, *, default: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(key, default)
    if value not in choices:
        raise RuntimeError(f"{key} must be one of: {', '.join(choices)}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
