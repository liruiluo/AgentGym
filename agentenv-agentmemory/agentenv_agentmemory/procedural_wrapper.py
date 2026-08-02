from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .env_wrapper import LATENT_PREFERENCE_PROMPT_MODE, MEMORY_PROMPT_MODES
from .memoryarena_webshop_env import (
    ACTION_LISTING_MODES,
    LTM_INVENTORY_KEY_MAX_CHARS,
    LTM_INVENTORY_MODES,
    LTM_TRANSITION_NOTICE_MODES,
    MemoryArenaWebShopEnv,
)
from .native_webshop_backend import (
    MemoryArenaNativeWebShopBackend,
    NativeWebShopBackend,
)
from .procedural import (
    NaturalAttributeChainGenerator,
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODES,
    ProductPool,
    VerifiedProceduralBundleProvider,
    file_sha256,
    load_certified_product_pool,
    require_unique_product_classification,
    verify_lucene_index_manifest,
)
from .procedural_webshop_env import PROCEDURAL_SURFACE, ProceduralMemoryWebShopEnv
from .reward_hierarchy import (
    FIRST_VALID_ADD_BONUS,
    FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
    build_memoryarena_reward_contract,
)


class ProceduralAgentMemoryWrapper:
    """AgentGym HTTP wrapper for machine-verified procedural memory training."""

    surface = PROCEDURAL_SURFACE
    environment_type = ProceduralMemoryWebShopEnv

    def __init__(self) -> None:
        self._initialize_native_training_runtime(
            forbidden_env_keys=(
                "AGENTMEMORY_MEMORYARENA_RAW_PATH",
                "AGENTMEMORY_ANNOTATION_GATE_MANIFEST",
                "AGENTMEMORY_ANNOTATION_MANUAL_EVIDENCE",
            )
        )
        if self.memory_prompt_mode == LATENT_PREFERENCE_PROMPT_MODE:
            raise RuntimeError(
                "The natural-chain surface cannot use the latent_preference_sop prompt."
            )
        pool = load_certified_product_pool(
            _required_file("AGENTMEMORY_PROCEDURAL_PRODUCT_POOL"),
            expected_file_sha256=_required_env(
                "AGENTMEMORY_PROCEDURAL_PRODUCT_POOL_SHA256"
            ),
        )
        attest_procedural_runtime_inputs(
            pool,
            self.backend,
            items_file=self.items_file,
            attributes_file=self.attributes_file,
            search_root=self.search_root,
            lucene_manifest=self.lucene_manifest,
        )

        self.provider = VerifiedProceduralBundleProvider(
            generator=NaturalAttributeChainGenerator(
                pool=pool,
                seed=_required_int("AGENTMEMORY_PROCEDURAL_GENERATOR_SEED"),
            ),
            split=_required_env("AGENTMEMORY_SPLIT"),
            task_count=_required_int("AGENTMEMORY_PROCEDURAL_TASK_COUNT"),
            mode=_env_choice(
                "AGENTMEMORY_PROCEDURAL_PROVIDER_MODE",
                default=PROVIDER_MODE_FIXED_WINDOW,
                choices=PROVIDER_MODES,
            ),
            start_orbit=_env_int("AGENTMEMORY_PROCEDURAL_START_ORBIT", 0),
        )
        self._initialize_wrapper_state()

    def _initialize_native_training_runtime(
        self,
        *,
        forbidden_env_keys: tuple[str, ...],
    ) -> None:
        if os.environ.get("AGENTMEMORY_SURFACE") != self.surface:
            raise RuntimeError(
                f"AGENTMEMORY_SURFACE must be explicitly set to {self.surface!r}."
            )
        forbidden = sorted(
            key
            for key in forbidden_env_keys
            if os.environ.get(key)
        )
        if forbidden:
            raise RuntimeError(
                "Programmatic training refuses frozen-evaluation or manual-gate "
                "inputs: "
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

        self.items_file = _required_file("MEMORYARENA_WEBSHOP_ITEMS_FILE")
        self.attributes_file = _required_file("MEMORYARENA_WEBSHOP_ATTR_FILE")
        self.search_root = _required_directory("MEMORYARENA_WEBSHOP_SEARCH_ROOT")
        self.lucene_manifest = _required_file("MEMORYARENA_LUCENE_INDEX_MANIFEST")
        self.backend = MemoryArenaNativeWebShopBackend(
            memoryarena_root=_required_directory("MEMORYARENA_ROOT"),
            items_file=self.items_file,
            attributes_file=self.attributes_file,
            search_root=self.search_root,
            java_home=_required_directory("MEMORYARENA_WEBSHOP_JAVA_HOME"),
            expected_memoryarena_commit=_required_env("MEMORYARENA_BASE_COMMIT"),
            price_seed=_env_int("AGENTMEMORY_WEBSHOP_PRICE_SEED", 233),
        )

    def _initialize_wrapper_state(self) -> None:
        self.max_id = 0
        self.envs: dict[int, MemoryArenaWebShopEnv] = {}
        self.info: dict[int, dict[str, Any]] = {}
        self.env_locks: dict[int, threading.RLock] = {}
        self.lock = threading.RLock()

    def create(self) -> dict[str, Any]:
        with self.lock:
            env_id = self.max_id
            self.max_id += 1
            env = self.environment_type(
                provider=self.provider,
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
            # Client creation is transport setup, not task selection.  The
            # rollout sends the authoritative absolute index in a later reset.
            # Keeping this bootstrap reset at zero also prevents monotonically
            # increasing environment ids from escaping a bounded eval window.
            observation, info = env.reset(data_idx=0)
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
        provider_metadata = self.provider.metadata()
        return {
            "surface": self.surface,
            "source": "agentmemory_programmatic_generator",
            "paper_eligible": False,
            "task_count": self.provider.task_count,
            "provider_mode": self.provider.mode,
            "accepted_index_domain": provider_metadata["accepted_index_domain"],
            "provider": provider_metadata,
            "reward_contract": dict(self.reward_contract),
            "ltm_inventory_mode": self.ltm_inventory_mode,
            "ltm_transition_notice_mode": self.ltm_transition_notice_mode,
            "action_listing_mode": self.action_listing_mode,
            "memory_prompt_mode": self.memory_prompt_mode,
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


def _required_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {key}")
    return value


def _required_int(key: str) -> int:
    value = _required_env(key)
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer.") from exc


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be an integer.") from exc


def _required_file(key: str) -> Path:
    path = Path(_required_env(key)).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Required file does not exist for {key}: {path}")
    return path


def _required_directory(key: str) -> Path:
    path = Path(_required_env(key)).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"Required directory does not exist for {key}: {path}")
    return path


def _env_choice(key: str, *, default: str, choices: tuple[str, ...]) -> str:
    value = os.environ.get(key, default)
    if value not in choices:
        raise RuntimeError(f"{key} must be one of: {', '.join(choices)}")
    return value


def _env_nonnegative_float(key: str, default: float) -> float:
    value = os.environ.get(key)
    try:
        parsed = default if value is None else float(value)
    except ValueError as exc:
        raise RuntimeError(f"{key} must be a float.") from exc
    if parsed < 0:
        raise RuntimeError(f"{key} must be non-negative.")
    return parsed


def _require_equal_hash(name: str, *, expected: str, observed: str) -> None:
    if expected != observed:
        raise RuntimeError(
            f"Certified {name} SHA256 mismatch: expected {expected}, observed {observed}."
        )


def attest_procedural_runtime_inputs(
    pool: ProductPool,
    backend: NativeWebShopBackend,
    *,
    items_file: Path,
    attributes_file: Path,
    search_root: Path,
    lucene_manifest: Path,
) -> None:
    """Fail closed if native runtime inputs differ from certification."""

    backend_metadata = backend.metadata()
    _require_equal_hash(
        "product catalog",
        expected=pool.catalog_sha256,
        observed=file_sha256(items_file),
    )
    _require_equal_hash(
        "product attributes",
        expected=pool.attributes_sha256,
        observed=file_sha256(attributes_file),
    )
    _require_equal_hash(
        "native price table",
        expected=pool.price_table_sha256,
        observed=str(backend_metadata["price_table_sha256"]),
    )
    _require_equal_hash(
        "Lucene index manifest",
        expected=pool.lucene_index_sha256,
        observed=file_sha256(lucene_manifest),
    )
    verify_lucene_index_manifest(
        lucene_manifest,
        index_dir=search_root / "indexes-full",
    )
    for product in pool.products:
        record = backend.product_record(product.asin)
        if backend.product_title(product.asin) != product.title:
            raise RuntimeError(
                f"Certified title no longer matches native product {product.asin}."
            )
        if backend.product_price_cents(product.asin) != product.price_cents:
            raise RuntimeError(
                f"Certified price no longer matches native product {product.asin}."
            )
        _require_equal_hash(
            f"catalog record for {product.asin}",
            expected=product.catalog_record_sha256,
            observed=backend.product_record_sha256(product.asin),
        )
        try:
            classification = require_unique_product_classification(
                record,
                scenario_ids=pool.scenario_ids,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"Certified natural classification no longer holds for {product.asin}."
            ) from exc
        if (
            classification.scenario_id != product.scenario_id
            or classification.slot_id != product.slot_id
            or classification.attribute_name != product.attribute_name
            or classification.attribute_value != product.attribute_value
            or classification.semantic_sha256 != product.classification_sha256
        ):
            raise RuntimeError(
                f"Certified natural attribute no longer matches native product {product.asin}."
            )
