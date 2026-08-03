from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from ..latent_preference.schema import canonical_json_bytes
from .generator import SelectiveMemoryUseGenerator
from .schema import (
    BRANCH_SPECS,
    SPLITS,
    SelectiveMemoryUseBundle,
    SelectiveMemoryUseDataError,
    SelectiveMemoryUseOrbit,
)
from .verifier import (
    SelectiveMemoryUseOrbitProof,
    verify_selective_memory_use_orbit,
)


PROVIDER_MODE_FIXED_WINDOW = "fixed_window"
PROVIDER_MODE_RESEEDED_STREAM = "reseeded_stream"
PROVIDER_MODES = (PROVIDER_MODE_FIXED_WINDOW, PROVIDER_MODE_RESEEDED_STREAM)
TASKS_PER_ORBIT = len(BRANCH_SPECS)


class VerifiedSelectiveMemoryUseBundleProvider:
    """Serve verified 2x2 memory-need/preference factorial orbits."""

    def __init__(
        self,
        *,
        generator: SelectiveMemoryUseGenerator,
        split: str,
        task_count: int,
        mode: str = PROVIDER_MODE_FIXED_WINDOW,
        start_orbit: int = 0,
        cache_orbits: int = 256,
    ) -> None:
        if split not in SPLITS:
            raise SelectiveMemoryUseDataError("invalid provider split.")
        if (
            isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count <= 0
            or task_count % TASKS_PER_ORBIT
        ):
            raise SelectiveMemoryUseDataError(
                "task_count must be a positive multiple of four."
            )
        if mode not in PROVIDER_MODES:
            raise SelectiveMemoryUseDataError("invalid provider mode.")
        if (
            isinstance(start_orbit, bool)
            or not isinstance(start_orbit, int)
            or start_orbit < 0
        ):
            raise SelectiveMemoryUseDataError("start_orbit must be non-negative.")
        if (
            isinstance(cache_orbits, bool)
            or not isinstance(cache_orbits, int)
            or cache_orbits < 1
        ):
            raise SelectiveMemoryUseDataError("cache_orbits must be positive.")
        orbit_count = task_count // TASKS_PER_ORBIT
        if mode == PROVIDER_MODE_FIXED_WINDOW:
            if start_orbit + orbit_count > generator.semantic_period_orbits:
                raise SelectiveMemoryUseDataError(
                    "fixed window exceeds the collision-free semantic period."
                )
        elif split != "train" or start_orbit != 0:
            raise SelectiveMemoryUseDataError(
                "reseeded_stream is train-only and starts at orbit zero."
            )
        self.generator = generator
        self.split = split
        self.task_count = task_count
        self.mode = mode
        self.start_orbit = start_orbit
        self.cache_orbits = cache_orbits
        self._cache: OrderedDict[
            tuple[int, int],
            tuple[SelectiveMemoryUseOrbit, SelectiveMemoryUseOrbitProof],
        ] = OrderedDict()
        self._seed_epoch_generators: OrderedDict[
            int, SelectiveMemoryUseGenerator
        ] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def orbit_count(self) -> int:
        return self.task_count // TASKS_PER_ORBIT

    @property
    def seed_epoch_orbit_count(self) -> int:
        return self.generator.semantic_period_orbits

    @property
    def seed_epoch_task_count(self) -> int:
        return self.generator.semantic_period_tasks

    def get(self, data_idx: int) -> SelectiveMemoryUseBundle:
        self._validate_data_idx(data_idx)
        stream_orbit, branch = divmod(data_idx, TASKS_PER_ORBIT)
        orbit, proof = self._verified_orbit(stream_orbit)
        task = orbit.tasks[branch]
        return SelectiveMemoryUseBundle(
            task_id=task.task_id,
            questions=task.questions,
            target_asins=task.target_asins,
            budget_cents=task.budget_cents,
            split=task.split,
            orbit_id=task.orbit_id,
            branch_kind=task.branch_kind,
            memory_requirement=task.memory_requirement,
            preferred_attribute_value=task.preferred_attribute_value,
            initial_memory=task.initial_memory,
            canonical_query=task.canonical_query,
            proof_sha256=proof.proof_sha256,
            generator_version=task.generator_version,
            product_pool_sha256=task.product_pool_sha256,
        )

    def proof_for_index(self, data_idx: int) -> SelectiveMemoryUseOrbitProof:
        self._validate_data_idx(data_idx)
        return self._verified_orbit(data_idx // TASKS_PER_ORBIT)[1]

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "agentmemory_verified_selective_memory_use_provider_v1",
            "split": self.split,
            "provider_mode": self.mode,
            "task_count": self.task_count,
            "virtual_task_count": self.task_count,
            "orbit_count": self.orbit_count,
            "tasks_per_orbit": TASKS_PER_ORBIT,
            "accepted_index_domain": (
                "all_nonnegative_integers"
                if self.mode == PROVIDER_MODE_RESEEDED_STREAM
                else f"0_to_{self.task_count - 1}_inclusive"
            ),
            "on_demand_generation": True,
            "generator_version": self.generator.version,
            "generator_base_seed": self.generator.seed,
            "product_pool_sha256": self.generator.pool.semantic_sha256,
            "phase_count_per_task": 6,
            "candidate_count_per_phase": 2,
            "factorial_branch_specs": [list(item) for item in BRANCH_SPECS],
            "memory_required_fraction": 0.5,
            "memory_not_required_fraction": 0.5,
            "required_branch_seeded_memory_state": "current",
            "not_required_branch_seeded_memory_state": "stale_opposite",
            "retrieve_policy": "query_top1",
            "memory_id_lookup_allowed": False,
            "ltm_inventory_visible": False,
            "memory_action_positive_shaping_allowed": False,
            "unnecessary_memory_action_penalty": -0.01,
            "memory_required_without_memory_counterfactually_ambiguous": True,
            "memory_not_required_current_request_explicit": True,
            "task_prompt_product_identity": "complete_native_title",
            "target_asin_in_task_prompt": False,
            "native_search_result_asin_handles_visible": True,
            "native_click_action_uses_asin_handle": True,
            "purchase_receipt_asin_verification": True,
            "semantic_period_orbits": self.generator.semantic_period_orbits,
            "semantic_period_tasks": self.generator.semantic_period_tasks,
            "human_review_required": False,
            "llm_judge_required": False,
            "paper_eligible": False,
            "fixed_window": (
                {
                    "start_orbit": self.start_orbit,
                    "end_orbit_exclusive": self.start_orbit + self.orbit_count,
                }
                if self.mode == PROVIDER_MODE_FIXED_WINDOW
                else None
            ),
            "reseeded_stream": (
                {
                    "tasks_per_seed_epoch": self.seed_epoch_task_count,
                    "orbits_per_seed_epoch": self.seed_epoch_orbit_count,
                    "factorial_orbit_never_crosses_seed_epoch": True,
                    "seed_epoch_zero_uses_base_seed": True,
                    "later_seed_epoch_derivation": "sha256_v1",
                    "collision_free_within_complete_seed_epoch": True,
                    "semantic_uniqueness_guaranteed_through_task_index": (
                        self.seed_epoch_task_count - 1
                    ),
                    "cross_seed_epoch_semantic_uniqueness_guaranteed": False,
                }
                if self.mode == PROVIDER_MODE_RESEEDED_STREAM
                else None
            ),
        }

    def _verified_orbit(self, stream_orbit_index: int):
        with self._lock:
            key, generator, source_index = self._source_orbit(stream_orbit_index)
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached
                return cached
            orbit = generator.generate_orbit(source_index, split=self.split)
            proof = verify_selective_memory_use_orbit(
                orbit,
                pool=generator.pool,
                expected_generator_version=generator.version,
                expected_generator_seed=generator.seed,
            )
            value = (orbit, proof)
            self._cache[key] = value
            while len(self._cache) > self.cache_orbits:
                self._cache.popitem(last=False)
            return value

    def _validate_data_idx(self, data_idx: int) -> None:
        invalid = (
            isinstance(data_idx, bool)
            or not isinstance(data_idx, int)
            or data_idx < 0
        )
        outside = (
            self.mode == PROVIDER_MODE_FIXED_WINDOW
            and isinstance(data_idx, int)
            and not isinstance(data_idx, bool)
            and data_idx >= self.task_count
        )
        if invalid or outside:
            domain = (
                f"[0, {self.task_count})"
                if self.mode == PROVIDER_MODE_FIXED_WINDOW
                else "the non-negative integers"
            )
            raise IndexError(
                f"selective memory use data_idx {data_idx!r} is outside {domain}."
            )

    def _source_orbit(self, stream_orbit_index: int):
        if self.mode == PROVIDER_MODE_FIXED_WINDOW:
            source = self.start_orbit + stream_orbit_index
            return (-1, source), self.generator, source
        epoch, source = divmod(stream_orbit_index, self.seed_epoch_orbit_count)
        return (epoch, source), self._generator_for_epoch(epoch), source

    def _generator_for_epoch(self, epoch: int) -> SelectiveMemoryUseGenerator:
        if epoch == 0:
            return self.generator
        cached = self._seed_epoch_generators.pop(epoch, None)
        if cached is not None:
            self._seed_epoch_generators[epoch] = cached
            return cached
        seed = int.from_bytes(
            hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": "agentmemory_selective_memory_use_seed_epoch_v1",
                        "generator_version": self.generator.version,
                        "generator_base_seed": self.generator.seed,
                        "product_pool_sha256": self.generator.pool.semantic_sha256,
                        "split": self.split,
                        "seed_epoch_index": epoch,
                    }
                )
            ).digest(),
            "big",
        )
        value = SelectiveMemoryUseGenerator(
            pool=self.generator.pool,
            seed=seed,
            version=self.generator.version,
        )
        self._seed_epoch_generators[epoch] = value
        while len(self._seed_epoch_generators) > 8:
            self._seed_epoch_generators.popitem(last=False)
        return value
