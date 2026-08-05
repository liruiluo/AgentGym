from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from .generator import NaturalAttributeChainGenerator
from .question_format import QUESTION_FORMAT_VERSION
from .schema import (
    SPLITS,
    CounterfactualOrbit,
    ProceduralMemoryBundle,
    ProceduralMemoryDataError,
    canonical_json_bytes,
)
from .verifier import OrbitProof, verify_counterfactual_orbit


PROVIDER_MODE_FIXED_WINDOW = "fixed_window"
PROVIDER_MODE_RESEEDED_STREAM = "reseeded_stream"
PROVIDER_MODES = (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
)


class VerifiedProceduralBundleProvider:
    """Serve deterministic tasks from a fixed window or an open training stream.

    ``task_count`` is always the virtual dataset length exposed to AgentGym. A
    fixed window rejects indices outside that range. A reseeded stream accepts
    every non-negative absolute index and exhausts the generator's complete
    collision-free semantic period before deriving the next deterministic seed.
    Both counterfactual branches always stay in the same seed epoch.
    """

    def __init__(
        self,
        *,
        generator: NaturalAttributeChainGenerator,
        split: str,
        task_count: int,
        mode: str = PROVIDER_MODE_FIXED_WINDOW,
        start_orbit: int = 0,
        cache_orbits: int = 256,
    ) -> None:
        if split not in SPLITS:
            raise ProceduralMemoryDataError(
                f"invalid provider split {split!r}; expected one of {SPLITS}."
            )
        if (
            isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count <= 0
            or task_count % 2
        ):
            raise ProceduralMemoryDataError(
                "task_count must be a positive even integer so every orbit is paired."
            )
        if mode not in PROVIDER_MODES:
            raise ProceduralMemoryDataError(
                f"invalid provider mode {mode!r}; expected one of {PROVIDER_MODES}."
            )
        if (
            isinstance(start_orbit, bool)
            or not isinstance(start_orbit, int)
            or start_orbit < 0
        ):
            raise ProceduralMemoryDataError("start_orbit must be non-negative.")
        if (
            isinstance(cache_orbits, bool)
            or not isinstance(cache_orbits, int)
            or cache_orbits < 1
        ):
            raise ProceduralMemoryDataError("cache_orbits must be positive.")
        if mode == PROVIDER_MODE_FIXED_WINDOW:
            if start_orbit + task_count // 2 > generator.semantic_period_orbits:
                raise ProceduralMemoryDataError(
                    "requested fixed window exceeds the collision-free semantic "
                    "task period."
                )
        else:
            if split != "train":
                raise ProceduralMemoryDataError(
                    "reseeded_stream is training-only; dev/test must use fixed_window."
                )
            if start_orbit != 0:
                raise ProceduralMemoryDataError(
                    "reseeded_stream uses absolute stream indices and requires "
                    "start_orbit=0."
                )
        self.generator = generator
        self.split = split
        self.task_count = task_count
        self.mode = mode
        self.start_orbit = start_orbit
        self.cache_orbits = cache_orbits
        self._cache: OrderedDict[
            tuple[int, int], tuple[CounterfactualOrbit, OrbitProof]
        ] = OrderedDict()
        self._seed_epoch_generators: OrderedDict[
            int, NaturalAttributeChainGenerator
        ] = (
            OrderedDict()
        )
        self._lock = threading.RLock()

    @property
    def orbit_count(self) -> int:
        return self.task_count // 2

    @property
    def seed_epoch_orbit_count(self) -> int:
        return self.generator.semantic_period_orbits

    @property
    def seed_epoch_task_count(self) -> int:
        return self.generator.semantic_period_tasks

    def get(self, data_idx: int) -> ProceduralMemoryBundle:
        self._validate_data_idx(data_idx)
        stream_orbit_index, branch_index = divmod(data_idx, 2)
        orbit, proof = self._verified_orbit(stream_orbit_index)
        task = orbit.tasks[branch_index]
        return ProceduralMemoryBundle(
            task_id=task.task_id,
            questions=task.questions,
            target_asins=task.target_asins,
            target_attribute_values=task.target_attribute_values,
            budget_cents=task.budget_cents,
            split=task.split,
            orbit_id=task.orbit_id,
            scenario_id=task.scenario_id,
            proof_sha256=proof.proof_sha256,
            generator_version=task.generator_version,
            product_pool_sha256=task.product_pool_sha256,
        )

    def proof_for_index(self, data_idx: int) -> OrbitProof:
        self._validate_data_idx(data_idx)
        return self._verified_orbit(data_idx // 2)[1]

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "agentmemory_verified_natural_chain_provider_v4",
            "split": self.split,
            "provider_mode": self.mode,
            "task_count": self.task_count,
            "virtual_task_count": self.task_count,
            "orbit_count": self.orbit_count,
            "tasks_per_orbit": 2,
            "accepted_index_domain": (
                "all_nonnegative_integers"
                if self.mode == PROVIDER_MODE_RESEEDED_STREAM
                else f"0_to_{self.task_count - 1}_inclusive"
            ),
            "on_demand_generation": True,
            "scenario_ids": list(self.generator.pool.scenario_ids),
            "generator_version": self.generator.version,
            "question_format_version": QUESTION_FORMAT_VERSION,
            "generator_base_seed": self.generator.seed,
            "product_pool_sha256": self.generator.pool.semantic_sha256,
            "products_per_attribute_cell": self.generator.pool.products_per_cell,
            "candidate_count_per_phase": 2,
            "task_prompt_product_identity": "complete_native_title",
            "target_asin_in_task_prompt": False,
            "native_search_result_asin_handles_visible": True,
            "native_click_action_uses_asin_handle": True,
            "purchase_receipt_asin_verification": True,
            "catalog_wide_normalized_title_uniqueness": True,
            "phase_count_per_task": 6,
            "semantic_period_orbits": self.generator.semantic_period_orbits,
            "semantic_period_tasks": self.generator.semantic_period_tasks,
            "conservative_task_capacity_without_candidate_order": (
                self.generator.conservative_task_capacity_without_candidate_order
            ),
            "memory_dependency": "previous_purchased_natural_attribute",
            "order_specific_bijection_count_per_task": 5,
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
                    "counterfactual_pair_never_crosses_seed_epoch": True,
                    "seed_epoch_zero_uses_base_seed": True,
                    "later_seed_epoch_derivation": "sha256_v2",
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

    def _verified_orbit(
        self, stream_orbit_index: int
    ) -> tuple[CounterfactualOrbit, OrbitProof]:
        with self._lock:
            cache_key, generator, source_orbit_index = self._source_orbit(
                stream_orbit_index
            )
            cached = self._cache.pop(cache_key, None)
            if cached is not None:
                self._cache[cache_key] = cached
                return cached
            orbit = generator.generate_orbit(source_orbit_index, split=self.split)
            proof = verify_counterfactual_orbit(
                orbit,
                pool=generator.pool,
                expected_generator_version=generator.version,
                expected_generator_seed=generator.seed,
            )
            value = (orbit, proof)
            self._cache[cache_key] = value
            while len(self._cache) > self.cache_orbits:
                self._cache.popitem(last=False)
            return value

    def _validate_data_idx(self, data_idx: int) -> None:
        invalid_type_or_sign = (
            isinstance(data_idx, bool)
            or not isinstance(data_idx, int)
            or data_idx < 0
        )
        outside_fixed_window = (
            self.mode == PROVIDER_MODE_FIXED_WINDOW
            and isinstance(data_idx, int)
            and not isinstance(data_idx, bool)
            and data_idx >= self.task_count
        )
        if invalid_type_or_sign or outside_fixed_window:
            domain = (
                f"[0, {self.task_count})"
                if self.mode == PROVIDER_MODE_FIXED_WINDOW
                else "the non-negative integers"
            )
            raise IndexError(
                f"procedural data_idx {data_idx!r} is outside {domain}."
            )

    def _source_orbit(
        self,
        stream_orbit_index: int,
    ) -> tuple[tuple[int, int], NaturalAttributeChainGenerator, int]:
        if self.mode == PROVIDER_MODE_FIXED_WINDOW:
            source_orbit_index = self.start_orbit + stream_orbit_index
            return (-1, source_orbit_index), self.generator, source_orbit_index

        seed_epoch_index, source_orbit_index = divmod(
            stream_orbit_index,
            self.seed_epoch_orbit_count,
        )
        return (
            (seed_epoch_index, source_orbit_index),
            self._generator_for_seed_epoch(seed_epoch_index),
            source_orbit_index,
        )

    def _generator_for_seed_epoch(
        self,
        seed_epoch_index: int,
    ) -> NaturalAttributeChainGenerator:
        if seed_epoch_index == 0:
            return self.generator
        cached = self._seed_epoch_generators.pop(seed_epoch_index, None)
        if cached is not None:
            self._seed_epoch_generators[seed_epoch_index] = cached
            return cached
        seed = int.from_bytes(
            hashlib.sha256(
                canonical_json_bytes(
                    {
                        "schema": "agentmemory_reseeded_stream_seed_epoch_v2",
                        "generator_version": self.generator.version,
                        "generator_base_seed": self.generator.seed,
                        "product_pool_sha256": self.generator.pool.semantic_sha256,
                        "split": self.split,
                        "seed_epoch_index": seed_epoch_index,
                    }
                )
            ).digest(),
            "big",
        )
        value = NaturalAttributeChainGenerator(
            pool=self.generator.pool,
            seed=seed,
            version=self.generator.version,
        )
        self._seed_epoch_generators[seed_epoch_index] = value
        while len(self._seed_epoch_generators) > 8:
            self._seed_epoch_generators.popitem(last=False)
        return value
