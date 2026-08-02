from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from .generator import LatentPreferenceGenerator
from .schema import (
    SPLITS,
    LatentPreferenceBundle,
    LatentPreferenceDataError,
    LatentPreferenceOrbit,
    canonical_json_bytes,
)
from .verifier import LatentPreferenceOrbitProof, verify_latent_preference_orbit


PROVIDER_MODE_FIXED_WINDOW = "fixed_window"
PROVIDER_MODE_RESEEDED_STREAM = "reseeded_stream"
PROVIDER_MODES = (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
)


class VerifiedLatentPreferenceBundleProvider:
    """Serve proof-carrying counterfactual tasks by absolute dataset index."""

    def __init__(
        self,
        *,
        generator: LatentPreferenceGenerator,
        split: str,
        task_count: int,
        mode: str = PROVIDER_MODE_FIXED_WINDOW,
        start_orbit: int = 0,
        cache_orbits: int = 256,
    ) -> None:
        if split not in SPLITS:
            raise LatentPreferenceDataError(
                f"invalid provider split {split!r}; expected one of {SPLITS}."
            )
        if (
            isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count <= 0
            or task_count % 2
        ):
            raise LatentPreferenceDataError(
                "task_count must be a positive even integer so every orbit is paired."
            )
        if mode not in PROVIDER_MODES:
            raise LatentPreferenceDataError(
                f"invalid provider mode {mode!r}; expected one of {PROVIDER_MODES}."
            )
        if (
            isinstance(start_orbit, bool)
            or not isinstance(start_orbit, int)
            or start_orbit < 0
        ):
            raise LatentPreferenceDataError("start_orbit must be non-negative.")
        if (
            isinstance(cache_orbits, bool)
            or not isinstance(cache_orbits, int)
            or cache_orbits < 1
        ):
            raise LatentPreferenceDataError("cache_orbits must be positive.")
        if mode == PROVIDER_MODE_FIXED_WINDOW:
            if start_orbit + task_count // 2 > generator.semantic_period_orbits:
                raise LatentPreferenceDataError(
                    "requested fixed window exceeds the collision-free semantic period."
                )
        else:
            if split != "train":
                raise LatentPreferenceDataError(
                    "reseeded_stream is training-only; dev/test use fixed windows."
                )
            if start_orbit != 0:
                raise LatentPreferenceDataError(
                    "reseeded_stream requires start_orbit=0."
                )

        self.generator = generator
        self.split = split
        self.task_count = task_count
        self.mode = mode
        self.start_orbit = start_orbit
        self.cache_orbits = cache_orbits
        self._cache: OrderedDict[
            tuple[int, int], tuple[LatentPreferenceOrbit, LatentPreferenceOrbitProof]
        ] = OrderedDict()
        self._seed_epoch_generators: OrderedDict[
            int, LatentPreferenceGenerator
        ] = OrderedDict()
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

    def get(self, data_idx: int) -> LatentPreferenceBundle:
        self._validate_data_idx(data_idx)
        stream_orbit_index, branch_index = divmod(data_idx, 2)
        orbit, proof = self._verified_orbit(stream_orbit_index)
        task = orbit.tasks[branch_index]
        recipe = self.generator.pool.recipe_by_id(task.recipe_id)
        return LatentPreferenceBundle(
            task_id=task.task_id,
            questions=task.questions,
            target_asins=task.target_asins,
            budget_cents=task.budget_cents,
            split=task.split,
            orbit_id=task.orbit_id,
            recipe_id=task.recipe_id,
            user_id=task.user_id,
            preference_axis=recipe.axis,
            supporting_evidence_count=task.supporting_evidence_count,
            proof_sha256=proof.proof_sha256,
            generator_version=task.generator_version,
            product_pool_sha256=task.product_pool_sha256,
        )

    def proof_for_index(self, data_idx: int) -> LatentPreferenceOrbitProof:
        self._validate_data_idx(data_idx)
        return self._verified_orbit(data_idx // 2)[1]

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "agentmemory_verified_latent_preference_provider_v1",
            "split": self.split,
            "provider_mode": self.mode,
            "task_count": self.task_count,
            "virtual_task_count": self.task_count,
            "orbit_count": self.orbit_count,
            "accepted_index_domain": (
                "all_nonnegative_integers"
                if self.mode == PROVIDER_MODE_RESEEDED_STREAM
                else f"0_to_{self.task_count - 1}_inclusive"
            ),
            "on_demand_generation": True,
            "recipe_ids": [
                recipe.recipe_id for recipe in self.generator.pool.recipes
            ],
            "generator_version": self.generator.version,
            "generator_base_seed": self.generator.seed,
            "product_pool_sha256": self.generator.pool.semantic_sha256,
            "products_per_attribute_cell": (
                self.generator.pool.products_per_cell
            ),
            "phase_count_per_task": 6,
            "candidate_count_per_phase": 2,
            "supporting_evidence_counts": [1, 2, 3],
            "resolution_step": 1,
            "preference_hypothesis": "one_value_on_one_natural_attribute_axis",
            "counterfactual_pairing": True,
            "application_observation_identity": True,
            "application_target_flip": True,
            "task_prompt_product_identity": "complete_native_title",
            "target_asin_in_task_prompt": False,
            "native_search_result_asin_handles_visible": True,
            "native_click_action_uses_asin_handle": True,
            "purchase_receipt_asin_verification": True,
            "semantic_period_orbits": self.generator.semantic_period_orbits,
            "semantic_period_tasks": self.generator.semantic_period_tasks,
            "conservative_task_capacity_without_candidate_order": (
                self.generator.conservative_task_capacity_without_candidate_order
            ),
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

    def _verified_orbit(
        self,
        stream_orbit_index: int,
    ) -> tuple[LatentPreferenceOrbit, LatentPreferenceOrbitProof]:
        with self._lock:
            cache_key, generator, source_orbit_index = self._source_orbit(
                stream_orbit_index
            )
            cached = self._cache.pop(cache_key, None)
            if cached is not None:
                self._cache[cache_key] = cached
                return cached
            orbit = generator.generate_orbit(source_orbit_index, split=self.split)
            proof = verify_latent_preference_orbit(
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
        invalid = (
            isinstance(data_idx, bool)
            or not isinstance(data_idx, int)
            or data_idx < 0
        )
        outside_fixed = (
            self.mode == PROVIDER_MODE_FIXED_WINDOW
            and isinstance(data_idx, int)
            and not isinstance(data_idx, bool)
            and data_idx >= self.task_count
        )
        if invalid or outside_fixed:
            domain = (
                f"[0, {self.task_count})"
                if self.mode == PROVIDER_MODE_FIXED_WINDOW
                else "the non-negative integers"
            )
            raise IndexError(
                f"latent preference data_idx {data_idx!r} is outside {domain}."
            )

    def _source_orbit(
        self,
        stream_orbit_index: int,
    ) -> tuple[tuple[int, int], LatentPreferenceGenerator, int]:
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
    ) -> LatentPreferenceGenerator:
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
                        "schema": "agentmemory_latent_preference_seed_epoch_v1",
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
        value = LatentPreferenceGenerator(
            pool=self.generator.pool,
            seed=seed,
            version=self.generator.version,
        )
        self._seed_epoch_generators[seed_epoch_index] = value
        while len(self._seed_epoch_generators) > 8:
            self._seed_epoch_generators.popitem(last=False)
        return value
