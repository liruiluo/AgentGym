from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from ..latent_preference.schema import canonical_json_bytes
from .generator import RecencyOverrideGenerator
from .schema import RecencyOverrideBundle, RecencyOverrideDataError, RecencyOverrideOrbit
from .verifier import RecencyOverrideOrbitProof, verify_recency_override_orbit


PROVIDER_MODE_FIXED_WINDOW = "fixed_window"
PROVIDER_MODE_RESEEDED_STREAM = "reseeded_stream"
PROVIDER_MODES = (PROVIDER_MODE_FIXED_WINDOW, PROVIDER_MODE_RESEEDED_STREAM)


class VerifiedRecencyOverrideBundleProvider:
    """Serve proof-carrying recency tasks by absolute data index."""

    def __init__(self, *, generator: RecencyOverrideGenerator, split: str, task_count: int, mode: str = PROVIDER_MODE_FIXED_WINDOW, start_orbit: int = 0, cache_orbits: int = 256) -> None:
        if split not in ("train", "dev", "test"):
            raise RecencyOverrideDataError("invalid provider split.")
        if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count <= 0 or task_count % 2:
            raise RecencyOverrideDataError("task_count must be a positive even integer.")
        if mode not in PROVIDER_MODES:
            raise RecencyOverrideDataError("invalid provider mode.")
        if isinstance(start_orbit, bool) or not isinstance(start_orbit, int) or start_orbit < 0:
            raise RecencyOverrideDataError("start_orbit must be non-negative.")
        if mode == PROVIDER_MODE_FIXED_WINDOW and start_orbit + task_count // 2 > generator.semantic_period_orbits:
            raise RecencyOverrideDataError("fixed window exceeds semantic period.")
        if mode == PROVIDER_MODE_RESEEDED_STREAM and (split != "train" or start_orbit != 0):
            raise RecencyOverrideDataError("reseeded_stream is train-only and starts at orbit zero.")
        if isinstance(cache_orbits, bool) or not isinstance(cache_orbits, int) or cache_orbits < 1:
            raise RecencyOverrideDataError("cache_orbits must be positive.")
        self.generator = generator
        self.split = split
        self.task_count = task_count
        self.mode = mode
        self.start_orbit = start_orbit
        self.cache_orbits = cache_orbits
        self._cache: OrderedDict[tuple[int, int], tuple[RecencyOverrideOrbit, RecencyOverrideOrbitProof]] = OrderedDict()
        self._seed_epoch_generators: OrderedDict[int, RecencyOverrideGenerator] = OrderedDict()
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

    def get(self, data_idx: int) -> RecencyOverrideBundle:
        self._validate_data_idx(data_idx)
        stream_orbit, branch = divmod(data_idx, 2)
        orbit, proof = self._verified_orbit(stream_orbit)
        task = orbit.tasks[branch]
        return RecencyOverrideBundle(
            task_id=task.task_id,
            questions=task.questions,
            target_asins=task.target_asins,
            active_values=task.active_values,
            budget_cents=task.budget_cents,
            split=task.split,
            orbit_id=task.orbit_id,
            recipe_id=task.recipe_id,
            user_id=task.user_id,
            branch_kind=task.branch_kind,
            old_attribute_value=task.old_attribute_value,
            new_attribute_value=task.new_attribute_value,
            canonical_memory_key=task.canonical_memory_key,
            override_mode=task.override_mode,
            proof_sha256=proof.proof_sha256,
            generator_version=task.generator_version,
            product_pool_sha256=task.product_pool_sha256,
        )

    def proof_for_index(self, data_idx: int) -> RecencyOverrideOrbitProof:
        self._validate_data_idx(data_idx)
        return self._verified_orbit(data_idx // 2)[1]

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "agentmemory_verified_recency_override_provider_v1",
            "split": self.split,
            "provider_mode": self.mode,
            "task_count": self.task_count,
            "virtual_task_count": self.task_count,
            "orbit_count": self.orbit_count,
            "accepted_index_domain": "all_nonnegative_integers" if self.mode == PROVIDER_MODE_RESEEDED_STREAM else f"0_to_{self.task_count - 1}_inclusive",
            "on_demand_generation": True,
            "recipe_ids": [recipe.recipe_id for recipe in self.generator.pool.recipes],
            "generator_version": self.generator.version,
            "generator_base_seed": self.generator.seed,
            "product_pool_sha256": self.generator.pool.semantic_sha256,
            "phase_count_per_task": 6,
            "candidate_count_per_phase": 2,
            "phase_schedule": ["evidence", "application", "override", "application", "application", "application"],
            "override_phase_index": 2,
            "canonical_memory_key": "user_preference",
            "counterfactual_pairing": True,
            "stay_branch": "old preference remains active",
            "flip_branch": "new preference replaces old canonical state",
            "update_contract": "UPDATE same memory_id or DELETE old then ADD new",
            "application_observation_identity": True,
            "application_target_flip": True,
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

    def _verified_orbit(self, stream_orbit_index: int):
        with self._lock:
            key, generator, source_index = self._source_orbit(stream_orbit_index)
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached
                return cached
            orbit = generator.generate_orbit(source_index, split=self.split)
            proof = verify_recency_override_orbit(orbit, pool=generator.pool, expected_generator_version=generator.version, expected_generator_seed=generator.seed)
            value = (orbit, proof)
            self._cache[key] = value
            while len(self._cache) > self.cache_orbits:
                self._cache.popitem(last=False)
            return value

    def _validate_data_idx(self, data_idx: int) -> None:
        invalid = isinstance(data_idx, bool) or not isinstance(data_idx, int) or data_idx < 0
        outside = self.mode == PROVIDER_MODE_FIXED_WINDOW and isinstance(data_idx, int) and not isinstance(data_idx, bool) and data_idx >= self.task_count
        if invalid or outside:
            domain = f"[0, {self.task_count})" if self.mode == PROVIDER_MODE_FIXED_WINDOW else "the non-negative integers"
            raise IndexError(f"recency override data_idx {data_idx!r} is outside {domain}.")

    def _source_orbit(self, stream_orbit_index: int):
        if self.mode == PROVIDER_MODE_FIXED_WINDOW:
            source = self.start_orbit + stream_orbit_index
            return (-1, source), self.generator, source
        epoch, source = divmod(stream_orbit_index, self.seed_epoch_orbit_count)
        return (epoch, source), self._generator_for_epoch(epoch), source

    def _generator_for_epoch(self, epoch: int) -> RecencyOverrideGenerator:
        if epoch == 0:
            return self.generator
        cached = self._seed_epoch_generators.pop(epoch, None)
        if cached is not None:
            self._seed_epoch_generators[epoch] = cached
            return cached
        seed = int.from_bytes(hashlib.sha256(canonical_json_bytes({"schema": "agentmemory_recency_override_seed_epoch_v1", "generator_version": self.generator.version, "generator_base_seed": self.generator.seed, "product_pool_sha256": self.generator.pool.semantic_sha256, "split": self.split, "seed_epoch_index": epoch})).digest(), "big")
        value = RecencyOverrideGenerator(pool=self.generator.pool, seed=seed, version=self.generator.version)
        self._seed_epoch_generators[epoch] = value
        while len(self._seed_epoch_generators) > 8:
            self._seed_epoch_generators.popitem(last=False)
        return value
