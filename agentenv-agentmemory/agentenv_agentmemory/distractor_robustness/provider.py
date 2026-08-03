from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from ..latent_preference.schema import canonical_json_bytes
from .generator import DistractorRobustnessGenerator
from .schema import (
    SPLITS,
    DistractorRobustnessBundle,
    DistractorRobustnessDataError,
    DistractorRobustnessOrbit,
)
from .verifier import (
    DistractorRobustnessOrbitProof,
    verify_distractor_robustness_orbit,
)


PROVIDER_MODE_FIXED_WINDOW = "fixed_window"
PROVIDER_MODE_RESEEDED_STREAM = "reseeded_stream"
PROVIDER_MODES = (PROVIDER_MODE_FIXED_WINDOW, PROVIDER_MODE_RESEEDED_STREAM)


class VerifiedDistractorRobustnessBundleProvider:
    """Serve proof-carrying clean/distracted pairs by absolute data index."""

    def __init__(
        self,
        *,
        generator: DistractorRobustnessGenerator,
        split: str,
        task_count: int,
        mode: str = PROVIDER_MODE_FIXED_WINDOW,
        start_orbit: int = 0,
        cache_orbits: int = 256,
    ) -> None:
        if split not in SPLITS:
            raise DistractorRobustnessDataError("invalid provider split.")
        if (
            isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count <= 0
            or task_count % 2
        ):
            raise DistractorRobustnessDataError(
                "task_count must be a positive even integer."
            )
        if mode not in PROVIDER_MODES:
            raise DistractorRobustnessDataError("invalid provider mode.")
        if (
            isinstance(start_orbit, bool)
            or not isinstance(start_orbit, int)
            or start_orbit < 0
        ):
            raise DistractorRobustnessDataError("start_orbit must be non-negative.")
        if (
            isinstance(cache_orbits, bool)
            or not isinstance(cache_orbits, int)
            or cache_orbits < 1
        ):
            raise DistractorRobustnessDataError("cache_orbits must be positive.")
        if mode == PROVIDER_MODE_FIXED_WINDOW:
            if start_orbit + task_count // 2 > generator.semantic_period_orbits:
                raise DistractorRobustnessDataError(
                    "fixed window exceeds the collision-free semantic period."
                )
        elif split != "train" or start_orbit != 0:
            raise DistractorRobustnessDataError(
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
            tuple[DistractorRobustnessOrbit, DistractorRobustnessOrbitProof],
        ] = OrderedDict()
        self._seed_epoch_generators: OrderedDict[
            int, DistractorRobustnessGenerator
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

    def get(self, data_idx: int) -> DistractorRobustnessBundle:
        self._validate_data_idx(data_idx)
        stream_orbit, branch = divmod(data_idx, 2)
        orbit, proof = self._verified_orbit(stream_orbit)
        task = orbit.tasks[branch]
        return DistractorRobustnessBundle(
            task_id=task.task_id,
            questions=task.questions,
            target_asins=task.target_asins,
            budget_cents=task.budget_cents,
            split=task.split,
            orbit_id=task.orbit_id,
            branch_kind=task.branch_kind,
            initial_memories=task.initial_memories,
            canonical_query=task.canonical_query,
            proof_sha256=proof.proof_sha256,
            generator_version=task.generator_version,
            product_pool_sha256=task.product_pool_sha256,
        )

    def proof_for_index(self, data_idx: int) -> DistractorRobustnessOrbitProof:
        self._validate_data_idx(data_idx)
        return self._verified_orbit(data_idx // 2)[1]

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "agentmemory_verified_distractor_robustness_provider_v1",
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
            "generator_version": self.generator.version,
            "generator_base_seed": self.generator.seed,
            "product_pool_sha256": self.generator.pool.semantic_sha256,
            "phase_count_per_task": 6,
            "candidate_count_per_phase": 2,
            "counterfactual_pairing": True,
            "branch_order": ["clean", "distracted"],
            "preloaded_distractor_count": self.generator.distractor_count,
            "correct_memory_preloaded": False,
            "correct_memory_policy_authored_after_evidence": True,
            "retrieve_policy": "query_top1",
            "memory_id_lookup_allowed": False,
            "initial_memory_inventory_visible": False,
            "strict_top1_certified": True,
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
            proof = verify_distractor_robustness_orbit(
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
                f"distractor robustness data_idx {data_idx!r} is outside {domain}."
            )

    def _source_orbit(self, stream_orbit_index: int):
        if self.mode == PROVIDER_MODE_FIXED_WINDOW:
            source = self.start_orbit + stream_orbit_index
            return (-1, source), self.generator, source
        epoch, source = divmod(stream_orbit_index, self.seed_epoch_orbit_count)
        return (epoch, source), self._generator_for_epoch(epoch), source

    def _generator_for_epoch(self, epoch: int) -> DistractorRobustnessGenerator:
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
                        "schema": "agentmemory_distractor_robustness_seed_epoch_v1",
                        "generator_version": self.generator.version,
                        "generator_base_seed": self.generator.seed,
                        "distractor_count": self.generator.distractor_count,
                        "product_pool_sha256": self.generator.pool.semantic_sha256,
                        "split": self.split,
                        "seed_epoch_index": epoch,
                    }
                )
            ).digest(),
            "big",
        )
        value = DistractorRobustnessGenerator(
            pool=self.generator.pool,
            seed=seed,
            distractor_count=self.generator.distractor_count,
            version=self.generator.version,
        )
        self._seed_epoch_generators[epoch] = value
        while len(self._seed_epoch_generators) > 8:
            self._seed_epoch_generators.popitem(last=False)
        return value
