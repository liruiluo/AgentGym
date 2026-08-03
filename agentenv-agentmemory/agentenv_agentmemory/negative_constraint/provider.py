from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

from ..latent_preference.schema import canonical_json_bytes
from .generator import NegativeConstraintGenerator
from .schema import (
    SPLITS,
    NegativeConstraintBundle,
    NegativeConstraintDataError,
    NegativeConstraintOrbit,
)
from .verifier import (
    NegativeConstraintOrbitProof,
    verify_negative_constraint_orbit,
)


PROVIDER_MODE_FIXED_WINDOW = "fixed_window"
PROVIDER_MODE_RESEEDED_STREAM = "reseeded_stream"
PROVIDER_MODES = (PROVIDER_MODE_FIXED_WINDOW, PROVIDER_MODE_RESEEDED_STREAM)
TASKS_PER_ORBIT = 3


class VerifiedNegativeConstraintBundleProvider:
    """Serve proof-carrying three-way negative-constraint tasks by data index."""

    def __init__(
        self,
        *,
        generator: NegativeConstraintGenerator,
        split: str,
        task_count: int,
        mode: str = PROVIDER_MODE_FIXED_WINDOW,
        start_orbit: int = 0,
        cache_orbits: int = 128,
    ) -> None:
        if split not in SPLITS:
            raise NegativeConstraintDataError("invalid provider split.")
        if (
            isinstance(task_count, bool)
            or not isinstance(task_count, int)
            or task_count <= 0
            or task_count % TASKS_PER_ORBIT
        ):
            raise NegativeConstraintDataError(
                "task_count must be a positive multiple of three."
            )
        if mode not in PROVIDER_MODES:
            raise NegativeConstraintDataError("invalid provider mode.")
        if (
            isinstance(start_orbit, bool)
            or not isinstance(start_orbit, int)
            or start_orbit < 0
        ):
            raise NegativeConstraintDataError("start_orbit must be non-negative.")
        if (
            isinstance(cache_orbits, bool)
            or not isinstance(cache_orbits, int)
            or cache_orbits < 1
        ):
            raise NegativeConstraintDataError("cache_orbits must be positive.")
        orbit_count = task_count // TASKS_PER_ORBIT
        if mode == PROVIDER_MODE_FIXED_WINDOW:
            if start_orbit + orbit_count > generator.semantic_period_orbits:
                raise NegativeConstraintDataError(
                    "fixed window exceeds the collision-free semantic period."
                )
        elif split != "train" or start_orbit != 0:
            raise NegativeConstraintDataError(
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
            tuple[NegativeConstraintOrbit, NegativeConstraintOrbitProof],
        ] = OrderedDict()
        self._seed_epoch_generators: OrderedDict[
            int,
            NegativeConstraintGenerator,
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

    def get(self, data_idx: int) -> NegativeConstraintBundle:
        self._validate_data_idx(data_idx)
        stream_orbit, branch = divmod(data_idx, TASKS_PER_ORBIT)
        orbit, proof = self._verified_orbit(stream_orbit)
        task = orbit.tasks[branch]
        return NegativeConstraintBundle(
            task_id=task.task_id,
            questions=task.questions,
            target_asins=task.target_asins,
            budget_cents=task.budget_cents,
            split=task.split,
            orbit_id=task.orbit_id,
            branch_kind=task.branch_kind,
            allowed_attribute_value=task.allowed_attribute_value,
            forbidden_attribute_values=task.forbidden_attribute_values,
            canonical_memory_key=task.canonical_memory_key,
            canonical_memory_value=task.canonical_memory_value,
            canonical_retrieval_query=task.canonical_retrieval_query,
            proof_sha256=proof.proof_sha256,
            generator_version=task.generator_version,
            product_pool_sha256=task.product_pool_sha256,
        )

    def proof_for_index(self, data_idx: int) -> NegativeConstraintOrbitProof:
        self._validate_data_idx(data_idx)
        return self._verified_orbit(data_idx // TASKS_PER_ORBIT)[1]

    def metadata(self) -> dict[str, object]:
        return {
            "schema": "agentmemory_verified_negative_constraint_provider_v1",
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
            "source_candidate_artifact_sha256": (
                self.generator.pool.candidate_artifact_sha256
            ),
            "phase_count_per_task": 6,
            "candidate_count_per_phase": 3,
            "distinct_values_per_phase": 3,
            "counterfactual_branches": 3,
            "retrieve_policy": "query_top1",
            "rules_only": True,
            "native_certified": False,
            "training_ready": False,
            "seed_epoch_orbit_count": self.seed_epoch_orbit_count,
            "seed_epoch_task_count": self.seed_epoch_task_count,
        }

    def _validate_data_idx(self, data_idx: int) -> None:
        if isinstance(data_idx, bool) or not isinstance(data_idx, int) or data_idx < 0:
            raise IndexError("data_idx must be a non-negative integer.")
        if self.mode == PROVIDER_MODE_FIXED_WINDOW and data_idx >= self.task_count:
            raise IndexError(data_idx)

    def _verified_orbit(
        self,
        stream_orbit: int,
    ) -> tuple[NegativeConstraintOrbit, NegativeConstraintOrbitProof]:
        generator, generator_epoch, orbit_index = self._generator_for_stream_orbit(
            stream_orbit
        )
        cache_key = (generator_epoch, orbit_index)
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached
        orbit = generator.generate_orbit(orbit_index, split=self.split)
        proof = verify_negative_constraint_orbit(
            orbit,
            pool=generator.pool,
            expected_generator_version=generator.version,
            expected_generator_seed=generator.seed,
        )
        with self._lock:
            self._cache[cache_key] = (orbit, proof)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_orbits:
                self._cache.popitem(last=False)
        return orbit, proof

    def _generator_for_stream_orbit(self, stream_orbit: int):
        if self.mode == PROVIDER_MODE_FIXED_WINDOW:
            return self.generator, 0, self.start_orbit + stream_orbit
        generator_epoch, orbit_index = divmod(
            stream_orbit,
            self.seed_epoch_orbit_count,
        )
        if generator_epoch == 0:
            return self.generator, 0, orbit_index
        with self._lock:
            generator = self._seed_epoch_generators.get(generator_epoch)
            if generator is None:
                digest = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "schema": "agentmemory_negative_constraint_seed_epoch_v1",
                            "base_seed": self.generator.seed,
                            "generator_epoch": generator_epoch,
                            "product_pool_sha256": self.generator.pool.semantic_sha256,
                        }
                    )
                ).digest()
                generator = NegativeConstraintGenerator(
                    pool=self.generator.pool,
                    seed=int.from_bytes(digest[:8], "big"),
                    version=self.generator.version,
                )
                self._seed_epoch_generators[generator_epoch] = generator
                while len(self._seed_epoch_generators) > 8:
                    self._seed_epoch_generators.popitem(last=False)
            else:
                self._seed_epoch_generators.move_to_end(generator_epoch)
        return generator, generator_epoch, orbit_index
