"""Proof-carrying two-hop profile recall tasks over frozen WebShop products."""

from .generator import (
    CANONICAL_MEMORY_KEY,
    DEFAULT_GENERATOR_VERSION,
    CompositionalRecallGenerator,
)
from .provider import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES,
    TASKS_PER_ORBIT,
    VerifiedCompositionalRecallBundleProvider,
)
from .schema import (
    BRANCH_COORDINATES,
    CanonicalMemoryFact,
    CompositionalRecallBundle,
    CompositionalRecallDataError,
    CompositionalRecallOrbit,
    CompositionalRecallTask,
)
from .verifier import (
    CompositionalRecallOrbitProof,
    verify_compositional_recall_orbit,
)

__all__ = [
    "BRANCH_COORDINATES",
    "CANONICAL_MEMORY_KEY",
    "DEFAULT_GENERATOR_VERSION",
    "PROVIDER_MODE_FIXED_WINDOW",
    "PROVIDER_MODE_RESEEDED_STREAM",
    "PROVIDER_MODES",
    "TASKS_PER_ORBIT",
    "CanonicalMemoryFact",
    "CompositionalRecallBundle",
    "CompositionalRecallDataError",
    "CompositionalRecallGenerator",
    "CompositionalRecallOrbit",
    "CompositionalRecallOrbitProof",
    "CompositionalRecallTask",
    "VerifiedCompositionalRecallBundleProvider",
    "verify_compositional_recall_orbit",
]
