"""Proof-carrying selective-retrieval tasks with hidden distractor memory."""

from .generator import (
    CANONICAL_MEMORY_KEY,
    DEFAULT_DISTRACTOR_COUNT,
    DEFAULT_GENERATOR_VERSION,
    DistractorRobustnessGenerator,
)
from .provider import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES,
    VerifiedDistractorRobustnessBundleProvider,
)
from .schema import (
    DistractorRobustnessBundle,
    DistractorRobustnessDataError,
    DistractorRobustnessOrbit,
    DistractorRobustnessTask,
    InitialMemory,
)
from .verifier import (
    DistractorRobustnessOrbitProof,
    verify_distractor_robustness_orbit,
)

__all__ = [
    "CANONICAL_MEMORY_KEY",
    "DEFAULT_DISTRACTOR_COUNT",
    "DEFAULT_GENERATOR_VERSION",
    "PROVIDER_MODE_FIXED_WINDOW",
    "PROVIDER_MODE_RESEEDED_STREAM",
    "PROVIDER_MODES",
    "DistractorRobustnessBundle",
    "DistractorRobustnessDataError",
    "DistractorRobustnessGenerator",
    "DistractorRobustnessOrbit",
    "DistractorRobustnessOrbitProof",
    "DistractorRobustnessTask",
    "InitialMemory",
    "VerifiedDistractorRobustnessBundleProvider",
    "verify_distractor_robustness_orbit",
]
