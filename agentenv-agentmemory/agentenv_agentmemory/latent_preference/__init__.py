"""Deterministic hidden-preference memory tasks over frozen WebShop products."""

from .certifier import (
    CERTIFIER_VERSION,
    PREFERENCE_RECIPES,
    PREFERENCE_RULES_SHA256,
    NativePreferenceCertificationConfig,
    NativePreferencePoolCertificationError,
    certify_native_preference_product_pool,
)
from .generator import (
    CATEGORY_SCHEDULES,
    DEFAULT_GENERATOR_VERSION,
    EVIDENCE_COUNTS,
    LatentPreferenceGenerator,
)
from .pool_io import (
    load_preference_product_pool,
    write_preference_product_pool_manifest,
)
from .provider import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES,
    VerifiedLatentPreferenceBundleProvider,
)
from .runtime_attestation import attest_latent_preference_runtime_inputs
from .schema import (
    CertifiedPreferenceProduct,
    LatentPreferenceBundle,
    LatentPreferenceDataError,
    LatentPreferenceOrbit,
    LatentPreferenceTask,
    PreferenceCandidate,
    PreferencePhase,
    PreferenceProductPool,
    PreferenceRecipe,
)
from .verifier import (
    LatentPreferenceOrbitProof,
    verify_latent_preference_orbit,
)

__all__ = [
    "CATEGORY_SCHEDULES",
    "CERTIFIER_VERSION",
    "DEFAULT_GENERATOR_VERSION",
    "EVIDENCE_COUNTS",
    "PROVIDER_MODE_FIXED_WINDOW",
    "PROVIDER_MODE_RESEEDED_STREAM",
    "PROVIDER_MODES",
    "PREFERENCE_RECIPES",
    "PREFERENCE_RULES_SHA256",
    "CertifiedPreferenceProduct",
    "LatentPreferenceBundle",
    "LatentPreferenceDataError",
    "LatentPreferenceGenerator",
    "LatentPreferenceOrbit",
    "LatentPreferenceOrbitProof",
    "LatentPreferenceTask",
    "NativePreferenceCertificationConfig",
    "NativePreferencePoolCertificationError",
    "PreferenceCandidate",
    "PreferencePhase",
    "PreferenceProductPool",
    "PreferenceRecipe",
    "VerifiedLatentPreferenceBundleProvider",
    "certify_native_preference_product_pool",
    "attest_latent_preference_runtime_inputs",
    "load_preference_product_pool",
    "verify_latent_preference_orbit",
    "write_preference_product_pool_manifest",
]
