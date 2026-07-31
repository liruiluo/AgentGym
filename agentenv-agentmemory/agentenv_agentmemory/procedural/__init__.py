"""Deterministic, proof-carrying natural-attribute memory task generation."""

from .certifier import (
    NativeCertificationConfig,
    NativeProductPoolCertificationError,
    certify_native_product_pool,
)
from .generator import NaturalAttributeChainGenerator
from .pool_io import load_certified_product_pool
from .provenance import file_sha256, verify_lucene_index_manifest
from .provider import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES,
    VerifiedProceduralBundleProvider,
)
from .scenarios import (
    SCENARIO_DEFINITION_SHA256,
    SCENARIO_DEFINITION_VERSION,
    SCENARIOS,
    AttributeValueSpec,
    ProductClassification,
    ScenarioSpec,
    SlotSpec,
    classify_product_record,
    require_unique_product_classification,
    scenario_by_id,
)
from .schema import (
    AttributeTransition,
    CertifiedProduct,
    CounterfactualOrbit,
    ProceduralCandidate,
    ProceduralMemoryBundle,
    ProceduralMemoryDataError,
    ProceduralPhase,
    ProceduralTask,
    ProductPool,
    assign_product_split,
    normalize_native_title,
)
from .verifier import OrbitProof, verify_counterfactual_orbit

__all__ = [
    "AttributeTransition",
    "AttributeValueSpec",
    "CertifiedProduct",
    "CounterfactualOrbit",
    "NaturalAttributeChainGenerator",
    "NativeCertificationConfig",
    "NativeProductPoolCertificationError",
    "OrbitProof",
    "ProceduralCandidate",
    "ProceduralMemoryBundle",
    "ProceduralMemoryDataError",
    "ProceduralPhase",
    "ProceduralTask",
    "ProductClassification",
    "ProductPool",
    "PROVIDER_MODE_FIXED_WINDOW",
    "PROVIDER_MODE_RESEEDED_STREAM",
    "PROVIDER_MODES",
    "SCENARIOS",
    "SCENARIO_DEFINITION_SHA256",
    "SCENARIO_DEFINITION_VERSION",
    "ScenarioSpec",
    "SlotSpec",
    "VerifiedProceduralBundleProvider",
    "assign_product_split",
    "certify_native_product_pool",
    "classify_product_record",
    "file_sha256",
    "load_certified_product_pool",
    "normalize_native_title",
    "require_unique_product_classification",
    "scenario_by_id",
    "verify_counterfactual_orbit",
    "verify_lucene_index_manifest",
]
