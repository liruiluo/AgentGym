from __future__ import annotations

from .env_wrapper import LATENT_PREFERENCE_PROMPT_MODE, NATURAL_FILESYSTEM_PROMPT_MODE
from .filesystem_wrapper import (
    FilesystemAgentMemoryWrapperMixin,
    WORKSPACE_PROMPT_FAMILY_INTENT,
)
from .intent_clarification import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODES,
    IntentClarificationGenerator,
    VerifiedIntentClarificationBundleProvider,
)
from .intent_clarification_webshop_env import (
    INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
    INTENT_CLARIFICATION_SURFACE,
    IntentClarificationFilesystemWebShopEnv,
    IntentClarificationWebShopEnv,
)
from .latent_preference import load_preference_product_pool
from .latent_preference.runtime_attestation import (
    attest_latent_preference_runtime_inputs,
)
from .procedural_wrapper import (
    ProceduralAgentMemoryWrapper,
    _env_choice,
    _env_int,
    _programmatic_runtime_inputs,
    _required_env,
    _required_file,
    _required_int,
)


class IntentClarificationAgentMemoryWrapper(ProceduralAgentMemoryWrapper):
    """AgentGym HTTP wrapper for ask-then-remember training."""

    surface = INTENT_CLARIFICATION_SURFACE
    environment_type = IntentClarificationWebShopEnv

    def __init__(self) -> None:
        self._initialize_intent_clarification_runtime(
            expected_prompt_mode=LATENT_PREFERENCE_PROMPT_MODE,
        )

    def _initialize_intent_clarification_runtime(
        self,
        *,
        expected_prompt_mode: str,
    ) -> None:
        self._initialize_native_training_runtime(
            forbidden_env_keys=(
                "AGENTMEMORY_MEMORYARENA_RAW_PATH",
                "AGENTMEMORY_ANNOTATION_GATE_MANIFEST",
                "AGENTMEMORY_ANNOTATION_MANUAL_EVIDENCE",
                "AGENTMEMORY_PROCEDURAL_PRODUCT_POOL",
                "AGENTMEMORY_PROCEDURAL_PRODUCT_POOL_SHA256",
                "AGENTMEMORY_LATENT_PREFERENCE_PRODUCT_POOL",
                "AGENTMEMORY_LATENT_PREFERENCE_PRODUCT_POOL_SHA256",
                "AGENTMEMORY_RECENCY_OVERRIDE_PRODUCT_POOL",
                "AGENTMEMORY_RECENCY_OVERRIDE_PRODUCT_POOL_SHA256",
            )
        )
        if self.memory_prompt_mode != expected_prompt_mode:
            raise RuntimeError(
                "The intent-clarification surface requires "
                f"memory_prompt_mode={expected_prompt_mode!r}."
            )
        pool = load_preference_product_pool(
            _required_file("AGENTMEMORY_INTENT_CLARIFICATION_PRODUCT_POOL"),
            expected_file_sha256=_required_env(
                "AGENTMEMORY_INTENT_CLARIFICATION_PRODUCT_POOL_SHA256"
            ),
        )
        self.runtime_inputs = _programmatic_runtime_inputs(
            pool,
            product_pool_file_sha256=_required_env(
                "AGENTMEMORY_INTENT_CLARIFICATION_PRODUCT_POOL_SHA256"
            ),
        )
        attest_latent_preference_runtime_inputs(
            pool,
            self.backend,
            items_file=self.items_file,
            attributes_file=self.attributes_file,
            search_root=self.search_root,
            lucene_manifest=self.lucene_manifest,
        )
        self.provider = VerifiedIntentClarificationBundleProvider(
            generator=IntentClarificationGenerator(
                pool=pool,
                seed=_required_int(
                    "AGENTMEMORY_INTENT_CLARIFICATION_GENERATOR_SEED"
                ),
            ),
            split=_required_env("AGENTMEMORY_SPLIT"),
            task_count=_required_int(
                "AGENTMEMORY_INTENT_CLARIFICATION_TASK_COUNT"
            ),
            mode=_env_choice(
                "AGENTMEMORY_INTENT_CLARIFICATION_PROVIDER_MODE",
                default=PROVIDER_MODE_FIXED_WINDOW,
                choices=PROVIDER_MODES,
            ),
            start_orbit=_env_int(
                "AGENTMEMORY_INTENT_CLARIFICATION_START_ORBIT",
                0,
            ),
        )
        self._initialize_wrapper_state()


class IntentClarificationFilesystemAgentMemoryWrapper(
    FilesystemAgentMemoryWrapperMixin,
    IntentClarificationAgentMemoryWrapper,
):
    """ASK plus ordinary persistent workspace files."""

    surface = INTENT_CLARIFICATION_FILESYSTEM_SURFACE
    environment_type = IntentClarificationFilesystemWebShopEnv
    workspace_intervention_boundary_index = 1
    workspace_prompt_family = WORKSPACE_PROMPT_FAMILY_INTENT

    def __init__(self) -> None:
        self._initialize_intent_clarification_runtime(
            expected_prompt_mode=NATURAL_FILESYSTEM_PROMPT_MODE,
        )
        self._initialize_filesystem_runtime()
