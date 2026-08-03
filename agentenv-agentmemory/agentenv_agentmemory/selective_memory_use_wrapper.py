from __future__ import annotations

from .env_wrapper import SELECTIVE_MEMORY_PROMPT_MODE
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
from .selective_memory_use import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODES,
    SelectiveMemoryUseGenerator,
    VerifiedSelectiveMemoryUseBundleProvider,
)
from .selective_memory_use_webshop_env import (
    SELECTIVE_MEMORY_USE_SURFACE,
    SelectiveMemoryUseWebShopEnv,
)


class SelectiveMemoryUseAgentMemoryWrapper(ProceduralAgentMemoryWrapper):
    """AgentGym HTTP wrapper for memory-use selection and abstention training."""

    surface = SELECTIVE_MEMORY_USE_SURFACE
    environment_type = SelectiveMemoryUseWebShopEnv

    def __init__(self) -> None:
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
        if self.memory_prompt_mode != SELECTIVE_MEMORY_PROMPT_MODE:
            raise RuntimeError(
                "The selective-memory-use surface requires "
                f"memory_prompt_mode={SELECTIVE_MEMORY_PROMPT_MODE!r}."
            )
        for key in (
            "first_valid_add_reward",
            "first_valid_later_session_retrieve_reward",
        ):
            if float(self.reward_contract[key]) != 0.0:
                raise RuntimeError(
                    "Selective memory use requires both memory-action bonuses to be zero."
                )
        pool = load_preference_product_pool(
            _required_file("AGENTMEMORY_SELECTIVE_MEMORY_USE_PRODUCT_POOL"),
            expected_file_sha256=_required_env(
                "AGENTMEMORY_SELECTIVE_MEMORY_USE_PRODUCT_POOL_SHA256"
            ),
        )
        self.runtime_inputs = _programmatic_runtime_inputs(
            pool,
            product_pool_file_sha256=_required_env(
                "AGENTMEMORY_SELECTIVE_MEMORY_USE_PRODUCT_POOL_SHA256"
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
        self.provider = VerifiedSelectiveMemoryUseBundleProvider(
            generator=SelectiveMemoryUseGenerator(
                pool=pool,
                seed=_required_int(
                    "AGENTMEMORY_SELECTIVE_MEMORY_USE_GENERATOR_SEED"
                ),
            ),
            split=_required_env("AGENTMEMORY_SPLIT"),
            task_count=_required_int(
                "AGENTMEMORY_SELECTIVE_MEMORY_USE_TASK_COUNT"
            ),
            mode=_env_choice(
                "AGENTMEMORY_SELECTIVE_MEMORY_USE_PROVIDER_MODE",
                default=PROVIDER_MODE_FIXED_WINDOW,
                choices=PROVIDER_MODES,
            ),
            start_orbit=_env_int(
                "AGENTMEMORY_SELECTIVE_MEMORY_USE_START_ORBIT",
                0,
            ),
        )
        self._initialize_wrapper_state()
