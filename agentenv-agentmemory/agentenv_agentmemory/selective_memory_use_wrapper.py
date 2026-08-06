from __future__ import annotations

from .env_wrapper import NATURAL_FILESYSTEM_PROMPT_MODE, SELECTIVE_MEMORY_PROMPT_MODE
from .filesystem_wrapper import (
    FilesystemAgentMemoryWrapperMixin,
    SOURCE_PAIRING_XOR_PREFERENCE_COORDINATE,
    WORKSPACE_PROMPT_FAMILY_SELECTIVE,
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
from .selective_memory_use import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODES,
    SelectiveMemoryUseGenerator,
    VerifiedSelectiveMemoryUseBundleProvider,
)
from .selective_memory_use_webshop_env import (
    SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
    SELECTIVE_MEMORY_USE_SURFACE,
    SelectiveMemoryUseFilesystemWebShopEnv,
    SelectiveMemoryUseWebShopEnv,
)


class SelectiveMemoryUseAgentMemoryWrapper(ProceduralAgentMemoryWrapper):
    """AgentGym HTTP wrapper for memory-use selection and abstention training."""

    surface = SELECTIVE_MEMORY_USE_SURFACE
    environment_type = SelectiveMemoryUseWebShopEnv

    def __init__(self) -> None:
        self._initialize_selective_memory_runtime(
            expected_prompt_mode=SELECTIVE_MEMORY_PROMPT_MODE,
        )

    def _initialize_selective_memory_runtime(
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
                "The selective-memory-use surface requires "
                f"memory_prompt_mode={expected_prompt_mode!r}."
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


class SelectiveMemoryUseFilesystemAgentMemoryWrapper(
    FilesystemAgentMemoryWrapperMixin,
    SelectiveMemoryUseAgentMemoryWrapper,
):
    """Selective-memory control backed by a seeded ordinary profile file."""

    surface = SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE
    environment_type = SelectiveMemoryUseFilesystemWebShopEnv
    workspace_intervention_boundary_index = 1
    workspace_source_pairing = SOURCE_PAIRING_XOR_PREFERENCE_COORDINATE
    workspace_tasks_per_orbit = 4
    workspace_prompt_family = WORKSPACE_PROMPT_FAMILY_SELECTIVE
    workspace_intervention_source_state = (
        "harness_seeded_branch_profile_with_optional_policy_edits"
    )
    workspace_seed_contract = "branch_conditioned_initial_profile_files_v1"
    workspace_evaluation_contract = (
        "selective_required_separation_not_required_invariance_v1"
    )

    def __init__(self) -> None:
        self._initialize_selective_memory_runtime(
            expected_prompt_mode=NATURAL_FILESYSTEM_PROMPT_MODE,
        )
        self._initialize_filesystem_runtime()
