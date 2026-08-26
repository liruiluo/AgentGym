from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

from .annotation_gate import ANNOTATION_GATE_MODES
from .domains import (
    BROWSECOMP_BM25_INTEGRATION_SURFACE,
    BROWSECOMP_SURFACES,
    FORMAL_REASONING_SURFACES_BY_MODE,
    TRAVEL_SURFACES,
    V3_SURFACES,
)
from .domains.browsecomp import BROWSECOMP_FROZEN_EMBEDDING_MODEL
from .reward_hierarchy import (
    FIRST_VALID_ADD_BONUS,
    FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
)
from .memoryarena_webshop_env import (
    ACTION_LISTING_MODES,
    LTM_INVENTORY_MODES,
    LTM_TRANSITION_NOTICE_MODES,
)
from .env_wrapper import (
    LATENT_PREFERENCE_PROMPT_MODE,
    MEMORY_PROMPT_MODES,
    NATURAL_FILESYSTEM_PROMPT_MODE,
    SELECTIVE_MEMORY_PROMPT_MODE,
)
from .filesystem_webshop_env import PROCEDURAL_FILESYSTEM_SURFACE
from .compositional_recall import (
    PROVIDER_MODE_FIXED_WINDOW as COMPOSITIONAL_PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM as COMPOSITIONAL_PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES as COMPOSITIONAL_PROVIDER_MODES,
    TASKS_PER_ORBIT as COMPOSITIONAL_TASKS_PER_ORBIT,
)
from .compositional_recall_webshop_env import (
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
    COMPOSITIONAL_RECALL_SURFACE,
)
from .distractor_robustness import (
    PROVIDER_MODE_FIXED_WINDOW as DISTRACTOR_PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM as DISTRACTOR_PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES as DISTRACTOR_PROVIDER_MODES,
)
from .distractor_robustness_webshop_env import (
    DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
    DISTRACTOR_ROBUSTNESS_SURFACE,
)
from .intent_clarification import (
    PROVIDER_MODE_FIXED_WINDOW as INTENT_PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM as INTENT_PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES as INTENT_PROVIDER_MODES,
)
from .intent_clarification_webshop_env import (
    INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
    INTENT_CLARIFICATION_SURFACE,
)
from .latent_preference import (
    PROVIDER_MODE_FIXED_WINDOW as LATENT_PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM as LATENT_PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES as LATENT_PROVIDER_MODES,
)
from .latent_preference_webshop_env import (
    LATENT_PREFERENCE_FILESYSTEM_SURFACE,
    LATENT_PREFERENCE_SURFACE,
)
from .literesearcher import (
    LITERESEARCHER_FORMAL_JUDGE_MODELS,
    LITERESEARCHER_FULLPOOL_SURFACE,
    LITERESEARCHER_SURFACE,
)
from .recency_override import (
    PROVIDER_MODE_FIXED_WINDOW as RECENCY_PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM as RECENCY_PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES as RECENCY_PROVIDER_MODES,
)
from .recency_override_webshop_env import (
    RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
    RECENCY_OVERRIDE_SURFACE,
)
from .selective_memory_use import (
    PROVIDER_MODE_FIXED_WINDOW as SELECTIVE_PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM as SELECTIVE_PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES as SELECTIVE_PROVIDER_MODES,
    TASKS_PER_ORBIT as SELECTIVE_TASKS_PER_ORBIT,
)
from .selective_memory_use_webshop_env import (
    SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
    SELECTIVE_MEMORY_USE_SURFACE,
)
from .negative_constraint import (
    PROVIDER_MODE_FIXED_WINDOW as NEGATIVE_PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM as NEGATIVE_PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES as NEGATIVE_PROVIDER_MODES,
    TASKS_PER_ORBIT as NEGATIVE_TASKS_PER_ORBIT,
)
from .negative_constraint_webshop_env import (
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
    NEGATIVE_CONSTRAINT_SURFACE,
)
from .procedural import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES,
)
from .procedural_webshop_env import PROCEDURAL_SURFACE
from .service_identity import SERVICE_ROLES


NATIVE_SURFACE = "memoryarena_webshop_native_v1"
FILESYSTEM_SURFACES = {
    PROCEDURAL_FILESYSTEM_SURFACE,
    LATENT_PREFERENCE_FILESYSTEM_SURFACE,
    RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
    DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
    INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
    SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
}


def launch() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--surface",
        choices=[
            NATIVE_SURFACE,
            PROCEDURAL_SURFACE,
            PROCEDURAL_FILESYSTEM_SURFACE,
            LATENT_PREFERENCE_SURFACE,
            LATENT_PREFERENCE_FILESYSTEM_SURFACE,
            RECENCY_OVERRIDE_SURFACE,
            RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
            DISTRACTOR_ROBUSTNESS_SURFACE,
            DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
            COMPOSITIONAL_RECALL_SURFACE,
            COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
            INTENT_CLARIFICATION_SURFACE,
            INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
            SELECTIVE_MEMORY_USE_SURFACE,
            SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
            NEGATIVE_CONSTRAINT_SURFACE,
            NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
            LITERESEARCHER_SURFACE,
            LITERESEARCHER_FULLPOOL_SURFACE,
            *V3_SURFACES,
        ],
        required=True,
    )
    parser.add_argument("--memoryarena-root")
    parser.add_argument("--raw-data")
    parser.add_argument("--items-file")
    parser.add_argument("--attributes-file")
    parser.add_argument("--search-root")
    parser.add_argument("--java-home")
    parser.add_argument("--domain-data-path")
    parser.add_argument("--lucene-index-manifest")
    parser.add_argument("--annotation-audit-summary")
    parser.add_argument("--annotation-audit-chains")
    parser.add_argument("--annotation-manual-evidence")
    parser.add_argument("--memoryarena-base-commit")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--service-role",
        choices=SERVICE_ROLES,
        default="formal",
    )
    parser.add_argument("--runtime-source-id")
    parser.add_argument(
        "--split", choices=["train", "dev", "test", "all"], default="train"
    )
    parser.add_argument("--price-seed", type=int, default=233)
    parser.add_argument(
        "--annotation-gate-mode",
        choices=ANNOTATION_GATE_MODES,
        default="provisional",
    )
    parser.add_argument("--annotation-gate-manifest")
    parser.add_argument("--annotation-gate-manifest-sha256")
    parser.add_argument("--procedural-product-pool")
    parser.add_argument("--procedural-product-pool-sha256")
    parser.add_argument("--procedural-task-count", type=int)
    parser.add_argument("--procedural-generator-seed", type=int)
    parser.add_argument(
        "--procedural-provider-mode",
        choices=PROVIDER_MODES,
    )
    parser.add_argument("--procedural-start-orbit", type=int, default=0)
    parser.add_argument("--workspace-rg-binary")
    parser.add_argument("--workspace-rg-sha256")
    parser.add_argument("--workspace-root-parent")
    parser.add_argument("--workspace-intervention-token-file")
    parser.add_argument("--literesearcher-coverage-manifest")
    parser.add_argument("--literesearcher-full-pool-manifest")
    parser.add_argument("--literesearcher-full-pool-rows")
    parser.add_argument("--literesearcher-source-root")
    parser.add_argument("--literesearcher-upstream-endpoint")
    parser.add_argument(
        "--literesearcher-backend-timeout-seconds",
        type=float,
        default=120.0,
    )
    parser.add_argument("--literesearcher-judge-api-base")
    parser.add_argument("--literesearcher-judge-model")
    parser.add_argument("--literesearcher-judge-api-key-file")
    parser.add_argument(
        "--literesearcher-judge-timeout-seconds",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--literesearcher-judge-max-retries",
        type=int,
        default=3,
    )
    parser.add_argument("--literesearcher-max-policy-steps", type=int, default=40)
    parser.add_argument("--literesearcher-top-k", type=int, default=5)
    parser.add_argument(
        "--literesearcher-filter-visitable",
        action="store_true",
        help="ask the upstream service to return only PostgreSQL-visitable URLs",
    )
    parser.add_argument("--latent-preference-product-pool")
    parser.add_argument("--latent-preference-product-pool-sha256")
    parser.add_argument("--latent-preference-task-count", type=int)
    parser.add_argument("--latent-preference-generator-seed", type=int)
    parser.add_argument(
        "--latent-preference-provider-mode",
        choices=LATENT_PROVIDER_MODES,
    )
    parser.add_argument("--latent-preference-start-orbit", type=int, default=0)
    parser.add_argument("--recency-override-product-pool")
    parser.add_argument("--recency-override-product-pool-sha256")
    parser.add_argument("--recency-override-task-count", type=int)
    parser.add_argument("--recency-override-generator-seed", type=int)
    parser.add_argument(
        "--recency-override-provider-mode",
        choices=RECENCY_PROVIDER_MODES,
    )
    parser.add_argument("--recency-override-start-orbit", type=int, default=0)
    parser.add_argument("--distractor-robustness-product-pool")
    parser.add_argument("--distractor-robustness-product-pool-sha256")
    parser.add_argument("--distractor-robustness-task-count", type=int)
    parser.add_argument("--distractor-robustness-generator-seed", type=int)
    parser.add_argument(
        "--distractor-robustness-provider-mode",
        choices=DISTRACTOR_PROVIDER_MODES,
    )
    parser.add_argument("--distractor-robustness-start-orbit", type=int, default=0)
    parser.add_argument("--compositional-recall-product-pool")
    parser.add_argument("--compositional-recall-product-pool-sha256")
    parser.add_argument("--compositional-recall-task-count", type=int)
    parser.add_argument("--compositional-recall-generator-seed", type=int)
    parser.add_argument(
        "--compositional-recall-provider-mode",
        choices=COMPOSITIONAL_PROVIDER_MODES,
    )
    parser.add_argument("--compositional-recall-start-orbit", type=int, default=0)
    parser.add_argument("--intent-clarification-product-pool")
    parser.add_argument("--intent-clarification-product-pool-sha256")
    parser.add_argument("--intent-clarification-task-count", type=int)
    parser.add_argument("--intent-clarification-generator-seed", type=int)
    parser.add_argument(
        "--intent-clarification-provider-mode",
        choices=INTENT_PROVIDER_MODES,
    )
    parser.add_argument("--intent-clarification-start-orbit", type=int, default=0)
    parser.add_argument("--selective-memory-use-product-pool")
    parser.add_argument("--selective-memory-use-product-pool-sha256")
    parser.add_argument("--selective-memory-use-task-count", type=int)
    parser.add_argument("--selective-memory-use-generator-seed", type=int)
    parser.add_argument(
        "--selective-memory-use-provider-mode",
        choices=SELECTIVE_PROVIDER_MODES,
    )
    parser.add_argument("--selective-memory-use-start-orbit", type=int, default=0)
    parser.add_argument("--negative-constraint-product-pool")
    parser.add_argument("--negative-constraint-product-pool-sha256")
    parser.add_argument("--negative-constraint-task-count", type=int)
    parser.add_argument("--negative-constraint-generator-seed", type=int)
    parser.add_argument(
        "--negative-constraint-provider-mode",
        choices=NEGATIVE_PROVIDER_MODES,
    )
    parser.add_argument("--negative-constraint-start-orbit", type=int, default=0)
    parser.add_argument("--travel-tasks-path")
    parser.add_argument("--travel-database-path")
    parser.add_argument("--formal-reasoning-tasks-path")
    parser.add_argument("--formal-reasoning-judge-model")
    parser.add_argument("--formal-reasoning-judge-base-url")
    parser.add_argument(
        "--formal-reasoning-judge-temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--formal-reasoning-judge-max-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument("--browsecomp-tasks-path")
    parser.add_argument("--browsecomp-index-path")
    parser.add_argument("--browsecomp-corpus-path")
    parser.add_argument("--browsecomp-corpus-manifest")
    parser.add_argument("--browsecomp-bm25-index-path")
    parser.add_argument(
        "--browsecomp-embedding-provider",
        choices=["openai", "openrouter"],
    )
    parser.add_argument(
        "--browsecomp-embedding-model",
        choices=[BROWSECOMP_FROZEN_EMBEDDING_MODEL],
        default=BROWSECOMP_FROZEN_EMBEDDING_MODEL,
    )
    parser.add_argument("--browsecomp-judge-model")
    parser.add_argument(
        "--browsecomp-judge-max-tokens",
        type=int,
        default=8000,
    )
    parser.add_argument("--browsecomp-api-base-url")
    parser.add_argument("--memory-first-add-reward", type=float)
    parser.add_argument("--memory-first-later-retrieve-reward", type=float)
    parser.add_argument("--memory-exact-repeat-reward", type=float, default=0.0)
    parser.add_argument("--invalid-action-reward", type=float, default=0.0)
    parser.add_argument(
        "--ltm-inventory-mode",
        choices=LTM_INVENTORY_MODES,
        default="hidden",
    )
    parser.add_argument(
        "--ltm-transition-notice-mode",
        choices=LTM_TRANSITION_NOTICE_MODES,
        default="none",
    )
    parser.add_argument(
        "--memory-prompt-mode",
        choices=MEMORY_PROMPT_MODES,
        default="legacy",
    )
    parser.add_argument(
        "--action-listing-mode",
        choices=ACTION_LISTING_MODES,
        default="separate",
    )
    args = parser.parse_args()

    if args.service_role in {"smoke", "intervention_eval"} and not args.runtime_source_id:
        parser.error(
            f"--service-role {args.service_role} requires --runtime-source-id"
        )

    preference_surfaces = {
        LATENT_PREFERENCE_SURFACE,
        RECENCY_OVERRIDE_SURFACE,
        DISTRACTOR_ROBUSTNESS_SURFACE,
        COMPOSITIONAL_RECALL_SURFACE,
        INTENT_CLARIFICATION_SURFACE,
        NEGATIVE_CONSTRAINT_SURFACE,
    }
    literesearcher_surfaces = {LITERESEARCHER_SURFACE, LITERESEARCHER_FULLPOOL_SURFACE}
    if args.surface in literesearcher_surfaces:
        _require_args(
            parser,
            args,
            "workspace_rg_binary",
            "workspace_rg_sha256",
        )
        if args.surface == LITERESEARCHER_SURFACE:
            _require_args(parser, args, "literesearcher_coverage_manifest")
        else:
            _require_args(
                parser,
                args,
                "literesearcher_full_pool_manifest",
                "literesearcher_full_pool_rows",
                "literesearcher_source_root",
                "literesearcher_upstream_endpoint",
                "literesearcher_judge_api_base",
                "literesearcher_judge_model",
            )
            if args.split != "train":
                parser.error("LiteResearcher full-pool formal currently requires --split train")
            if (
                args.literesearcher_judge_model
                not in LITERESEARCHER_FORMAL_JUDGE_MODELS
            ):
                parser.error(
                    "LiteResearcher full-pool formal requires a frozen judge model: "
                    f"{', '.join(sorted(LITERESEARCHER_FORMAL_JUDGE_MODELS))}"
                )
        if args.split not in {"train", "test"}:
            parser.error("LiteResearcher requires --split train or --split test")
        if args.literesearcher_max_policy_steps < 1:
            parser.error("--literesearcher-max-policy-steps must be positive")
        if args.literesearcher_top_k < 1:
            parser.error("--literesearcher-top-k must be positive")
        if args.literesearcher_top_k > 50:
            parser.error("--literesearcher-top-k must not exceed 50")
        if args.literesearcher_backend_timeout_seconds <= 0:
            parser.error("--literesearcher-backend-timeout-seconds must be positive")
        if args.literesearcher_judge_timeout_seconds <= 0:
            parser.error("--literesearcher-judge-timeout-seconds must be positive")
        if args.literesearcher_judge_max_retries < 1:
            parser.error("--literesearcher-judge-max-retries must be positive")
        if args.service_role == "intervention_eval":
            parser.error("LiteResearcher does not expose workspace interventions")
        if args.workspace_intervention_token_file:
            parser.error(
                "LiteResearcher refuses --workspace-intervention-token-file"
            )
        if any(
            value not in {None, 0.0}
            for value in (
                args.memory_first_add_reward,
                args.memory_first_later_retrieve_reward,
                args.memory_exact_repeat_reward,
            )
        ):
            parser.error("LiteResearcher refuses memory-specific reward shaping")
        if args.invalid_action_reward not in {None, 0.0, -0.01}:
            parser.error(
                "LiteResearcher invalid-action reward must be 0 or the frozen -0.01"
            )
        if args.memory_prompt_mode != "legacy":
            parser.error("LiteResearcher owns its prompt and refuses memory prompt modes")
    elif args.surface in FILESYSTEM_SURFACES:
        if not args.workspace_rg_binary or not args.workspace_rg_sha256:
            parser.error(
                "the Codex workspace surface requires --workspace-rg-binary and "
                "--workspace-rg-sha256"
            )
        if args.memory_prompt_mode != NATURAL_FILESYSTEM_PROMPT_MODE:
            parser.error(
                "the filesystem-v2 surface requires "
                f"--memory-prompt-mode {NATURAL_FILESYSTEM_PROMPT_MODE}"
            )
        if any(
            value not in {None, 0.0}
            for value in (
                args.memory_first_add_reward,
                args.memory_first_later_retrieve_reward,
            )
        ):
            parser.error(
                "the filesystem-v2 surface refuses dedicated write/read reward shaping"
            )
        if (
            args.ltm_inventory_mode != "hidden"
            or args.ltm_transition_notice_mode != "none"
            or args.action_listing_mode != "separate"
        ):
            parser.error(
                "the filesystem-v2 surface refuses legacy LTM inventory, transition, "
                "or unified action-listing modes"
            )
        if args.service_role == "intervention_eval":
            if not args.workspace_intervention_token_file:
                parser.error(
                    "--service-role intervention_eval requires "
                    "--workspace-intervention-token-file"
                )
        elif args.workspace_intervention_token_file:
            parser.error(
                "--workspace-intervention-token-file is valid only for the "
                "intervention_eval service role"
            )
    elif args.service_role == "intervention_eval":
        parser.error(
            "--service-role intervention_eval is valid only for a filesystem-v2 surface"
        )
    elif args.surface == SELECTIVE_MEMORY_USE_SURFACE:
        if args.memory_prompt_mode != SELECTIVE_MEMORY_PROMPT_MODE:
            parser.error(
                "the selective-memory-use surface requires "
                f"--memory-prompt-mode {SELECTIVE_MEMORY_PROMPT_MODE}"
            )
    elif args.surface in preference_surfaces:
        if args.memory_prompt_mode != LATENT_PREFERENCE_PROMPT_MODE:
            parser.error(
                "this programmatic preference surface requires "
                f"--memory-prompt-mode {LATENT_PREFERENCE_PROMPT_MODE}"
            )
    elif args.memory_prompt_mode in {
        LATENT_PREFERENCE_PROMPT_MODE,
        SELECTIVE_MEMORY_PROMPT_MODE,
        NATURAL_FILESYSTEM_PROMPT_MODE,
    }:
        parser.error(
            "specialized --memory-prompt-mode is only valid for its approved "
            "programmatic memory surface"
        )

    configured = {
        "AGENTMEMORY_SURFACE": args.surface,
        "AGENTMEMORY_RUN_ID": args.run_id,
        "AGENTMEMORY_SERVICE_ROLE": args.service_role,
    }
    if args.surface not in literesearcher_surfaces:
        _require_args(parser, args, "memoryarena_root", "memoryarena_base_commit")
        configured.update(
            {
                "MEMORYARENA_ROOT": args.memoryarena_root,
                "MEMORYARENA_BASE_COMMIT": args.memoryarena_base_commit,
            }
        )
    if args.workspace_root_parent:
        configured["AGENTMEMORY_WORKSPACE_ROOT_PARENT"] = args.workspace_root_parent
    if args.runtime_source_id:
        configured["AGENTMEMORY_RUNTIME_SOURCE_ID"] = args.runtime_source_id
    if args.workspace_intervention_token_file:
        token_path = Path(args.workspace_intervention_token_file).expanduser()
        try:
            token_info = token_path.lstat()
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"cannot read workspace intervention token file: {exc}")
        if (
            token_path.is_symlink()
            or not stat.S_ISREG(token_info.st_mode)
            or token_info.st_mode & 0o077
        ):
            parser.error(
                "workspace intervention token file must be a private regular file"
            )
        if len(token) < 32 or any(character.isspace() for character in token):
            parser.error(
                "workspace intervention token must contain at least 32 non-whitespace characters"
            )
        configured["AGENTMEMORY_WORKSPACE_INTERVENTION_TOKEN"] = token
    if args.surface in literesearcher_surfaces:
        configured.update({
                "AGENTMEMORY_LITERESEARCHER_COVERAGE_MANIFEST": (
                    args.literesearcher_coverage_manifest or ""
                ),
                "AGENTMEMORY_LITERESEARCHER_SPLIT": args.split,
                "AGENTMEMORY_LITERESEARCHER_MAX_POLICY_STEPS": str(
                    args.literesearcher_max_policy_steps
                ),
                "AGENTMEMORY_LITERESEARCHER_TOP_K": str(
                    args.literesearcher_top_k
                ),
                "AGENTMEMORY_LITERESEARCHER_FILTER_VISITABLE": (
                    "1" if args.literesearcher_filter_visitable else "0"
                ),
                "AGENTMEMORY_WORKSPACE_RG_BINARY": args.workspace_rg_binary,
                "AGENTMEMORY_WORKSPACE_RG_SHA256": args.workspace_rg_sha256,
                "AGENTMEMORY_INVALID_ACTION_REWARD": str(
                    0.0
                    if args.invalid_action_reward is None
                    else args.invalid_action_reward
                ),
        })
        if args.surface == LITERESEARCHER_FULLPOOL_SURFACE:
            judge_api_key = "EMPTY"
            if args.literesearcher_judge_api_key_file:
                key_path = Path(args.literesearcher_judge_api_key_file).expanduser()
                try:
                    key_info = key_path.lstat()
                    judge_api_key = key_path.read_text(encoding="utf-8").strip()
                except OSError as exc:
                    parser.error(f"cannot read LiteResearcher judge API key file: {exc}")
                if (
                    key_path.is_symlink()
                    or not stat.S_ISREG(key_info.st_mode)
                    or key_info.st_mode & 0o077
                ):
                    parser.error(
                        "LiteResearcher judge API key file must be a private regular file"
                    )
                if not judge_api_key or any(
                    character.isspace() for character in judge_api_key
                ):
                    parser.error(
                        "LiteResearcher judge API key must be nonempty without whitespace"
                    )
            configured.update({
                "AGENTMEMORY_LITERESEARCHER_FULL_POOL_MANIFEST": args.literesearcher_full_pool_manifest,
                "AGENTMEMORY_LITERESEARCHER_FULL_POOL_ROWS": args.literesearcher_full_pool_rows,
                "AGENTMEMORY_LITERESEARCHER_SOURCE_ROOT": args.literesearcher_source_root,
                "AGENTMEMORY_LITERESEARCHER_UPSTREAM_ENDPOINT": args.literesearcher_upstream_endpoint,
                "AGENTMEMORY_LITERESEARCHER_BACKEND_TIMEOUT_SECONDS": str(
                    args.literesearcher_backend_timeout_seconds
                ),
                "AGENTMEMORY_LITERESEARCHER_JUDGE_API_BASE": args.literesearcher_judge_api_base,
                "AGENTMEMORY_LITERESEARCHER_JUDGE_MODEL": args.literesearcher_judge_model,
                "AGENTMEMORY_LITERESEARCHER_JUDGE_API_KEY": judge_api_key,
                "AGENTMEMORY_LITERESEARCHER_JUDGE_TIMEOUT_SECONDS": str(
                    args.literesearcher_judge_timeout_seconds
                ),
                "AGENTMEMORY_LITERESEARCHER_JUDGE_MAX_RETRIES": str(
                    args.literesearcher_judge_max_retries
                ),
            })
    elif args.surface == NATIVE_SURFACE:
        _require_args(
            parser,
            args,
            "raw_data",
            "items_file",
            "attributes_file",
            "search_root",
            "java_home",
            "domain_data_path",
            "lucene_index_manifest",
            "annotation_audit_summary",
            "annotation_audit_chains",
            "annotation_manual_evidence",
            "annotation_gate_manifest",
            "annotation_gate_manifest_sha256",
        )
        configured.update(
            {
                "AGENTMEMORY_MEMORYARENA_RAW_PATH": args.raw_data,
                "MEMORYARENA_WEBSHOP_ITEMS_FILE": args.items_file,
                "MEMORYARENA_WEBSHOP_ATTR_FILE": args.attributes_file,
                "MEMORYARENA_WEBSHOP_SEARCH_ROOT": args.search_root,
                "MEMORYARENA_WEBSHOP_JAVA_HOME": args.java_home,
                "MEMORYARENA_WEBSHOP_DOMAIN_DATA_PATH": args.domain_data_path,
                "MEMORYARENA_LUCENE_INDEX_MANIFEST": args.lucene_index_manifest,
                "AGENTMEMORY_ANNOTATION_AUDIT_SUMMARY": args.annotation_audit_summary,
                "AGENTMEMORY_ANNOTATION_AUDIT_CHAINS": args.annotation_audit_chains,
                "AGENTMEMORY_ANNOTATION_MANUAL_EVIDENCE": args.annotation_manual_evidence,
                "AGENTMEMORY_SPLIT": args.split,
                "AGENTMEMORY_WEBSHOP_PRICE_SEED": str(args.price_seed),
                "AGENTMEMORY_ANNOTATION_GATE_MODE": args.annotation_gate_mode,
                "AGENTMEMORY_ANNOTATION_GATE_MANIFEST": args.annotation_gate_manifest,
                "AGENTMEMORY_ANNOTATION_GATE_MANIFEST_SHA256": (
                    args.annotation_gate_manifest_sha256
                ),
                "AGENTMEMORY_FIRST_VALID_ADD_REWARD": str(
                    FIRST_VALID_ADD_BONUS
                    if args.memory_first_add_reward is None
                    else args.memory_first_add_reward
                ),
                "AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD": str(
                    FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS
                    if args.memory_first_later_retrieve_reward is None
                    else args.memory_first_later_retrieve_reward
                ),
                "AGENTMEMORY_LTM_INVENTORY_MODE": args.ltm_inventory_mode,
                "AGENTMEMORY_LTM_TRANSITION_NOTICE_MODE": (
                    args.ltm_transition_notice_mode
                ),
                "AGENTMEMORY_MEMORY_PROMPT_MODE": args.memory_prompt_mode,
                "AGENTMEMORY_ACTION_LISTING_MODE": args.action_listing_mode,
            }
        )
    elif args.surface in {PROCEDURAL_SURFACE, PROCEDURAL_FILESYSTEM_SURFACE}:
        _require_args(
            parser,
            args,
            "items_file",
            "attributes_file",
            "search_root",
            "java_home",
            "lucene_index_manifest",
            "procedural_product_pool",
            "procedural_product_pool_sha256",
            "procedural_task_count",
        )
        if args.procedural_generator_seed is None:
            parser.error("surface requires: --procedural-generator-seed")
        if args.split == "all":
            parser.error("procedural data requires one explicit split")
        if args.procedural_task_count <= 0 or args.procedural_task_count % 2:
            parser.error("--procedural-task-count must be a positive even integer")
        provider_mode = args.procedural_provider_mode or (
            PROVIDER_MODE_RESEEDED_STREAM
            if args.split == "train"
            else PROVIDER_MODE_FIXED_WINDOW
        )
        if args.procedural_start_orbit < 0:
            parser.error("--procedural-start-orbit must be non-negative")
        if provider_mode == PROVIDER_MODE_RESEEDED_STREAM:
            if args.split != "train":
                parser.error(
                    "--procedural-provider-mode reseeded_stream is training-only"
                )
            if args.procedural_start_orbit != 0:
                parser.error(
                    "reseeded_stream requires --procedural-start-orbit 0"
                )
        configured.update(
            {
                "MEMORYARENA_WEBSHOP_ITEMS_FILE": args.items_file,
                "MEMORYARENA_WEBSHOP_ATTR_FILE": args.attributes_file,
                "MEMORYARENA_WEBSHOP_SEARCH_ROOT": args.search_root,
                "MEMORYARENA_WEBSHOP_JAVA_HOME": args.java_home,
                "MEMORYARENA_LUCENE_INDEX_MANIFEST": args.lucene_index_manifest,
                "AGENTMEMORY_PROCEDURAL_PRODUCT_POOL": (
                    args.procedural_product_pool
                ),
                "AGENTMEMORY_PROCEDURAL_PRODUCT_POOL_SHA256": (
                    args.procedural_product_pool_sha256
                ),
                "AGENTMEMORY_PROCEDURAL_TASK_COUNT": str(
                    args.procedural_task_count
                ),
                "AGENTMEMORY_PROCEDURAL_GENERATOR_SEED": str(
                    args.procedural_generator_seed
                ),
                "AGENTMEMORY_PROCEDURAL_PROVIDER_MODE": provider_mode,
                "AGENTMEMORY_PROCEDURAL_START_ORBIT": str(
                    args.procedural_start_orbit
                ),
                "AGENTMEMORY_SPLIT": args.split,
                "AGENTMEMORY_WEBSHOP_PRICE_SEED": str(args.price_seed),
                "AGENTMEMORY_FIRST_VALID_ADD_REWARD": str(
                    0.0
                    if args.surface == PROCEDURAL_FILESYSTEM_SURFACE
                    else FIRST_VALID_ADD_BONUS
                    if args.memory_first_add_reward is None
                    else args.memory_first_add_reward
                ),
                "AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD": str(
                    0.0
                    if args.surface == PROCEDURAL_FILESYSTEM_SURFACE
                    else FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS
                    if args.memory_first_later_retrieve_reward is None
                    else args.memory_first_later_retrieve_reward
                ),
                "AGENTMEMORY_LTM_INVENTORY_MODE": args.ltm_inventory_mode,
                "AGENTMEMORY_LTM_TRANSITION_NOTICE_MODE": (
                    args.ltm_transition_notice_mode
                ),
                "AGENTMEMORY_MEMORY_PROMPT_MODE": args.memory_prompt_mode,
                "AGENTMEMORY_ACTION_LISTING_MODE": args.action_listing_mode,
            }
        )
        if args.surface == PROCEDURAL_FILESYSTEM_SURFACE:
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"] = (
                args.workspace_rg_binary
            )
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"] = (
                args.workspace_rg_sha256
            )
    elif args.surface in {
        LATENT_PREFERENCE_SURFACE,
        LATENT_PREFERENCE_FILESYSTEM_SURFACE,
    }:
        _require_args(
            parser,
            args,
            "items_file",
            "attributes_file",
            "search_root",
            "java_home",
            "lucene_index_manifest",
            "latent_preference_product_pool",
            "latent_preference_product_pool_sha256",
            "latent_preference_task_count",
        )
        if args.latent_preference_generator_seed is None:
            parser.error("surface requires: --latent-preference-generator-seed")
        if args.split == "all":
            parser.error("latent-preference data requires one explicit split")
        if (
            args.latent_preference_task_count <= 0
            or args.latent_preference_task_count % 2
        ):
            parser.error(
                "--latent-preference-task-count must be a positive even integer"
            )
        provider_mode = args.latent_preference_provider_mode or (
            LATENT_PROVIDER_MODE_RESEEDED_STREAM
            if args.split == "train"
            else LATENT_PROVIDER_MODE_FIXED_WINDOW
        )
        if args.latent_preference_start_orbit < 0:
            parser.error("--latent-preference-start-orbit must be non-negative")
        if provider_mode == LATENT_PROVIDER_MODE_RESEEDED_STREAM:
            if args.split != "train":
                parser.error(
                    "--latent-preference-provider-mode reseeded_stream is "
                    "training-only"
                )
            if args.latent_preference_start_orbit != 0:
                parser.error(
                    "reseeded_stream requires --latent-preference-start-orbit 0"
                )
        configured.update(
            {
                "MEMORYARENA_WEBSHOP_ITEMS_FILE": args.items_file,
                "MEMORYARENA_WEBSHOP_ATTR_FILE": args.attributes_file,
                "MEMORYARENA_WEBSHOP_SEARCH_ROOT": args.search_root,
                "MEMORYARENA_WEBSHOP_JAVA_HOME": args.java_home,
                "MEMORYARENA_LUCENE_INDEX_MANIFEST": args.lucene_index_manifest,
                "AGENTMEMORY_LATENT_PREFERENCE_PRODUCT_POOL": (
                    args.latent_preference_product_pool
                ),
                "AGENTMEMORY_LATENT_PREFERENCE_PRODUCT_POOL_SHA256": (
                    args.latent_preference_product_pool_sha256
                ),
                "AGENTMEMORY_LATENT_PREFERENCE_TASK_COUNT": str(
                    args.latent_preference_task_count
                ),
                "AGENTMEMORY_LATENT_PREFERENCE_GENERATOR_SEED": str(
                    args.latent_preference_generator_seed
                ),
                "AGENTMEMORY_LATENT_PREFERENCE_PROVIDER_MODE": provider_mode,
                "AGENTMEMORY_LATENT_PREFERENCE_START_ORBIT": str(
                    args.latent_preference_start_orbit
                ),
                "AGENTMEMORY_SPLIT": args.split,
                "AGENTMEMORY_WEBSHOP_PRICE_SEED": str(args.price_seed),
                "AGENTMEMORY_FIRST_VALID_ADD_REWARD": str(
                    0.0
                    if args.surface == LATENT_PREFERENCE_FILESYSTEM_SURFACE
                    else FIRST_VALID_ADD_BONUS
                    if args.memory_first_add_reward is None
                    else args.memory_first_add_reward
                ),
                "AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD": str(
                    0.0
                    if args.surface == LATENT_PREFERENCE_FILESYSTEM_SURFACE
                    else FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS
                    if args.memory_first_later_retrieve_reward is None
                    else args.memory_first_later_retrieve_reward
                ),
                "AGENTMEMORY_LTM_INVENTORY_MODE": args.ltm_inventory_mode,
                "AGENTMEMORY_LTM_TRANSITION_NOTICE_MODE": (
                    args.ltm_transition_notice_mode
                ),
                "AGENTMEMORY_MEMORY_PROMPT_MODE": args.memory_prompt_mode,
                "AGENTMEMORY_ACTION_LISTING_MODE": args.action_listing_mode,
            }
        )
        if args.surface == LATENT_PREFERENCE_FILESYSTEM_SURFACE:
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"] = args.workspace_rg_binary
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"] = args.workspace_rg_sha256
    elif args.surface in {
        RECENCY_OVERRIDE_SURFACE,
        RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
    }:
        _require_args(
            parser,
            args,
            "items_file",
            "attributes_file",
            "search_root",
            "java_home",
            "lucene_index_manifest",
            "recency_override_product_pool",
            "recency_override_product_pool_sha256",
            "recency_override_task_count",
        )
        if args.recency_override_generator_seed is None:
            parser.error("surface requires --recency-override-generator-seed")
        if args.split == "all":
            parser.error("recency-override data requires one explicit split")
        if (
            args.recency_override_task_count <= 0
            or args.recency_override_task_count % 2
        ):
            parser.error(
                "--recency-override-task-count must be a positive even integer"
            )
        provider_mode = args.recency_override_provider_mode or (
            RECENCY_PROVIDER_MODE_RESEEDED_STREAM
            if args.split == "train"
            else RECENCY_PROVIDER_MODE_FIXED_WINDOW
        )
        if args.recency_override_start_orbit < 0:
            parser.error(
                "--recency-override-start-orbit must be non-negative"
            )
        if provider_mode == RECENCY_PROVIDER_MODE_RESEEDED_STREAM:
            if args.split != "train":
                parser.error(
                    "--recency-override-provider-mode reseeded_stream is "
                    "training-only"
                )
            if args.recency_override_start_orbit != 0:
                parser.error(
                    "reseeded_stream requires --recency-override-start-orbit 0"
                )
        configured.update(
            {
                "MEMORYARENA_WEBSHOP_ITEMS_FILE": args.items_file,
                "MEMORYARENA_WEBSHOP_ATTR_FILE": args.attributes_file,
                "MEMORYARENA_WEBSHOP_SEARCH_ROOT": args.search_root,
                "MEMORYARENA_WEBSHOP_JAVA_HOME": args.java_home,
                "MEMORYARENA_LUCENE_INDEX_MANIFEST": args.lucene_index_manifest,
                "AGENTMEMORY_RECENCY_OVERRIDE_PRODUCT_POOL": (
                    args.recency_override_product_pool
                ),
                "AGENTMEMORY_RECENCY_OVERRIDE_PRODUCT_POOL_SHA256": (
                    args.recency_override_product_pool_sha256
                ),
                "AGENTMEMORY_RECENCY_OVERRIDE_TASK_COUNT": str(
                    args.recency_override_task_count
                ),
                "AGENTMEMORY_RECENCY_OVERRIDE_GENERATOR_SEED": str(
                    args.recency_override_generator_seed
                ),
                "AGENTMEMORY_RECENCY_OVERRIDE_PROVIDER_MODE": provider_mode,
                "AGENTMEMORY_RECENCY_OVERRIDE_START_ORBIT": str(
                    args.recency_override_start_orbit
                ),
                "AGENTMEMORY_SPLIT": args.split,
                "AGENTMEMORY_WEBSHOP_PRICE_SEED": str(args.price_seed),
                "AGENTMEMORY_FIRST_VALID_ADD_REWARD": str(
                    0.0
                    if args.surface == RECENCY_OVERRIDE_FILESYSTEM_SURFACE
                    else FIRST_VALID_ADD_BONUS
                    if args.memory_first_add_reward is None
                    else args.memory_first_add_reward
                ),
                "AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD": str(
                    0.0
                    if args.surface == RECENCY_OVERRIDE_FILESYSTEM_SURFACE
                    else FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS
                    if args.memory_first_later_retrieve_reward is None
                    else args.memory_first_later_retrieve_reward
                ),
                "AGENTMEMORY_LTM_INVENTORY_MODE": args.ltm_inventory_mode,
                "AGENTMEMORY_LTM_TRANSITION_NOTICE_MODE": (
                    args.ltm_transition_notice_mode
                ),
                "AGENTMEMORY_MEMORY_PROMPT_MODE": args.memory_prompt_mode,
                "AGENTMEMORY_ACTION_LISTING_MODE": args.action_listing_mode,
            }
        )
        if args.surface == RECENCY_OVERRIDE_FILESYSTEM_SURFACE:
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"] = (
                args.workspace_rg_binary
            )
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"] = (
                args.workspace_rg_sha256
            )
    elif args.surface in {
        DISTRACTOR_ROBUSTNESS_SURFACE,
        DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
    }:
        configured.update(
            _configure_programmatic_memory_surface(
                parser,
                args,
                cli_prefix="distractor_robustness",
                env_prefix="AGENTMEMORY_DISTRACTOR_ROBUSTNESS",
                tasks_per_orbit=2,
                fixed_window_mode=DISTRACTOR_PROVIDER_MODE_FIXED_WINDOW,
                reseeded_stream_mode=DISTRACTOR_PROVIDER_MODE_RESEEDED_STREAM,
                zero_memory_rewards=(
                    args.surface == DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE
                ),
            )
        )
        if args.surface == DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE:
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"] = (
                args.workspace_rg_binary
            )
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"] = (
                args.workspace_rg_sha256
            )
    elif args.surface in {
        COMPOSITIONAL_RECALL_SURFACE,
        COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
    }:
        configured.update(
            _configure_programmatic_memory_surface(
                parser,
                args,
                cli_prefix="compositional_recall",
                env_prefix="AGENTMEMORY_COMPOSITIONAL_RECALL",
                tasks_per_orbit=COMPOSITIONAL_TASKS_PER_ORBIT,
                fixed_window_mode=COMPOSITIONAL_PROVIDER_MODE_FIXED_WINDOW,
                reseeded_stream_mode=COMPOSITIONAL_PROVIDER_MODE_RESEEDED_STREAM,
                zero_memory_rewards=(
                    args.surface == COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE
                ),
            )
        )
        if args.surface == COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE:
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"] = (
                args.workspace_rg_binary
            )
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"] = (
                args.workspace_rg_sha256
            )
    elif args.surface in {
        INTENT_CLARIFICATION_SURFACE,
        INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
    }:
        configured.update(
            _configure_programmatic_memory_surface(
                parser,
                args,
                cli_prefix="intent_clarification",
                env_prefix="AGENTMEMORY_INTENT_CLARIFICATION",
                tasks_per_orbit=2,
                fixed_window_mode=INTENT_PROVIDER_MODE_FIXED_WINDOW,
                reseeded_stream_mode=INTENT_PROVIDER_MODE_RESEEDED_STREAM,
                zero_memory_rewards=(
                    args.surface == INTENT_CLARIFICATION_FILESYSTEM_SURFACE
                ),
            )
        )
        if args.surface == INTENT_CLARIFICATION_FILESYSTEM_SURFACE:
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"] = args.workspace_rg_binary
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"] = args.workspace_rg_sha256
    elif args.surface in {
        SELECTIVE_MEMORY_USE_SURFACE,
        SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
    }:
        configured.update(
            _configure_programmatic_memory_surface(
                parser,
                args,
                cli_prefix="selective_memory_use",
                env_prefix="AGENTMEMORY_SELECTIVE_MEMORY_USE",
                tasks_per_orbit=SELECTIVE_TASKS_PER_ORBIT,
                fixed_window_mode=SELECTIVE_PROVIDER_MODE_FIXED_WINDOW,
                reseeded_stream_mode=SELECTIVE_PROVIDER_MODE_RESEEDED_STREAM,
                zero_memory_rewards=True,
            )
        )
        if args.surface == SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE:
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"] = args.workspace_rg_binary
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"] = args.workspace_rg_sha256
    elif args.surface in {
        NEGATIVE_CONSTRAINT_SURFACE,
        NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
    }:
        configured.update(
            _configure_programmatic_memory_surface(
                parser,
                args,
                cli_prefix="negative_constraint",
                env_prefix="AGENTMEMORY_NEGATIVE_CONSTRAINT",
                tasks_per_orbit=NEGATIVE_TASKS_PER_ORBIT,
                fixed_window_mode=NEGATIVE_PROVIDER_MODE_FIXED_WINDOW,
                reseeded_stream_mode=NEGATIVE_PROVIDER_MODE_RESEEDED_STREAM,
                zero_memory_rewards=(
                    args.surface == NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE
                ),
            )
        )
        if args.surface == NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE:
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"] = (
                args.workspace_rg_binary
            )
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"] = (
                args.workspace_rg_sha256
            )
    elif args.surface in TRAVEL_SURFACES.values():
        _require_args(parser, args, "travel_tasks_path", "travel_database_path")
        configured.update(
            {
                "AGENTMEMORY_TRAVEL_TASKS_PATH": args.travel_tasks_path,
                "MEMORYARENA_TRAVEL_DATABASE_PATH": args.travel_database_path,
            }
        )
    elif args.surface in {
        surface
        for surfaces in FORMAL_REASONING_SURFACES_BY_MODE.values()
        for surface in surfaces.values()
    }:
        _require_args(
            parser,
            args,
            "formal_reasoning_tasks_path",
            "formal_reasoning_judge_model",
            "formal_reasoning_judge_base_url",
        )
        if args.formal_reasoning_judge_max_tokens < 1:
            parser.error("--formal-reasoning-judge-max-tokens must be positive")
        configured.update(
            {
                "AGENTMEMORY_FORMAL_REASONING_TASKS_PATH": (
                    args.formal_reasoning_tasks_path
                ),
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_MODEL": (
                    args.formal_reasoning_judge_model
                ),
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_BASE_URL": (
                    args.formal_reasoning_judge_base_url
                ),
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_TEMPERATURE": str(
                    args.formal_reasoning_judge_temperature
                ),
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_MAX_TOKENS": str(
                    args.formal_reasoning_judge_max_tokens
                ),
            }
        )
    elif args.surface == BROWSECOMP_BM25_INTEGRATION_SURFACE:
        _require_args(
            parser,
            args,
            "browsecomp_tasks_path",
            "browsecomp_bm25_index_path",
            "browsecomp_judge_model",
            "browsecomp_api_base_url",
        )
        if args.browsecomp_judge_max_tokens < 1:
            parser.error("--browsecomp-judge-max-tokens must be positive")
        configured.update(
            {
                "AGENTMEMORY_BROWSECOMP_TASKS_PATH": args.browsecomp_tasks_path,
                "MEMORYARENA_BROWSECOMP_BM25_INDEX_PATH": (
                    args.browsecomp_bm25_index_path
                ),
                "AGENTMEMORY_BROWSECOMP_JUDGE_MODEL": args.browsecomp_judge_model,
                "AGENTMEMORY_BROWSECOMP_JUDGE_MAX_TOKENS": str(
                    args.browsecomp_judge_max_tokens
                ),
                "OPENAI_BASE_URL": args.browsecomp_api_base_url,
            }
        )
    elif args.surface in BROWSECOMP_SURFACES.values():
        _require_args(
            parser,
            args,
            "browsecomp_tasks_path",
            "browsecomp_index_path",
            "browsecomp_corpus_path",
            "browsecomp_corpus_manifest",
            "browsecomp_embedding_provider",
            "browsecomp_judge_model",
            "browsecomp_api_base_url",
        )
        if args.browsecomp_judge_max_tokens < 1:
            parser.error("--browsecomp-judge-max-tokens must be positive")
        configured.update(
            {
                "AGENTMEMORY_BROWSECOMP_TASKS_PATH": args.browsecomp_tasks_path,
                "MEMORYARENA_BROWSECOMP_INDEX_PATH": args.browsecomp_index_path,
                "MEMORYARENA_BROWSECOMP_CORPUS_PATH": args.browsecomp_corpus_path,
                "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST": (
                    args.browsecomp_corpus_manifest
                ),
                "AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER": (
                    args.browsecomp_embedding_provider
                ),
                "AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL": (
                    args.browsecomp_embedding_model
                ),
                "AGENTMEMORY_BROWSECOMP_JUDGE_MODEL": args.browsecomp_judge_model,
                "AGENTMEMORY_BROWSECOMP_JUDGE_MAX_TOKENS": str(
                    args.browsecomp_judge_max_tokens
                ),
                "OPENAI_BASE_URL": args.browsecomp_api_base_url,
            }
        )
    else:  # pragma: no cover - argparse choices keep this unreachable.
        parser.error(f"unsupported AgentMemoryGym surface {args.surface!r}")

    if args.surface in V3_SURFACES:
        first_add_reward = (
            0.0
            if args.memory_first_add_reward is None
            else args.memory_first_add_reward
        )
        first_later_retrieve_reward = (
            0.0
            if args.memory_first_later_retrieve_reward is None
            else args.memory_first_later_retrieve_reward
        )
        configured.update(
            {
                "AGENTMEMORY_FIRST_ADD_REWARD": str(first_add_reward),
                "AGENTMEMORY_FIRST_LATER_RETRIEVE_REWARD": str(
                    first_later_retrieve_reward
                ),
                "AGENTMEMORY_EXACT_REPEAT_REWARD": str(args.memory_exact_repeat_reward),
                "AGENTMEMORY_INVALID_ACTION_REWARD": str(args.invalid_action_reward),
            }
        )
    for key, value in configured.items():
        os.environ[key] = value

    for legacy_key in [
        "AGENTMEMORY_CATALOG_INDEX_PATH",
        "AGENTMEMORY_SEARCH_TIMEOUT_MS",
    ]:
        if os.environ.get(legacy_key):
            parser.error(f"native launch refuses legacy SQLite variable {legacy_key}")

    uvicorn.run(
        "agentenv_agentmemory.server:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


def _require_args(parser: argparse.ArgumentParser, args, *names: str) -> None:
    missing = [
        f"--{name.replace('_', '-')}" for name in names if not getattr(args, name)
    ]
    if missing:
        parser.error(f"surface {args.surface!r} requires: " + ", ".join(missing))


def _configure_programmatic_memory_surface(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    cli_prefix: str,
    env_prefix: str,
    tasks_per_orbit: int,
    fixed_window_mode: str,
    reseeded_stream_mode: str,
    zero_memory_rewards: bool = False,
) -> dict[str, str]:
    product_pool_name = f"{cli_prefix}_product_pool"
    product_pool_sha_name = f"{cli_prefix}_product_pool_sha256"
    task_count_name = f"{cli_prefix}_task_count"
    generator_seed_name = f"{cli_prefix}_generator_seed"
    provider_mode_name = f"{cli_prefix}_provider_mode"
    start_orbit_name = f"{cli_prefix}_start_orbit"
    _require_args(
        parser,
        args,
        "items_file",
        "attributes_file",
        "search_root",
        "java_home",
        "lucene_index_manifest",
        product_pool_name,
        product_pool_sha_name,
        task_count_name,
    )
    generator_seed = getattr(args, generator_seed_name)
    rendered_prefix = cli_prefix.replace("_", "-")
    if generator_seed is None:
        parser.error(f"surface requires --{rendered_prefix}-generator-seed")
    if args.split == "all":
        parser.error(f"{rendered_prefix} data requires one explicit split")
    task_count = getattr(args, task_count_name)
    if task_count <= 0 or task_count % tasks_per_orbit:
        parser.error(
            f"--{rendered_prefix}-task-count must be a positive multiple of "
            f"{tasks_per_orbit}"
        )
    provider_mode = getattr(args, provider_mode_name) or (
        reseeded_stream_mode if args.split == "train" else fixed_window_mode
    )
    start_orbit = getattr(args, start_orbit_name)
    if start_orbit < 0:
        parser.error(f"--{rendered_prefix}-start-orbit must be non-negative")
    if provider_mode == reseeded_stream_mode:
        if args.split != "train":
            parser.error(
                f"--{rendered_prefix}-provider-mode reseeded_stream is training-only"
            )
        if start_orbit != 0:
            parser.error(
                f"reseeded_stream requires --{rendered_prefix}-start-orbit 0"
            )
    if zero_memory_rewards and any(
        value is not None and float(value) != 0.0
        for value in (
            args.memory_first_add_reward,
            args.memory_first_later_retrieve_reward,
        )
    ):
        parser.error(
            f"{rendered_prefix} requires --memory-first-add-reward and "
            "--memory-first-later-retrieve-reward to remain zero"
        )
    first_add_reward = 0.0 if zero_memory_rewards else (
        FIRST_VALID_ADD_BONUS
        if args.memory_first_add_reward is None
        else args.memory_first_add_reward
    )
    first_later_retrieve_reward = 0.0 if zero_memory_rewards else (
        FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS
        if args.memory_first_later_retrieve_reward is None
        else args.memory_first_later_retrieve_reward
    )
    return {
        "MEMORYARENA_WEBSHOP_ITEMS_FILE": args.items_file,
        "MEMORYARENA_WEBSHOP_ATTR_FILE": args.attributes_file,
        "MEMORYARENA_WEBSHOP_SEARCH_ROOT": args.search_root,
        "MEMORYARENA_WEBSHOP_JAVA_HOME": args.java_home,
        "MEMORYARENA_LUCENE_INDEX_MANIFEST": args.lucene_index_manifest,
        f"{env_prefix}_PRODUCT_POOL": getattr(args, product_pool_name),
        f"{env_prefix}_PRODUCT_POOL_SHA256": getattr(args, product_pool_sha_name),
        f"{env_prefix}_TASK_COUNT": str(task_count),
        f"{env_prefix}_GENERATOR_SEED": str(generator_seed),
        f"{env_prefix}_PROVIDER_MODE": provider_mode,
        f"{env_prefix}_START_ORBIT": str(start_orbit),
        "AGENTMEMORY_SPLIT": args.split,
        "AGENTMEMORY_WEBSHOP_PRICE_SEED": str(args.price_seed),
        "AGENTMEMORY_FIRST_VALID_ADD_REWARD": str(first_add_reward),
        "AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD": str(
            first_later_retrieve_reward
        ),
        "AGENTMEMORY_LTM_INVENTORY_MODE": args.ltm_inventory_mode,
        "AGENTMEMORY_LTM_TRANSITION_NOTICE_MODE": args.ltm_transition_notice_mode,
        "AGENTMEMORY_MEMORY_PROMPT_MODE": args.memory_prompt_mode,
        "AGENTMEMORY_ACTION_LISTING_MODE": args.action_listing_mode,
    }


if __name__ == "__main__":
    launch()
