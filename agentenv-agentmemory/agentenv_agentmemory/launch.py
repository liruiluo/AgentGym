from __future__ import annotations

import argparse
import os

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
from .env_wrapper import LATENT_PREFERENCE_PROMPT_MODE, MEMORY_PROMPT_MODES
from .latent_preference import (
    PROVIDER_MODE_FIXED_WINDOW as LATENT_PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM as LATENT_PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES as LATENT_PROVIDER_MODES,
)
from .latent_preference_webshop_env import LATENT_PREFERENCE_SURFACE
from .procedural import (
    PROVIDER_MODE_FIXED_WINDOW,
    PROVIDER_MODE_RESEEDED_STREAM,
    PROVIDER_MODES,
)
from .procedural_webshop_env import PROCEDURAL_SURFACE


NATIVE_SURFACE = "memoryarena_webshop_native_v1"


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
            LATENT_PREFERENCE_SURFACE,
            *V3_SURFACES,
        ],
        required=True,
    )
    parser.add_argument("--memoryarena-root", required=True)
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
    parser.add_argument("--memoryarena-base-commit", required=True)
    parser.add_argument("--run-id", required=True)
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
    parser.add_argument("--latent-preference-product-pool")
    parser.add_argument("--latent-preference-product-pool-sha256")
    parser.add_argument("--latent-preference-task-count", type=int)
    parser.add_argument("--latent-preference-generator-seed", type=int)
    parser.add_argument(
        "--latent-preference-provider-mode",
        choices=LATENT_PROVIDER_MODES,
    )
    parser.add_argument("--latent-preference-start-orbit", type=int, default=0)
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

    if args.surface == LATENT_PREFERENCE_SURFACE:
        if args.memory_prompt_mode != LATENT_PREFERENCE_PROMPT_MODE:
            parser.error(
                "the latent-preference surface requires "
                f"--memory-prompt-mode {LATENT_PREFERENCE_PROMPT_MODE}"
            )
    elif args.memory_prompt_mode == LATENT_PREFERENCE_PROMPT_MODE:
        parser.error(
            "--memory-prompt-mode latent_preference_sop is only valid for the "
            "latent-preference surface"
        )

    configured = {
        "AGENTMEMORY_SURFACE": args.surface,
        "MEMORYARENA_ROOT": args.memoryarena_root,
        "MEMORYARENA_BASE_COMMIT": args.memoryarena_base_commit,
        "AGENTMEMORY_RUN_ID": args.run_id,
    }
    if args.surface == NATIVE_SURFACE:
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
    elif args.surface == PROCEDURAL_SURFACE:
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
    elif args.surface == LATENT_PREFERENCE_SURFACE:
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


if __name__ == "__main__":
    launch()
