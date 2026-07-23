from __future__ import annotations

import argparse
import os

from .annotation_gate import ANNOTATION_GATE_MODES
from .domains import (
    BROWSECOMP_SURFACES,
    FORMAL_REASONING_SURFACES,
    TRAVEL_SURFACES,
    V3_SURFACES,
)
from .domains.browsecomp import BROWSECOMP_FROZEN_EMBEDDING_MODEL
from .reward_hierarchy import (
    FIRST_VALID_ADD_BONUS,
    FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
)


NATIVE_SURFACE = "memoryarena_webshop_native_v1"


def launch() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--surface",
        choices=[NATIVE_SURFACE, *V3_SURFACES],
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
    args = parser.parse_args()

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
    elif args.surface in FORMAL_REASONING_SURFACES.values():
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
