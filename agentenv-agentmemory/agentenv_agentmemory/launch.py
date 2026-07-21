from __future__ import annotations

import argparse
import os

from .annotation_gate import ANNOTATION_GATE_MODES


NATIVE_SURFACE = "memoryarena_webshop_native_v1"


def launch() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--surface", choices=[NATIVE_SURFACE], required=True)
    parser.add_argument("--memoryarena-root", required=True)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--items-file", required=True)
    parser.add_argument("--attributes-file", required=True)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--java-home", required=True)
    parser.add_argument("--domain-data-path", required=True)
    parser.add_argument("--lucene-index-manifest", required=True)
    parser.add_argument("--annotation-audit-summary", required=True)
    parser.add_argument("--annotation-audit-chains", required=True)
    parser.add_argument("--annotation-manual-evidence", required=True)
    parser.add_argument("--memoryarena-base-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="train")
    parser.add_argument("--price-seed", type=int, default=233)
    parser.add_argument(
        "--annotation-gate-mode",
        choices=ANNOTATION_GATE_MODES,
        default="provisional",
    )
    parser.add_argument("--annotation-gate-manifest", required=True)
    parser.add_argument("--annotation-gate-manifest-sha256", required=True)
    args = parser.parse_args()

    configured = {
        "AGENTMEMORY_SURFACE": args.surface,
        "MEMORYARENA_ROOT": args.memoryarena_root,
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
        "MEMORYARENA_BASE_COMMIT": args.memoryarena_base_commit,
        "AGENTMEMORY_RUN_ID": args.run_id,
        "AGENTMEMORY_SPLIT": args.split,
        "AGENTMEMORY_WEBSHOP_PRICE_SEED": str(args.price_seed),
        "AGENTMEMORY_ANNOTATION_GATE_MODE": args.annotation_gate_mode,
        "AGENTMEMORY_ANNOTATION_GATE_MANIFEST": args.annotation_gate_manifest,
        "AGENTMEMORY_ANNOTATION_GATE_MANIFEST_SHA256": args.annotation_gate_manifest_sha256,
    }
    for key, value in configured.items():
        os.environ[key] = value

    for legacy_key in ["AGENTMEMORY_CATALOG_INDEX_PATH", "AGENTMEMORY_SEARCH_TIMEOUT_MS"]:
        if os.environ.get(legacy_key):
            parser.error(f"native launch refuses legacy SQLite variable {legacy_key}")

    uvicorn.run(
        "agentenv_agentmemory.server:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    launch()
