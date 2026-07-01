from __future__ import annotations

import argparse
from pathlib import Path

from agentenv_agentmemory.catalog_search import build_sqlite_catalog_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact SQLite/FTS index for AgentMemoryGym SEARCH.")
    parser.add_argument(
        "--product-db-root",
        action="append",
        default=[],
        type=Path,
        help="MemoryArena product DB root or product_catalog directory. Can be repeated.",
    )
    parser.add_argument(
        "--catalog-path",
        action="append",
        default=[],
        type=Path,
        help="Explicit catalog JSON file/directory. Can be repeated.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output SQLite index path.")
    parser.add_argument("--limit", type=int, help="Optional product limit for smoke tests.")
    parser.add_argument("--no-replace", action="store_true", help="Do not delete an existing output index first.")
    args = parser.parse_args()

    catalog_paths = [*args.product_db_root, *args.catalog_path]
    if not catalog_paths:
        raise SystemExit("At least one --product-db-root or --catalog-path is required.")
    count = build_sqlite_catalog_index(catalog_paths, args.output, replace=not args.no_replace, limit=args.limit)
    print(f"AGENTMEMORY_CATALOG_SEARCH_INDEX_OK products={count} index={args.output}")


if __name__ == "__main__":
    main()
