from __future__ import annotations

import argparse
from pathlib import Path

from agentenv_agentmemory.memoryarena_converter import convert_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MemoryArena bundled_shopping JSONL to AgentMemoryGym JSONL.")
    parser.add_argument("--input", required=True, help="MemoryArena bundled_shopping data.jsonl path or HTTPS URL.")
    parser.add_argument("--output", required=True, type=Path, help="Output AgentMemoryGym JSONL path.")
    parser.add_argument("--split-dir", required=True, type=Path, help="Directory for generated train/dev/test split files.")
    parser.add_argument("--report", type=Path, help="Optional JSONL report with target-match audit rows.")
    parser.add_argument("--limit", type=int, help="Optional max number of source records to convert.")
    parser.add_argument("--split-mode", choices=["ratio", "cycle"], default="ratio")
    parser.add_argument("--min-match-score", type=int, default=1)
    parser.add_argument(
        "--catalog-path",
        action="append",
        default=[],
        help=(
            "Optional MemoryArena product DB JSON file or directory. "
            "If a product DB root is passed, product_catalog/*.json is used and huge items_shuffle.json is skipped."
        ),
    )
    parser.add_argument(
        "--ambiguous-policy",
        choices=["first", "fail"],
        default="first",
        help="How to handle tied target-option matches. Ties are always recorded in the report.",
    )
    args = parser.parse_args()

    stats = convert_file(
        args.input,
        args.output,
        split_dir=args.split_dir,
        report_path=args.report,
        limit=args.limit,
        split_mode=args.split_mode,
        min_match_score=args.min_match_score,
        ambiguous_policy=args.ambiguous_policy,
        catalog_paths=args.catalog_path,
    )
    print(stats.marker())


if __name__ == "__main__":
    main()
