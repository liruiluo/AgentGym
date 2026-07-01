from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from agentenv_agentmemory.memoryarena_converter import collect_target_asins, convert_file, read_jsonl

ASIN_BYTES_PATTERN = re.compile(rb"\b[A-Z0-9]{10}\b")
DEFAULT_INPUT_URL = "https://huggingface.co/datasets/ZexueHe/memoryarena/resolve/main/bundled_shopping/data.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze MemoryArena bundled-shopping data with catalog-assisted target resolution."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_URL, help="MemoryArena bundled_shopping data.jsonl path or URL.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for frozen data/report/splits/manifest.")
    parser.add_argument(
        "--product-db-root",
        type=Path,
        help="MemoryArena product DB root containing product_catalog/*.json. Used to discover relevant catalog shards.",
    )
    parser.add_argument(
        "--catalog-path",
        action="append",
        default=[],
        type=Path,
        help="Explicit catalog JSON file. If omitted, relevant shards are discovered from --product-db-root.",
    )
    parser.add_argument("--split-mode", choices=["ratio", "cycle"], default="ratio")
    parser.add_argument("--min-match-score", type=int, default=1)
    parser.add_argument("--ambiguous-policy", choices=["first", "fail"], default="first")
    parser.add_argument("--source-url", default=DEFAULT_INPUT_URL, help="Canonical source URL to record in manifest.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "memoryarena_agentmemory.jsonl"
    report_path = args.output_dir / "report.jsonl"
    manifest_path = args.output_dir / "freeze_manifest.json"

    records = read_jsonl(args.input)
    target_asins = collect_target_asins(records)
    catalog_paths = list(args.catalog_path)
    if not catalog_paths:
        if args.product_db_root is None:
            raise SystemExit("Either --catalog-path or --product-db-root is required for formal freeze.")
        catalog_paths = discover_relevant_catalog_paths(args.product_db_root, target_asins)
    if not catalog_paths:
        raise SystemExit("No catalog paths were selected for formal freeze.")

    stats = convert_file(
        args.input,
        output_path,
        split_dir=split_dir,
        report_path=report_path,
        split_mode=args.split_mode,
        min_match_score=args.min_match_score,
        ambiguous_policy=args.ambiguous_policy,
        catalog_paths=catalog_paths,
    )
    print(stats.marker(), flush=True)
    run_validator(output_path, split_dir)
    manifest = build_manifest(
        input_source=args.input,
        source_url=args.source_url,
        product_db_root=args.product_db_root,
        catalog_paths=catalog_paths,
        output_path=output_path,
        report_path=report_path,
        split_dir=split_dir,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        manifest["marker"],
        f"tasks={manifest['task_count']}",
        f"rows={manifest['report_rows']}",
        "splits=" + ",".join(f"{key}:{value}" for key, value in manifest["splits"].items()),
        f"ambiguous={manifest['ambiguous_matches']}",
        "resolvers=" + json.dumps(manifest["resolver_counts"], sort_keys=True),
        f"catalog_paths={len(catalog_paths)}",
    )


def discover_relevant_catalog_paths(product_db_root: Path, target_asins: set[str]) -> list[Path]:
    catalog_dir = product_db_root / "product_catalog" if (product_db_root / "product_catalog").is_dir() else product_db_root
    if not catalog_dir.is_dir():
        raise FileNotFoundError(f"Product catalog directory not found: {catalog_dir}")
    remaining = {asin.encode("ascii") for asin in target_asins if is_scan_candidate_asin(asin)}
    selected: list[Path] = []
    for catalog_path in sorted(catalog_dir.glob("*.json")):
        found = scan_catalog_file_for_asins(catalog_path, remaining)
        if found:
            selected.append(catalog_path)
            remaining -= found
        if not remaining:
            break
    return selected


def is_scan_candidate_asin(asin: str) -> bool:
    return len(asin) == 10 and asin.isascii() and asin.isalnum()


def scan_catalog_file_for_asins(catalog_path: Path, remaining_asins: set[bytes]) -> set[bytes]:
    found: set[bytes] = set()
    tail = b""
    with catalog_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            payload = tail + chunk
            for match in ASIN_BYTES_PATTERN.finditer(payload):
                token = match.group(0)
                if token in remaining_asins:
                    found.add(token)
            if remaining_asins <= found:
                return found
            tail = payload[-9:]
    return found


def run_validator(data_path: Path, split_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("validate_agentmemory_data.py")),
            "--data",
            str(data_path),
            "--split-dir",
            str(split_dir),
        ],
        check=True,
    )


def build_manifest(
    *,
    input_source: str,
    source_url: str,
    product_db_root: Path | None,
    catalog_paths: Iterable[Path],
    output_path: Path,
    report_path: Path,
    split_dir: Path,
) -> dict[str, object]:
    tasks = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in report_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    split_counts = {split: split_file_count(split_dir / f"{split}.txt") for split in ["train", "dev", "test"]}
    resolver_counts: dict[str, int] = {}
    for row in rows:
        resolver = str(row.get("resolver", "unknown"))
        resolver_counts[resolver] = resolver_counts.get(resolver, 0) + 1
    product_db_status = read_product_db_status(product_db_root) if product_db_root is not None else {}
    return {
        "marker": "AGENTMEMORY_MEMORYARENA_FORMAL_FREEZE_OK",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_url": source_url,
        "source_input": input_source,
        "source_sha256": sha256_file(Path(input_source)) if not input_source.startswith(("http://", "https://")) else None,
        "product_db_root": str(product_db_root) if product_db_root is not None else None,
        "product_db_status": product_db_status,
        "catalog_paths": [str(path) for path in catalog_paths],
        "task_count": len(tasks),
        "report_rows": len(rows),
        "splits": split_counts,
        "ambiguous_matches": sum(bool(row.get("ambiguous_match_ids")) for row in rows),
        "resolver_counts": resolver_counts,
        "target_asin_found": sum(bool(row.get("target_asin_found")) for row in rows),
        "min_match_score": min(row["match_score"] for row in rows),
        "max_match_score": max(row["match_score"] for row in rows),
        "part_files": count_part_files(output_path.parent),
    }


def read_product_db_status(product_db_root: Path) -> dict[str, object]:
    status_path = product_db_root / "download_status.json"
    if not status_path.exists():
        return {}
    return json.loads(status_path.read_text(encoding="utf-8"))


def split_file_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_part_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file() and path.name.endswith((".part", ".part.priority")))


if __name__ == "__main__":
    main()
