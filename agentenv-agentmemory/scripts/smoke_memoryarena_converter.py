from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentenv_agentmemory.environment import AgentMemoryEnv, load_task_dataset
from agentenv_agentmemory.catalog_search import build_sqlite_catalog_index
from agentenv_agentmemory.memoryarena_converter import convert_file


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    fixture = package_root / "agentenv_agentmemory" / "data" / "fixtures" / "memoryarena_bundled_shopping_sample.jsonl"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        output = temp_root / "agentmemory.jsonl"
        split_dir = temp_root / "splits"
        report = temp_root / "report.jsonl"
        stats = convert_file(
            fixture,
            output,
            split_dir=split_dir,
            report_path=report,
            split_mode="cycle",
            min_match_score=1,
        )
        assert stats.task_count == 3, stats
        assert stats.split_counts == {"train": 1, "dev": 1, "test": 1}, stats
        for split in ["train", "dev", "test"]:
            tasks = load_task_dataset(data_path=output, split=split, split_dir=split_dir)
            assert len(tasks) == 1, (split, tasks)
            assert tasks[0].task_id.startswith("memoryarena_bundled_shopping_"), tasks[0].task_id
        run_target_plan_smoke(output)
        report_lines = report.read_text(encoding="utf-8").splitlines()
        assert len(report_lines) == 9, len(report_lines)
        run_catalog_resolver_smoke(temp_root)
        run_candidate_metadata_smoke(temp_root)
        run_no_partial_candidate_metadata_smoke(temp_root)
        run_catalog_search_smoke(temp_root)
    print("AGENTMEMORY_MEMORYARENA_CONVERTER_SMOKE_OK")


def run_target_plan_smoke(data_path: Path) -> None:
    tasks = load_task_dataset(data_path=data_path)
    for data_idx, task in enumerate(tasks):
        env = AgentMemoryEnv(data_path=data_path)
        _, info = env.reset(data_idx=data_idx)
        for subtask in task.subtasks:
            _, _, done, _, info = env.step(f'BUY {{"product_id": "{subtask.target_product_id}"}}')
        assert done, info
        assert info["episode_success"], info

        first_subtask = task.subtasks[0]
        wrong_products = [
            product.product_id
            for product in first_subtask.candidate_products
            if product.product_id != first_subtask.target_product_id
        ]
        if wrong_products:
            env = AgentMemoryEnv(data_path=data_path)
            env.reset(data_idx=data_idx)
            _, reward, done, _, info = env.step(f'BUY {{"product_id": "{wrong_products[0]}"}}')
            assert reward < 0 and not done and info["compatibility_violations"], info


def run_catalog_resolver_smoke(temp_root: Path) -> None:
    source = temp_root / "ambiguous_memoryarena.jsonl"
    catalog = temp_root / "catalog.json"
    output = temp_root / "catalog_resolved.jsonl"
    split_dir = temp_root / "catalog_splits"
    report = temp_root / "catalog_report.jsonl"
    record = {
        "id": "catalog_tiebreak_0",
        "category": "unit_test",
        "questions": [
            "\n".join(
                [
                    "Product 1:",
                    "### Select compatible mount",
                    "**Goal:** Pick the catalog target.",
                    "**Available Options:**",
                    "- A white steel wall mount for a 75 inch television.",
                    "- A black articulating wall mount for a 75 inch television.",
                ]
            )
        ],
        "answers": [
            {
                "target_asin": "BUNIT0002",
                "attributes": ["75 inch television wall mount"],
            }
        ],
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    catalog.write_text(
        json.dumps(
            [
                {
                    "asin": "BUNIT0002",
                    "name": "Black articulating wall mount for 75 inch television",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    stats = convert_file(
        source,
        output,
        split_dir=split_dir,
        report_path=report,
        split_mode="cycle",
        min_match_score=1,
        catalog_paths=[catalog],
    )
    assert stats.task_count == 1, stats
    assert stats.ambiguous_matches == 0, stats
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["resolver"] == "asin_catalog", rows[0]
    task = load_task_dataset(data_path=output)[0]
    target = task.subtasks[0].target_product_id
    target_title = next(product.title for product in task.subtasks[0].candidate_products if product.product_id == target)
    assert "black articulating" in target_title.lower(), target_title


def run_candidate_metadata_smoke(temp_root: Path) -> None:
    source = temp_root / "candidate_metadata_memoryarena.jsonl"
    catalog = temp_root / "candidate_metadata_catalog.json"
    output = temp_root / "candidate_metadata_agentmemory.jsonl"
    split_dir = temp_root / "candidate_metadata_splits"
    report = temp_root / "candidate_metadata_report.jsonl"
    record = {
        "id": "candidate_metadata_0",
        "category": "unit_test",
        "questions": [
            "\n".join(
                [
                    "Product 1:",
                    "### Select cake mix",
                    "**Goal:** Buy the highest-priced one in available options.",
                    "**Available Options:**",
                    "- Alpha vanilla cake mix 16 oz.",
                    "- Beta chocolate cake mix 18 oz.",
                ]
            )
        ],
        "answers": [{"target_asin": "BUNITMETA2", "attributes": ["Beta", "chocolate", "18 oz"]}],
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    catalog.write_text(
        json.dumps(
            [
                {
                    "asin": "BUNITMETA1",
                    "name": "Alpha vanilla cake mix 16 oz",
                    "average_rating": 4.1,
                    "pricing": "$5.50",
                    "total_reviews": 12,
                },
                {
                    "asin": "BUNITMETA2",
                    "name": "Beta chocolate cake mix 18 oz",
                    "average_rating": 4.8,
                    "pricing": "$8.99",
                    "total_reviews": 99,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    stats = convert_file(
        source,
        output,
        split_dir=split_dir,
        report_path=report,
        split_mode="cycle",
        min_match_score=1,
        catalog_paths=[catalog],
        enrich_candidate_metadata=True,
        candidate_metadata_min_score=90,
    )
    assert stats.candidate_metadata_full_steps == 1, stats
    task = load_task_dataset(data_path=output)[0]
    for product in task.subtasks[0].candidate_products:
        assert product.attributes["average_rating"] in {4.1, 4.8}, product
        assert product.attributes["price_usd"] in {5.5, 8.99}, product
        assert product.attributes["total_reviews"] in {12, 99}, product
        assert "asin" not in product.attributes, product
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["candidate_metadata_status"] == "full", rows[0]
    assert rows[0]["candidate_metadata_enriched_count"] == 2, rows[0]


def run_no_partial_candidate_metadata_smoke(temp_root: Path) -> None:
    source = temp_root / "partial_candidate_metadata_memoryarena.jsonl"
    catalog = temp_root / "partial_candidate_metadata_catalog.json"
    output = temp_root / "partial_candidate_metadata_agentmemory.jsonl"
    split_dir = temp_root / "partial_candidate_metadata_splits"
    report = temp_root / "partial_candidate_metadata_report.jsonl"
    record = {
        "id": "candidate_metadata_partial_0",
        "category": "unit_test",
        "questions": [
            "\n".join(
                [
                    "Product 1:",
                    "### Select cake mix",
                    "**Goal:** Buy the highest-priced one in available options.",
                    "**Available Options:**",
                    "- Alpha vanilla cake mix 16 oz.",
                    "- Beta chocolate cake mix 18 oz.",
                ]
            )
        ],
        "answers": [{"target_asin": "BUNITMETA2", "attributes": ["Beta", "chocolate", "18 oz"]}],
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    catalog.write_text(
        json.dumps(
            [
                {
                    "asin": "BUNITMETA2",
                    "name": "Beta chocolate cake mix 18 oz",
                    "average_rating": 4.8,
                    "pricing": "$8.99",
                    "total_reviews": 99,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    convert_file(
        source,
        output,
        split_dir=split_dir,
        report_path=report,
        split_mode="cycle",
        min_match_score=1,
        catalog_paths=[catalog],
        enrich_candidate_metadata=True,
        candidate_metadata_min_score=90,
    )
    task = load_task_dataset(data_path=output)[0]
    for product in task.subtasks[0].candidate_products:
        assert "average_rating" not in product.attributes, product
        assert "price_usd" not in product.attributes, product
        assert "total_reviews" not in product.attributes, product
    rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["candidate_metadata_status"] == "partial", rows[0]
    assert rows[0]["candidate_metadata_enriched_count"] == 0, rows[0]


def run_catalog_search_smoke(temp_root: Path) -> None:
    source = temp_root / "search_memoryarena.jsonl"
    catalog = temp_root / "search_catalog.json"
    output = temp_root / "search_agentmemory.jsonl"
    split_dir = temp_root / "search_splits"
    report = temp_root / "search_report.jsonl"
    index_path = temp_root / "search_catalog.sqlite"
    record = {
        "id": "search_0",
        "category": "unit_test",
        "questions": [
            "\n".join(
                [
                    "Product 1:",
                    "### Select cake mix",
                    "**Goal:** Search the catalog and buy the highest-priced one in available options.",
                    "**Available Options:**",
                    "- Alpha vanilla cake mix 16 oz.",
                    "- Beta chocolate cake mix 18 oz.",
                ]
            )
        ],
        "answers": [{"target_asin": "BUNITSRCH2", "attributes": ["Beta", "chocolate", "18 oz"]}],
    }
    source.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    catalog.write_text(
        json.dumps(
            [
                {
                    "asin": "BUNITSRCH1",
                    "name": "Alpha vanilla cake mix 16 oz",
                    "average_rating": 4.1,
                    "pricing": "$5.50",
                    "total_reviews": 12,
                },
                {
                    "asin": "BUNITSRCH2",
                    "name": "Beta chocolate cake mix 18 oz",
                    "average_rating": 4.8,
                    "pricing": "$8.99",
                    "total_reviews": 99,
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    convert_file(
        source,
        output,
        split_dir=split_dir,
        report_path=report,
        split_mode="cycle",
        min_match_score=1,
        catalog_paths=[catalog],
    )
    build_sqlite_catalog_index([catalog], index_path)
    env = AgentMemoryEnv(data_path=output, catalog_index_path=index_path)
    observation, _ = env.reset()
    assert 'SEARCH {"query": "...", "top_k": 3}' in observation, observation
    observation, reward, done, _, info = env.step('SEARCH {"query":"Beta chocolate cake mix 18 oz","top_k":2}')
    assert reward == 0.0 and not done, info
    assert "Product search results" in observation, observation
    assert "Beta chocolate cake mix 18 oz" in observation, observation
    assert "price_usd=8.99" in observation, observation
    assert info["tool_ops"][0]["op"] == "SEARCH", info
    assert info["memory_ops"] == [], info


if __name__ == "__main__":
    main()
