from __future__ import annotations

from pathlib import Path

import agentenv_agentmemory.environment as environment_module
from agentenv_agentmemory.catalog_search import CatalogSearchResult
from agentenv_agentmemory.environment import (
    AgentMemoryEnv,
    Product,
    ShoppingSubtask,
    ShoppingTask,
)


def task() -> ShoppingTask:
    return ShoppingTask(
        task_id="search_tool_observability",
        title="SEARCH structured result evidence",
        memory_dependency="cross_session_bundled_shopping_attributes",
        subtasks=(
            ShoppingSubtask(
                instruction="Select a source product for a later session.",
                target_product_id="product_a",
                candidate_products=(
                    Product("product_a", "Ultra Pro Source", {"source_option": "a"}),
                    Product("product_b", "Basic Source", {"source_option": "b"}),
                ),
            ),
            ShoppingSubtask(
                instruction="Select a dependent product.",
                target_product_id="dependent_a",
                candidate_products=(
                    Product("dependent_a", "Dependent A", {"source_option": "a"}),
                ),
            ),
        ),
    )


def result(title: str) -> CatalogSearchResult:
    return CatalogSearchResult(
        title=title,
        average_rating=4.8,
        price_usd=19.5,
        total_reviews=123,
        match_score=90,
    )


def main() -> None:
    original_search = environment_module.search_sqlite_catalog
    try:
        environment_module.search_sqlite_catalog = (
            lambda index_path, query, top_k: [result("Ultra Pro catalog row")]
        )
        env = AgentMemoryEnv(
            tasks=[task()],
            catalog_index_path=Path("/unused/catalog.sqlite"),
        )
        env.reset(data_idx=0)
        _, _, _, _, info = env.step(
            'SEARCH {"query": "product_a Ultra Pro Source", "top_k": 3}'
        )
        tool_op = info["tool_ops"][0]
        assert tool_op["result_count"] == 1, tool_op
        assert tool_op["matched_visible_candidate_ids"] == ["product_a"], tool_op
        assert "result_product_ids" not in tool_op, tool_op

        environment_module.search_sqlite_catalog = (
            lambda index_path, query, top_k: []
        )
        _, _, _, _, info = env.step(
            'SEARCH {"query": "no catalog match", "top_k": 3}'
        )
        tool_op = info["tool_ops"][0]
        assert tool_op["result_count"] == 0, tool_op
        assert tool_op["matched_visible_candidate_ids"] == [], tool_op
        assert "result_product_ids" not in tool_op, tool_op
    finally:
        environment_module.search_sqlite_catalog = original_search

    print("AGENTMEMORY_SEARCH_TOOL_OBSERVABILITY_OK")


if __name__ == "__main__":
    main()
