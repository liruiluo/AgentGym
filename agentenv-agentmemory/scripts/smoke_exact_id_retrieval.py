from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentenv_agentmemory.environment import (
    AgentMemoryEnv,
    MemoryEntry,
    rank_memory_entries_bm25,
)


def memory(memory_id: str, product_id: str, step: int, *, suffix: str = "") -> MemoryEntry:
    return MemoryEntry(
        memory_id=memory_id,
        key="selected_product",
        value=f"product_id={product_id}; {suffix}".strip(),
        created_step=step,
        updated_step=step,
    )


def assert_exact_underscore_id_beats_bag_of_words_collisions() -> None:
    entries = [
        memory("mem_0000", "ma_c_b_d", 1),
        memory("mem_0001", "ma_d_b_c", 2),
        memory("mem_0002", "ma_b_c_d", 3),
        memory("mem_0003", "ma_d_c_b", 4),
        memory("mem_0004", "ma_c_d_b", 5),
        memory("mem_0005", "ma_c_d_b", 6),
    ]
    ranked = rank_memory_entries_bm25("ma_c_d_b", entries, top_k=3)
    ranked_ids = [entry.memory_id for entry, _ in ranked]
    assert ranked_ids[:2] == ["mem_0005", "mem_0004"], ranked_ids


def assert_explicit_product_id_query_uses_exact_value() -> None:
    entries = [
        memory("mem_0000", "ma_c_b_d", 1),
        memory("mem_0001", "ma_d_b_c", 2),
        memory("mem_0002", "ma_c_d_b", 3),
    ]
    ranked = rank_memory_entries_bm25(
        "retrieve the prior product_id=ma_c_d_b now",
        entries,
        top_k=1,
    )
    assert ranked[0][0].memory_id == "mem_0002", ranked


def assert_explicit_product_id_dominates_other_opaque_text() -> None:
    entries = [
        memory("mem_target", "ma_target_id", 1, suffix="title=plain"),
        memory(
            "mem_distractor",
            "ma_other_id",
            2,
            suffix="title=ultra-pro",
        ),
    ]
    ranked = rank_memory_entries_bm25(
        "retrieve product_id=ma_target_id and prior ultra-pro note",
        entries,
        top_k=2,
    )
    assert ranked[0][0].memory_id == "mem_target", ranked


def assert_non_product_opaque_text_keeps_bm25_order() -> None:
    entries = [
        memory(
            "mem_older_better_bm25",
            "ma_old_id",
            1,
            suffix="title=ultra-pro ultra-pro ultra-pro",
        ),
        memory(
            "mem_newer_weaker_bm25",
            "ma_new_id",
            2,
            suffix="title=ultra-pro",
        ),
    ]
    ranked = rank_memory_entries_bm25("ultra-pro", entries, top_k=2)
    assert ranked[0][1] > ranked[1][1], ranked
    assert ranked[0][0].memory_id == "mem_older_better_bm25", ranked


def assert_schema_terms_do_not_trigger_exact_recency_priority() -> None:
    entries = [
        memory("mem_0000", "old", 1, suffix="source_option=d d d d"),
        memory("mem_0001", "new", 2, suffix="source_option=d"),
    ]
    ranked = rank_memory_entries_bm25("source_option=d", entries, top_k=2)
    assert ranked[0][0].memory_id == "mem_0000", ranked


def assert_formal_train_task_2(
    data_path: Path,
    split_dir: Path,
) -> dict[str, str]:
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    os.environ["AGENTMEMORY_ROUND_ECONOMY_REPAIR"] = "1"
    os.environ["AGENTMEMORY_SUPPRESS_CANONICAL_BUY_TEMPLATE"] = "1"
    os.environ["AGENTMEMORY_SUPPRESS_DIRECT_BUY_SCHEMA_TEMPLATE"] = "1"
    for name in (
        "AGENTMEMORY_SOURCE_KEY_CLOSE_LOOP_DIAGNOSTIC",
        "AGENTMEMORY_FUZZY_PRODUCT_REFERENCE_DIAGNOSTIC",
        "AGENTMEMORY_SOURCE_SEARCH_SUMMARY",
        "AGENTMEMORY_SHOW_CONCRETE_MEMORY_BUY_ACTIONS",
        "AGENTMEMORY_SHOW_CONCRETE_BUY_ACTIONS",
    ):
        os.environ.pop(name, None)

    env = AgentMemoryEnv(
        data_path=data_path,
        split="train",
        split_dir=split_dir,
    )
    assert len(env.tasks) == 120, len(env.tasks)
    task = env.tasks[2]
    assert task.task_id == "memoryarena_bundled_shopping_c", task.task_id
    env.reset(data_idx=2)
    regression_query = ""
    regression_expected_memory_id = ""
    regression_expected_product_id = ""

    for subtask_index, subtask in enumerate(task.subtasks):
        if subtask_index > 0:
            previous_id = env.purchase_history[-1]["product_id"]
            query = f"selected product_id {previous_id}"
            _, _, done, _, info = env.step(
                "RETRIEVE "
                + json.dumps({"query": query, "top_k": 3})
            )
            assert not done, info
            expected_memory_id = f"mem_{subtask_index - 1:04d}"
            expected_prefix = f"[{expected_memory_id}] "
            assert env.short_term_context[0].startswith(expected_prefix), {
                "task_index": 2,
                "task_id": task.task_id,
                "subtask_index": subtask_index,
                "query": query,
                "expected_memory_id": expected_memory_id,
                "expected_product_id": previous_id,
                "actual_context": env.short_term_context,
            }
            if subtask_index == 4:
                regression_query = query
                regression_expected_memory_id = expected_memory_id
                regression_expected_product_id = previous_id

        target_id = subtask.target_product_id
        if subtask_index < len(task.subtasks) - 1:
            target = next(
                product
                for product in subtask.candidate_products
                if product.product_id == target_id
            )
            authored_value = (
                f"product_id={target.product_id}; "
                f"source_option={target.attributes.get('source_option')}"
            )
            _, _, done, _, info = env.step(
                "ADD "
                + json.dumps(
                    {
                        "key": f"selected_product_{subtask_index}",
                        "value": authored_value,
                    }
                )
            )
            assert not done, info

        _, reward, done, _, info = env.step(
            "BUY " + json.dumps({"product_id": target_id})
        )
        assert reward > 0.0, info
        if subtask_index < len(task.subtasks) - 1:
            assert not done, info
        else:
            assert done and info["episode_success"], info

    assert regression_query == "selected product_id ma_c_d_b", regression_query
    assert regression_expected_memory_id == "mem_0003", regression_expected_memory_id
    return {
        "task_id": task.task_id,
        "query": regression_query,
        "expected_memory_id": regression_expected_memory_id,
        "expected_product_id": regression_expected_product_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-data", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()

    task_2 = assert_formal_train_task_2(
        args.formal_data,
        args.split_dir,
    )
    assert_exact_underscore_id_beats_bag_of_words_collisions()
    assert_explicit_product_id_query_uses_exact_value()
    assert_explicit_product_id_dominates_other_opaque_text()
    assert_non_product_opaque_text_keeps_bm25_order()
    assert_schema_terms_do_not_trigger_exact_recency_priority()
    print(
        "AGENTMEMORY_EXACT_ID_RETRIEVAL_OK",
        "task_index=2",
        f"task_id={task_2['task_id']}",
        f"query={task_2['query']!r}",
        f"expected_memory_id={task_2['expected_memory_id']}",
        f"expected_product_id={task_2['expected_product_id']}",
    )


if __name__ == "__main__":
    main()
