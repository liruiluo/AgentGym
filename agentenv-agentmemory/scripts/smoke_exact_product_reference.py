from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from agentenv_agentmemory.environment import (
    AgentMemoryEnv,
    Product,
    ShoppingSubtask,
    ShoppingTask,
)


def configure_formal() -> None:
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    os.environ["AGENTMEMORY_ROUND_ECONOMY_REPAIR"] = "1"
    for name in (
        "AGENTMEMORY_SOURCE_KEY_CLOSE_LOOP_DIAGNOSTIC",
        "AGENTMEMORY_FUZZY_PRODUCT_REFERENCE_DIAGNOSTIC",
        "AGENTMEMORY_SOURCE_SEARCH_SUMMARY",
        "AGENTMEMORY_SHOW_CONCRETE_MEMORY_BUY_ACTIONS",
        "AGENTMEMORY_SHOW_CONCRETE_BUY_ACTIONS",
        "AGENTMEMORY_LEGACY_R4_PROTOCOL",
    ):
        os.environ.pop(name, None)


def source_products(order: tuple[str, ...] = ("a", "b", "c")) -> tuple[Product, ...]:
    products = {
        "a": Product(
            "ma_a_a_a",
            "A Duncan Hines Signature Cake Mix with Strawberry Supreme flavor in 16.5 oz.",
            {"source_option": "a"},
        ),
        "b": Product(
            "ma_a_a_b",
            "A Sweet N Low lemon cake mix with sugar-free ingredients.",
            {"source_option": "b"},
        ),
        "c": Product(
            "ma_a_a_c",
            "A Simple Mills Almond Flour Baking Mix with gluten free vanilla cake mix.",
            {"source_option": "c"},
        ),
    }
    return tuple(products[key] for key in order)


def build_regression_task(order: tuple[str, ...] = ("a", "b", "c")) -> ShoppingTask:
    return ShoppingTask(
        task_id="exact_product_reference_regression",
        title="Exact product reference regression",
        memory_dependency="cross_session_bundled_shopping_attributes",
        subtasks=(
            ShoppingSubtask(
                instruction="Select Cake Base; source_option is visible.",
                target_product_id="ma_a_a_c",
                candidate_products=source_products(order),
            ),
            ShoppingSubtask(
                instruction="Select dependent item using prior memory.",
                target_product_id="dependent_c",
                candidate_products=(
                    Product("dependent_c", "Dependent C", {"source_option": "c"}),
                ),
            ),
        ),
    )


def concrete_buy_ids(observation: str) -> list[str]:
    return re.findall(r'^BUY \{"product_id": "([^".]+)"\}$', observation, flags=re.MULTILINE)


def assert_exact_add_and_buy(order: tuple[str, ...]) -> None:
    env = AgentMemoryEnv(tasks=[build_regression_task(order)])
    initial_observation, _ = env.reset(data_idx=0)
    assert "complete product_id" in initial_observation, initial_observation
    observation, reward, done, _, info = env.step(
        'ADD {"key":"selected_product",'
        '"value":"product_id=ma_a_a_c; source_option=c"}'
    )
    assert not done and reward > 0.0
    assert info["tool_ops"][0]["referenced_product_ids"] == ["ma_a_a_c"], info
    assert env.memory_product_refs == {"ma_a_a_c": {"mem_0000"}}
    assert concrete_buy_ids(observation) == ["ma_a_a_c"], observation
    _, reward, done, _, info = env.step('BUY {"product_id":"ma_a_a_c"}')
    assert reward > 0.0 and not done, (reward, info)
    assert info["current_subtask_index"] == 1


def assert_title_and_attribute_only_do_not_link(order: tuple[str, ...]) -> None:
    env = AgentMemoryEnv(tasks=[build_regression_task(order)])
    env.reset(data_idx=0)
    selected = next(
        product for product in env.current_subtask().candidate_products
        if product.product_id == "ma_a_a_c"
    )
    observation, reward, done, _, info = env.step(
        "ADD "
        + json.dumps(
            {
                "key": "ambiguous_without_id",
                "value": f"title={selected.title}; source_option=c",
            }
        )
    )
    assert not done and reward == 0.0, (reward, info)
    assert info["tool_ops"][0]["referenced_product_ids"] == [], info
    assert env.memory_product_refs == {}, env.memory_product_refs
    assert concrete_buy_ids(observation) == [], observation


def assert_exact_update_relinks() -> None:
    env = AgentMemoryEnv(tasks=[build_regression_task()])
    env.reset(data_idx=0)
    env.step(
        'ADD {"key":"selected_product",'
        '"value":"product_id=ma_a_a_a; source_option=a"}'
    )
    observation, _, _, _, info = env.step(
        'UPDATE {"memory_id":"mem_0000",'
        '"value":"product_id=ma_a_a_c; source_option=c"}'
    )
    assert info["tool_ops"][0]["referenced_product_ids"] == ["ma_a_a_c"], info
    assert env.memory_product_refs == {"ma_a_a_c": {"mem_0000"}}
    assert concrete_buy_ids(observation) == ["ma_a_a_c"], observation
    _, reward, _, _, info = env.step('BUY {"product_id":"ma_a_a_c"}')
    assert reward > 0.0 and info["current_subtask_index"] == 1


def assert_complete_id_boundary() -> None:
    products = [
        Product("item", "Short", {"source_option": "a"}),
        Product("item-long", "Long", {"source_option": "b"}),
        Product("item-longer", "Longer", {"source_option": "c"}),
    ]
    referenced = AgentMemoryEnv.exact_product_ids_in_text(
        "policy wrote product_id=item-long; keep it", products
    )
    assert referenced == {"item-long"}, referenced


def assert_legacy_fuzzy_path_is_opt_in() -> None:
    os.environ["AGENTMEMORY_FUZZY_PRODUCT_REFERENCE_DIAGNOSTIC"] = "1"
    try:
        env = AgentMemoryEnv(tasks=[build_regression_task()])
        env.reset(data_idx=0)
        observation, _, _, _, info = env.step(
            'ADD {"key":"selected_product",'
            '"value":"product_id=ma_a_a_c; source_option=c"}'
        )
        assert info["tool_ops"][0]["referenced_product_ids"] == [
            "ma_a_a_a",
            "ma_a_a_c",
        ], info
        assert concrete_buy_ids(observation) == ["ma_a_a_a"], observation
        _, reward, done, _, info = env.step('BUY {"product_id":"ma_a_a_c"}')
        assert reward < 0.0 and not done
        assert info["reward_components"][0]["name"] == (
            "source_purchase_blocked_by_stale_memory_anchor"
        )
    finally:
        os.environ.pop("AGENTMEMORY_FUZZY_PRODUCT_REFERENCE_DIAGNOSTIC", None)


def audit_formal_train(data_path: Path, split_dir: Path) -> tuple[int, int]:
    env = AgentMemoryEnv(
        data_path=data_path,
        split="train",
        split_dir=split_dir,
    )
    task_count = len(env.tasks)
    subtask_count = 0
    assert task_count == 120, task_count
    for task_index in range(task_count):
        initial_observation, _ = env.reset(data_idx=task_index)
        assert "complete product_id" in initial_observation, task_index
        expected_subtasks = len(env.require_task().subtasks)
        for subtask_index in range(expected_subtasks):
            assert env.current_subtask_index == subtask_index
            subtask = env.current_subtask()
            target_id = subtask.target_product_id
            target = next(
                product for product in subtask.candidate_products
                if product.product_id == target_id
            )
            if env.current_subtask_requires_prior_memory() and env.long_term_memory:
                env.step(
                    'RETRIEVE {"query":"selected_product product_id source_option",'
                    '"top_k":3}'
                )
            source_option = target.attributes.get("source_option", "unknown")
            observation, _, done, _, info = env.step(
                "ADD "
                + json.dumps(
                    {
                        "key": f"selected_product_{subtask_index}",
                        "value": (
                            f"product_id={target_id}; "
                            f"source_option={source_option}"
                        ),
                    }
                )
            )
            assert not done
            refs = info["tool_ops"][0]["referenced_product_ids"]
            assert refs == [target_id], (task_index, subtask_index, refs, target_id)
            wrong_menu_ids = [
                product_id for product_id in concrete_buy_ids(observation)
                if product_id != target_id
            ]
            assert not wrong_menu_ids, (
                task_index,
                subtask_index,
                target_id,
                wrong_menu_ids,
            )
            _, reward, done, _, info = env.step(
                "BUY " + json.dumps({"product_id": target_id})
            )
            assert reward > 0.0, (
                task_index,
                subtask_index,
                target_id,
                reward,
                info["reward_components"],
            )
            assert info["current_subtask_index"] == subtask_index + 1
            subtask_count += 1
        assert done and info["episode_success"], (task_index, info)
    return task_count, subtask_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-data", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_formal()
    assert_exact_add_and_buy(("a", "b", "c"))
    assert_exact_add_and_buy(("c", "b", "a"))
    assert_exact_add_and_buy(("b", "a", "c"))
    assert_title_and_attribute_only_do_not_link(("a", "b", "c"))
    assert_title_and_attribute_only_do_not_link(("c", "b", "a"))
    assert_exact_update_relinks()
    assert_complete_id_boundary()
    assert_legacy_fuzzy_path_is_opt_in()
    task_count, subtask_count = audit_formal_train(
        args.formal_data, args.split_dir
    )
    print(
        "AGENTMEMORY_EXACT_PRODUCT_REFERENCE_OK",
        f"tasks={task_count}",
        f"subtasks={subtask_count}",
        "permutations=3",
    )


if __name__ == "__main__":
    main()
