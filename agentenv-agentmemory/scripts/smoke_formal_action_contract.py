#!/usr/bin/env python3
"""Regression for the formal AgentMemoryGym action and memory-credit contract."""

from __future__ import annotations

import os

from agentenv_agentmemory.environment import (
    AgentMemoryEnv,
    Product,
    ShoppingSubtask,
    ShoppingTask,
)


def make_task() -> ShoppingTask:
    source = Product("source_a", "Source A", {"source_option": "Alpha"})
    dependent = Product("dependent_b", "Dependent B", {"source_option": "Beta"})
    return ShoppingTask(
        task_id="memoryarena_formal_action_contract",
        title="MemoryArena formal action contract",
        source="memoryarena_formal",
        memory_dependency="cross_session_product_attribute",
        curriculum_flags=frozenset(
            {
                "require_memory_before_source_buy",
                "require_retrieved_memory_for_dependent_buy",
            }
        ),
        subtasks=(
            ShoppingSubtask("Choose the source product.", (source,), "source_a"),
            ShoppingSubtask(
                "Choose the product compatible with the previous purchase.",
                (dependent,),
                "dependent_b",
            ),
        ),
    )


def main() -> None:
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    # Removed compatibility flags must not resurrect an old action surface.
    os.environ["AGENTMEMORY_ENABLE_LEGACY_GROUND"] = "1"
    os.environ["AGENTMEMORY_DISABLE_GROUND"] = "0"

    env = AgentMemoryEnv(tasks=[make_task()])
    observation, _ = env.reset()
    assert "GROUND" not in observation
    assert not hasattr(env, "action_ground")

    observation, reward, done, _, info = env.step(
        'GROUND {"candidate_id": "source_a", "memory_ids": ["C0"], "why": "legacy"}'
    )
    assert not done
    assert reward < 0
    assert "Unsupported action 'GROUND'" in observation
    assert info["tool_ops"] == []

    _, _, done, _, info = env.step(
        'ADD {"key": "selected_product", "value": "product_id=source_a, source_option=Alpha"}'
    )
    assert not done
    assert info["memory_state_diff"]["added"]

    observation, _, done, _, _ = env.step('BUY {"product_id": "source_a"}')
    assert not done
    assert "new shopping session" in observation

    observation, _, done, _, info = env.step(
        'RETRIEVE {"query": "source_a Alpha", "top_k": 3}'
    )
    assert not done
    assert info["memory_ops"][0]["op"] == "RETRIEVE"
    assert "source_a" in observation

    observation, reward, done, _, info = env.step(
        'BUY {"product_id": "dependent_b", "memory_ids": ["C0"]}'
    )
    assert not done
    assert reward < 0
    assert "BUY accepts only product_id" in observation
    assert info["current_subtask_index"] == 1

    _, reward, done, _, info = env.step('BUY {"product_id": "dependent_b"}')
    assert done
    assert info["episode_success"]
    component_names = {item["name"] for item in info["reward_components"]}
    assert "retrieved_memory_before_dependent_purchase" in component_names
    assert reward >= 2.45, (reward, info["reward_components"])
    print("AGENTMEMORY_FORMAL_ACTION_CONTRACT_OK")


if __name__ == "__main__":
    main()
