from __future__ import annotations

import os

from agentenv_agentmemory.environment import (
    AgentMemoryEnv,
    InitialMemorySpec,
    Product,
    ShoppingSubtask,
    ShoppingTask,
)


def run_plan(data_idx: int, actions: list[str]) -> dict:
    env = AgentMemoryEnv()
    _, info = env.reset(data_idx=data_idx)
    reward_sum = 0.0
    done = False
    for action in actions:
        _, reward, done, _, info = env.step(action)
        reward_sum += reward
    assert done, f"plan did not finish for data_idx={data_idx}: {info}"
    assert info["episode_success"], f"episode_success not set: {info}"
    assert info["progress_score"] == 1.0, f"unexpected progress: {info}"
    assert reward_sum > 0, f"unexpected reward_sum={reward_sum}"
    for key in [
        "task_id",
        "split",
        "source",
        "difficulty",
        "memory_dependency",
        "tool_ops",
        "memory_ops",
        "memory_state_diff",
        "compatibility_violations",
        "purchase_history",
        "session_trace",
    ]:
        assert key in info, f"missing info key {key}: {info}"
    return info


def assert_session_trace_boundary() -> None:
    env = AgentMemoryEnv()
    observation, info = env.reset(data_idx=0)
    assert info["session_trace"] == [], info
    assert "Current session short-term history: <empty>" in observation, observation

    observation, _, done, _, info = env.step(
        'ADD {"key": "tv_profile", "value": "Purchased TV: 75 inches, 32kg, VESA 400x400."}'
    )
    assert not done, info
    assert info["session_trace"], info
    assert "Action: ADD" in observation, observation
    assert "Active retrieved/summary context: <empty>" in observation, observation

    observation, _, done, _, info = env.step('BUY {"product_id": "tv_b"}')
    assert not done, info
    assert info["session_trace"] == [], info
    assert "Current session short-term history: <empty>" in observation, observation


def assert_memory_tool_contract() -> None:
    env = AgentMemoryEnv()
    observation, _ = env.reset(data_idx=0)

    observation, _, _, _, info = env.step(
        'ADD {"key": "tv_profile", "value": "Purchased TV: 75 inches, 32kg, VESA 400x400."}'
    )
    assert info["tool_ops"][0]["op"] == "ADD", info
    assert info["memory_ops"][0]["op"] == "ADD", info
    memory_id = info["memory_state_diff"]["added"][0]["memory_id"]

    observation, _, _, _, info = env.step(
        f'UPDATE {{"memory_id": "{memory_id}", "value": "Updated TV memory: 75-inch TV, 32kg, VESA 400x400."}}'
    )
    assert info["memory_ops"][0]["op"] == "UPDATE", info
    assert info["memory_state_diff"]["updated"][0]["before"]["value"].startswith("Purchased TV"), info
    assert info["memory_state_diff"]["updated"][0]["after"]["value"].startswith("Updated TV"), info

    observation, _, _, _, info = env.step('RETRIEVE {"query": "updated tv vesa", "top_k": 1}')
    assert info["memory_ops"][0]["op"] == "RETRIEVE", info
    assert "Updated TV memory" in observation, observation
    assert "C0: [mem_" in observation, observation

    observation, _, _, _, info = env.step(
        'SUMMARY {"text": "The selected TV is 75-inch, 32kg, VESA 400x400.", "source_ids": ["S0", "C0"]}'
    )
    assert info["memory_ops"][0]["op"] == "SUMMARY", info
    assert info["memory_ops"][0]["source"] == "policy_text", info
    assert info["memory_ops"][0]["source_ids"] == ["S0", "C0"], info
    assert "Summary (" in observation, observation
    assert "Active retrieved/summary context:" in observation, observation
    assert "C0: Summary (policy_text)" in observation, observation

    observation, _, _, _, info = env.step('FILTER {"keep_ids": ["C0"], "scope": "active"}')
    assert info["memory_ops"][0]["op"] == "FILTER", info
    assert info["memory_ops"][0]["keep_ids"] == ["C0"], info
    assert "Summary (" in observation, observation

    observation, reward, _, _, info = env.step('FILTER {"keep_ids": ["C999"], "scope": "active"}')
    assert reward < 0, info
    assert info["memory_ops"] == [], info
    assert "Invalid action" in observation, observation

    observation, _, _, _, info = env.step('FILTER {"drop_ids": ["C0"], "scope": "active"}')
    assert info["memory_ops"][0]["op"] == "FILTER", info
    assert info["memory_ops"][0]["drop_ids"] == ["C0"], info
    assert "Active retrieved/summary context: <empty>" in observation, observation

    observation, _, _, _, info = env.step(f'DELETE {{"memory_id": "{memory_id}"}}')
    assert info["memory_ops"][0]["op"] == "DELETE", info
    assert info["memory_state_diff"]["deleted"][0]["memory_id"] == memory_id, info

    observation, _, _, _, info = env.step('RETRIEVE {"query": "updated tv vesa", "top_k": 1}')
    assert info["memory_ops"][0]["op"] == "RETRIEVE", info
    assert "No relevant memory retrieved." in observation, observation

    env = AgentMemoryEnv()
    env.reset(data_idx=0)
    env.step('ADD {"key": "tv_profile", "value": "Purchased TV: 75 inches."}')
    observation, _, _, _, info = env.step('FILTER {"keep_ids": ["S0"], "scope": "session"}')
    assert info["memory_ops"][0]["op"] == "FILTER", info
    assert info["memory_ops"][0]["scope"] == "session", info
    assert "Action: ADD" in observation, observation
    observation, _, _, _, info = env.step('FILTER {"drop_ids": ["S0"], "scope": "session"}')
    assert info["memory_ops"][0]["removed"] >= 1, info
    assert "Action: ADD" not in observation, observation


def assert_dynamic_action_menu_hides_invisible_context_ids() -> None:
    env = AgentMemoryEnv()
    observation, _ = env.reset(data_idx=0)
    assert 'BUY {"product_id": "...", "memory_ids": ["C0"], "why": "..."}' not in observation, observation
    assert 'GROUND {"candidate_id": "...", "memory_ids": ["C0"], "why": "..."}' not in observation, observation
    assert 'UPDATE {"memory_id": "mem_0000", "value": "..."}' not in observation, observation
    assert "Do not use memory_ids yet" in observation, observation

    observation, _, _, _, info = env.step(
        'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV."}'
    )
    memory_id = info["memory_state_diff"]["added"][0]["memory_id"]
    assert f'UPDATE {{"memory_id": "{memory_id}", "value": "..."}}' in observation, observation
    assert 'BUY {"product_id": "...", "memory_ids": ["C0"], "why": "..."}' not in observation, observation

    observation, _, _, _, _ = env.step('BUY {"product_id": "tv_b"}')
    assert 'BUY {"product_id": "...", "memory_ids": ["C0"], "why": "..."}' not in observation, observation
    assert "RETRIEVE prior selected product/source_option memory" in observation, observation

    observation, _, _, _, _ = env.step('RETRIEVE {"query": "tv_b vesa", "top_k": 1}')
    assert 'BUY {"product_id": "...", "memory_ids": ["C0"], "why": "..."}' in observation, observation
    assert 'GROUND {"candidate_id": "...", "memory_ids": ["C0"], "why": "..."}' in observation, observation
    assert "not raw mem_0000 ids" in observation, observation


def assert_memory_product_ref_index_consistency() -> None:
    env = AgentMemoryEnv()
    env.reset(data_idx=0)
    _, _, _, _, info = env.step(
        'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV."}'
    )
    memory_id = info["memory_state_diff"]["added"][0]["memory_id"]
    assert env.source_session_has_stored_current_product()
    assert env.memory_ids_for_product("tv_b") == {memory_id}

    _, _, _, _, info = env.step('RETRIEVE {"query": "selected tv", "top_k": 1}')
    assert env.retrieved_memory_ids_this_session == {memory_id}
    assert "Nebula 4K TV" in env.short_term_context[0]

    observation, _, _, _, info = env.step(
        f'UPDATE {{"memory_id": "{memory_id}", "value": "Selected product_id tv_a Orion 4K TV."}}'
    )
    assert env.memory_ids_for_product("tv_b") == set()
    assert env.memory_ids_for_product("tv_a") == {memory_id}
    assert info["memory_ops"][0]["referenced_product_ids"] == ["tv_a"], info
    assert "Orion 4K TV" in observation, observation
    assert "Orion 4K TV" in env.short_term_context[0]

    observation, _, _, _, info = env.step(f'DELETE {{"memory_id": "{memory_id}"}}')
    assert env.memory_product_refs == {}, env.memory_product_refs
    assert env.retrieved_memory_ids_this_session == set()
    assert env.short_term_context == []
    assert not env.source_session_has_stored_current_product()
    assert "Active retrieved/summary context: <empty>" in observation, observation


def assert_bm25_retrieve() -> None:
    env = AgentMemoryEnv()
    env.reset(data_idx=0)
    env.step(
        'ADD {"key": "tv_profile", "value": "Purchased TV: 75 inches, 32kg, VESA 400x400 mounting pattern."}'
    )
    env.step('ADD {"key": "laptop_profile", "value": "Purchased laptop: 14 inches with usb-c port."}')
    _, _, _, _, info = env.step('RETRIEVE {"query": "vesa tv mount", "top_k": 1}')
    assert info["memory_ops"][0]["op"] == "RETRIEVE", info
    assert len(env.short_term_context) == 1, env.short_term_context
    assert "tv_profile: Purchased TV: 75 inches, 32kg, VESA 400x400 mounting pattern." in env.short_term_context[0]


def assert_wrong_purchase(data_idx: int, actions: list[str]) -> None:
    env = AgentMemoryEnv(buy_semantics="retry")
    env.reset(data_idx=data_idx)
    last = None
    for action in actions:
        last = env.step(action)
    assert last is not None
    _, reward, done, _, info = last
    assert reward < 0, f"wrong purchase should be penalized: reward={reward}, info={info}"
    assert not done, f"wrong purchase should not finish: {info}"
    assert info["compatibility_violations"], f"missing compatibility violation: {info}"


def assert_memory_chain_reward_shaping() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        memory_chain_reward = 0.0
        reward_events: list[str] = []
        for action in [
            'ADD {"key": "tv_profile", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 tv_weight_kg 32 vesa 400x400."}',
            'BUY {"product_id": "tv_b"}',
            'RETRIEVE {"query": "tv profile size weight vesa", "top_k": 3}',
            'BUY {"product_id": "mount_b"}',
            'RETRIEVE {"query": "tv profile size", "top_k": 3}',
            'BUY {"product_id": "console_b"}',
        ]:
            _, reward, done, _, info = env.step(action)
            memory_chain_reward += reward
            reward_events.extend(item["name"] for item in info.get("reward_components", []))
        assert done, info
        assert "memory_written_before_source_purchase" in reward_events, reward_events
        assert "retrieved_memory_before_dependent_purchase" in reward_events, reward_events

        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        no_memory_reward = 0.0
        for action in [
            'BUY {"product_id": "tv_b"}',
            'BUY {"product_id": "mount_b"}',
            'BUY {"product_id": "console_b"}',
        ]:
            _, reward, done, _, info = env.step(action)
            no_memory_reward += reward
        assert done, info
        assert memory_chain_reward > no_memory_reward, (memory_chain_reward, no_memory_reward)
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_candidate_memory_grounding_reward_shaping() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        reward_events: list[str] = []
        for action in [
            'ADD {"key": "tv_profile", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 tv_weight_kg 32 vesa 400x400."}',
            'BUY {"product_id": "tv_b"}',
            'RETRIEVE {"query": "tv profile size weight vesa", "top_k": 3}',
            'GROUND {"candidate_id": "mount_b", "memory_ids": ["C0"], "why": "C0 says the prior selected TV is tv_b, 75-inch, 32kg, VESA 400x400; mount_b supports 65-85 inch, 45kg, and 400x400."}',
            'BUY {"product_id": "mount_b"}',
            'RETRIEVE {"query": "tv profile size", "top_k": 3}',
            'GROUND {"candidate_id": "console_b", "memory_ids": ["C0"], "why": "C0 says the prior selected TV is 75-inch; console_b supports 70-85 inch TVs."}',
            'BUY {"product_id": "console_b"}',
        ]:
            observation, _, done, _, info = env.step(action)
            reward_events.extend(item["name"] for item in info.get("reward_components", []))
            if action.startswith("GROUND"):
                assert "Grounded candidate" in observation, observation
                assert env.last_grounding_this_session is not None
        assert done, info
        assert "candidate_memory_grounding_valid" in reward_events, reward_events
        assert "grounded_dependent_buy" in reward_events, reward_events

        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "tv_profile", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 vesa 400x400."}'
        )
        env.step('BUY {"product_id": "tv_b"}')
        env.step('RETRIEVE {"query": "tv profile size weight vesa", "top_k": 3}')
        observation, reward, _, _, info = env.step(
            'GROUND {"candidate_id": "mount_b", "memory_ids": ["C0"], "why": "C0 says tv_b is 75-inch with VESA 400x400; mount_b supports that compatibility."}'
        )
        assert reward > 0.0, (reward, info)
        assert "candidate_memory_grounding_valid" in [
            item["name"] for item in info.get("reward_components", [])
        ], info
        observation, reward, _, _, info = env.step(
            'GROUND {"candidate_id": "mount_b", "memory_ids": ["C0"], "why": "C0 still says tv_b is 75-inch with VESA 400x400; mount_b supports that compatibility."}'
        )
        repeat_names = [item["name"] for item in info.get("reward_components", [])]
        assert reward < 0.0, (reward, info)
        assert "candidate_memory_grounding_repeat_noop" in repeat_names, info
        assert "Repeated same GROUND" in observation and "no-progress action" in observation, observation
        observation, reward, _, _, info = env.step(
            'GROUND {"candidate_id": "mount_b", "memory_ids": ["C0"], "why": "C0 says tv_b is useful, but mount_a is also mentioned here."}'
        )
        assert reward < 0.0, (reward, info)
        assert "Invalid action" in observation, observation

        shared_attr_task = ShoppingTask(
            task_id="ground_shared_attribute_smoke",
            title="GROUND should allow shared compatibility attribute words",
            source="unit",
            memory_dependency="cross_session_bundled_shopping_attributes",
            subtasks=(
                ShoppingSubtask(
                    instruction="Product 1: select cake base and remember it.",
                    target_product_id="ma_src_a",
                    candidate_products=(
                        Product("ma_src_a", "A carrot cake base option.", {"source_option": "a"}),
                        Product("ma_src_b", "A chocolate cake base option.", {"source_option": "b"}),
                    ),
                ),
                ShoppingSubtask(
                    instruction="Product 2: choose frosting compatible with the previous cake base.",
                    target_product_id="ma_dep_a",
                    candidate_products=(
                        Product(
                            "ma_dep_a",
                            "A cream cheese frosting option.",
                            {"flavor": "Cream Cheese", "pairs_with": "Carrot"},
                        ),
                        Product(
                            "ma_dep_b",
                            "Another cream cheese frosting option.",
                            {"flavor": "Cream Cheese", "pairs_with": "Carrot"},
                        ),
                    ),
                ),
            ),
        )
        env = AgentMemoryEnv(tasks=[shared_attr_task])
        env.reset(data_idx=0)
        env.step('ADD {"key": "cake_base", "value": "Selected product_id ma_src_a: Carrot cake base."}')
        env.step('BUY {"product_id": "ma_src_a"}')
        env.step('RETRIEVE {"query": "carrot cake base", "top_k": 1}')
        observation, reward, _, _, info = env.step(
            'GROUND {"candidate_id": "ma_dep_a", "memory_ids": ["C0"], "why": "Cream Cheese pairs well with Carrot from C0."}'
        )
        assert reward >= 0.0, (reward, info)
        assert "Grounded candidate ma_dep_a" in observation, observation
        observation, reward, _, _, info = env.step(
            'GROUND {"candidate_id": "ma_dep_a", "memory_ids": ["C0"], "why": "ma_dep_b is another option, but ma_dep_a is selected."}'
        )
        assert reward < 0.0, (reward, info)
        assert "Invalid action" in observation, observation

        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "tv_profile", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 tv_weight_kg 32 vesa 400x400."}'
        )
        env.step('BUY {"product_id": "tv_b"}')
        env.step('RETRIEVE {"query": "tv profile size weight vesa", "top_k": 3}')
        observation, reward, _, _, info = env.step(
            'BUY {"product_id": "mount_b", "memory_ids": ["C0"], "why": "C0 says tv_b is 75-inch, 32kg, VESA 400x400; mount_b supports that TV size, weight, and VESA pattern."}'
        )
        event_names = [item["name"] for item in info.get("reward_components", [])]
        assert reward > 1.0, (reward, info)
        assert "direct_grounded_dependent_buy" in event_names, info
        assert "missing_grounding_before_dependent_purchase" not in event_names, info
        assert "Purchase accepted" in observation, observation
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_memoryarena_source_option_reward_shaping() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        task = ShoppingTask(
            task_id="memoryarena_source_option_smoke",
            title="MemoryArena source_option shaping smoke",
            source="memoryarena_bundled_shopping_v0",
            memory_dependency="cross_session_bundled_shopping_attributes",
            subtasks=(
                ShoppingSubtask(
                    instruction="Product 1: select a source product. Use memory tools to preserve attributes.",
                    target_product_id="ma_x_a",
                    candidate_products=(
                        Product("ma_x_a", "A vanilla cake base option.", {"source_option": "a"}),
                        Product("ma_x_b", "A lemon cake base option.", {"source_option": "b"}),
                    ),
                ),
                ShoppingSubtask(
                    instruction="Product 2: Constraint: Must be compatible with the previous products.",
                    target_product_id="ma_y_a",
                    candidate_products=(
                        Product("ma_y_a", "A white frosting option.", {"source_option": "a"}),
                        Product("ma_y_b", "A fudge frosting option.", {"source_option": "b"}),
                    ),
                ),
            ),
        )

        env = AgentMemoryEnv(tasks=[task], buy_semantics="retry")
        env.reset(data_idx=0)
        with_memory_reward = 0.0
        reward_events: list[str] = []
        first_retrieve_reward = None
        repeat_retrieve_reward = None
        for action in [
            'ADD {"key": "selected_base", "value": "Selected product_id ma_x_a: A vanilla cake base option."}',
            'BUY {"product_id": "ma_x_a"}',
            'RETRIEVE {"query": "previous selected source_option", "top_k": 3}',
            'RETRIEVE {"query": "previous selected source_option", "top_k": 3}',
            'BUY {"product_id": "ma_y_a"}',
        ]:
            observation, reward, done, _, info = env.step(action)
            if action.startswith("ADD"):
                assert env.memory_ids_for_product("ma_x_a") == {"mem_0000"}, env.memory_product_refs
                assert env.memory_ids_for_product("ma_x_b") == set(), env.memory_product_refs
            if action.startswith('RETRIEVE') and first_retrieve_reward is None:
                first_retrieve_reward = reward
                assert "Canonical visible product facts for retrieval" in observation, observation
                assert "source_option=a" in observation, observation
                assert "prior-session memory has been retrieved" in observation, observation
            elif action.startswith('RETRIEVE'):
                repeat_retrieve_reward = reward
            with_memory_reward += reward
            reward_events.extend(item["name"] for item in info.get("reward_components", []))
        assert done, info
        assert first_retrieve_reward and first_retrieve_reward > 0, first_retrieve_reward
        assert repeat_retrieve_reward and repeat_retrieve_reward < 0.0, repeat_retrieve_reward
        assert "memory_written_before_source_purchase" in reward_events, reward_events
        assert "memory_retrieve_nonempty_repeat_same_session" in reward_events, reward_events
        assert "retrieved_memory_before_dependent_purchase" in reward_events, reward_events

        env = AgentMemoryEnv(tasks=[task])
        env.reset(data_idx=0)
        no_memory_reward = 0.0
        for action in [
            'BUY {"product_id": "ma_x_a"}',
            'BUY {"product_id": "ma_y_a"}',
        ]:
            _, reward, done, _, info = env.step(action)
            no_memory_reward += reward
        assert done, info
        assert with_memory_reward > no_memory_reward, (with_memory_reward, no_memory_reward)
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_semantic_compatibility_precision_reward() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        task = ShoppingTask(
            task_id="semantic_compatibility_precision_smoke",
            title="Semantic compatibility precision smoke",
            source="memoryarena_bundled_shopping_unit",
            memory_dependency="cross_session_bundled_shopping_attributes",
            subtasks=(
                ShoppingSubtask(
                    instruction="Product 1: select cake base and remember it.",
                    target_product_id="cake_carrot",
                    candidate_products=(
                    Product("cake_carrot", "A carrot cake base option.", {"source_option": "Carrot"}),
                    Product("cake_chocolate", "A chocolate cake base option.", {"source_option": "Chocolate"}),
                    ),
                ),
                ShoppingSubtask(
                    instruction=(
                        "Product 2:\n"
                        "### Select Frosting\n"
                        "**Goal:** Compatibility notes: Carrot pairs well with Cream Cheese. "
                        "Chocolate pairs well with Fudge.\n"
                        "**Preference:** Pick the highest-rated option among those compatible with the notes.\n"
                        "**Avoid:** Carrot avoids Fudge. Chocolate avoids Cream Cheese.\n"
                        "**Constraint:** Must be compatible with the previous (ground truth) products."
                    ),
                    target_product_id="frosting_cream",
                    candidate_products=(
                        Product("frosting_cream", "A cream cheese frosting option.", {"source_option": "a"}),
                        Product("frosting_fudge", "A chocolate fudge frosting option.", {"source_option": "b"}),
                    ),
                ),
            ),
        )

        env = AgentMemoryEnv(tasks=[task], buy_semantics="retry")
        env.reset(data_idx=0)
        env.step('ADD {"key": "selected_cake", "value": "Selected product_id cake_carrot: Carrot cake base."}')
        env.step('BUY {"product_id": "cake_carrot"}')
        env.step('RETRIEVE {"query": "carrot cake base", "top_k": 1}')
        observation, reward, done, _, info = env.step(
            'BUY {"product_id": "frosting_cream", "memory_ids": ["C0"], '
            '"why": "C0 says the prior cake is Carrot; Cream Cheese pairs well with Carrot."}'
        )
        event_names = [item["name"] for item in info.get("reward_components", [])]
        assert done, info
        assert "All bundled shopping subtasks are complete" in observation, observation
        assert reward > 1.0, (reward, info)
        assert "direct_grounded_dependent_buy" in event_names, info
        assert "direct_semantic_compatible_memory_buy" in event_names, info
        assert "direct_grounded_dependent_buy_weak_semantic_evidence" not in event_names, info

        env = AgentMemoryEnv(tasks=[task])
        env.reset(data_idx=0)
        env.step('ADD {"key": "selected_cake", "value": "Selected product_id cake_carrot: Carrot cake base."}')
        env.step('BUY {"product_id": "cake_carrot"}')
        env.step('RETRIEVE {"query": "carrot cake base", "top_k": 1}')
        observation, reward, done, _, info = env.step(
            'BUY {"product_id": "frosting_cream", "memory_ids": ["C0"], '
            '"why": "C0 is useful, so buy this candidate."}'
        )
        event_names = [item["name"] for item in info.get("reward_components", [])]
        assert done, info
        assert "All bundled shopping subtasks are complete" in observation, observation
        assert "direct_grounded_dependent_buy_weak_semantic_evidence" in event_names, info
        assert "direct_semantic_compatible_memory_buy" not in event_names, info

        env = AgentMemoryEnv(tasks=[task], buy_semantics="retry")
        env.reset(data_idx=0)
        env.step('ADD {"key": "selected_cake", "value": "Selected product_id cake_carrot: Carrot cake base."}')
        env.step('BUY {"product_id": "cake_carrot"}')
        env.step('RETRIEVE {"query": "carrot cake base", "top_k": 1}')
        observation, reward, done, _, info = env.step(
            'BUY {"product_id": "frosting_fudge", "memory_ids": ["C0"], '
            '"why": "C0 says the prior cake is Carrot; this Fudge candidate is compatible."}'
        )
        event_names = [item["name"] for item in info.get("reward_components", [])]
        assert not done, info
        assert reward <= -0.6, (reward, info)
        assert "Purchase rejected" in observation, observation
        assert "direct_memory_buy_rejected_weak_semantic_evidence" in event_names, info
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_close_loop_noop_guards() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        env = AgentMemoryEnv(buy_semantics="retry")
        env.reset(data_idx=0)
        _, first_add_reward, _, _, info = env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV."}'
        )
        _, duplicate_add_reward, _, _, info = env.step(
            'ADD {"key": "selected_tv_again", "value": "Selected product_id tv_b Nebula 4K TV."}'
        )
        assert first_add_reward > 0, first_add_reward
        assert duplicate_add_reward < 0, (duplicate_add_reward, info)
        assert any(
            item["name"] == "memory_add_duplicate_visible_product_reference"
            for item in info.get("reward_components", [])
        ), info

        _, first_retrieve_reward, _, _, info = env.step('RETRIEVE {"query": "selected tv", "top_k": 1}')
        _, repeat_retrieve_reward, _, _, info = env.step('RETRIEVE {"query": "selected tv", "top_k": 1}')
        assert first_retrieve_reward == 0.0, (first_retrieve_reward, info)
        assert repeat_retrieve_reward < 0.0, (repeat_retrieve_reward, info)
        assert any(
            item["name"] == "memory_retrieve_source_same_session_repeat_noop"
            for item in info.get("reward_components", [])
        ), info

        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 vesa 400x400."}'
        )
        env.step('ADD {"key": "room_profile", "value": "Room profile: concrete wall and wide console space."}')
        env.step('BUY {"product_id": "tv_b"}')
        _, first_dependent_retrieve_reward, _, _, info = env.step(
            'RETRIEVE {"query": "selected tv vesa", "top_k": 1}'
        )
        _, additional_dependent_retrieve_reward, _, _, info = env.step(
            'RETRIEVE {"query": "room concrete wall", "top_k": 1}'
        )
        assert first_dependent_retrieve_reward > 0.0, (first_dependent_retrieve_reward, info)
        assert additional_dependent_retrieve_reward >= 0.0, (additional_dependent_retrieve_reward, info)
        assert any(
            item["name"] == "memory_retrieve_additional_nonempty_dependent_context"
            for item in info.get("reward_components", [])
        ), info

        env = AgentMemoryEnv(buy_semantics="retry")
        env.reset(data_idx=0)
        observation, first_reject_reward, _, _, info = env.step('BUY {"product_id": "tv_a"}')
        observation, repeat_reject_reward, _, _, info = env.step('BUY {"product_id": "tv_a"}')
        assert first_reject_reward < 0.0, (first_reject_reward, info)
        assert repeat_reject_reward < first_reject_reward, (first_reject_reward, repeat_reject_reward, info)
        assert "Rejected product_ids this session: tv_a" in observation, observation
        assert any(item["name"] == "buy_repeats_rejected_product_noop" for item in info.get("reward_components", [])), info
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_premature_answer_is_penalized_without_memory_tool_tax() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        env.step('ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV."}')
        observation, reward, _, _, info = env.step(
            'ANSWER {"text": "The selected TV is tv_b."}'
        )
        assert reward < 0.0, (reward, info)
        assert "BUY with the exact stored product_id" in observation, observation
        assert any(
            item["name"] == "source_memory_ready_answer_instead_of_buy"
            for item in info.get("reward_components", [])
        ), info

        env = AgentMemoryEnv(buy_semantics="retry")
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 vesa 400x400."}'
        )
        env.step('BUY {"product_id": "tv_b"}')
        env.step('RETRIEVE {"query": "selected tv vesa", "top_k": 1}')
        observation, reward, _, _, info = env.step(
            'ANSWER {"text": "The remembered TV is tv_b."}'
        )
        assert reward < 0.0, (reward, info)
        assert "BUY a compatible product_id" in observation, observation
        event_names = [item["name"] for item in info.get("reward_components", [])]
        assert "dependent_memory_ready_answer_instead_of_buy" in event_names, info
        assert "dependent_memory_ready_answer_no_progress" in event_names, info
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_destructive_memory_edits_are_targeted() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 vesa 400x400."}'
        )
        _, destructive_update_reward, _, _, info = env.step(
            'UPDATE {"memory_id": "mem_0000", "value": "Selected display profile: large screen for the living room."}'
        )
        assert destructive_update_reward < 0.0, (destructive_update_reward, info)
        assert any(
            item["name"] == "memory_update_drops_product_anchor"
            for item in info.get("reward_components", [])
        ), info

        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 vesa 400x400."}'
        )
        _, preserving_update_reward, _, _, info = env.step(
            'UPDATE {"memory_id": "mem_0000", "value": "Selected product_id tv_b; refine attributes: 75 inch, VESA 400x400."}'
        )
        assert preserving_update_reward >= 0.0, (preserving_update_reward, info)
        assert not any(
            item["name"] == "memory_update_drops_product_anchor"
            for item in info.get("reward_components", [])
        ), info

        env = AgentMemoryEnv()
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV tv_size_in 75 vesa 400x400."}'
        )
        _, delete_reward, _, _, info = env.step('DELETE {"memory_id": "mem_0000"}')
        assert delete_reward < 0.0, (delete_reward, info)
        assert any(
            item["name"] == "memory_delete_product_anchor"
            for item in info.get("reward_components", [])
        ), info
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_stale_source_memory_anchor_repair() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        env = AgentMemoryEnv(buy_semantics="retry")
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_a Orion 4K TV."}'
        )
        observation, reject_reward, _, _, info = env.step('BUY {"product_id": "tv_a"}')
        assert reject_reward < 0.0, (reject_reward, info)
        assert "Purchase rejected" in observation, observation
        observation, blocked_reward, done, _, info = env.step('BUY {"product_id": "tv_b"}')
        assert not done, info
        assert blocked_reward < 0.0, (blocked_reward, info)
        assert "Source-memory mismatch" in observation, observation
        assert any(
            item["name"] == "source_purchase_blocked_by_stale_memory_anchor"
            for item in info.get("reward_components", [])
        ), info

        observation, repair_reward, _, _, info = env.step(
            'UPDATE {"memory_id": "mem_0000", "value": "Selected product_id tv_b Nebula 4K TV."}'
        )
        assert repair_reward > 0.0, (repair_reward, info)
        assert "repaired memory" in observation, observation
        assert any(
            item["name"] == "memory_update_replaces_rejected_source_anchor"
            for item in info.get("reward_components", [])
        ), info
        observation, buy_reward, done, _, info = env.step('BUY {"product_id": "tv_b"}')
        assert buy_reward > 0.0 and not done, (buy_reward, info)
        assert "Purchase accepted" in observation, observation

        env = AgentMemoryEnv(buy_semantics="retry")
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_a Orion 4K TV."}'
        )
        env.step('BUY {"product_id": "tv_a"}')
        observation, delete_reward, _, _, info = env.step('DELETE {"memory_id": "mem_0000"}')
        assert delete_reward > 0.0, (delete_reward, info)
        assert "removed a stored product/source anchor that had already been rejected" in observation, observation
        assert any(
            item["name"] == "memory_delete_rejected_source_anchor"
            for item in info.get("reward_components", [])
        ), info
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_direct_memory_buy_precision_reward() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        env = AgentMemoryEnv(buy_semantics="retry")
        env.reset(data_idx=0)
        env.step('BUY {"product_id": "tv_b"}')
        env.step(
            'ADD {"key": "generic_anchor", "value": "Old note from a previous attempt without an accepted prior product_id."}'
        )
        env.step('RETRIEVE {"query": "old note previous attempt", "top_k": 1}')
        observation, reward, done, _, info = env.step(
            'BUY {"product_id": "mount_b", "memory_ids": ["C0"], "why": "C0 is a generic old note, so buy mount_b."}'
        )
        assert not done, info
        event_names = [item["name"] for item in info.get("reward_components", [])]
        assert "retrieved_memory_irrelevant_to_dependent_purchase" in event_names, info
        assert "direct_grounded_dependent_buy_stale_or_irrelevant" in event_names, info
        assert "direct_grounded_dependent_buy" not in event_names, info
        assert "Purchase accepted" in observation, observation
        assert reward < 1.0, (reward, info)

        env = AgentMemoryEnv(buy_semantics="retry")
        env.reset(data_idx=0)
        env.step(
            'ADD {"key": "selected_tv", "value": "Selected product_id tv_b Nebula 4K TV."}'
        )
        env.step('BUY {"product_id": "tv_b"}')
        env.step('RETRIEVE {"query": "selected tv", "top_k": 1}')
        observation, reward, _, _, info = env.step(
            'BUY {"product_id": "mount_a", "memory_ids": ["C0"], "why": "C0 supports this mount."}'
        )
        assert reward <= -0.5, (reward, info)
        assert "Purchase rejected" in observation, observation
        assert any(
            item["name"] == "dependent_buy_after_retrieved_memory_rejected_try_next"
            for item in info.get("reward_components", [])
        ), info
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_memskill_warmstart_curriculum() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        task = ShoppingTask(
            task_id="retrieve_buy_skill_v0_smoke",
            title="Warm-start retrieve then memory-backed BUY skill",
            source="memoryarena_curriculum_unit",
            memory_dependency="cross_session_bundled_shopping_attributes",
            start_subtask_index=1,
            initial_purchase_product_ids=("cake_carrot",),
            initial_memories=(
                InitialMemorySpec(
                    key="selected_cake",
                    value="Selected product_id cake_carrot: Carrot cake base with source_option=Carrot.",
                    product_ids=("cake_carrot",),
                ),
            ),
            curriculum_flags=frozenset({"require_retrieved_memory_for_dependent_buy"}),
            subtasks=(
                ShoppingSubtask(
                    instruction="Product 1: select cake base and remember it.",
                    target_product_id="cake_carrot",
                    candidate_products=(
                        Product("cake_carrot", "A carrot cake base option.", {"source_option": "Carrot"}),
                        Product("cake_chocolate", "A chocolate cake base option.", {"source_option": "Chocolate"}),
                    ),
                ),
                ShoppingSubtask(
                    instruction=(
                        "Product 2:\n"
                        "### Select Frosting\n"
                        "**Goal:** Compatibility notes: Carrot pairs well with Cream Cheese. "
                        "Chocolate pairs well with Fudge.\n"
                        "**Avoid:** Carrot avoids Fudge. Chocolate avoids Cream Cheese.\n"
                        "**Constraint:** Must be compatible with the previous products."
                    ),
                    target_product_id="frosting_cream",
                    candidate_products=(
                        Product("frosting_cream", "A cream cheese frosting option.", {"source_option": "Cream Cheese"}),
                        Product("frosting_fudge", "A fudge frosting option.", {"source_option": "Fudge"}),
                    ),
                ),
            ),
        )

        env = AgentMemoryEnv(tasks=[task], buy_semantics="retry")
        observation, info = env.reset(data_idx=0)
        assert info["current_subtask_index"] == 1, info
        assert info["purchase_history"][0]["product_id"] == "cake_carrot", info
        assert "Progress: 1/2" in observation, observation
        assert "Active retrieved/summary context: <empty>" in observation, observation
        assert "C0:" not in observation, observation
        assert "mem_0000" not in observation, observation
        assert 'UPDATE {"memory_id": "mem_0000"' not in observation, observation

        observation, reward, done, _, info = env.step('BUY {"product_id": "frosting_cream"}')
        assert not done, info
        assert reward < 0.0, (reward, info)
        assert "Curriculum memory gate" in observation, observation
        assert any(
            item["name"] == "curriculum_dependent_buy_without_retrieved_memory"
            for item in info.get("reward_components", [])
        ), info

        observation, reward, done, _, info = env.step('RETRIEVE {"query": "carrot cake base", "top_k": 1}')
        assert not done, info
        assert reward > 0.0, (reward, info)
        assert "C0: [mem_0000]" in observation, observation
        assert 'BUY {"product_id": "...", "memory_ids": ["C0"], "why": "..."}' in observation, observation

        observation, reward, done, _, info = env.step(
            'BUY {"product_id": "frosting_cream", "memory_ids": ["C0"], '
            '"why": "C0 says the prior cake is Carrot; Cream Cheese pairs well with Carrot."}'
        )
        event_names = [item["name"] for item in info.get("reward_components", [])]
        assert done, info
        assert reward > 1.0, (reward, info)
        assert "All bundled shopping subtasks are complete" in observation, observation
        assert "retrieved_memory_before_dependent_purchase" in event_names, info
        assert "direct_semantic_compatible_memory_buy" in event_names, info
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def assert_end_to_end_strict_memory_curriculum() -> None:
    previous_mode = os.environ.get("AGENTMEMORY_MEMORY_SHAPING")
    os.environ["AGENTMEMORY_MEMORY_SHAPING"] = "chain_v1"
    try:
        task = ShoppingTask(
            task_id="e2e_strict_memory_curr2_smoke",
            title="End-to-end strict memory curriculum",
            source="memoryarena_curriculum_unit",
            memory_dependency="cross_session_bundled_shopping_attributes",
            curriculum_flags=frozenset(
                {
                    "require_memory_before_source_buy",
                    "require_retrieved_memory_for_dependent_buy",
                }
            ),
            subtasks=(
                ShoppingSubtask(
                    instruction="Product 1: select cake base and remember it.",
                    target_product_id="cake_carrot",
                    candidate_products=(
                        Product("cake_carrot", "A carrot cake base option.", {"source_option": "Carrot"}),
                        Product("cake_chocolate", "A chocolate cake base option.", {"source_option": "Chocolate"}),
                    ),
                ),
                ShoppingSubtask(
                    instruction=(
                        "Product 2:\n"
                        "### Select Frosting\n"
                        "**Goal:** Compatibility notes: Carrot pairs well with Cream Cheese. "
                        "Chocolate pairs well with Fudge.\n"
                        "**Avoid:** Carrot avoids Fudge. Chocolate avoids Cream Cheese.\n"
                        "**Constraint:** Must be compatible with the previous products."
                    ),
                    target_product_id="frosting_cream",
                    candidate_products=(
                        Product("frosting_cream", "A cream cheese frosting option.", {"source_option": "Cream Cheese"}),
                        Product("frosting_fudge", "A fudge frosting option.", {"source_option": "Fudge"}),
                    ),
                ),
            ),
        )

        env = AgentMemoryEnv(tasks=[task], buy_semantics="retry")
        observation, info = env.reset(data_idx=0)
        assert info["current_subtask_index"] == 0, info
        assert "Progress: 0/2" in observation, observation

        observation, reward, done, _, info = env.step('BUY {"product_id": "cake_carrot"}')
        assert not done, info
        assert reward < 0.0, (reward, info)
        assert "Curriculum memory gate" in observation, observation
        assert 'ADD {"key": "selected_product"' in observation, observation
        assert "product_id=cake_carrot" in observation, observation
        assert any(
            item["name"] == "curriculum_source_buy_without_written_memory"
            for item in info.get("reward_components", [])
        ), info

        observation, reward, done, _, info = env.step(
            'ADD {"key": "selected_cake", "value": "Selected product_id cake_carrot: Carrot cake base with source_option=Carrot."}'
        )
        assert not done, info
        assert reward >= 0.20, (reward, info)
        assert "mem_0000" in observation, observation

        observation, reward, done, _, info = env.step('BUY {"product_id": "cake_carrot"}')
        assert not done, info
        assert reward > 1.0, (reward, info)
        assert "Purchase accepted" in observation, observation
        assert info["current_subtask_index"] == 1, info

        observation, reward, done, _, info = env.step('BUY {"product_id": "frosting_cream"}')
        assert not done, info
        assert reward < 0.0, (reward, info)
        assert "Curriculum memory gate" in observation, observation
        assert 'RETRIEVE {"query": "selected_product product_id source_option", "top_k": 3}' in observation, observation
        assert any(
            item["name"] == "curriculum_dependent_buy_without_retrieved_memory"
            for item in info.get("reward_components", [])
        ), info

        observation, reward, done, _, info = env.step('RETRIEVE {"query": "does not match memory", "top_k": 1}')
        assert not done, info
        assert reward == 0.0, (reward, info)
        assert "Next action template: RETRIEVE" in observation, observation

        observation, reward, done, _, info = env.step('RETRIEVE {"query": "does not match memory", "top_k": 1}')
        assert not done, info
        assert reward < 0.0, (reward, info)
        assert any(
            item["name"] == "memory_retrieve_empty_repeat_same_query_noop"
            for item in info.get("reward_components", [])
        ), info

        observation, reward, done, _, info = env.step('RETRIEVE {"query": "carrot cake source_option", "top_k": 1}')
        assert not done, info
        assert reward > 0.0, (reward, info)
        assert "C0: [mem_0000]" in observation, observation

        observation, reward, done, _, info = env.step(
            'GROUND {"candidate_id": "frosting_cream", "memory_ids": ["C0"], '
            '"why": "C0 says the prior cake is Carrot; Cream Cheese pairs well with Carrot."}'
        )
        assert not done, info
        assert reward > 0.0, (reward, info)
        assert (
            'BUY {"product_id": "frosting_cream", "memory_ids": ["C0"], '
            '"why": "C0 says the prior cake is Carrot; Cream Cheese pairs well with Carrot."}'
            in observation
        ), observation

        observation, reward, done, _, info = env.step(
            'BUY {"product_id": "frosting_cream", "memory_ids": ["C0"], '
            '"why": "C0 says the prior cake is Carrot; Cream Cheese pairs well with Carrot."}'
        )
        event_names = [item["name"] for item in info.get("reward_components", [])]
        assert done, info
        assert reward > 1.0, (reward, info)
        assert "All bundled shopping subtasks are complete" in observation, observation
        assert "direct_semantic_compatible_memory_buy" in event_names, info
    finally:
        if previous_mode is None:
            os.environ.pop("AGENTMEMORY_MEMORY_SHAPING", None)
        else:
            os.environ["AGENTMEMORY_MEMORY_SHAPING"] = previous_mode


def main() -> None:
    tv_info = run_plan(
        0,
        [
            'BUY {"product_id": "tv_b"}',
            'ADD {"key": "tv_profile", "value": "Purchased TV: 75 inches, 32kg, VESA 400x400."}',
            'RETRIEVE {"query": "tv size weight vesa", "top_k": 1}',
            'BUY {"product_id": "mount_b"}',
            'RETRIEVE {"query": "tv size", "top_k": 1}',
            'BUY {"product_id": "console_b"}',
        ],
    )
    laptop_info = run_plan(
        1,
        [
            'BUY {"product_id": "laptop_a"}',
            'ADD {"key": "laptop_profile", "value": "Purchased laptop: 14 inches with usb-c port."}',
            'RETRIEVE {"query": "laptop size", "top_k": 1}',
            'BUY {"product_id": "sleeve_b"}',
            'RETRIEVE {"query": "laptop port", "top_k": 1}',
            'BUY {"product_id": "dock_a"}',
        ],
    )
    monitor_info = run_plan(
        2,
        [
            'BUY {"product_id": "monitor_b"}',
            'ADD {"key": "monitor_profile", "value": "Purchased monitor: 27 inches, 6kg, VESA 100x100, ports usb-c and hdmi."}',
            'RETRIEVE {"query": "monitor size weight vesa", "top_k": 1}',
            'BUY {"product_id": "arm_b"}',
            'RETRIEVE {"query": "monitor port", "top_k": 1}',
            'BUY {"product_id": "cable_b"}',
        ],
    )
    assert_wrong_purchase(0, ['BUY {"product_id": "tv_b"}', 'BUY {"product_id": "mount_a"}'])
    assert_wrong_purchase(
        1,
        ['BUY {"product_id": "laptop_a"}', 'BUY {"product_id": "sleeve_b"}', 'BUY {"product_id": "dock_b"}'],
    )
    assert_wrong_purchase(
        2,
        ['BUY {"product_id": "monitor_b"}', 'BUY {"product_id": "arm_a"}'],
    )
    assert_session_trace_boundary()
    assert_memory_tool_contract()
    assert_dynamic_action_menu_hides_invisible_context_ids()
    assert_memory_product_ref_index_consistency()
    assert_bm25_retrieve()
    assert_memory_chain_reward_shaping()
    assert_candidate_memory_grounding_reward_shaping()
    assert_memoryarena_source_option_reward_shaping()
    assert_semantic_compatibility_precision_reward()
    assert_close_loop_noop_guards()
    assert_premature_answer_is_penalized_without_memory_tool_tax()
    assert_destructive_memory_edits_are_targeted()
    assert_stale_source_memory_anchor_repair()
    assert_direct_memory_buy_precision_reward()
    assert_memskill_warmstart_curriculum()
    assert_end_to_end_strict_memory_curriculum()
    print("AGENTMEMORY_DIRECT_SMOKE_OK", tv_info["task_id"], laptop_info["task_id"], monitor_info["task_id"])
    print("AGENTMEMORY_BM25_RETRIEVE_SMOKE_OK")
    print("AGENTMEMORY_MEMORY_PRODUCT_REF_INDEX_OK")
    print("AGENTMEMORY_DYNAMIC_ACTION_MENU_OK")
    print("AGENTMEMORY_MEMORY_CHAIN_SHAPING_SMOKE_OK")
    print("AGENTMEMORY_CANDIDATE_MEMORY_GROUNDING_SMOKE_OK")
    print("AGENTMEMORY_MEMORYARENA_SOURCE_OPTION_SHAPING_SMOKE_OK")
    print("AGENTMEMORY_SEMANTIC_COMPAT_PRECISION_OK")
    print("AGENTMEMORY_CLOSE_LOOP_NOOP_GUARDS_OK")
    print("AGENTMEMORY_PREMATURE_ANSWER_NO_TOOL_TAX_OK")
    print("AGENTMEMORY_DESTRUCTIVE_MEMORY_EDIT_TARGETED_OK")
    print("AGENTMEMORY_STALE_SOURCE_MEMORY_REPAIR_OK")
    print("AGENTMEMORY_DIRECT_MEMORY_BUY_PRECISION_OK")
    print("AGENTMEMORY_MEMSKILL_WARMSTART_OK")
    print("AGENTMEMORY_E2E_STRICT_MEMORY_CURRICULUM_OK")


if __name__ == "__main__":
    main()
