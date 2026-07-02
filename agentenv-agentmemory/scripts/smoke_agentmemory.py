from __future__ import annotations

from agentenv_agentmemory.environment import AgentMemoryEnv


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


def assert_wrong_purchase(data_idx: int, actions: list[str]) -> None:
    env = AgentMemoryEnv()
    env.reset(data_idx=data_idx)
    last = None
    for action in actions:
        last = env.step(action)
    assert last is not None
    _, reward, done, _, info = last
    assert reward < 0, f"wrong purchase should be penalized: reward={reward}, info={info}"
    assert not done, f"wrong purchase should not finish: {info}"
    assert info["compatibility_violations"], f"missing compatibility violation: {info}"


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
    print("AGENTMEMORY_DIRECT_SMOKE_OK", tv_info["task_id"], laptop_info["task_id"], monitor_info["task_id"])


if __name__ == "__main__":
    main()
