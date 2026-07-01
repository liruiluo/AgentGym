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
        "memory_ops",
        "memory_state_diff",
        "compatibility_violations",
        "purchase_history",
    ]:
        assert key in info, f"missing info key {key}: {info}"
    return info


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
    print("AGENTMEMORY_DIRECT_SMOKE_OK", tv_info["task_id"], laptop_info["task_id"], monitor_info["task_id"])


if __name__ == "__main__":
    main()
