#!/usr/bin/env python3
from __future__ import annotations

import argparse

from agentenv_agentmemory.memoryarena_dataset import load_memoryarena_dataset
from agentenv_agentmemory.memoryarena_webshop_env import MemoryArenaWebShopEnv
from agentenv_agentmemory.native_webshop_backend import MemoryArenaNativeWebShopBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memoryarena-root", required=True)
    parser.add_argument("--raw-data", required=True)
    parser.add_argument("--items-file", required=True)
    parser.add_argument("--attributes-file", required=True)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--java-home", required=True)
    parser.add_argument("--task-id", default="baking_item_0")
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--price-seed", type=int, default=233)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.sessions <= 6:
        raise SystemExit("--sessions must be between 1 and 6")

    backend = MemoryArenaNativeWebShopBackend(
        memoryarena_root=args.memoryarena_root,
        items_file=args.items_file,
        attributes_file=args.attributes_file,
        search_root=args.search_root,
        java_home=args.java_home,
        price_seed=args.price_seed,
    )
    dataset = load_memoryarena_dataset(
        args.raw_data,
        frozen_product_asins=backend.product_asins(),
    )
    bundle = dataset.get(args.task_id)
    env = MemoryArenaWebShopEnv(bundles=[bundle], backend=backend, env_uid="native_smoke")
    try:
        observation, _ = env.reset()
        if bundle.task_id in observation:
            raise AssertionError("stable task id leaked into the policy observation")

        authored_memory = "native-smoke-authored-value"
        observation, reward, done, _, _ = env.step(
            f'ADD {{"key":"native-smoke","value":"{authored_memory}"}}'
        )
        if reward != 0.0 or done or authored_memory not in observation:
            raise AssertionError("native smoke ADD contract failed")

        for session_index in range(args.sessions):
            target_asin = bundle.target_asins[session_index]
            title = backend.product_title(target_asin)
            _search_and_open_target(env, title, target_asin)
            observation, reward, done, _, info = env.step("click[Buy Now]")
            expected_reward = 2.0 if session_index == 5 else 1.0
            if reward != expected_reward:
                raise AssertionError(
                    f"session {session_index} purchase reward {reward} != {expected_reward}: {info}"
                )
            if not info["tool_ops"][0]["purchase_correct"]:
                raise AssertionError(f"session {session_index} oracle purchase failed: {info}")
            if session_index + 1 < args.sessions:
                if authored_memory in observation:
                    raise AssertionError("long-term memory leaked across the session boundary")
                observation, retrieve_reward, retrieve_done, _, retrieve_info = env.step(
                    'RETRIEVE {"query":"native smoke authored","top_k":3}'
                )
                if retrieve_reward != 0.0 or retrieve_done or authored_memory not in observation:
                    raise AssertionError(f"cross-session exact-memory retrieval failed: {retrieve_info}")

        print(
            "AGENTMEMORY_MEMORYARENA_NATIVE_WEBSHOP_SMOKE_OK "
            f"task_id={bundle.task_id} sessions={args.sessions} "
            f"price_seed={args.price_seed} surface={env.surface}"
        )
    finally:
        env.close()
        backend.close()


def _search_and_open_target(
    env: MemoryArenaWebShopEnv,
    title: str,
    target_asin: str,
) -> None:
    _, reward, done, _, info = env.step(f"search[{_native_argument(title)}]")
    if reward != 0.0 or done:
        raise AssertionError(f"native search failed: {info}")
    target_key = target_asin.lower()
    for _ in range(5):
        page = env.native_page
        if page is None:
            raise AssertionError("native page missing after search")
        if target_key in {value.lower() for value in page.clickables}:
            _, click_reward, click_done, _, click_info = env.step(f"click[{target_asin}]")
            if click_reward != 0.0 or click_done:
                raise AssertionError(f"native ASIN click failed: {click_info}")
            return
        if "next >" not in {value.lower() for value in page.clickables}:
            break
        env.step("click[Next >]")
    raise AssertionError(f"target ASIN {target_asin} absent from original Lucene top-50 for exact title")


def _native_argument(value: str) -> str:
    text = " ".join(value.split())
    if any(char in text for char in "[]\r\n"):
        raise ValueError(f"native WebShop argument contains a bracket/newline: {text!r}")
    return text


if __name__ == "__main__":
    main()
