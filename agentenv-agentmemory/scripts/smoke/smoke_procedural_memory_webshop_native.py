#!/usr/bin/env python3
"""Run one machine-solved procedural memory chain on native WebShop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentenv_agentmemory.native_webshop_backend import MemoryArenaNativeWebShopBackend
from agentenv_agentmemory.procedural import (
    NaturalAttributeChainGenerator,
    VerifiedProceduralBundleProvider,
    load_certified_product_pool,
    scenario_by_id,
)
from agentenv_agentmemory.procedural_webshop_env import ProceduralMemoryWebShopEnv
from agentenv_agentmemory.procedural_wrapper import attest_procedural_runtime_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memoryarena-root", required=True, type=Path)
    parser.add_argument("--memoryarena-base-commit", required=True)
    parser.add_argument("--items-file", required=True, type=Path)
    parser.add_argument("--attributes-file", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    parser.add_argument("--java-home", required=True, type=Path)
    parser.add_argument("--lucene-index-manifest", required=True, type=Path)
    parser.add_argument("--product-pool", required=True, type=Path)
    parser.add_argument("--product-pool-sha256", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--generator-seed", type=int, default=0)
    parser.add_argument(
        "--data-index",
        dest="data_indices",
        type=int,
        action="append",
        help="Task index to smoke; repeat the flag to reuse one native backend.",
    )
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--price-seed", type=int, default=233)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_indices = args.data_indices or [0]
    if any(data_index < 0 for data_index in data_indices):
        raise SystemExit("every --data-index must be non-negative")
    if len(data_indices) != len(set(data_indices)):
        raise SystemExit("repeated --data-index values are not allowed")
    if not 1 <= args.sessions <= 6:
        raise SystemExit("--sessions must be between 1 and 6")

    pool = load_certified_product_pool(
        args.product_pool,
        expected_file_sha256=args.product_pool_sha256,
    )
    backend = MemoryArenaNativeWebShopBackend(
        memoryarena_root=args.memoryarena_root,
        items_file=args.items_file,
        attributes_file=args.attributes_file,
        search_root=args.search_root,
        java_home=args.java_home,
        expected_memoryarena_commit=args.memoryarena_base_commit,
        price_seed=args.price_seed,
    )
    task_count = max(data_indices) + 1
    if task_count % 2:
        task_count += 1
    provider = VerifiedProceduralBundleProvider(
        generator=NaturalAttributeChainGenerator(
            pool=pool,
            seed=args.generator_seed,
        ),
        split=args.split,
        task_count=task_count,
    )
    env = ProceduralMemoryWebShopEnv(
        provider=provider,
        backend=backend,
        env_uid="procedural_native_smoke",
        first_valid_add_reward=0.0,
        first_valid_later_session_retrieve_reward=0.0,
    )
    try:
        attest_procedural_runtime_inputs(
            pool,
            backend,
            items_file=args.items_file.resolve(),
            attributes_file=args.attributes_file.resolve(),
            search_root=args.search_root.resolve(),
            lucene_manifest=args.lucene_index_manifest.resolve(),
        )
        scenario_ids = []
        for data_index in data_indices:
            scenario_ids.append(
                _run_task(
                    env,
                    backend=backend,
                    provider=provider,
                    data_index=data_index,
                    sessions=args.sessions,
                    split=args.split,
                )
            )
        print(
            "AGENTMEMORY_PROCEDURAL_NATIVE_SMOKE_BATCH_OK "
            f"tasks={len(data_indices)} data_indices={','.join(map(str, data_indices))} "
            f"scenario_ids={','.join(scenario_ids)} sessions_per_task={args.sessions}"
        )
    finally:
        env.close()
        backend.close()


def _run_task(
    env: ProceduralMemoryWebShopEnv,
    *,
    backend: MemoryArenaNativeWebShopBackend,
    provider: VerifiedProceduralBundleProvider,
    data_index: int,
    sessions: int,
    split: str,
) -> str:
    bundle = provider.get(data_index)
    _, info = env.reset(data_idx=data_index)
    if info["candidate_count_per_phase"] != 2:
        raise AssertionError("procedural smoke requires exactly two candidates")
    if info["paper_eligible"]:
        raise AssertionError("procedural training surface became paper eligible")
    scenario = scenario_by_id(bundle.scenario_id)

    done = False
    purchase_info = info
    for session_index in range(sessions):
        if session_index > 0:
            previous_slot = scenario.slots[session_index - 1]
            previous_value = bundle.target_attribute_values[session_index - 1]
            previous_display = previous_slot.value(previous_value).display_name
            retrieve = json.dumps(
                {"query": previous_slot.slot_id, "top_k": 1},
                separators=(",", ":"),
            )
            observation, reward, done, _, retrieve_info = env.step(
                f"RETRIEVE {retrieve}"
            )
            if reward != 0.0 or done or previous_display not in observation:
                raise AssertionError(
                    "previous natural-attribute retrieval failed: "
                    f"{retrieve_info}"
                )

        target_asin = bundle.target_asins[session_index]
        slot = scenario.slots[session_index]
        target_value = bundle.target_attribute_values[session_index]
        target_display = slot.value(target_value).display_name
        memory_value = (
            f"Bought {slot.display_name}; {slot.attribute_name}={target_display}"
        )
        memory_action = json.dumps(
            {"key": slot.slot_id, "value": memory_value},
            ensure_ascii=True,
            separators=(",", ":"),
        )
        observation, reward, done, _, add_info = env.step(f"ADD {memory_action}")
        if reward != 0.0 or done or memory_value not in observation:
            raise AssertionError(
                f"session {session_index} attribute ADD failed: {add_info}"
            )

        target_product = next(
            product
            for product in provider.generator.pool.products
            if product.asin == target_asin
        )
        if backend.product_title(target_asin) != target_product.title:
            raise AssertionError("certified target title changed before native smoke")
        _search_and_open_target(
            env,
            query=target_product.search_query,
            target_asin=target_asin,
        )
        _, reward, done, _, purchase_info = env.step("click[Buy Now]")
        expected_reward = 2.0 if session_index == 5 else 1.0
        if reward != expected_reward:
            raise AssertionError(
                f"session {session_index} reward {reward} != {expected_reward}: "
                f"{purchase_info}"
            )
        if env.current_session_index != session_index + 1:
            raise AssertionError(
                f"session {session_index} oracle purchase failed: {purchase_info}"
            )
        if "actual_asin" in json.dumps(purchase_info, sort_keys=True):
            raise AssertionError("purchase state leaked through procedural runtime info")

    expected_done = sessions == 6
    if done != expected_done:
        raise AssertionError(
            f"terminal state mismatch after {sessions} sessions: {purchase_info}"
        )
    print(
        "AGENTMEMORY_PROCEDURAL_NATIVE_SMOKE_OK "
        f"task_id={bundle.task_id} split={split} scenario_id={bundle.scenario_id} "
        f"data_index={data_index} sessions={sessions} "
        "candidate_count_per_phase=2 dependency=previous_natural_attribute "
        "human_review_required=false llm_judge_required=false"
    )
    return bundle.scenario_id


def _search_and_open_target(
    env: ProceduralMemoryWebShopEnv,
    *,
    query: str,
    target_asin: str,
) -> None:
    _, reward, done, _, info = env.step(f"search[{_native_argument(query)}]")
    if reward != 0.0 or done:
        raise AssertionError(f"native search failed: {info}")
    page = env.native_page
    if page is None:
        raise AssertionError("native page missing after search")
    if target_asin.lower() not in {value.lower() for value in page.clickables}:
        raise AssertionError(
            f"certified target {target_asin} is absent from the native first page"
        )
    _, click_reward, click_done, _, click_info = env.step(f"click[{target_asin}]")
    if click_reward != 0.0 or click_done:
        raise AssertionError(f"native ASIN click failed: {click_info}")


def _native_argument(value: str) -> str:
    text = " ".join(value.split())
    if any(char in text for char in "[]\r\n"):
        raise ValueError(f"native WebShop argument is unsafe: {text!r}")
    return text


if __name__ == "__main__":
    main()
