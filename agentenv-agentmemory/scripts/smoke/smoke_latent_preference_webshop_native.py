#!/usr/bin/env python3
"""Run machine-solved hidden-preference tasks on native WebShop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentenv_agentmemory.latent_preference import (
    LatentPreferenceGenerator,
    VerifiedLatentPreferenceBundleProvider,
    attest_latent_preference_runtime_inputs,
    load_preference_product_pool,
)
from agentenv_agentmemory.latent_preference_webshop_env import (
    LatentPreferenceWebShopEnv,
)
from agentenv_agentmemory.native_webshop_backend import (
    MemoryArenaNativeWebShopBackend,
)


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
    parser.add_argument("--generator-seed", type=int, default=233)
    parser.add_argument("--start-orbit", type=int, default=0)
    parser.add_argument(
        "--data-index",
        dest="data_indices",
        type=int,
        action="append",
        help="Task index to smoke; defaults to the two branches of orbit zero.",
    )
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--price-seed", type=int, default=233)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_indices = args.data_indices or [0, 1]
    if args.start_orbit < 0:
        raise SystemExit("--start-orbit must be non-negative")
    if any(data_index < 0 for data_index in data_indices):
        raise SystemExit("every --data-index must be non-negative")
    if len(data_indices) != len(set(data_indices)):
        raise SystemExit("repeated --data-index values are not allowed")
    if not 1 <= args.sessions <= 6:
        raise SystemExit("--sessions must be between 1 and 6")

    pool = load_preference_product_pool(
        args.product_pool,
        expected_file_sha256=args.product_pool_sha256,
    )
    generator = LatentPreferenceGenerator(pool=pool, seed=args.generator_seed)
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
    provider = VerifiedLatentPreferenceBundleProvider(
        generator=generator,
        split=args.split,
        task_count=task_count,
        start_orbit=args.start_orbit,
    )
    env = LatentPreferenceWebShopEnv(
        provider=provider,
        backend=backend,
        env_uid="latent_preference_native_smoke",
        first_valid_add_reward=0.0,
        first_valid_later_session_retrieve_reward=0.0,
    )
    try:
        attest_latent_preference_runtime_inputs(
            pool,
            backend,
            items_file=args.items_file.resolve(),
            attributes_file=args.attributes_file.resolve(),
            search_root=args.search_root.resolve(),
            lucene_manifest=args.lucene_index_manifest.resolve(),
        )
        orbit_ids = []
        for data_index in data_indices:
            proof = provider.proof_for_index(data_index)
            if proof.valid_solution_counts != (1, 1):
                raise AssertionError("latent-preference proof lost its unique solutions")
            orbit_ids.append(
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
            "AGENTMEMORY_LATENT_PREFERENCE_NATIVE_SMOKE_BATCH_OK "
            f"tasks={len(data_indices)} "
            f"data_indices={','.join(map(str, data_indices))} "
            f"orbit_ids={','.join(orbit_ids)} sessions_per_task={args.sessions}"
        )
    finally:
        env.close()
        backend.close()


def _run_task(
    env: LatentPreferenceWebShopEnv,
    *,
    backend: MemoryArenaNativeWebShopBackend,
    provider: VerifiedLatentPreferenceBundleProvider,
    data_index: int,
    sessions: int,
    split: str,
) -> str:
    bundle = provider.get(data_index)
    source_orbit_index = provider.start_orbit + data_index // 2
    orbit = provider.generator.generate_orbit(source_orbit_index, split=split)
    task = orbit.tasks[data_index % 2]
    if task.task_id != bundle.task_id:
        raise AssertionError("provider bundle disagrees with generated task")
    recipe = provider.generator.pool.recipe_by_id(task.recipe_id)
    preferred_display = recipe.value_display_name(task.preferred_attribute_value)
    memory_value = (
        f"Customer preference for {recipe.axis_display_name}: {preferred_display}"
    )

    _, info = env.reset(data_idx=data_index)
    if info["candidate_count_per_phase"] != 2:
        raise AssertionError("latent-preference smoke requires two candidates")
    if info["paper_eligible"]:
        raise AssertionError("latent-preference training surface became paper eligible")

    done = False
    purchase_info = info
    memory_written = False
    for session_index in range(sessions):
        phase = task.phases[session_index]
        page_observation = (
            "" if env.native_page is None else env.native_page.observation
        )
        if phase.question not in page_observation:
            raise AssertionError(
                "native session omitted the verified task instruction: "
                f"data_index={data_index} session_index={session_index} "
                f"expected={phase.question!r} observed={page_observation!r}"
            )
        for candidate in phase.candidates:
            if candidate.title not in phase.question:
                raise AssertionError("approved native title is absent from task prompt")
            if candidate.asin.casefold() in phase.question.casefold():
                raise AssertionError("task prompt leaked an internal ASIN")

        if phase.phase_kind == "evidence" and not memory_written:
            action = json.dumps(
                {"key": "user_preference", "value": memory_value},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            observation, reward, done, _, add_info = env.step(f"ADD {action}")
            if reward != 0.0 or done or memory_value not in observation:
                raise AssertionError(
                    f"preference memory ADD failed in session {session_index}: {add_info}"
                )
            memory_written = True
        elif phase.phase_kind == "application":
            retrieve = json.dumps(
                {"query": "user preference", "top_k": 1},
                separators=(",", ":"),
            )
            observation, reward, done, _, retrieve_info = env.step(
                f"RETRIEVE {retrieve}"
            )
            if reward != 0.0 or done or memory_value not in observation:
                raise AssertionError(
                    "preference memory RETRIEVE failed in session "
                    f"{session_index}: {retrieve_info}"
                )

        target_asin = bundle.target_asins[session_index]
        target_product = provider.generator.pool.product_by_asin(target_asin)
        if backend.product_title(target_asin) != target_product.title:
            raise AssertionError("certified target title changed before native smoke")
        _search_and_open_target(
            env,
            query=target_product.search_query,
            target_asin=target_asin,
        )
        _, reward, done, _, purchase_info = env.step("click[Buy Now]")
        if not env.purchase_ledger:
            raise AssertionError("native purchase produced no verifier receipt")
        receipt = env.purchase_ledger[-1]
        if receipt.get("actual_asin") != target_asin.upper():
            raise AssertionError(
                "native purchase receipt ASIN mismatch: "
                f"expected {target_asin.upper()}, "
                f"observed {receipt.get('actual_asin')}"
            )
        if receipt.get("purchase_correct") is not True:
            raise AssertionError(f"native purchase receipt rejected target: {receipt}")
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
            raise AssertionError("purchase state leaked through latent runtime info")

    expected_done = sessions == 6
    if done != expected_done:
        raise AssertionError(
            f"terminal state mismatch after {sessions} sessions: {purchase_info}"
        )
    print(
        "AGENTMEMORY_LATENT_PREFERENCE_NATIVE_SMOKE_OK "
        f"task_id={bundle.task_id} split={split} recipe_id={bundle.recipe_id} "
        f"data_index={data_index} sessions={sessions} "
        f"evidence_count={bundle.supporting_evidence_count} "
        "candidate_count_per_phase=2 target_asin_in_task_prompt=false "
        "native_search_result_asin_handles_visible=true "
        "native_click_action_uses_asin_handle=true "
        "purchase_receipt_asin_verification=true "
        "human_review_required=false llm_judge_required=false"
    )
    return bundle.orbit_id


def _search_and_open_target(
    env: LatentPreferenceWebShopEnv,
    *,
    query: str,
    target_asin: str,
) -> None:
    search_observation, reward, done, _, info = env.step(
        f"search[{_native_argument(query)}]"
    )
    if reward != 0.0 or done:
        raise AssertionError(f"native search failed: {info}")
    if target_asin.casefold() not in search_observation.casefold():
        raise AssertionError(
            f"policy-visible native search observation omitted ASIN {target_asin}"
        )
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
