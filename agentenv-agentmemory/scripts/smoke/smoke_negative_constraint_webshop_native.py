#!/usr/bin/env python3
"""Run certified negative-constraint branches on the native WebShop runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentenv_agentmemory.native_webshop_backend import (
    MemoryArenaNativeWebShopBackend,
)
from agentenv_agentmemory.negative_constraint import (
    NegativeConstraintGenerator,
    VerifiedNegativeConstraintBundleProvider,
    load_negative_constraint_native_product_pool,
)
from agentenv_agentmemory.negative_constraint.runtime_attestation import (
    attest_negative_constraint_runtime_inputs,
)
from agentenv_agentmemory.negative_constraint_webshop_env import (
    NegativeConstraintWebShopEnv,
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
        help="Task index to smoke; defaults to all three branches of orbit zero.",
    )
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--price-seed", type=int, default=233)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_indices = args.data_indices or [0, 1, 2]
    if args.start_orbit < 0:
        raise SystemExit("--start-orbit must be non-negative")
    if any(data_index < 0 for data_index in data_indices):
        raise SystemExit("every --data-index must be non-negative")
    if len(data_indices) != len(set(data_indices)):
        raise SystemExit("repeated --data-index values are not allowed")
    if not 1 <= args.sessions <= 6:
        raise SystemExit("--sessions must be between 1 and 6")

    pool = load_negative_constraint_native_product_pool(
        args.product_pool,
        expected_file_sha256=args.product_pool_sha256,
    )
    generator = NegativeConstraintGenerator(pool=pool, seed=args.generator_seed)
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
    if task_count % 3:
        task_count += 3 - (task_count % 3)
    provider = VerifiedNegativeConstraintBundleProvider(
        generator=generator,
        split=args.split,
        task_count=task_count,
        start_orbit=args.start_orbit,
    )
    env = NegativeConstraintWebShopEnv(
        provider=provider,
        backend=backend,
        env_uid="negative_constraint_native_smoke",
        first_valid_add_reward=0.0,
        first_valid_later_session_retrieve_reward=0.0,
    )
    try:
        attest_negative_constraint_runtime_inputs(
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
            if proof.valid_solution_counts != (1, 1, 1):
                raise AssertionError("negative proof lost its unique solutions")
            orbit_ids.append(
                _run_task(
                    env,
                    backend=backend,
                    pool=pool,
                    provider=provider,
                    data_index=data_index,
                    sessions=args.sessions,
                    split=args.split,
                )
            )
        if backend.active_session_count() != 0:
            raise AssertionError("native session leaked at smoke endpoint")
        print(
            "AGENTMEMORY_NEGATIVE_CONSTRAINT_NATIVE_SMOKE_BATCH_OK "
            f"tasks={len(data_indices)} "
            f"data_indices={','.join(map(str, data_indices))} "
            f"orbit_ids={','.join(orbit_ids)} sessions_per_task={args.sessions} "
            "candidate_count_per_phase=3 counterfactual_branches=3 "
            "query_top1=true target_asin_in_task_prompt=false "
            "native_search_result_asin_handles_visible=true "
            "native_click_action_uses_asin_handle=true "
            "purchase_receipt_asin_verification=true"
        )
    finally:
        env.close()
        backend.close()


def _run_task(
    env: NegativeConstraintWebShopEnv,
    *,
    backend: MemoryArenaNativeWebShopBackend,
    pool,
    provider: VerifiedNegativeConstraintBundleProvider,
    data_index: int,
    sessions: int,
    split: str,
) -> str:
    bundle = provider.get(data_index)
    source_orbit_index = provider.start_orbit + data_index // 3
    orbit = provider.generator.generate_orbit(source_orbit_index, split=split)
    task = orbit.tasks[data_index % 3]
    if task.task_id != bundle.task_id:
        raise AssertionError("provider bundle disagrees with generated task")

    candidates = {candidate.asin: candidate for candidate in pool.candidates}
    _, info = env.reset(data_idx=data_index)
    if info["candidate_count_per_phase"] != 3:
        raise AssertionError("negative smoke requires three candidates")
    if info["counterfactual_branch_count"] != 3:
        raise AssertionError("negative smoke lost its three-way orbit")
    if info["training_ready"] is not True:
        raise AssertionError("native negative smoke is not training-ready")

    done = False
    purchase_info = info
    memory_written = False
    for session_index in range(sessions):
        question = bundle.questions[session_index]
        observation = env.render_observation()
        if question not in observation:
            raise AssertionError(
                f"native observation omitted task instruction at session {session_index}"
            )
        target_asin = bundle.target_asins[session_index]
        if target_asin.casefold() in question.casefold():
            raise AssertionError("negative task prompt leaked a target ASIN")
        info_text = json.dumps(env.build_info(), ensure_ascii=True, sort_keys=True)
        if target_asin.casefold() in info_text.casefold():
            raise AssertionError("negative policy metadata leaked a target ASIN")

        if session_index == 0:
            add_payload = json.dumps(
                {
                    "key": bundle.canonical_memory_key,
                    "value": bundle.canonical_memory_value,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
            observation, reward, done, truncated, add_info = env.step(
                f"ADD {add_payload}"
            )
            if (
                reward != 0.0
                or done
                or truncated
                or add_info.get("memory_ops", [{}])[0].get("memory_id")
                != "mem_0000"
            ):
                raise AssertionError(f"negative canonical ADD failed: {add_info}")
            memory_written = True
        elif memory_written:
            retrieve_payload = json.dumps(
                {"query": bundle.canonical_retrieval_query},
                ensure_ascii=True,
                separators=(",", ":"),
            )
            observation, reward, done, truncated, retrieve_info = env.step(
                f"RETRIEVE {retrieve_payload}"
            )
            event = retrieve_info.get("memory_ops", [{}])[0]
            if (
                reward != 0.0
                or done
                or truncated
                or event.get("retrieved_memory_ids") != ["mem_0000"]
                or bundle.canonical_memory_value not in observation
            ):
                raise AssertionError(
                    f"negative query_top1 RETRIEVE failed: {retrieve_info}"
                )

        certificate = pool.certificate_for(target_asin)
        candidate = candidates[target_asin]
        if backend.product_title(target_asin) != candidate.title:
            raise AssertionError("certified negative title changed before smoke")
        _search_and_open_target(
            env,
            query=certificate.search_query,
            target_asin=target_asin,
        )
        _, reward, done, truncated, purchase_info = env.step("click[Buy Now]")
        if truncated:
            raise AssertionError("native negative BUY truncated unexpectedly")
        if not env.purchase_ledger:
            raise AssertionError("native negative BUY produced no receipt")
        receipt = env.purchase_ledger[-1]
        if (
            receipt.get("actual_asin") != target_asin.upper()
            or receipt.get("purchase_correct") is not True
        ):
            raise AssertionError(f"negative receipt rejected target: {receipt}")
        if "actual_asin" in json.dumps(purchase_info, sort_keys=True):
            raise AssertionError("negative purchase state leaked through info")
        expected_reward = 2.0 if session_index == 5 else 1.0
        if reward != expected_reward or done != (session_index == 5):
            raise AssertionError("negative BUY reward or terminal flag mismatch")
        if env.current_session_index != session_index + 1:
            raise AssertionError("negative BUY did not advance exactly one session")

    if done != (sessions == 6):
        raise AssertionError("negative terminal state mismatch")
    if sessions == 6 and not purchase_info.get("episode_success"):
        raise AssertionError("negative task did not report episode success")
    if env.native_session_token is not None:
        raise AssertionError("negative task retained a native session token")
    if backend.active_session_count() != 0:
        raise AssertionError("native session leaked at negative task endpoint")
    print(
        "AGENTMEMORY_NEGATIVE_CONSTRAINT_NATIVE_SMOKE_OK "
        f"task_id={bundle.task_id} split={split} data_index={data_index} "
        f"sessions={sessions} branch={bundle.branch_kind} "
        "add_then_query_top1=true candidate_count_per_phase=3 "
        "target_asin_in_task_prompt=false purchase_receipt_asin_verification=true"
    )
    return bundle.orbit_id


def _search_and_open_target(
    env: NegativeConstraintWebShopEnv,
    *,
    query: str,
    target_asin: str,
) -> None:
    search_observation, reward, done, truncated, info = env.step(
        f"search[{_native_argument(query)}]"
    )
    if reward != 0.0 or done or truncated:
        raise AssertionError(f"native negative search failed: {info}")
    if target_asin.casefold() not in search_observation.casefold():
        raise AssertionError("native search omitted the certified ASIN handle")
    page = env.native_page
    if page is None or target_asin.casefold() not in {
        value.casefold() for value in page.clickables
    }:
        raise AssertionError("certified target is absent from native first page")
    _, reward, done, truncated, info = env.step(f"click[{target_asin}]")
    if reward != 0.0 or done or truncated:
        raise AssertionError(f"native negative ASIN click failed: {info}")


def _native_argument(value: str) -> str:
    text = " ".join(value.split())
    if any(char in text for char in "[]\r\n"):
        raise ValueError(f"native WebShop argument is unsafe: {text!r}")
    return text


if __name__ == "__main__":
    main()
