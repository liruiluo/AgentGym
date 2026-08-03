#!/usr/bin/env python3
"""Smoke four programmatic memory surfaces with one native WebShop load."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agentenv_agentmemory.compositional_recall import (
    CompositionalRecallGenerator,
    VerifiedCompositionalRecallBundleProvider,
)
from agentenv_agentmemory.compositional_recall_webshop_env import (
    COMPOSITIONAL_RECALL_SURFACE,
    CompositionalRecallWebShopEnv,
)
from agentenv_agentmemory.distractor_robustness import (
    DistractorRobustnessGenerator,
    VerifiedDistractorRobustnessBundleProvider,
)
from agentenv_agentmemory.distractor_robustness_webshop_env import (
    DISTRACTOR_ROBUSTNESS_SURFACE,
    DistractorRobustnessWebShopEnv,
)
from agentenv_agentmemory.intent_clarification import (
    IntentClarificationGenerator,
    VerifiedIntentClarificationBundleProvider,
)
from agentenv_agentmemory.intent_clarification_webshop_env import (
    INTENT_CLARIFICATION_SURFACE,
    IntentClarificationWebShopEnv,
)
from agentenv_agentmemory.latent_preference import (
    attest_latent_preference_runtime_inputs,
    load_preference_product_pool,
)
from agentenv_agentmemory.native_webshop_backend import (
    MemoryArenaNativeWebShopBackend,
)
from agentenv_agentmemory.selective_memory_use import (
    SelectiveMemoryUseGenerator,
    VerifiedSelectiveMemoryUseBundleProvider,
)
from agentenv_agentmemory.selective_memory_use_webshop_env import (
    SELECTIVE_MEMORY_USE_SURFACE,
    SelectiveMemoryUseWebShopEnv,
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
    parser.add_argument("--price-seed", type=int, default=233)
    parser.add_argument("--evidence-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_orbit < 0:
        raise SystemExit("--start-orbit must be non-negative")
    pool = load_preference_product_pool(
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
    started_at = time.time()
    cold_started_at = time.monotonic()
    try:
        backend.start()
        cold_start_seconds = time.monotonic() - cold_started_at
        attest_latent_preference_runtime_inputs(
            pool,
            backend,
            items_file=args.items_file.resolve(),
            attributes_file=args.attributes_file.resolve(),
            search_root=args.search_root.resolve(),
            lucene_manifest=args.lucene_index_manifest.resolve(),
        )
        surface_results = [
            _run_distractor_suite(
                backend,
                pool=pool,
                split=args.split,
                seed=args.generator_seed,
                start_orbit=args.start_orbit,
            ),
            _run_compositional_suite(
                backend,
                pool=pool,
                split=args.split,
                seed=args.generator_seed,
                start_orbit=args.start_orbit,
            ),
            _run_intent_suite(
                backend,
                pool=pool,
                split=args.split,
                seed=args.generator_seed,
                start_orbit=args.start_orbit,
            ),
            _run_selective_memory_suite(
                backend,
                pool=pool,
                split=args.split,
                seed=args.generator_seed,
                start_orbit=args.start_orbit,
            ),
        ]
        _require_no_native_sessions(backend, boundary="suite endpoint")
        evidence = {
            "schema": "agentmemory_programmatic_memory_native_suite_v1",
            "started_at_unix": started_at,
            "elapsed_seconds": round(time.time() - started_at, 6),
            "cold_start_seconds": round(cold_start_seconds, 6),
            "native_backend_instances": 1,
            "native_backend_start_calls": 1,
            "surface_order": [item["surface"] for item in surface_results],
            "product_pool": {
                "file": str(args.product_pool.resolve()),
                "file_sha256": args.product_pool_sha256,
                "semantic_sha256": pool.semantic_sha256,
            },
            "runtime": {
                "memoryarena_base_commit": args.memoryarena_base_commit,
                "split": args.split,
                "generator_seed": args.generator_seed,
                "start_orbit": args.start_orbit,
                "price_seed": args.price_seed,
                "backend": backend.metadata(),
            },
            "surfaces": surface_results,
            "validation": {
                "all_twelve_tasks_completed": sum(
                    item["task_count"] for item in surface_results
                )
                == 12,
                "all_six_sessions_completed": all(
                    all(task["sessions_completed"] == 6 for task in item["tasks"])
                    for item in surface_results
                ),
                "target_asin_absent_from_prompt_and_metadata": True,
                "native_search_result_asin_handles_visible": True,
                "query_top1_rejects_top_k": True,
                "query_top1_rejects_memory_id": True,
                "dependent_buy_receipts_verified": True,
                "native_active_session_count_final": backend.active_session_count(),
            },
        }
        args.evidence_json.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_json.write_text(
            json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "AGENTMEMORY_PROGRAMMATIC_MEMORY_NATIVE_SUITE_OK "
            "surfaces=4 tasks=12 sessions=72 cold_loads=1 "
            f"cold_start_seconds={cold_start_seconds:.3f} "
            "active_native_sessions=0"
        )
    finally:
        backend.close()


def _run_distractor_suite(
    backend: MemoryArenaNativeWebShopBackend,
    *,
    pool,
    split: str,
    seed: int,
    start_orbit: int,
) -> dict[str, Any]:
    generator = DistractorRobustnessGenerator(pool=pool, seed=seed)
    provider = VerifiedDistractorRobustnessBundleProvider(
        generator=generator,
        split=split,
        task_count=2,
        start_orbit=start_orbit,
    )
    env = DistractorRobustnessWebShopEnv(
        provider=provider,
        backend=backend,
        env_uid="programmatic_suite_distractor",
        first_valid_add_reward=0.0,
        first_valid_later_session_retrieve_reward=0.0,
    )
    results = []
    reset_observations = []
    try:
        orbit = generator.generate_orbit(start_orbit, split=split)
        for data_index, task in enumerate(orbit.tasks):
            proof = provider.proof_for_index(data_index)
            if proof.source_recency_valid_solution_counts != (1, 1):
                raise AssertionError("distractor proof lost unique legal paths")
            observation, info = env.reset(data_idx=data_index)
            reset_observations.append(observation)
            _assert_private_phase(env, task, phase_index=0, info=info)
            _assert_top1_rejections(env)
            expected_memory_id = f"mem_{len(task.initial_memories):04d}"
            memory_id = _add_memory(
                env,
                key=task.canonical_memory_key,
                value=task.canonical_memory_value,
            )
            if memory_id != expected_memory_id:
                raise AssertionError("distractor canonical memory id changed")
            purchase_info = info
            for phase_index in range(6):
                observation = _retrieve_top1(
                    env,
                    query=task.canonical_query,
                    expected_memory_id=memory_id,
                )
                if task.canonical_memory_value not in observation:
                    raise AssertionError("distractor top1 omitted canonical memory")
                if any(item.value in observation for item in task.initial_memories):
                    raise AssertionError("distractor top1 returned an irrelevant memory")
                purchase_info = _buy_target(
                    env,
                    backend=backend,
                    pool=pool,
                    task=task,
                    phase_index=phase_index,
                )
            _assert_task_complete(env, backend, purchase_info)
            results.append(_task_result(task, purchase_info))
        if reset_observations[0] != reset_observations[1]:
            raise AssertionError("clean/distracted reset observations differ")
    finally:
        env.close()
    _require_no_native_sessions(backend, boundary=DISTRACTOR_ROBUSTNESS_SURFACE)
    return {
        "surface": DISTRACTOR_ROBUSTNESS_SURFACE,
        "task_count": len(results),
        "branch_reset_observations_identical": True,
        "tasks": results,
    }


def _run_compositional_suite(
    backend: MemoryArenaNativeWebShopBackend,
    *,
    pool,
    split: str,
    seed: int,
    start_orbit: int,
) -> dict[str, Any]:
    generator = CompositionalRecallGenerator(pool=pool, seed=seed)
    provider = VerifiedCompositionalRecallBundleProvider(
        generator=generator,
        split=split,
        task_count=4,
        start_orbit=start_orbit,
    )
    env = CompositionalRecallWebShopEnv(
        provider=provider,
        backend=backend,
        env_uid="programmatic_suite_compositional",
        first_valid_add_reward=0.0,
        first_valid_later_session_retrieve_reward=0.0,
    )
    results = []
    try:
        orbit = generator.generate_orbit(start_orbit, split=split)
        for data_index, task in enumerate(orbit.tasks):
            proof = provider.proof_for_index(data_index)
            if proof.valid_solution_counts != (1, 1, 1, 1):
                raise AssertionError("compositional proof lost unique legal paths")
            _, info = env.reset(data_idx=data_index)
            _assert_private_phase(env, task, phase_index=0, info=info)
            _assert_top1_rejections(env)
            mapping, directory = task.canonical_memories
            mapping_id = _add_memory(env, key=mapping.key, value=mapping.value)
            directory_id = None
            purchase_info = info
            for phase_index in range(6):
                if phase_index == 1:
                    directory_id = _add_memory(
                        env,
                        key=directory.key,
                        value=directory.value,
                    )
                if phase_index >= 1:
                    first = _retrieve_top1(
                        env,
                        query=mapping.query,
                        expected_memory_id=mapping_id,
                    )
                    if task.active_profile_token not in first:
                        raise AssertionError("compositional hop1 omitted bridge token")
                    if directory_id is None:
                        raise AssertionError("compositional directory was not stored")
                    second = _retrieve_top1(
                        env,
                        query=directory.query,
                        expected_memory_id=directory_id,
                    )
                    recipe = pool.recipe_by_id(task.source_task.recipe_id)
                    expected_display = recipe.value_display_name(
                        task.preferred_attribute_value
                    )
                    if expected_display not in second:
                        raise AssertionError("compositional hop2 omitted preferred value")
                purchase_info = _buy_target(
                    env,
                    backend=backend,
                    pool=pool,
                    task=task,
                    phase_index=phase_index,
                )
            _assert_task_complete(env, backend, purchase_info)
            result = _task_result(task, purchase_info)
            result["sequential_retrievals_per_dependent_session"] = 2
            results.append(result)
    finally:
        env.close()
    _require_no_native_sessions(backend, boundary=COMPOSITIONAL_RECALL_SURFACE)
    return {
        "surface": COMPOSITIONAL_RECALL_SURFACE,
        "task_count": len(results),
        "factorial_branch_count": 4,
        "tasks": results,
    }


def _run_intent_suite(
    backend: MemoryArenaNativeWebShopBackend,
    *,
    pool,
    split: str,
    seed: int,
    start_orbit: int,
) -> dict[str, Any]:
    generator = IntentClarificationGenerator(pool=pool, seed=seed)
    provider = VerifiedIntentClarificationBundleProvider(
        generator=generator,
        split=split,
        task_count=2,
        start_orbit=start_orbit,
    )
    env = IntentClarificationWebShopEnv(
        provider=provider,
        backend=backend,
        env_uid="programmatic_suite_intent",
        first_valid_add_reward=0.0,
        first_valid_later_session_retrieve_reward=0.0,
    )
    results = []
    reset_observations = []
    try:
        orbit = generator.generate_orbit(start_orbit, split=split)
        for data_index, task in enumerate(orbit.tasks):
            proof = provider.proof_for_index(data_index)
            if proof.valid_solution_counts != (1, 1):
                raise AssertionError("intent proof lost unique legal paths")
            observation, info = env.reset(data_idx=data_index)
            reset_observations.append(observation)
            _assert_private_phase(env, task, phase_index=0, info=info)
            _assert_top1_rejections(env)
            observation, reward, done, truncated, ask_info = env.step(
                "ASK "
                + json.dumps(
                    {"field": task.clarification_field},
                    separators=(",", ":"),
                )
            )
            if reward != 0.0 or done or truncated:
                raise AssertionError("intent ASK failed")
            if task.clarification_answer not in observation:
                raise AssertionError("intent ASK omitted the clarification answer")
            ask_event = ask_info.get("tool_ops", [{}])[0]
            if ask_event.get("op") != "CLARIFY" or "answer" in ask_event:
                raise AssertionError("intent ASK metadata leaked or lost its event")
            fact = task.canonical_memory
            memory_id = _add_memory(env, key=fact.key, value=fact.value)
            purchase_info = info
            for phase_index in range(6):
                if phase_index > 0:
                    retrieved = _retrieve_top1(
                        env,
                        query=fact.query,
                        expected_memory_id=memory_id,
                    )
                    if fact.value not in retrieved:
                        raise AssertionError("intent later session omitted clarification memory")
                purchase_info = _buy_target(
                    env,
                    backend=backend,
                    pool=pool,
                    task=task,
                    phase_index=phase_index,
                )
            _assert_task_complete(env, backend, purchase_info)
            result = _task_result(task, purchase_info)
            result["ask_then_add_then_later_retrieve"] = True
            results.append(result)
        if reset_observations[0] != reset_observations[1]:
            raise AssertionError("intent counterfactual pre-ASK observations differ")
    finally:
        env.close()
    _require_no_native_sessions(backend, boundary=INTENT_CLARIFICATION_SURFACE)
    return {
        "surface": INTENT_CLARIFICATION_SURFACE,
        "task_count": len(results),
        "counterfactual_pre_ask_observations_identical": True,
        "tasks": results,
    }


def _run_selective_memory_suite(
    backend: MemoryArenaNativeWebShopBackend,
    *,
    pool,
    split: str,
    seed: int,
    start_orbit: int,
) -> dict[str, Any]:
    generator = SelectiveMemoryUseGenerator(pool=pool, seed=seed)
    provider = VerifiedSelectiveMemoryUseBundleProvider(
        generator=generator,
        split=split,
        task_count=4,
        start_orbit=start_orbit,
    )
    env = SelectiveMemoryUseWebShopEnv(
        provider=provider,
        backend=backend,
        env_uid="programmatic_suite_selective_memory",
        first_valid_add_reward=0.0,
        first_valid_later_session_retrieve_reward=0.0,
    )
    results = []
    required_observations = []
    try:
        orbit = generator.generate_orbit(start_orbit, split=split)
        for data_index, task in enumerate(orbit.tasks):
            proof = provider.proof_for_index(data_index)
            if proof.valid_solution_counts != (1, 1, 1, 1):
                raise AssertionError("selective-memory proof lost unique legal paths")
            observation, info = env.reset(data_idx=data_index)
            _assert_private_phase(env, task, phase_index=0, info=info)
            _assert_top1_rejections(env)
            if task.initial_memory.value in observation:
                raise AssertionError("selective-memory reset exposed hidden profile")
            if task.memory_requirement == "memory_required":
                required_observations.append(observation)

            memory_action_count = 0
            purchase_info = info
            for phase_index in range(6):
                if task.memory_requirement == "memory_required":
                    observation = _retrieve_top1(
                        env,
                        query=task.canonical_query,
                        expected_memory_id="mem_0000",
                    )
                    memory_action_count += 1
                    if task.initial_memory.value not in observation:
                        raise AssertionError(
                            "selective-memory required branch omitted current profile"
                        )
                purchase_info = _buy_target(
                    env,
                    backend=backend,
                    pool=pool,
                    task=task,
                    phase_index=phase_index,
                )
            expected_memory_actions = (
                6 if task.memory_requirement == "memory_required" else 0
            )
            if memory_action_count != expected_memory_actions:
                raise AssertionError("selective-memory action count changed")
            _assert_task_complete(env, backend, purchase_info)
            result = _task_result(task, purchase_info)
            result.update(
                {
                    "memory_requirement": task.memory_requirement,
                    "memory_action_count": memory_action_count,
                    "memory_abstention_verified": memory_action_count == 0,
                }
            )
            results.append(result)
        if len(required_observations) != 2 or (
            required_observations[0] != required_observations[1]
        ):
            raise AssertionError(
                "selective-memory required A/B reset observations differ"
            )
    finally:
        env.close()
    _require_no_native_sessions(backend, boundary=SELECTIVE_MEMORY_USE_SURFACE)
    return {
        "surface": SELECTIVE_MEMORY_USE_SURFACE,
        "task_count": len(results),
        "factorial_branch_count": 4,
        "required_branch_reset_observations_identical": True,
        "memory_not_required_abstention_verified": True,
        "tasks": results,
    }


def _assert_top1_rejections(env) -> None:
    memory_count = len(env.long_term_memory)
    session_index = env.current_session_index
    actions = (
        'RETRIEVE {"query":"probe","top_k":1}',
        'RETRIEVE {"memory_id":"mem_0000"}',
    )
    for action in actions:
        _, reward, done, truncated, info = env.step(action)
        components = info.get("reward_components", [])
        if reward >= 0.0 or done or truncated:
            raise AssertionError(f"query_top1 accepted forbidden action: {action}")
        if not components or components[0].get("name") != "invalid_action":
            raise AssertionError("forbidden top1 action lacked invalid-action evidence")
        if env.current_session_index != session_index:
            raise AssertionError("forbidden top1 action advanced the task")
        if len(env.long_term_memory) != memory_count:
            raise AssertionError("forbidden top1 action mutated memory")


def _add_memory(env, *, key: str, value: str) -> str:
    action = "ADD " + json.dumps(
        {"key": key, "value": value},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    observation, reward, done, truncated, info = env.step(action)
    event = info.get("memory_ops", [{}])[0]
    memory_id = event.get("memory_id")
    if (
        reward != 0.0
        or done
        or truncated
        or not isinstance(memory_id, str)
        or value not in observation
    ):
        raise AssertionError("canonical memory ADD failed")
    return memory_id


def _retrieve_top1(env, *, query: str, expected_memory_id: str) -> str:
    action = "RETRIEVE " + json.dumps(
        {"query": query},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    observation, reward, done, truncated, info = env.step(action)
    event = info.get("memory_ops", [{}])[0]
    if reward != 0.0 or done or truncated:
        raise AssertionError("canonical query_top1 RETRIEVE failed")
    if event.get("retrieved_memory_ids") != [expected_memory_id]:
        raise AssertionError("canonical query_top1 retrieved the wrong memory")
    if event.get("top_k") != 1:
        raise AssertionError("query_top1 runtime did not fix result count to one")
    return observation


def _buy_target(
    env,
    *,
    backend: MemoryArenaNativeWebShopBackend,
    pool,
    task,
    phase_index: int,
) -> dict[str, Any]:
    _assert_private_phase(env, task, phase_index=phase_index, info=env.build_info())
    target_asin = task.target_asins[phase_index]
    target_product = pool.product_by_asin(target_asin)
    if backend.product_title(target_asin) != target_product.title:
        raise AssertionError("certified target title changed before native smoke")
    search_observation, reward, done, truncated, info = env.step(
        f"search[{_native_argument(target_product.search_query)}]"
    )
    if reward != 0.0 or done or truncated:
        raise AssertionError(f"native search failed: {info}")
    if target_asin.casefold() not in search_observation.casefold():
        raise AssertionError("native search omitted the certified ASIN handle")
    page = env.native_page
    if page is None or target_asin.casefold() not in {
        value.casefold() for value in page.clickables
    }:
        raise AssertionError("certified target is absent from native first page")
    page_observation, reward, done, truncated, info = env.step(
        f"click[{target_asin}]"
    )
    if reward != 0.0 or done or truncated or "Buy Now" not in page_observation:
        raise AssertionError(f"native ASIN click failed: {info}")
    _, reward, done, truncated, info = env.step("click[Buy Now]")
    if truncated:
        raise AssertionError("native BUY truncated unexpectedly")
    event = info.get("tool_ops", [{}])[0]
    if event.get("op") != "BUY" or event.get("purchase_correct") is not True:
        raise AssertionError("dependent BUY was rejected")
    if any(key in event for key in ("actual_asin", "actual_price_cents")):
        raise AssertionError("sanitized BUY metadata leaked private receipt state")
    receipt = env.purchase_ledger[-1]
    if receipt.get("actual_asin") != target_asin.upper():
        raise AssertionError("internal native BUY receipt ASIN mismatch")
    expected_reward = 2.0 if phase_index == 5 else 1.0
    if reward != expected_reward or done != (phase_index == 5):
        raise AssertionError("dependent BUY reward or terminal flag mismatch")
    if info.get("current_subtask_index") != phase_index + 1:
        raise AssertionError("dependent BUY did not advance exactly one session")
    return info


def _assert_private_phase(env, task, *, phase_index: int, info: dict[str, Any]) -> None:
    observation = env.render_observation()
    question = task.questions[phase_index]
    if question not in observation:
        raise AssertionError("native observation omitted the certified task prompt")
    info_text = json.dumps(info, ensure_ascii=True, sort_keys=True)
    for asin in task.target_asins:
        if asin.casefold() in question.casefold():
            raise AssertionError("task prompt leaked a target ASIN")
        if asin.casefold() in info_text.casefold():
            raise AssertionError("policy metadata leaked a target ASIN")
    for private_key in ("task_id", "target_asins", "purchase_history"):
        if private_key in info:
            raise AssertionError(f"policy metadata leaked private field {private_key}")


def _assert_task_complete(
    env,
    backend: MemoryArenaNativeWebShopBackend,
    info: dict[str, Any],
) -> None:
    if not info.get("episode_success") or env.current_session_index != 6:
        raise AssertionError("programmatic memory task did not complete six sessions")
    if env.native_session_token is not None:
        raise AssertionError("terminal task retained a native session token")
    _require_no_native_sessions(backend, boundary="task endpoint")


def _require_no_native_sessions(
    backend: MemoryArenaNativeWebShopBackend,
    *,
    boundary: str,
) -> None:
    count = backend.active_session_count()
    if count != 0:
        raise AssertionError(
            f"native session leak at {boundary}: active_session_count={count}"
        )


def _task_result(task, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "orbit_id": task.orbit_id,
        "branch_kind": task.branch_kind,
        "sessions_completed": info.get("current_subtask_index"),
        "episode_success": info.get("episode_success"),
        "native_session_cleanup_verified": True,
        "target_asin_absent_from_prompt_and_metadata": True,
        "dependent_buy_receipts_verified": 6,
    }


def _native_argument(value: str) -> str:
    text = " ".join(value.split())
    if any(char in text for char in "[]\r\n"):
        raise ValueError(f"unsafe native WebShop argument: {text!r}")
    return text


if __name__ == "__main__":
    main()
