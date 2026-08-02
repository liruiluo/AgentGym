#!/usr/bin/env python3
"""Run machine-solved programmatic WebShop smoke over a resident HTTP server."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from agentenv_agentmemory.latent_preference import (
    LatentPreferenceGenerator,
    VerifiedLatentPreferenceBundleProvider,
    load_preference_product_pool,
)
from agentenv_agentmemory.procedural import (
    NaturalAttributeChainGenerator,
    VerifiedProceduralBundleProvider,
    load_certified_product_pool,
    scenario_by_id,
)
from agentenv_agentmemory.recency_override import (
    RecencyOverrideGenerator,
    VerifiedRecencyOverrideBundleProvider,
)
from agentenv_agentmemory.smoke_http import (
    AgentMemorySmokeHttpClient,
    SmokeHttpError,
    SmokeServiceExpectation,
    require_correct_buy,
    validate_smoke_service,
)


LATENT_SURFACE = "agentmemory_webshop_latent_preference_train_v1"
PROCEDURAL_SURFACE = "agentmemory_webshop_procedural_natural_chain_train_v1"
RECENCY_SURFACE = "agentmemory_webshop_recency_override_train_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument(
        "--surface",
        choices=(LATENT_SURFACE, PROCEDURAL_SURFACE, RECENCY_SURFACE),
        required=True,
    )
    parser.add_argument("--memoryarena-base-commit", required=True)
    parser.add_argument("--product-pool", required=True, type=Path)
    parser.add_argument("--product-pool-sha256", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="train")
    parser.add_argument("--generator-seed", type=int)
    parser.add_argument("--start-orbit", type=int, default=0)
    parser.add_argument(
        "--data-index",
        dest="data_indices",
        type=int,
        action="append",
        help="Repeat for more tasks; each task gets a fresh HTTP environment/session.",
    )
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--price-seed", type=int, default=233)
    parser.add_argument("--expected-runtime-source-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_indices = args.data_indices or (
        [0, 1] if args.surface in {LATENT_SURFACE, RECENCY_SURFACE} else [0]
    )
    if args.start_orbit < 0:
        raise SystemExit("--start-orbit must be non-negative")
    if any(index < 0 for index in data_indices):
        raise SystemExit("every --data-index must be non-negative")
    if len(data_indices) != len(set(data_indices)):
        raise SystemExit("repeated --data-index values are not allowed")
    if not 1 <= args.sessions <= 6:
        raise SystemExit("--sessions must be between 1 and 6")
    generator_seed = args.generator_seed
    if generator_seed is None:
        generator_seed = 233 if args.surface in {LATENT_SURFACE, RECENCY_SURFACE} else 0

    if args.surface in {LATENT_SURFACE, RECENCY_SURFACE}:
        pool = load_preference_product_pool(
            args.product_pool,
            expected_file_sha256=args.product_pool_sha256,
        )
        if args.surface == LATENT_SURFACE:
            generator = LatentPreferenceGenerator(pool=pool, seed=generator_seed)
            provider = VerifiedLatentPreferenceBundleProvider(
                generator=generator,
                split=args.split,
                task_count=_task_count(data_indices),
                start_orbit=args.start_orbit,
            )
        else:
            generator = RecencyOverrideGenerator(pool=pool, seed=generator_seed)
            provider = VerifiedRecencyOverrideBundleProvider(
                generator=generator,
                split=args.split,
                task_count=_task_count(data_indices),
                start_orbit=args.start_orbit,
            )
        memory_prompt_mode = "latent_preference_sop"
    else:
        pool = load_certified_product_pool(
            args.product_pool,
            expected_file_sha256=args.product_pool_sha256,
        )
        generator = NaturalAttributeChainGenerator(pool=pool, seed=generator_seed)
        provider = VerifiedProceduralBundleProvider(
            generator=generator,
            split=args.split,
            task_count=_task_count(data_indices),
            start_orbit=args.start_orbit,
        )
        memory_prompt_mode = "legacy"

    runtime_source_id = args.expected_runtime_source_id or _git_head()
    client = AgentMemorySmokeHttpClient(args.server_url)
    expected = SmokeServiceExpectation(
        surface=args.surface,
        runtime_source_id=runtime_source_id,
        memoryarena_base_commit=args.memoryarena_base_commit,
        product_pool_file_sha256=args.product_pool_sha256,
        product_pool_semantic_sha256=pool.semantic_sha256,
        catalog_sha256=pool.catalog_sha256,
        attributes_sha256=pool.attributes_sha256,
        lucene_manifest_sha256=pool.lucene_index_sha256,
        generator_seed=generator_seed,
        split=args.split,
        price_seed=args.price_seed,
        memory_prompt_mode=memory_prompt_mode,
        minimum_task_count=max(data_indices) + 1,
        start_orbit=args.start_orbit,
    )
    fingerprint = validate_smoke_service(client.metadata(), expected)
    print(
        "AGENTMEMORY_RESIDENT_SMOKE_SERVICE_OK "
        f"surface={args.surface} fingerprint={fingerprint} "
        f"tasks={len(data_indices)} cold_start_already_paid=true"
    )

    for data_index in data_indices:
        with client.open(data_index) as env:
            if args.surface == LATENT_SURFACE:
                orbit_id = _run_latent_task(
                    env,
                    provider=provider,
                    generator=generator,
                    data_index=data_index,
                    sessions=args.sessions,
                    split=args.split,
                    start_orbit=args.start_orbit,
                )
            elif args.surface == RECENCY_SURFACE:
                orbit_id = _run_recency_task(
                    env,
                    provider=provider,
                    generator=generator,
                    data_index=data_index,
                    sessions=args.sessions,
                    split=args.split,
                    start_orbit=args.start_orbit,
                )
            else:
                orbit_id = _run_procedural_task(
                    env,
                    provider=provider,
                    generator=generator,
                    data_index=data_index,
                    sessions=args.sessions,
                    split=args.split,
                )
            print(
                "AGENTMEMORY_RESIDENT_SMOKE_TASK_OK "
                f"surface={args.surface} data_index={data_index} orbit_or_scenario={orbit_id}"
            )

    print(
        "AGENTMEMORY_RESIDENT_SMOKE_BATCH_OK "
        f"surface={args.surface} tasks={len(data_indices)} "
        f"data_indices={','.join(map(str, data_indices))}"
    )


def _run_latent_task(
    env,
    *,
    provider,
    generator,
    data_index: int,
    sessions: int,
    split: str,
    start_orbit: int,
) -> str:
    bundle = provider.get(data_index)
    orbit = generator.generate_orbit(start_orbit + data_index // 2, split=split)
    task = orbit.tasks[data_index % 2]
    if task.task_id != bundle.task_id:
        raise SmokeHttpError("latent provider and local orbit disagree")
    proof = provider.proof_for_index(data_index)
    if proof.valid_solution_counts != (1, 1):
        raise SmokeHttpError("latent-preference proof lost unique solutions")
    recipe = generator.pool.recipe_by_id(task.recipe_id)
    preferred_display = recipe.value_display_name(task.preferred_attribute_value)
    memory_value = f"Customer preference for {recipe.axis_display_name}: {preferred_display}"
    memory_written = False
    done = False
    for session_index in range(sessions):
        phase = task.phases[session_index]
        observation = env.last_payload["observation"]
        if phase.question not in observation:
            raise SmokeHttpError("latent task prompt missing from reset observation")
        if any(
            candidate.asin.casefold() in phase.question.casefold()
            for candidate in phase.candidates
        ):
            raise SmokeHttpError("latent task prompt leaked a candidate ASIN")
        if phase.phase_kind == "evidence" and not memory_written:
            action = json.dumps(
                {"key": "user_preference", "value": memory_value},
                separators=(",", ":"),
            )
            observation, reward, done, _, _ = env.step(f"ADD {action}")
            if reward != 0.0 or done or memory_value not in observation:
                raise SmokeHttpError("latent ADD failed over HTTP")
            memory_written = True
        elif phase.phase_kind == "application":
            action = json.dumps({"query": "user preference", "top_k": 1}, separators=(",", ":"))
            observation, reward, done, _, _ = env.step(f"RETRIEVE {action}")
            if reward != 0.0 or done or memory_value not in observation:
                raise SmokeHttpError("latent RETRIEVE failed over HTTP")

        target_asin = bundle.target_asins[session_index]
        target_product = next(
            product for product in generator.pool.products if product.asin == target_asin
        )
        _search_and_open(env, query=target_product.search_query, target_asin=target_asin)
        observation, reward, done, _, info = env.step("click[Buy Now]")
        del observation
        require_correct_buy(info, session_index=session_index)
        expected_reward = 2.0 if session_index == 5 else 1.0
        if reward != expected_reward or done != (session_index == 5):
            raise SmokeHttpError("latent BUY reward or terminal flag mismatch")
        if info.get("current_subtask_index") != session_index + 1:
            raise SmokeHttpError("latent HTTP BUY did not advance the session")
    return bundle.orbit_id


def _run_procedural_task(
    env,
    *,
    provider,
    generator,
    data_index: int,
    sessions: int,
    split: str,
) -> str:
    bundle = provider.get(data_index)
    scenario = scenario_by_id(bundle.scenario_id)
    for session_index in range(sessions):
        question = bundle.questions[session_index]
        current_observation = env.last_payload["observation"]
        if question not in current_observation:
            raise SmokeHttpError("procedural task prompt missing from HTTP observation")
        if bundle.target_asins[session_index].casefold() in question.casefold():
            raise SmokeHttpError("procedural task prompt leaked the target ASIN")
        if session_index > 0:
            previous_slot = scenario.slots[session_index - 1]
            previous_value = bundle.target_attribute_values[session_index - 1]
            previous_display = previous_slot.value(previous_value).display_name
            action = json.dumps(
                {"query": previous_slot.slot_id, "top_k": 1},
                separators=(",", ":"),
            )
            observation, reward, done, _, _ = env.step(f"RETRIEVE {action}")
            if reward != 0.0 or done or previous_display not in observation:
                raise SmokeHttpError("procedural RETRIEVE failed over HTTP")

        slot = scenario.slots[session_index]
        target_value = bundle.target_attribute_values[session_index]
        target_display = slot.value(target_value).display_name
        memory_value = f"Bought {slot.display_name}; {slot.attribute_name}={target_display}"
        action = json.dumps(
            {"key": slot.slot_id, "value": memory_value},
            separators=(",", ":"),
        )
        observation, reward, done, _, _ = env.step(f"ADD {action}")
        if reward != 0.0 or done or memory_value not in observation:
            raise SmokeHttpError("procedural ADD failed over HTTP")

        target_asin = bundle.target_asins[session_index]
        target_product = next(
            product for product in generator.pool.products if product.asin == target_asin
        )
        _search_and_open(env, query=target_product.search_query, target_asin=target_asin)
        observation, reward, done, _, info = env.step("click[Buy Now]")
        del observation
        require_correct_buy(info, session_index=session_index)
        expected_reward = 2.0 if session_index == 5 else 1.0
        if reward != expected_reward or done != (session_index == 5):
            raise SmokeHttpError("procedural BUY reward or terminal flag mismatch")
        if info.get("current_subtask_index") != session_index + 1:
            raise SmokeHttpError("procedural HTTP BUY did not advance the session")
    return bundle.scenario_id


def _run_recency_task(
    env,
    *,
    provider,
    generator,
    data_index: int,
    sessions: int,
    split: str,
    start_orbit: int,
) -> str:
    bundle = provider.get(data_index)
    orbit = generator.generate_orbit(start_orbit + data_index // 2, split=split)
    task = orbit.tasks[data_index % 2]
    if task.task_id != bundle.task_id:
        raise SmokeHttpError("recency provider and local orbit disagree")
    proof = provider.proof_for_index(data_index)
    if proof.valid_solution_counts != (1, 1):
        raise SmokeHttpError("recency proof lost unique solutions")
    recipe = generator.pool.recipe_by_id(task.recipe_id)
    old_display = recipe.value_display_name(task.old_attribute_value)
    new_display = recipe.value_display_name(task.new_attribute_value)
    memory_value = f"Customer current preference for {recipe.axis_display_name}: {old_display}"
    memory_id = None
    for session_index in range(sessions):
        phase = task.phases[session_index]
        observation = env.last_payload["observation"]
        if phase.question not in observation:
            raise SmokeHttpError("recency task prompt missing from HTTP observation")
        if any(
            candidate.asin.casefold() in phase.question.casefold()
            for candidate in phase.candidates
        ):
            raise SmokeHttpError("recency task prompt leaked a candidate ASIN")

        if session_index == 0:
            observation, reward, done, _, info = env.step(
                "ADD "
                + json.dumps(
                    {"key": "user_preference", "value": memory_value},
                    separators=(",", ":"),
                )
            )
            added = info.get("memory_state_diff", {}).get("added", [])
            if reward != 0.0 or done or len(added) != 1:
                raise SmokeHttpError("recency initial ADD failed over HTTP")
            memory_id = added[0].get("memory_id")
            if not isinstance(memory_id, str) or memory_value not in observation:
                raise SmokeHttpError("recency initial ADD did not expose memory id/value")
        elif session_index == 2 and bundle.branch_kind == "flip":
            if memory_id is None:
                raise SmokeHttpError("recency flip lacks initial memory id")
            observation, reward, done, _, info = env.step(
                "UPDATE "
                + json.dumps(
                    {
                        "memory_id": memory_id,
                        "value": (
                            f"Customer current preference for "
                            f"{recipe.axis_display_name}: {new_display}"
                        ),
                    },
                    separators=(",", ":"),
                )
            )
            updated = info.get("memory_state_diff", {}).get("updated", [])
            if reward != 0.0 or done or len(updated) != 1:
                raise SmokeHttpError("recency flip UPDATE failed over HTTP")
            before = updated[0].get("before", {})
            after = updated[0].get("after", {})
            if before.get("value") != memory_value or after.get("value") == memory_value:
                raise SmokeHttpError("recency UPDATE did not replace old canonical value")
            if new_display not in observation or old_display in observation:
                raise SmokeHttpError("recency UPDATE observation has wrong value")
        else:
            if memory_id is None:
                raise SmokeHttpError("recency task lacks initial memory id")
            observation, reward, done, _, info = env.step(
                "RETRIEVE "
                + json.dumps({"memory_id": memory_id}, separators=(",", ":"))
            )
            expected_display = (
                new_display
                if bundle.branch_kind == "flip" and session_index >= 3
                else old_display
            )
            if reward != 0.0 or done or expected_display not in observation:
                raise SmokeHttpError("recency RETRIEVE returned the wrong canonical value")
            retrieved = info.get("tool_ops", [{}])[0].get("retrieved_memory_ids", [])
            if retrieved != [memory_id]:
                raise SmokeHttpError("recency RETRIEVE did not return the canonical memory")

        target_asin = bundle.target_asins[session_index]
        target_product = next(
            product for product in generator.pool.products if product.asin == target_asin
        )
        _search_and_open(env, query=target_product.search_query, target_asin=target_asin)
        observation, reward, done, _, info = env.step("click[Buy Now]")
        del observation
        require_correct_buy(info, session_index=session_index)
        expected_reward = 2.0 if session_index == 5 else 1.0
        if reward != expected_reward or done != (session_index == 5):
            raise SmokeHttpError("recency BUY reward or terminal flag mismatch")
        if info.get("current_subtask_index") != session_index + 1:
            raise SmokeHttpError("recency HTTP BUY did not advance the session")
    return bundle.orbit_id


def _search_and_open(env, *, query: str, target_asin: str) -> None:
    action = f"search[{_native_argument(query)}]"
    observation, reward, done, _, _ = env.step(action)
    if reward != 0.0 or done or target_asin.casefold() not in observation.casefold():
        raise SmokeHttpError("native search did not expose the certified ASIN handle")
    observation, reward, done, _, _ = env.step(f"click[{target_asin}]")
    if reward != 0.0 or done or "Buy Now" not in observation:
        raise SmokeHttpError("native ASIN click did not open a Buy Now page")


def _native_argument(value: str) -> str:
    text = " ".join(value.split())
    if any(char in text for char in "[]\r\n"):
        raise ValueError(f"unsafe native WebShop argument: {text!r}")
    return text


def _task_count(data_indices: list[int]) -> int:
    count = max(data_indices) + 1
    return count + (count % 2)


def _git_head() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "--expected-runtime-source-id is required when the smoke client is "
            "not running from a git worktree"
        ) from exc


if __name__ == "__main__":
    main()
