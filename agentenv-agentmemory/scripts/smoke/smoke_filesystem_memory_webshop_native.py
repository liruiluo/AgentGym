#!/usr/bin/env python3
"""Run native WebShop filesystem-memory chain and intervention gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from agentenv_agentmemory.filesystem_webshop_env import (
    ProceduralFilesystemWebShopEnv,
)
from agentenv_agentmemory.native_webshop_backend import MemoryArenaNativeWebShopBackend
from agentenv_agentmemory.procedural import (
    NaturalAttributeChainGenerator,
    VerifiedProceduralBundleProvider,
    load_certified_product_pool,
    scenario_by_id,
)
from agentenv_agentmemory.procedural_wrapper import attest_procedural_runtime_inputs
from agentenv_agentmemory.persistent_workspace import WorkspaceLimits
from agentenv_agentmemory.reward_hierarchy import (
    INVALID_ACTION_PENALTY,
    WRONG_BUY_TERMINAL_FAILURE,
)
from agentenv_agentmemory.workspace_sandbox import LinuxNamespaceShellSandbox


MEMORY_PATH = ".agent_memory/MEMORY.md"
INTERVENTION_ARMS = ("correct", "blank", "swapped", "no_workspace")


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
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--generator-seed", type=int, default=233)
    parser.add_argument("--data-index", type=int, default=0)
    parser.add_argument("--price-seed", type=int, default=233)
    parser.add_argument("--workspace-root-parent", type=Path)
    parser.add_argument("--positive-task-reward-scale", type=float, default=1.0)
    parser.add_argument("--workspace-rg-binary", required=True, type=Path)
    parser.add_argument("--workspace-rg-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_index < 0:
        raise SystemExit("--data-index must be non-negative")

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
    task_count = args.data_index + 1
    if task_count % 2:
        task_count += 1
    provider = VerifiedProceduralBundleProvider(
        generator=NaturalAttributeChainGenerator(pool=pool, seed=args.generator_seed),
        split=args.split,
        task_count=task_count,
    )
    workspace_limits = WorkspaceLimits()
    shell_sandbox = LinuxNamespaceShellSandbox.from_environment(
        limits=workspace_limits.shell_limits(),
        rg_binary=args.workspace_rg_binary,
        expected_rg_sha256=args.workspace_rg_sha256,
    )
    env = ProceduralFilesystemWebShopEnv(
        provider=provider,
        backend=backend,
        env_uid="filesystem_native_intervention_smoke",
        shell_sandbox=shell_sandbox,
        workspace_root_parent=args.workspace_root_parent,
        workspace_limits=workspace_limits,
        positive_task_reward_scale=args.positive_task_reward_scale,
    )

    final_root: Path | None = None
    try:
        attest_procedural_runtime_inputs(
            pool,
            backend,
            items_file=args.items_file.resolve(),
            attributes_file=args.attributes_file.resolve(),
            search_root=args.search_root.resolve(),
            lucene_manifest=args.lucene_index_manifest.resolve(),
        )
        task = _task_for_index(provider, args.data_index)
        results = []
        previous_root: Path | None = None
        for arm in INTERVENTION_ARMS:
            result, current_root = _run_arm(
                env,
                backend=backend,
                provider=provider,
                task=task,
                data_index=args.data_index,
                arm=arm,
                previous_root=previous_root,
            )
            results.append(result)
            previous_root = current_root

        final_root = previous_root
        env.set_workspace_enabled(True)
        _, reset_info = env.reset(data_idx=args.data_index)
        if final_root is not None and final_root.exists():
            raise AssertionError("final intervention workspace survived reset")
        if reset_info["workspace_snapshot"]["file_count"] != 0:
            raise AssertionError("reset did not create an empty workspace")
        if reset_info["workspace_audit_event_count"] != 0:
            raise AssertionError("reset retained workspace audit events")
        final_root = env.workspace.host_root

        summary = {
            "schema": "agentmemory_codex_workspace_native_interventions_v2",
            "evidence_scope": "scripted_runtime_only_not_model_capability",
            "surface": env.surface,
            "data_index": args.data_index,
            "split": args.split,
            "generator_seed": args.generator_seed,
            "task_id": provider.get(args.data_index).task_id,
            "scenario_id": provider.get(args.data_index).scenario_id,
            "product_pool_file_sha256": args.product_pool_sha256,
            "product_pool_semantic_sha256": pool.semantic_sha256,
            "reward_contract": env.reward_contract(),
            "arms": results,
            "reset_cleanup_verified": True,
            "close_cleanup_verified": False,
            "native_session_count_before_close": backend.active_session_count(),
        }
    finally:
        env.close()
        backend.close()

    if final_root is None or final_root.exists():
        raise AssertionError("workspace survived environment close")
    if backend.active_session_count() != 0:
        raise AssertionError("native WebShop session survived environment close")
    summary["close_cleanup_verified"] = True
    summary["native_session_count_after_close"] = backend.active_session_count()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "AGENTMEMORY_FILESYSTEM_NATIVE_INTERVENTIONS_OK "
        f"data_index={args.data_index} arms={','.join(INTERVENTION_ARMS)} "
        f"output_sha256={_file_sha256(args.output_json)}"
    )


def _task_for_index(provider: VerifiedProceduralBundleProvider, data_index: int):
    orbit_index, branch_index = divmod(data_index, 2)
    orbit = provider.generator.generate_orbit(
        provider.start_orbit + orbit_index,
        split=provider.split,
    )
    return orbit.tasks[branch_index]


def _run_arm(
    env: ProceduralFilesystemWebShopEnv,
    *,
    backend: MemoryArenaNativeWebShopBackend,
    provider: VerifiedProceduralBundleProvider,
    task,
    data_index: int,
    arm: str,
    previous_root: Path | None,
) -> tuple[dict[str, Any], Path | None]:
    if arm not in INTERVENTION_ARMS:
        raise ValueError(f"unknown intervention arm: {arm}")
    env.set_workspace_enabled(arm != "no_workspace")
    _, reset_info = env.reset(data_idx=data_index)
    if previous_root is not None and previous_root.exists():
        raise AssertionError(f"previous workspace survived reset before {arm}")
    if reset_info["workspace_snapshot"]["file_count"] != 0:
        raise AssertionError(f"{arm} did not start with an empty workspace")
    workspace_root = None if arm == "no_workspace" else env.workspace.host_root

    if arm == "correct":
        _, reward, done, _, legacy_info = env.step(
            'ADD {"key":"legacy","value":"must fail"}'
        )
        if reward != INVALID_ACTION_PENALTY or done:
            raise AssertionError("filesystem v2 accepted a legacy memory action")
        if legacy_info["workspace_snapshot"]["file_count"] != 0:
            raise AssertionError("legacy action mutated the filesystem workspace")

    source_phase = task.phases[0]
    source_display = source_phase.target_attribute_value
    source_line = (
        f"source_session=0 slot={source_phase.slot_id} "
        f"value={source_display}"
    )
    opposite_value = next(
        value
        for value in scenario_by_id(task.scenario_id).slots[0].value_ids
        if value != source_display
    )
    contents = {
        "correct": source_line,
        "blank": "",
        "swapped": (
            f"source_session=0 slot={source_phase.slot_id} "
            f"value={opposite_value}"
        ),
    }

    write_sha256: str | None = None
    if arm != "no_workspace":
        content = contents[arm]
        _, reward, done, _, write_info = env.step(_add_file_patch(MEMORY_PATH, content))
        if reward != 0.0 or done:
            raise AssertionError(
                f"{arm} apply_patch received task reward or terminated"
            )
        if write_info["memory_ops"]:
            raise AssertionError(
                f"{arm} apply_patch was reported as a legacy memory op"
            )
        write_event = write_info["workspace_ops"][0]
        if write_event["op"] != "APPLY_PATCH":
            raise AssertionError(f"{arm} did not execute APPLY_PATCH")
        write_sha256 = _snapshot_file_sha256(
            write_info["workspace_snapshot"],
            MEMORY_PATH,
        )

    source_product = _product_for_asin(provider, source_phase.target_asin)
    _search_and_open_target(env, source_product)
    _, source_reward, source_done, _, source_buy_info = env.step("click[Buy Now]")
    if source_reward != 1.0 or source_done or env.current_session_index != 1:
        raise AssertionError(f"{arm} source BUY did not advance: {source_buy_info}")
    _assert_purchase(env, source_phase.target_asin)

    read_tree_sha256: str | None = None
    grep_tree_sha256: str | None = None
    if arm == "no_workspace":
        _, read_reward, read_done, _, read_info = env.step(
            _shell_action(f"cat -- {shlex.quote(MEMORY_PATH)}")
        )
        if read_reward != INVALID_ACTION_PENALTY or read_done:
            raise AssertionError("no_workspace shell_command did not fail closed")
        if read_info["workspace_audit_event_count"] != 0:
            raise AssertionError(
                "failed no_workspace shell_command created an audit event"
            )
        recovered_content = ""
    else:
        read_observation, read_reward, read_done, _, read_info = env.step(
            _shell_action(f"cat -- {shlex.quote(MEMORY_PATH)}")
        )
        if read_reward != 0.0 or read_done:
            raise AssertionError(
                f"{arm} shell_command read received task reward or terminated"
            )
        read_event = read_info["workspace_ops"][0]
        if read_event["op"] != "SHELL_COMMAND" or read_event["exit_code"] != 0:
            raise AssertionError(f"{arm} shell_command read failed")
        recovered_content = read_event["stdout"].rstrip("\n")
        if recovered_content != contents[arm]:
            raise AssertionError(f"{arm} shell_command read returned wrong content")
        if recovered_content and recovered_content not in read_observation:
            raise AssertionError(
                f"{arm} shell_command observation omitted the file content"
            )
        read_tree_sha256 = read_event["workspace_tree_sha256_after"]
        if read_event["workspace_diff"] != _empty_workspace_diff():
            raise AssertionError(f"{arm} shell_command read mutated the workspace")

        rg_observation, rg_reward, rg_done, _, rg_info = env.step(
            _shell_action(
                "rg -n --fixed-strings -- "
                f"{shlex.quote(source_phase.slot_id)} {shlex.quote(MEMORY_PATH)}"
            )
        )
        if rg_reward != 0.0 or rg_done:
            raise AssertionError(
                f"{arm} shell_command rg received task reward or terminated"
            )
        rg_event = rg_info["workspace_ops"][0]
        grep_tree_sha256 = rg_event["workspace_tree_sha256_after"]
        if rg_event["workspace_diff"] != _empty_workspace_diff():
            raise AssertionError(f"{arm} shell_command rg mutated the workspace")
        if recovered_content:
            if rg_event["exit_code"] != 0 or recovered_content not in rg_observation:
                raise AssertionError(
                    f"{arm} shell_command rg did not recover the written record"
                )
        elif rg_event["exit_code"] != 1 or rg_event["stdout"]:
            raise AssertionError("blank workspace unexpectedly matched rg")

    dependent_phase = task.phases[1]
    has_correct_memory = f"value={source_display}" in recovered_content
    chosen_asin = (
        dependent_phase.target_asin
        if has_correct_memory
        else next(
            candidate.asin
            for candidate in dependent_phase.candidates
            if candidate.asin != dependent_phase.target_asin
        )
    )
    chosen_product = _product_for_asin(provider, chosen_asin)
    _search_and_open_target(env, chosen_product)
    _, dependent_reward, dependent_done, _, dependent_buy_info = env.step(
        "click[Buy Now]"
    )
    _assert_purchase(env, chosen_asin)

    expected_success = arm == "correct"
    observed_success = (
        dependent_reward == 1.0
        and not dependent_done
        and env.current_session_index == 2
        and env.purchase_ledger[-1]["purchase_correct"] is True
    )
    if expected_success != observed_success:
        raise AssertionError(
            f"{arm} intervention outcome mismatch: {dependent_buy_info}"
        )
    if not expected_success:
        if dependent_reward != WRONG_BUY_TERMINAL_FAILURE or not dependent_done:
            raise AssertionError(f"{arm} wrong BUY did not fail fast")
        if env.current_session_index != 1:
            raise AssertionError(f"{arm} wrong BUY advanced the session")

    snapshot = env.workspace.snapshot()
    events = list(env.workspace.audit_events)
    _assert_audit_chain(events, snapshot=snapshot, arm=arm)
    if arm == "correct":
        if [event["op"] for event in events] != [
            "APPLY_PATCH",
            "SHELL_COMMAND",
            "SHELL_COMMAND",
        ]:
            raise AssertionError("correct arm lost the exact Codex tool evidence chain")
        if not (
            write_sha256
            and read_tree_sha256 == snapshot["tree_sha256"]
            and grep_tree_sha256 == snapshot["tree_sha256"]
        ):
            raise AssertionError("correct arm workspace version evidence diverged")
    if arm in {"correct", "blank", "swapped"}:
        if [item["path"] for item in snapshot["files"]] != [MEMORY_PATH]:
            raise AssertionError(f"{arm} changed the intervention file tree")
    elif snapshot["file_count"] != 0:
        raise AssertionError("no_workspace arm unexpectedly contains files")

    return (
        {
            "arm": arm,
            "expected_success": expected_success,
            "observed_success": observed_success,
            "source_buy_correct": source_buy_info["tool_ops"][0][
                "purchase_correct"
            ],
            "dependent_buy_correct": dependent_buy_info["tool_ops"][0][
                "purchase_correct"
            ],
            "dependent_reward": dependent_reward,
            "dependent_done": dependent_done,
            "chosen_asin": chosen_asin,
            "target_asin": dependent_phase.target_asin,
            "workspace_snapshot": snapshot,
            "workspace_event_count": len(events),
            "workspace_event_ops": [event["op"] for event in events],
            "written_content_sha256": write_sha256,
            "read_tree_sha256": read_tree_sha256,
            "rg_tree_sha256": grep_tree_sha256,
            "legacy_action_rejected": arm == "correct",
            "evidence_scope": "scripted_runtime_only_not_model_capability",
        },
        workspace_root,
    )


def _shell_action(command: str) -> str:
    return "shell_command " + json.dumps(
        {"command": command, "workdir": ".", "timeout_ms": 10_000},
        separators=(",", ":"),
    )


def _add_file_patch(path: str, content: str) -> str:
    lines = ["*** Begin Patch", f"*** Add File: {path}"]
    lines.extend("+" + line for line in content.splitlines())
    lines.append("*** End Patch")
    return "apply_patch\n" + "\n".join(lines)


def _snapshot_file_sha256(snapshot: dict[str, Any], path: str) -> str:
    matches = [item["sha256"] for item in snapshot["files"] if item["path"] == path]
    if len(matches) != 1:
        raise AssertionError(f"workspace snapshot does not contain exactly one {path}")
    return matches[0]


def _empty_workspace_diff() -> dict[str, Any]:
    return {
        "added": [],
        "modified": [],
        "deleted": [],
        "directories_added": [],
        "directories_deleted": [],
    }


def _assert_audit_chain(
    events: list[dict[str, Any]],
    *,
    snapshot: dict[str, Any],
    arm: str,
) -> None:
    if [event["event_id"] for event in events] != list(range(len(events))):
        raise AssertionError(f"{arm} workspace event IDs are not contiguous")
    for previous, current in zip(events, events[1:]):
        if previous["workspace_tree_sha256_after"] != current[
            "workspace_tree_sha256_before"
        ]:
            raise AssertionError(f"{arm} workspace tree continuity failed")
    if events:
        if events[-1]["workspace_tree_sha256_after"] != snapshot["tree_sha256"]:
            raise AssertionError(f"{arm} final workspace tree hash mismatch")


def _product_for_asin(provider: VerifiedProceduralBundleProvider, asin: str):
    return next(
        product
        for product in provider.generator.pool.products
        if product.asin == asin
    )


def _search_and_open_target(env: ProceduralFilesystemWebShopEnv, product) -> None:
    observation, reward, done, _, info = env.step(
        f"search[{_native_argument(product.search_query)}]"
    )
    if reward != 0.0 or done:
        raise AssertionError(f"native search failed: {info}")
    if product.asin.casefold() not in observation.casefold():
        raise AssertionError(f"native search omitted ASIN {product.asin}")
    page = env.native_page
    if page is None or product.asin.casefold() not in {
        value.casefold() for value in page.clickables
    }:
        raise AssertionError(f"native search page cannot click ASIN {product.asin}")
    _, click_reward, click_done, _, click_info = env.step(
        f"click[{product.asin}]"
    )
    if click_reward != 0.0 or click_done:
        raise AssertionError(f"native ASIN click failed: {click_info}")


def _assert_purchase(env: ProceduralFilesystemWebShopEnv, asin: str) -> None:
    if not env.purchase_ledger:
        raise AssertionError("native purchase produced no receipt")
    receipt = env.purchase_ledger[-1]
    if receipt.get("actual_asin") != asin.upper():
        raise AssertionError(
            f"purchase receipt ASIN mismatch: {receipt.get('actual_asin')} != {asin}"
        )


def _native_argument(value: str) -> str:
    text = " ".join(value.split())
    if any(char in text for char in "[]\r\n"):
        raise ValueError(f"unsafe native WebShop argument: {text!r}")
    return text


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
