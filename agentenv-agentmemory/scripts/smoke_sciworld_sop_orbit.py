from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agentenv_agentmemory.domains.sciworld import (
    SCIWORLD_SOP_MEMORY_SURFACE,
    SciWorldMemoryFactory,
)
from agentenv_agentmemory.runtime.wrapper import DomainEnvWrapper


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _step(wrapper, env_id: int, action: str, ledger: list[dict[str, Any]]):
    payload = wrapper.step(env_id, f"Action: {action}")
    info = payload["info"]
    evidence = info["domain_evidence"]
    ledger.append(
        {
            "action": action,
            "reward": payload["reward"],
            "done": payload["done"],
            "status": info["status"],
            "phase_index": info["phase_index"],
            "task_name": evidence.get("task_name"),
            "variation_idx": evidence.get("variation_idx"),
            "memory_inventory_count": evidence["memory_inventory_count"],
            "session_trace_count": len(evidence["session_trace"]),
            "observation_sha256": _sha256_text(payload["observation"]),
            "reward_components": info["reward_components"],
        }
    )
    return payload


def _gold_actions(probe: dict[str, Any], task_name: str) -> list[str]:
    for task in probe["tasks"]:
        if task["task_name"] == task_name:
            return list(task["gold_actions"])
    raise RuntimeError(f"Probe is missing gold actions for {task_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    probe = json.loads(args.probe_json.read_text(encoding="utf-8"))
    known_task = "test-conductivity"
    unknown_task = "test-conductivity-of-unknown-substances"
    known_actions = _gold_actions(probe, known_task)
    unknown_actions = _gold_actions(probe, unknown_task)
    last_known_look = max(
        index for index, action in enumerate(known_actions) if action == "look around"
    )

    factory = SciWorldMemoryFactory(
        surface=SCIWORLD_SOP_MEMORY_SURFACE,
        backend="scienceworld",
        split="train",
        task_count=1,
    )
    wrapper = DomainEnvWrapper(factory)
    created = wrapper.create()
    env_id = created["id"]
    ledger: list[dict[str, Any]] = []
    try:
        created_evidence = created["info"]["domain_evidence"]
        if created_evidence["task_name"] != known_task:
            raise RuntimeError("SOP orbit did not begin with the known-material task")
        if created_evidence["variation_idx"] != 0:
            raise RuntimeError("Frozen train orbit 0 did not select known variation 0")

        for action in known_actions[: last_known_look + 1]:
            payload = _step(
                wrapper,
                env_id,
                f'SCI_ACTION {json.dumps({"action": action}, ensure_ascii=True)}',
                ledger,
            )
            if payload["info"]["phase_index"] != 0:
                raise RuntimeError("Known episode ended before the policy could store its SOP")

        sop_value = (
            "In the workshop, connect battery anode/cathode through two wires and "
            "a light bulb, bridge the sample between the remaining wire terminals, "
            "wait, then inspect whether the bulb lights before sorting the sample."
        )
        _step(
            wrapper,
            env_id,
            "ADD "
            + json.dumps(
                {"key": "conductivity test SOP", "value": sop_value},
                ensure_ascii=True,
            ),
            ledger,
        )

        transition = None
        for action in known_actions[last_known_look + 1 :]:
            transition = _step(
                wrapper,
                env_id,
                f'SCI_ACTION {json.dumps({"action": action}, ensure_ascii=True)}',
                ledger,
            )
            if transition["info"]["phase_index"] == 1:
                break
        if transition is None or transition["info"]["phase_index"] != 1:
            raise RuntimeError("Known episode did not advance to the unknown task")
        transition_evidence = transition["info"]["domain_evidence"]
        if transition_evidence["task_name"] != unknown_task:
            raise RuntimeError("SOP orbit advanced to the wrong native task")
        if transition_evidence["variation_idx"] != 0:
            raise RuntimeError("Frozen train orbit 0 did not select unknown variation 0")
        if transition_evidence["memory_inventory_count"] != 1:
            raise RuntimeError("Policy-authored LTM did not survive the native boundary")
        if transition_evidence["session_trace"]:
            raise RuntimeError("Per-episode trace did not reset at the native boundary")

        retrieved = _step(
            wrapper,
            env_id,
            'RETRIEVE {"query": "conductivity test SOP", "top_k": 1}',
            ledger,
        )
        if sop_value not in retrieved["observation"]:
            raise RuntimeError("The later native episode could not retrieve the stored SOP")

        completed = None
        for action in unknown_actions:
            completed = _step(
                wrapper,
                env_id,
                f'SCI_ACTION {json.dumps({"action": action}, ensure_ascii=True)}',
                ledger,
            )
            if completed["done"]:
                break
        if completed is None or not completed["done"]:
            raise RuntimeError("Unknown-material gold path did not terminate")
        if not completed["info"]["episode_success"]:
            raise RuntimeError("Unknown-material gold path was not a strict success")
        if completed["info"]["phase_index"] != 2:
            raise RuntimeError("SOP orbit did not close both native episodes")

        result = {
            "schema_version": "agentmemory_sciworld_sop_native_smoke_v1",
            "source_commit": args.source_commit,
            "surface": SCIWORLD_SOP_MEMORY_SURFACE,
            "backend": "scienceworld",
            "split": "train",
            "orbit_index": 0,
            "native_tasks": [known_task, unknown_task],
            "probe_json": str(args.probe_json),
            "probe_sha256": hashlib.sha256(args.probe_json.read_bytes()).hexdigest(),
            "smoke_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "factory_metadata": factory.metadata(),
            "checks": {
                "known_episode_completed": True,
                "native_boundary_advanced": True,
                "local_trace_reset": True,
                "policy_ltm_persisted": True,
                "later_top1_retrieve_succeeded": True,
                "unknown_episode_strict_success": True,
                "reward_is_native_delta": all(
                    component.get("name") != "scienceworld_score_delta_or_score"
                    for row in ledger
                    for component in row["reward_components"]
                ),
            },
            "ledger": ledger,
        }
    finally:
        if env_id in wrapper.envs:
            wrapper.close(env_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
