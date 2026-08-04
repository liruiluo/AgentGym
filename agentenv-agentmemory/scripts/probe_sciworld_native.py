from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    "test-conductivity",
    "test-conductivity-of-unknown-substances",
)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def probe_task(env, task_name: str, variation: int, generate_gold: bool) -> dict[str, Any]:
    env.load(task_name, variation, generateGoldPath=generate_gold)
    observation, info = env.reset()
    train = list(env.get_variations_train())
    dev = list(env.get_variations_dev())
    test = list(env.get_variations_test())
    gold_actions = list(env.get_gold_action_sequence()) if generate_gold else []
    return {
        "task_name": task_name,
        "variation": variation,
        "max_variations": int(env.get_max_variations(task_name)),
        "splits": {"train": train, "dev": dev, "test": test},
        "split_counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "split_sha256": _sha256_json({"train": train, "dev": dev, "test": test}),
        "task_description": str(info.get("taskDesc", env.get_task_description())),
        "initial_observation": observation,
        "gold_action_count": len(gold_actions),
        "gold_actions": gold_actions,
        "gold_actions_sha256": _sha256_json(gold_actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", dest="tasks")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--generate-gold", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv()
    try:
        tasks = args.tasks or list(DEFAULT_TASKS)
        result = {
            "schema_version": "agentmemory_sciworld_native_probe_v1",
            "scienceworld_version": "1.2.3",
            "tasks": [
                probe_task(env, task, args.variation, args.generate_gold)
                for task in tasks
            ],
        }
    finally:
        env.close()

    rendered = json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
