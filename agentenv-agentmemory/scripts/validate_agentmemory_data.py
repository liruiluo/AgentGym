from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from agentenv_agentmemory.environment import (
    default_smoke_data_path,
    default_split_dir,
    load_split_task_ids,
    load_task_dataset,
    load_tasks_from_jsonl,
    select_tasks_by_split,
)

LEAKY_PRODUCT_ID_PATTERN = re.compile(
    r"(\d|large|small|inch|kg|vesa|usb|hdmi|displayport|barrel|tv|laptop|monitor)$|"
    r"(\d|large|small|inch|kg|vesa|usb|hdmi|displayport|barrel)",
    re.IGNORECASE,
)


def iter_product_ids(tasks) -> Iterable[tuple[str, str]]:
    for task in tasks:
        for subtask in task.subtasks:
            for product in subtask.candidate_products:
                yield task.task_id, product.product_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate AgentMemoryGym bundled-shopping JSONL smoke data.")
    parser.add_argument("--data", type=Path, default=default_smoke_data_path())
    parser.add_argument("--split-dir", type=Path, default=default_split_dir())
    parser.add_argument("--allow-empty-split", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks_from_jsonl(args.data)
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise SystemExit("Duplicate task_id found in data file.")

    split_to_ids = {split: load_split_task_ids(split, args.split_dir) for split in ["train", "dev", "test"]}
    for split, ids in split_to_ids.items():
        if not ids and not args.allow_empty_split:
            raise SystemExit(f"Split {split} is empty.")
        missing = [task_id for task_id in ids if task_id not in task_by_id]
        if missing:
            raise SystemExit(f"Split {split} references unknown task ids: {missing}")
        mismatch = [task_id for task_id in ids if task_by_id[task_id].split != split]
        if mismatch:
            raise SystemExit(f"Split {split} has task ids whose record split differs: {mismatch}")
        selected_tasks = select_tasks_by_split(tasks, split, split_dir=args.split_dir)
        loaded_tasks = load_task_dataset(data_path=args.data, split=split, split_dir=args.split_dir)
        if [task.task_id for task in selected_tasks] != [task.task_id for task in loaded_tasks]:
            raise SystemExit(f"Split {split} loader order mismatch.")

    covered = {task_id for ids in split_to_ids.values() for task_id in ids}
    uncovered = sorted(set(task_by_id) - covered)
    if uncovered:
        raise SystemExit(f"Tasks missing from split files: {uncovered}")

    leaky_ids = [(task_id, product_id) for task_id, product_id in iter_product_ids(tasks) if LEAKY_PRODUCT_ID_PATTERN.search(product_id)]
    if leaky_ids:
        raise SystemExit(f"Potential answer-leaking product ids found: {leaky_ids}")

    print(
        "AGENTMEMORY_DATA_VALIDATE_OK",
        f"tasks={len(tasks)}",
        "splits=" + ",".join(f"{split}:{len(ids)}" for split, ids in split_to_ids.items()),
    )


if __name__ == "__main__":
    main()
