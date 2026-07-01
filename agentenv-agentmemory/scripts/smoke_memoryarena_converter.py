from __future__ import annotations

import tempfile
from pathlib import Path

from agentenv_agentmemory.environment import AgentMemoryEnv, load_task_dataset
from agentenv_agentmemory.memoryarena_converter import convert_file


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    fixture = package_root / "agentenv_agentmemory" / "data" / "fixtures" / "memoryarena_bundled_shopping_sample.jsonl"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        output = temp_root / "agentmemory.jsonl"
        split_dir = temp_root / "splits"
        report = temp_root / "report.jsonl"
        stats = convert_file(
            fixture,
            output,
            split_dir=split_dir,
            report_path=report,
            split_mode="cycle",
            min_match_score=1,
        )
        assert stats.task_count == 3, stats
        assert stats.split_counts == {"train": 1, "dev": 1, "test": 1}, stats
        for split in ["train", "dev", "test"]:
            tasks = load_task_dataset(data_path=output, split=split, split_dir=split_dir)
            assert len(tasks) == 1, (split, tasks)
            assert tasks[0].task_id.startswith("memoryarena_bundled_shopping_"), tasks[0].task_id
        run_target_plan_smoke(output)
        report_lines = report.read_text(encoding="utf-8").splitlines()
        assert len(report_lines) == 9, len(report_lines)
    print("AGENTMEMORY_MEMORYARENA_CONVERTER_SMOKE_OK")


def run_target_plan_smoke(data_path: Path) -> None:
    tasks = load_task_dataset(data_path=data_path)
    for data_idx, task in enumerate(tasks):
        env = AgentMemoryEnv(data_path=data_path)
        _, info = env.reset(data_idx=data_idx)
        for subtask in task.subtasks:
            _, _, done, _, info = env.step(f'BUY {{"product_id": "{subtask.target_product_id}"}}')
        assert done, info
        assert info["episode_success"], info

        first_subtask = task.subtasks[0]
        wrong_products = [
            product.product_id
            for product in first_subtask.candidate_products
            if product.product_id != first_subtask.target_product_id
        ]
        if wrong_products:
            env = AgentMemoryEnv(data_path=data_path)
            env.reset(data_idx=data_idx)
            _, reward, done, _, info = env.step(f'BUY {{"product_id": "{wrong_products[0]}"}}')
            assert reward < 0 and not done and info["compatibility_violations"], info


if __name__ == "__main__":
    main()
