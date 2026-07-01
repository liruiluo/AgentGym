from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MARKER = "AGENTMEMORY_SCRIPTED_SEARCH_FAILURE_AUDIT_OK"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_decisions_by_step(decisions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for decision in decisions:
        grouped[int(decision["subtask_index"])].append(decision)
    return dict(grouped)


def classify_failed_step(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    last = decisions[-1]
    target = last.get("target_product_id")
    attempts = [item.get("chosen_product_id") for item in decisions]
    compatible = last.get("compatible_product_ids") or []
    fallback = last.get("fallback_product_ids") or []
    ranked = last.get("ranked_product_ids") or []
    target_rank = last.get("target_rank")
    target_in_compatible = last.get("target_in_compatible_pool")
    if target is None:
        failure_type = "no_target_audit"
    elif target in attempts:
        failure_type = "target_attempted_but_rejected"
    elif compatible and target not in compatible and not fallback:
        failure_type = "compatibility_filter_excluded_target"
    elif ranked and target not in ranked:
        failure_type = "target_absent_from_ranked_pool"
    elif target_rank is not None and len(attempts) < target_rank:
        failure_type = "attempt_budget_below_target_rank"
    elif fallback and target in fallback:
        failure_type = "fallback_target_not_reached"
    else:
        failure_type = "unclassified"
    return {
        "failure_type": failure_type,
        "subtask_index": last.get("subtask_index"),
        "target_product_id": target,
        "attempted_product_ids": attempts,
        "compatible_product_ids": compatible,
        "fallback_product_ids": fallback,
        "ranked_product_ids": ranked,
        "target_in_compatible_pool": target_in_compatible,
        "target_rank": target_rank,
        "allowed_values": last.get("allowed_values") or [],
        "active_compatibility_keys": last.get("active_compatibility_keys") or [],
        "preference": last.get("preference"),
        "compatibility_fallback": last.get("compatibility_fallback"),
        "chosen_titles": [item.get("chosen_title") for item in decisions],
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "summary.json")
    episodes = load_jsonl(run_dir / "episodes.jsonl")
    failed_steps: list[dict[str, Any]] = []
    for episode in episodes:
        if episode.get("episode_success"):
            continue
        grouped = group_decisions_by_step(episode.get("decisions", []))
        for step_index, decisions in sorted(grouped.items()):
            if not decisions or decisions[-1].get("buy_accepted"):
                continue
            item = classify_failed_step(decisions)
            item.update(
                {
                    "task_id": episode.get("task_id"),
                    "episode_progress_score": episode.get("progress_score"),
                    "episode_rejected_buys": episode.get("rejected_buys"),
                    "failed_step_index": step_index,
                }
            )
            failed_steps.append(item)
    counts = Counter(item["failure_type"] for item in failed_steps)
    return {
        "run_dir": str(run_dir),
        "source_marker": summary.get("marker"),
        "episodes": summary.get("episodes"),
        "successes": summary.get("successes"),
        "success_rate": summary.get("success_rate"),
        "mean_progress_score": summary.get("mean_progress_score"),
        "max_buy_attempts": summary.get("max_buy_attempts"),
        "compatibility_fallback": summary.get("compatibility_fallback"),
        "failure_type_counts": dict(sorted(counts.items())),
        "failed_steps": failed_steps,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Scripted SEARCH failure audit",
        "",
        f"Marker: `{MARKER}`",
        "",
        "## Runs",
        "",
    ]
    for run in report["runs"]:
        lines.extend(
            [
                f"### {Path(run['run_dir']).name}",
                "",
                f"- success: `{run['successes']}/{run['episodes']}`",
                f"- success_rate: `{run['success_rate']}`",
                f"- mean_progress_score: `{run['mean_progress_score']}`",
                f"- max_buy_attempts: `{run.get('max_buy_attempts')}`",
                f"- compatibility_fallback: `{run.get('compatibility_fallback')}`",
                f"- failure_type_counts: `{json.dumps(run['failure_type_counts'], ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
        if not run["failed_steps"]:
            lines.append("No failed steps.\n")
            continue
        lines.extend(["| task | step | type | target | attempts | target_rank | allowed |", "|---|---:|---|---|---|---:|---|"])
        for item in run["failed_steps"]:
            lines.append(
                "| {task_id} | {failed_step_index} | {failure_type} | {target_product_id} | {attempts} | {target_rank} | {allowed} |".format(
                    task_id=item.get("task_id"),
                    failed_step_index=item.get("failed_step_index"),
                    failure_type=item.get("failure_type"),
                    target_product_id=item.get("target_product_id"),
                    attempts=", ".join(str(x) for x in item.get("attempted_product_ids") or []),
                    target_rank=item.get("target_rank"),
                    allowed=", ".join(str(x) for x in item.get("allowed_values") or []),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze scripted SEARCH baseline failure modes.")
    parser.add_argument("--run-dir", action="append", required=True, help="Baseline evidence directory containing summary.json and episodes.jsonl. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [analyze_run(Path(item)) for item in args.run_dir]
    aggregate = Counter()
    for run in runs:
        aggregate.update(run["failure_type_counts"])
    report = {
        "marker": MARKER,
        "runs": runs,
        "aggregate_failure_type_counts": dict(sorted(aggregate.items())),
    }
    (output_dir / "failure_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(output_dir / "failure_audit.md", report)
    print(MARKER, f"runs={len(runs)}", f"failure_types={dict(sorted(aggregate.items()))}")
    print("SUMMARY_PATH", output_dir / "failure_audit.json")


if __name__ == "__main__":
    main()
