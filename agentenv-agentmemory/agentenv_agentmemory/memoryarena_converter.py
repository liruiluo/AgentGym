from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SPLITS = ("train", "dev", "test")
OPTION_PATTERN = re.compile(r"^\s*-\s+(?P<title>.+?)\s*$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ConversionStats:
    task_count: int
    split_counts: dict[str, int]
    min_match_score: int
    ambiguous_matches: int
    report_path: Path | None = None

    def marker(self) -> str:
        split_text = ",".join(f"{split}:{self.split_counts.get(split, 0)}" for split in SPLITS)
        report_text = f" report={self.report_path}" if self.report_path else ""
        return (
            "AGENTMEMORY_MEMORYARENA_CONVERT_OK "
            f"tasks={self.task_count} splits={split_text} min_match_score={self.min_match_score} "
            f"ambiguous_matches={self.ambiguous_matches}{report_text}"
        )


def read_jsonl(source: str | Path) -> list[dict[str, Any]]:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        with urllib.request.urlopen(source_text, timeout=60) as response:
            lines = response.read().decode("utf-8").splitlines()
    else:
        lines = Path(source).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def convert_file(
    source: str | Path,
    output_path: str | Path,
    *,
    split_dir: str | Path,
    report_path: str | Path | None = None,
    limit: int | None = None,
    split_mode: str = "ratio",
    min_match_score: int = 1,
    ambiguous_policy: str = "first",
) -> ConversionStats:
    records = read_jsonl(source)
    if limit is not None:
        records = records[:limit]
    tasks, report_rows = convert_records(
        records,
        split_mode=split_mode,
        min_match_score=min_match_score,
        ambiguous_policy=ambiguous_policy,
    )
    write_jsonl(output_path, tasks)
    write_split_files(split_dir, tasks)
    resolved_report_path = Path(report_path) if report_path is not None else None
    if resolved_report_path is not None:
        write_jsonl(resolved_report_path, report_rows)
    return ConversionStats(
        task_count=len(tasks),
        split_counts=Counter(task["split"] for task in tasks),
        min_match_score=min(row["match_score"] for row in report_rows) if report_rows else 0,
        ambiguous_matches=sum(1 for row in report_rows if row["ambiguous_match_ids"]),
        report_path=resolved_report_path,
    )


def convert_records(
    records: list[dict[str, Any]],
    *,
    split_mode: str = "ratio",
    min_match_score: int = 1,
    ambiguous_policy: str = "first",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if ambiguous_policy not in {"first", "fail"}:
        raise ValueError(f"Unsupported ambiguous_policy={ambiguous_policy!r}; expected 'first' or 'fail'.")
    tasks = []
    report_rows = []
    for position, record in enumerate(records):
        task, task_report_rows = convert_record(
            record,
            position=position,
            split_mode=split_mode,
            min_match_score=min_match_score,
            ambiguous_policy=ambiguous_policy,
        )
        tasks.append(task)
        report_rows.extend(task_report_rows)
    return tasks, report_rows


def convert_record(
    record: dict[str, Any],
    *,
    position: int,
    split_mode: str,
    min_match_score: int,
    ambiguous_policy: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for key in ["id", "questions", "answers", "category"]:
        if key not in record:
            raise ValueError(f"MemoryArena record at position {position} missing required key '{key}'.")
    questions = record["questions"]
    answers = record["answers"]
    if not isinstance(questions, list) or not isinstance(answers, list) or len(questions) != len(answers):
        raise ValueError(f"MemoryArena record {record['id']} must have aligned questions and answers lists.")
    if not questions:
        raise ValueError(f"MemoryArena record {record['id']} has no questions.")

    record_code = alpha_code(position)
    subtasks = []
    report_rows = []
    for step_index, (question, answer) in enumerate(zip(questions, answers), start=1):
        option_titles = extract_available_options(str(question))
        if not option_titles:
            raise ValueError(f"MemoryArena record {record['id']} step {step_index} has no available options.")
        candidates = [
            {
                "product_id": f"ma_{record_code}_{alpha_code(step_index - 1)}_{alpha_code(option_index)}",
                "title": title,
                "attributes": {"source_option": alpha_code(option_index)},
            }
            for option_index, title in enumerate(option_titles)
        ]
        target_product_id, match_score, candidate_scores, ambiguous_match_ids = infer_target_product(
            candidates,
            answer,
            ambiguous_policy=ambiguous_policy,
        )
        if match_score < min_match_score:
            raise ValueError(
                f"MemoryArena record {record['id']} step {step_index} target match score {match_score} "
                f"is below min_match_score={min_match_score}."
            )
        subtasks.append(
            {
                "instruction": extract_instruction(str(question)),
                "target_product_id": target_product_id,
                "candidate_products": candidates,
            }
        )
        report_rows.append(
            {
                "source_id": record["id"],
                "category": record["category"],
                "step_index": step_index,
                "target_asin": answer.get("target_asin") if isinstance(answer, dict) else None,
                "answer_attributes": answer.get("attributes", []) if isinstance(answer, dict) else [],
                "matched_product_id": target_product_id,
                "match_score": match_score,
                "ambiguous_match_ids": ambiguous_match_ids,
                "candidate_scores": candidate_scores,
            }
        )

    task = {
        "task_id": f"memoryarena_bundled_shopping_{record_code}",
        "title": f"MemoryArena bundled shopping: {record['category']} / {record['id']}",
        "split": assign_split(position, split_mode),
        "source": "memoryarena_bundled_shopping_v0",
        "difficulty": f"memoryarena_dependency_distance_{max(len(subtasks) - 1, 0)}",
        "memory_dependency": "cross_session_bundled_shopping_attributes",
        "subtasks": subtasks,
    }
    return task, report_rows


def extract_available_options(question: str) -> list[str]:
    if "**Available Options:**" not in question:
        return []
    tail = question.split("**Available Options:**", maxsplit=1)[1]
    titles = []
    for line in tail.splitlines():
        match = OPTION_PATTERN.match(line)
        if match:
            titles.append(match.group("title").strip())
        elif titles and line.strip():
            break
    return titles


def extract_instruction(question: str) -> str:
    section = question.split("----------------------------------------------------------------")[-1].strip()
    if "**Available Options:**" in section:
        section = section.split("**Available Options:**", maxsplit=1)[0].strip()
    return section + "\nUse memory tools to preserve any product attributes needed by later products."


def infer_target_product(
    candidates: list[dict[str, Any]],
    answer: Any,
    *,
    ambiguous_policy: str = "first",
) -> tuple[str, int, list[dict[str, Any]], list[str]]:
    if not isinstance(answer, dict):
        raise ValueError(f"MemoryArena answer must be an object, got {type(answer).__name__}.")
    answer_attributes = [str(item) for item in answer.get("attributes", [])]
    if not answer_attributes:
        raise ValueError("MemoryArena answer has no attributes for target matching.")
    scored = []
    for candidate in candidates:
        title = str(candidate["title"])
        score = target_match_score(title, answer_attributes)
        scored.append({"product_id": candidate["product_id"], "score": score, "title": title})
    best = max(scored, key=lambda item: item["score"])
    tied = [item for item in scored if item["score"] == best["score"]]
    if len(tied) > 1:
        tied_ids = [item["product_id"] for item in tied]
        if ambiguous_policy == "fail":
            raise ValueError(f"Ambiguous MemoryArena target match among {tied_ids} at score {best['score']}.")
        if ambiguous_policy != "first":
            raise ValueError(f"Unsupported ambiguous_policy={ambiguous_policy!r}; expected 'first' or 'fail'.")
        return str(tied[0]["product_id"]), int(best["score"]), scored, tied_ids
    return str(best["product_id"]), int(best["score"]), scored, []


def target_match_score(title: str, answer_attributes: list[str]) -> int:
    normalized_title = normalize_text(title)
    title_tokens = set(tokenize(title))
    score = 0
    for attribute in answer_attributes:
        normalized_attribute = normalize_text(attribute)
        if normalized_attribute and normalized_attribute in normalized_title:
            score += 5
        score += len(set(tokenize(attribute)) & title_tokens)
    return score


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def assign_split(position: int, split_mode: str) -> str:
    if split_mode == "cycle":
        return SPLITS[position % len(SPLITS)]
    if split_mode == "ratio":
        bucket = position % 10
        if bucket < 8:
            return "train"
        if bucket == 8:
            return "dev"
        return "test"
    raise ValueError(f"Unsupported split_mode={split_mode!r}; expected 'ratio' or 'cycle'.")


def write_split_files(split_dir: str | Path, tasks: list[dict[str, Any]]) -> None:
    root = Path(split_dir)
    root.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = {split: [] for split in SPLITS}
    for task in tasks:
        by_split[str(task["split"])].append(str(task["task_id"]))
    for split, task_ids in by_split.items():
        (root / f"{split}.txt").write_text("\n".join(task_ids) + ("\n" if task_ids else ""), encoding="utf-8")


def alpha_code(index: int) -> str:
    if index < 0:
        raise ValueError(f"alpha_code index must be non-negative, got {index}.")
    letters = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(ord("a") + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(letters))
