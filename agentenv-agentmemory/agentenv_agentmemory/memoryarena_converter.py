from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

SPLITS = ("train", "dev", "test")
OPTION_PATTERN = re.compile(r"^\s*-\s+(?P<title>.+?)\s*$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
DEFAULT_SKIPPED_CATALOG_FILENAMES = {"items_shuffle.json"}


@dataclass(frozen=True)
class CatalogProduct:
    asin: str
    title: str
    source_path: str


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
    catalog_paths: Iterable[str | Path] | None = None,
) -> ConversionStats:
    records = read_jsonl(source)
    if limit is not None:
        records = records[:limit]
    catalog_index = build_catalog_index(catalog_paths or [], target_asins=collect_target_asins(records))
    tasks, report_rows = convert_records(
        records,
        split_mode=split_mode,
        min_match_score=min_match_score,
        ambiguous_policy=ambiguous_policy,
        catalog_index=catalog_index,
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
    catalog_index: dict[str, CatalogProduct] | None = None,
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
            catalog_index=catalog_index or {},
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
    catalog_index: dict[str, CatalogProduct],
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
        resolution = infer_target_product(
            candidates,
            answer,
            ambiguous_policy=ambiguous_policy,
            catalog_index=catalog_index,
        )
        target_product_id = resolution["product_id"]
        match_score = resolution["match_score"]
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
                "resolver": resolution["resolver"],
                "target_asin_found": resolution["target_asin_found"],
                "catalog_title": resolution["catalog_title"],
                "catalog_source_path": resolution["catalog_source_path"],
                "ambiguous_match_ids": resolution["ambiguous_match_ids"],
                "candidate_scores": resolution["candidate_scores"],
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
    catalog_index: dict[str, CatalogProduct] | None = None,
) -> dict[str, Any]:
    if not isinstance(answer, dict):
        raise ValueError(f"MemoryArena answer must be an object, got {type(answer).__name__}.")
    answer_attributes = [str(item) for item in answer.get("attributes", [])]
    if not answer_attributes:
        raise ValueError("MemoryArena answer has no attributes for target matching.")
    catalog_product = None
    target_asin = normalize_asin(answer.get("target_asin"))
    if target_asin and catalog_index:
        catalog_product = catalog_index.get(target_asin)
    if catalog_product is not None:
        catalog_resolution = infer_target_product_from_catalog(
            candidates,
            catalog_product,
            answer_attributes=answer_attributes,
            ambiguous_policy=ambiguous_policy,
        )
        if catalog_resolution is not None:
            return catalog_resolution
    return infer_target_product_from_attributes(
        candidates,
        answer_attributes=answer_attributes,
        ambiguous_policy=ambiguous_policy,
        target_asin_found=bool(catalog_product),
    )


def infer_target_product_from_catalog(
    candidates: list[dict[str, Any]],
    catalog_product: CatalogProduct,
    *,
    answer_attributes: list[str],
    ambiguous_policy: str,
) -> dict[str, Any] | None:
    scored = []
    for candidate in candidates:
        title = str(candidate["title"])
        catalog_score = catalog_title_match_score(title, catalog_product.title)
        attribute_score = target_match_score(title, answer_attributes)
        scored.append(
            {
                "product_id": candidate["product_id"],
                "score": catalog_score,
                "rank_score": catalog_score * 1000 + attribute_score,
                "catalog_score": catalog_score,
                "attribute_score": attribute_score,
                "title": title,
            }
        )
    best = max(scored, key=lambda item: item["rank_score"])
    if best["catalog_score"] <= 0:
        return None
    tied = [item for item in scored if item["rank_score"] == best["rank_score"]]
    ambiguous_match_ids: list[str] = []
    if len(tied) > 1:
        ambiguous_match_ids = [item["product_id"] for item in tied]
        if ambiguous_policy == "fail":
            raise ValueError(
                f"Ambiguous MemoryArena catalog target match among {ambiguous_match_ids} "
                f"at score {best['catalog_score']}."
            )
        if ambiguous_policy != "first":
            raise ValueError(f"Unsupported ambiguous_policy={ambiguous_policy!r}; expected 'first' or 'fail'.")
        best = tied[0]
    return {
        "product_id": str(best["product_id"]),
        "match_score": int(best["catalog_score"]),
        "candidate_scores": scored,
        "ambiguous_match_ids": ambiguous_match_ids,
        "resolver": "asin_catalog",
        "target_asin_found": True,
        "catalog_title": catalog_product.title,
        "catalog_source_path": catalog_product.source_path,
    }


def infer_target_product_from_attributes(
    candidates: list[dict[str, Any]],
    *,
    answer_attributes: list[str],
    ambiguous_policy: str = "first",
    target_asin_found: bool = False,
) -> dict[str, Any]:
    scored = []
    for candidate in candidates:
        title = str(candidate["title"])
        score = target_match_score(title, answer_attributes)
        scored.append(
            {
                "product_id": candidate["product_id"],
                "score": score,
                "rank_score": score,
                "catalog_score": None,
                "attribute_score": score,
                "title": title,
            }
        )
    best = max(scored, key=lambda item: item["score"])
    tied = [item for item in scored if item["score"] == best["score"]]
    ambiguous_match_ids: list[str] = []
    if len(tied) > 1:
        ambiguous_match_ids = [item["product_id"] for item in tied]
        if ambiguous_policy == "fail":
            raise ValueError(f"Ambiguous MemoryArena target match among {ambiguous_match_ids} at score {best['score']}.")
        if ambiguous_policy != "first":
            raise ValueError(f"Unsupported ambiguous_policy={ambiguous_policy!r}; expected 'first' or 'fail'.")
        best = tied[0]
    return {
        "product_id": str(best["product_id"]),
        "match_score": int(best["score"]),
        "candidate_scores": scored,
        "ambiguous_match_ids": ambiguous_match_ids,
        "resolver": "attribute_heuristic",
        "target_asin_found": target_asin_found,
        "catalog_title": None,
        "catalog_source_path": None,
    }


def catalog_title_match_score(option_title: str, catalog_title: str) -> int:
    normalized_option = normalize_text(option_title)
    normalized_catalog = normalize_text(catalog_title)
    if not normalized_option or not normalized_catalog:
        return 0
    option_tokens = set(tokenize(option_title))
    catalog_tokens = set(tokenize(catalog_title))
    token_overlap = len(option_tokens & catalog_tokens)
    ratio = SequenceMatcher(None, normalized_option, normalized_catalog).ratio()
    containment_bonus = 0
    if normalized_option in normalized_catalog or normalized_catalog in normalized_option:
        containment_bonus = 50
    return int(round(ratio * 100)) + token_overlap * 3 + containment_bonus


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


def normalize_asin(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def collect_target_asins(records: Iterable[dict[str, Any]]) -> set[str]:
    target_asins: set[str] = set()
    for record in records:
        for answer in record.get("answers", []):
            if isinstance(answer, dict):
                asin = normalize_asin(answer.get("target_asin"))
                if asin:
                    target_asins.add(asin)
    return target_asins


def build_catalog_index(
    catalog_paths: Iterable[str | Path],
    *,
    target_asins: set[str] | None = None,
) -> dict[str, CatalogProduct]:
    paths = list(expand_catalog_paths(catalog_paths))
    if not paths:
        return {}
    remaining = set(target_asins or [])
    index: dict[str, CatalogProduct] = {}
    for catalog_path in paths:
        for product in iter_catalog_products(catalog_path):
            asin = normalize_asin(extract_product_asin(product))
            if not asin:
                continue
            if remaining and asin not in remaining:
                continue
            title = extract_product_title(product)
            if not title:
                continue
            index.setdefault(asin, CatalogProduct(asin=asin, title=title, source_path=str(catalog_path)))
            remaining.discard(asin)
        if target_asins and not remaining:
            break
    return index


def expand_catalog_paths(catalog_paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in catalog_paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
            continue
        if not path.exists():
            raise FileNotFoundError(f"Catalog path does not exist: {path}")
        if not path.is_dir():
            continue
        if (path / "product_catalog").is_dir():
            yield from sorted((path / "product_catalog").glob("*.json"))
            continue
        for candidate in sorted(path.rglob("*.json")):
            if "search_engine" in candidate.parts:
                continue
            if candidate.name in DEFAULT_SKIPPED_CATALOG_FILENAMES:
                continue
            yield candidate


def iter_catalog_products(catalog_path: Path) -> Iterable[dict[str, Any]]:
    first_char = first_non_whitespace_char(catalog_path)
    if first_char == "[":
        for item in iter_json_array_items(catalog_path):
            yield from iter_product_dicts(item)
        return
    if first_char == "{":
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        yield from iter_product_dicts(payload)
        return
    with catalog_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def first_non_whitespace_char(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        while True:
            char = handle.read(1)
            if not char:
                return ""
            if not char.isspace():
                return char


def iter_json_array_items(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        started = False
        eof = False
        while True:
            if not eof and len(buffer) < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            buffer = buffer.lstrip()
            if not started:
                if not buffer and eof:
                    return
                if not buffer:
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"Expected JSON array in {path}.")
                buffer = buffer[1:]
                started = True
                continue
            buffer = buffer.lstrip()
            if not buffer:
                if eof:
                    return
                continue
            if buffer[0] == "]":
                return
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            try:
                item, index = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                    continue
                eof = True
                continue
            yield item
            buffer = buffer[index:]


def iter_product_dicts(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from iter_product_dicts(item)
        return
    if not isinstance(payload, dict):
        return
    if extract_product_asin(payload) and extract_product_title(payload):
        yield payload
        return
    for key in ("products", "items", "data"):
        value = payload.get(key)
        if isinstance(value, (list, dict)):
            yield from iter_product_dicts(value)


def extract_product_asin(product: dict[str, Any]) -> str | None:
    for key in ("asin", "ASIN", "product_asin"):
        if product.get(key):
            return str(product[key])
    product_information = product.get("product_information")
    if isinstance(product_information, dict):
        for key, value in product_information.items():
            if "asin" in normalize_text(str(key)).split():
                return str(value).replace("\u200e", "").strip()
    return None


def extract_product_title(product: dict[str, Any]) -> str | None:
    for key in ("name", "title", "product_title"):
        value = product.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
