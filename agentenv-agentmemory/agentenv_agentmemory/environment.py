from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .catalog_search import search_sqlite_catalog


@dataclass(frozen=True)
class Product:
    product_id: str
    title: str
    attributes: dict[str, Any]

    def render(self) -> str:
        attrs = ", ".join(f"{key}={render_attr_value(value)}" for key, value in self.attributes.items())
        return f"- {self.product_id}: {self.title} ({attrs})"


@dataclass(frozen=True)
class ShoppingSubtask:
    instruction: str
    candidate_products: tuple[Product, ...]
    target_product_id: str


@dataclass(frozen=True)
class ShoppingTask:
    task_id: str
    title: str
    subtasks: tuple[ShoppingSubtask, ...]
    split: str = "train"
    source: str = "handcrafted_v0"
    difficulty: str = "smoke"
    memory_dependency: str = "cross_session_product_attribute"


@dataclass
class MemoryEntry:
    memory_id: str
    key: str
    value: str
    created_step: int
    updated_step: int
    access_count: int = 0

    def render(self) -> str:
        return f"[{self.memory_id}] {self.key}: {self.value}"


@dataclass
class StepInfo:
    task_id: str = ""
    task_family: str = "bundled_shopping"
    split: str = "train"
    source: str = "handcrafted_v0"
    difficulty: str = "smoke"
    memory_dependency: str = "cross_session_product_attribute"
    progress_score: float = 0.0
    episode_success: bool = False
    memory_ops: list[dict[str, Any]] = field(default_factory=list)
    memory_state_diff: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    compatibility_violations: list[dict[str, Any]] = field(default_factory=list)
    purchase_history: list[dict[str, Any]] = field(default_factory=list)
    current_subtask_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family,
            "split": self.split,
            "source": self.source,
            "difficulty": self.difficulty,
            "memory_dependency": self.memory_dependency,
            "progress_score": self.progress_score,
            "episode_success": self.episode_success,
            "memory_ops": self.memory_ops,
            "memory_state_diff": self.memory_state_diff,
            "compatibility_violations": self.compatibility_violations,
            "purchase_history": self.purchase_history,
            "current_subtask_index": self.current_subtask_index,
        }


class InvalidAction(ValueError):
    pass


class AgentMemoryEnv:
    """Minimal memory-dependent bundled shopping environment.

    The environment keeps long-term memory hidden unless the policy explicitly
    calls RETRIEVE. Current v0 tasks are small handcrafted smoke items; they are
    placeholders for converted MemoryArena/WebShop-style data.
    """

    action_pattern = re.compile(r"^\s*([A-Za-z_]+)\s*(.*)\s*$", re.DOTALL)

    def __init__(
        self,
        tasks: list[ShoppingTask] | None = None,
        *,
        data_path: str | Path | None = None,
        split: str | None = None,
        split_dir: str | Path | None = None,
        catalog_index_path: str | Path | None = None,
    ) -> None:
        self.tasks = tasks or load_task_dataset(data_path=data_path, split=split, split_dir=split_dir)
        resolved_catalog_index = catalog_index_path or os.environ.get("AGENTMEMORY_CATALOG_INDEX_PATH")
        self.catalog_index_path = Path(resolved_catalog_index) if resolved_catalog_index else None
        self.task: ShoppingTask | None = None
        self.data_idx = 0
        self.step_count = 0
        self.current_subtask_index = 0
        self.long_term_memory: dict[str, MemoryEntry] = {}
        self.short_term_context: list[str] = []
        self.purchase_history: list[dict[str, Any]] = []
        self.bundle_state: dict[str, Any] = {}
        self.memory_id_counter = 0
        self.done = False
        self.last_info = StepInfo()

    def reset(self, seed: int | None = None, data_idx: int = 0):
        del seed  # Deterministic smoke tasks; data_idx selects the item.
        self.data_idx = data_idx
        self.task = self.tasks[data_idx % len(self.tasks)]
        self.step_count = 0
        self.current_subtask_index = 0
        self.long_term_memory = {}
        self.short_term_context = []
        self.purchase_history = []
        self.bundle_state = {}
        self.memory_id_counter = 0
        self.done = False
        self.last_info = self.build_info(memory_state_diff=empty_memory_diff())
        return self.render_observation(), self.last_info.as_dict()

    def step(self, action: str):
        if self.done:
            return self.render_observation("Episode is already done."), 0.0, True, False, self.last_info.as_dict()

        self.step_count += 1
        try:
            op, payload = self.parse_action(action)
            observation, reward, done, memory_diff, violations, memory_op = self.dispatch_action(op, payload)
        except InvalidAction as exc:
            observation = self.render_observation(f"Invalid action: {exc}")
            reward = -0.1
            done = False
            memory_diff = empty_memory_diff()
            violations = []
            memory_op = None

        self.done = done
        memory_ops = [memory_op] if memory_op is not None else []
        self.last_info = self.build_info(
            memory_state_diff=memory_diff,
            compatibility_violations=violations,
            memory_ops=memory_ops,
        )
        return observation, reward, done, False, self.last_info.as_dict()

    def dispatch_action(self, op: str, payload: dict[str, Any]):
        if op == "ADD":
            return self.action_add(payload)
        if op == "UPDATE":
            return self.action_update(payload)
        if op == "DELETE":
            return self.action_delete(payload)
        if op == "RETRIEVE":
            return self.action_retrieve(payload)
        if op == "SUMMARY":
            return self.action_summary(payload)
        if op == "FILTER":
            return self.action_filter(payload)
        if op == "SEARCH":
            return self.action_search(payload)
        if op == "BUY":
            return self.action_buy(payload)
        if op == "ANSWER":
            return self.action_answer(payload)
        raise InvalidAction(f"Unsupported action '{op}'.")

    def action_add(self, payload: dict[str, Any]):
        key = require_str(payload, "key")
        value = require_str(payload, "value")
        memory_id = f"mem_{self.memory_id_counter:04d}"
        self.memory_id_counter += 1
        entry = MemoryEntry(
            memory_id=memory_id,
            key=key,
            value=value,
            created_step=self.step_count,
            updated_step=self.step_count,
        )
        self.long_term_memory[memory_id] = entry
        diff = empty_memory_diff()
        diff["added"].append(memory_entry_dict(entry))
        memory_op = {"op": "ADD", "memory_id": memory_id, "key": key, "step": self.step_count}
        return self.render_observation(f"Stored memory {entry.render()}"), -0.01, False, diff, [], memory_op

    def action_update(self, payload: dict[str, Any]):
        entry = self.find_memory_entry(payload)
        old_entry = memory_entry_dict(entry)
        if "key" in payload:
            entry.key = require_str(payload, "key")
        entry.value = require_str(payload, "value")
        entry.updated_step = self.step_count
        diff = empty_memory_diff()
        diff["updated"].append({"before": old_entry, "after": memory_entry_dict(entry)})
        memory_op = {"op": "UPDATE", "memory_id": entry.memory_id, "key": entry.key, "step": self.step_count}
        return self.render_observation(f"Updated memory {entry.render()}"), -0.01, False, diff, [], memory_op

    def action_delete(self, payload: dict[str, Any]):
        entry = self.find_memory_entry(payload)
        del self.long_term_memory[entry.memory_id]
        diff = empty_memory_diff()
        diff["deleted"].append(memory_entry_dict(entry))
        memory_op = {"op": "DELETE", "memory_id": entry.memory_id, "key": entry.key, "step": self.step_count}
        return self.render_observation(f"Deleted memory {entry.memory_id}."), -0.01, False, diff, [], memory_op

    def action_retrieve(self, payload: dict[str, Any]):
        query = require_str(payload, "query")
        top_k = int(payload.get("top_k", 3))
        ranked = sorted(
            self.long_term_memory.values(),
            key=lambda entry: retrieval_score(query, f"{entry.key} {entry.value}"),
            reverse=True,
        )
        retrieved = [entry for entry in ranked[:top_k] if retrieval_score(query, f"{entry.key} {entry.value}") > 0]
        for entry in retrieved:
            entry.access_count += 1
        if retrieved:
            self.short_term_context = [entry.render() for entry in retrieved]
            message = "Retrieved memories:\n" + "\n".join(self.short_term_context)
        else:
            self.short_term_context = []
            message = "No relevant memory retrieved."
        memory_op = {"op": "RETRIEVE", "query": query, "top_k": top_k, "step": self.step_count}
        return self.render_observation(message), -0.01, False, empty_memory_diff(), [], memory_op

    def action_summary(self, payload: dict[str, Any]):
        text = require_str(payload, "text")
        summary = text.strip()
        if not summary:
            raise InvalidAction("SUMMARY text must be non-empty.")
        self.short_term_context = [f"Summary: {summary[:500]}"]
        memory_op = {"op": "SUMMARY", "step": self.step_count}
        return self.render_observation("Short-term context replaced by summary."), -0.01, False, empty_memory_diff(), [], memory_op

    def action_filter(self, payload: dict[str, Any]):
        query = require_str(payload, "query")
        query_tokens = tokenize(query)
        self.short_term_context = [
            item for item in self.short_term_context if tokenize(item) & query_tokens
        ]
        memory_op = {"op": "FILTER", "query": query, "step": self.step_count}
        return self.render_observation("Filtered active short-term context."), -0.01, False, empty_memory_diff(), [], memory_op

    def action_search(self, payload: dict[str, Any]):
        if self.catalog_index_path is None:
            raise InvalidAction("SEARCH requires AGENTMEMORY_CATALOG_INDEX_PATH or catalog_index_path.")
        query = require_str(payload, "query")
        top_k = max(1, min(int(payload.get("top_k", 3)), 5))
        results = search_sqlite_catalog(self.catalog_index_path, query, top_k=top_k)
        if results:
            message = "Product search results:\n" + "\n".join(result.render() for result in results)
        else:
            message = "Product search returned no results."
        memory_op = {"op": "SEARCH", "query": query, "top_k": top_k, "step": self.step_count}
        return self.render_observation(message), -0.01, False, empty_memory_diff(), [], memory_op

    def action_buy(self, payload: dict[str, Any]):
        product_id = require_str(payload, "product_id")
        subtask = self.current_subtask()
        products = {product.product_id: product for product in subtask.candidate_products}
        if product_id not in products:
            raise InvalidAction(f"Unknown product_id '{product_id}' for current subtask.")

        product = products[product_id]
        compatible, reason = self.is_compatible_purchase(product)
        if not compatible:
            violation = {"product_id": product_id, "reason": reason, "step": self.step_count}
            return self.render_observation(f"Purchase rejected: {reason}"), -0.5, False, empty_memory_diff(), [violation], None

        self.record_purchase(product)
        self.current_subtask_index += 1
        done = self.current_subtask_index >= len(self.require_task().subtasks)
        progress_reward = 1.0
        final_bonus = 1.0 if done else 0.0
        self.short_term_context = []
        if done:
            message = "All bundled shopping subtasks are complete and compatible."
        else:
            message = "Purchase accepted. A new shopping session starts; use memory tools if prior attributes are needed."
        return self.render_observation(message), progress_reward + final_bonus, done, empty_memory_diff(), [], None

    def action_answer(self, payload: dict[str, Any]):
        text = require_str(payload, "text")
        required_ids = {subtask.target_product_id for subtask in self.require_task().subtasks}
        purchased_ids = {item["product_id"] for item in self.purchase_history}
        if required_ids <= purchased_ids and all(product_id in text for product_id in required_ids):
            self.done = True
            return self.render_observation("Answer accepted."), 1.0, True, empty_memory_diff(), [], None
        return self.render_observation("Answer recorded, but the bundle is not complete yet."), 0.0, False, empty_memory_diff(), [], None

    def parse_action(self, action: str) -> tuple[str, dict[str, Any]]:
        if not isinstance(action, str):
            raise InvalidAction(f"Action must be a string, got {type(action).__name__}.")
        match = self.action_pattern.match(action.strip())
        if not match:
            raise InvalidAction("Expected '<TOOL> {json payload}'.")
        op = match.group(1).upper()
        payload_text = match.group(2).strip()
        if not payload_text:
            return op, {}
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise InvalidAction(f"Payload must be valid JSON: {exc.msg}.") from exc
        if not isinstance(payload, dict):
            raise InvalidAction("Payload must be a JSON object.")
        return op, payload

    def find_memory_entry(self, payload: dict[str, Any]) -> MemoryEntry:
        memory_id = payload.get("memory_id")
        key = payload.get("key")
        if memory_id is not None:
            if memory_id not in self.long_term_memory:
                raise InvalidAction(f"Unknown memory_id '{memory_id}'.")
            return self.long_term_memory[str(memory_id)]
        if key is not None:
            matches = [entry for entry in self.long_term_memory.values() if entry.key == key]
            if not matches:
                raise InvalidAction(f"Unknown memory key '{key}'.")
            if len(matches) > 1:
                raise InvalidAction(f"Memory key '{key}' is ambiguous; use memory_id.")
            return matches[0]
        raise InvalidAction("Expected memory_id or key.")

    def is_compatible_purchase(self, product: Product) -> tuple[bool, str]:
        subtask = self.current_subtask()
        if product.product_id == subtask.target_product_id and not has_compatibility_constraints(product):
            return True, "target product without external compatibility constraints"
        attrs = product.attributes
        if "compatible_tv_min" in attrs:
            tv_size = self.bundle_state.get("tv_size_in")
            if tv_size is None:
                return False, "no previously purchased TV size exists in hidden bundle state"
            min_size = attrs["compatible_tv_min"]
            max_size = attrs["compatible_tv_max"]
            if not min_size <= tv_size <= max_size:
                return False, f"TV size {tv_size} is outside supported range {min_size}-{max_size}"
        if "compatible_monitor_min" in attrs:
            monitor_size = self.bundle_state.get("monitor_size_in")
            if monitor_size is None:
                return False, "no previously purchased monitor size exists in hidden bundle state"
            min_size = attrs["compatible_monitor_min"]
            max_size = attrs["compatible_monitor_max"]
            if not min_size <= monitor_size <= max_size:
                return False, f"monitor size {monitor_size} is outside supported range {min_size}-{max_size}"
        if "max_weight_kg" in attrs:
            mounted_weight = self.bundle_state.get("tv_weight_kg", self.bundle_state.get("monitor_weight_kg"))
            if mounted_weight is None:
                return False, "no previously purchased mounted-device weight exists in hidden bundle state"
            max_weight = attrs["max_weight_kg"]
            if mounted_weight > max_weight:
                return False, f"mounted-device weight {mounted_weight}kg exceeds supported weight {max_weight}kg"
        if "supported_vesa" in attrs:
            tv_vesa = self.bundle_state.get("vesa")
            if tv_vesa is None:
                return False, "no previously purchased TV VESA mount pattern exists in hidden bundle state"
            if tv_vesa not in attrs["supported_vesa"]:
                return False, f"TV VESA {tv_vesa} is not supported by {render_attr_value(attrs['supported_vesa'])}"
        if "compatible_laptop_min" in attrs:
            laptop_size = self.bundle_state.get("laptop_size_in")
            if laptop_size is None:
                return False, "no previously purchased laptop size exists in hidden bundle state"
            min_size = attrs["compatible_laptop_min"]
            max_size = attrs["compatible_laptop_max"]
            if not min_size <= laptop_size <= max_size:
                return False, f"laptop size {laptop_size} is outside supported range {min_size}-{max_size}"
        if "required_port" in attrs:
            available_ports = self.bundle_state.get("laptop_ports", self.bundle_state.get("monitor_ports"))
            if available_ports is None:
                return False, "no previously purchased device port type exists in hidden bundle state"
            if attrs["required_port"] not in str(available_ports).split(","):
                return False, f"required port {attrs['required_port']} does not match device ports {available_ports}"
        return product.product_id == subtask.target_product_id or has_compatibility_constraints(product), "compatible"

    def record_purchase(self, product: Product) -> None:
        purchase = {
            "step": self.step_count,
            "subtask_index": self.current_subtask_index,
            "product_id": product.product_id,
            "title": product.title,
            "attributes": dict(product.attributes),
        }
        self.purchase_history.append(purchase)
        for key, value in product.attributes.items():
            if key in {
                "tv_size_in",
                "tv_weight_kg",
                "vesa",
                "laptop_size_in",
                "laptop_ports",
                "monitor_size_in",
                "monitor_weight_kg",
                "monitor_ports",
            }:
                self.bundle_state[key] = value

    def render_observation(self, prefix: str | None = None) -> str:
        task = self.require_task()
        lines = [
            f"Task family: bundled_shopping",
            f"Task: {task.title}",
            f"Progress: {self.current_subtask_index}/{len(task.subtasks)}",
        ]
        if prefix:
            lines.append(f"Last tool/result: {prefix}")
        if self.current_subtask_index < len(task.subtasks):
            subtask = self.current_subtask()
            lines.extend(
                [
                    "",
                    f"Current shopping session {self.current_subtask_index + 1}/{len(task.subtasks)}:",
                    subtask.instruction,
                    "",
                    "Visible candidate products:",
                ]
            )
            lines.extend(product.render() for product in subtask.candidate_products)
        if self.short_term_context:
            lines.extend(["", "Active short-term memory/context:"])
            lines.extend(f"- {item}" for item in self.short_term_context)
        else:
            lines.extend(["", "Active short-term memory/context: <empty>"])
        lines.extend(
            [
                "",
                "Available actions:",
                'ADD {"key": "...", "value": "..."}',
                'UPDATE {"memory_id": "mem_0000", "value": "..."}',
                'DELETE {"memory_id": "mem_0000"}',
                'RETRIEVE {"query": "...", "top_k": 3}',
                'SUMMARY {"text": "..."}',
                'FILTER {"query": "..."}',
                'SEARCH {"query": "...", "top_k": 3}',
                'BUY {"product_id": "..."}',
                'ANSWER {"text": "..."}',
                "Long-term memory is hidden until RETRIEVE brings entries into active context.",
            ]
        )
        return "\n".join(lines)

    def build_info(
        self,
        memory_state_diff: dict[str, list[dict[str, Any]]],
        compatibility_violations: list[dict[str, Any]] | None = None,
        memory_ops: list[dict[str, Any]] | None = None,
    ) -> StepInfo:
        task = self.require_task()
        return StepInfo(
            task_id=task.task_id,
            split=task.split,
            source=task.source,
            difficulty=task.difficulty,
            memory_dependency=task.memory_dependency,
            progress_score=self.current_subtask_index / len(task.subtasks),
            episode_success=self.done,
            memory_ops=memory_ops or [],
            memory_state_diff=memory_state_diff,
            compatibility_violations=compatibility_violations or [],
            purchase_history=list(self.purchase_history),
            current_subtask_index=self.current_subtask_index,
        )

    def current_subtask(self) -> ShoppingSubtask:
        task = self.require_task()
        if self.current_subtask_index >= len(task.subtasks):
            return task.subtasks[-1]
        return task.subtasks[self.current_subtask_index]

    def require_task(self) -> ShoppingTask:
        if self.task is None:
            raise RuntimeError("Environment must be reset before use.")
        return self.task

    def close(self) -> None:
        self.long_term_memory.clear()
        self.short_term_context.clear()
        self.purchase_history.clear()
        self.bundle_state.clear()


def build_default_tasks() -> list[ShoppingTask]:
    return load_task_dataset()


def load_task_dataset(
    data_path: str | Path | None = None,
    split: str | None = None,
    split_dir: str | Path | None = None,
) -> list[ShoppingTask]:
    resolved_data_path = Path(data_path or os.environ.get("AGENTMEMORY_DATA_PATH", default_smoke_data_path()))
    resolved_split = split if split is not None else os.environ.get("AGENTMEMORY_SPLIT")
    resolved_split_dir = split_dir or os.environ.get("AGENTMEMORY_SPLIT_DIR")

    if resolved_data_path.exists():
        tasks = load_tasks_from_jsonl(resolved_data_path)
    else:
        tasks = [build_tv_bundle_task(), build_laptop_bundle_task()]
    if resolved_split:
        return select_tasks_by_split(tasks, resolved_split, split_dir=resolved_split_dir)
    return tasks


def default_smoke_data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "bundled_shopping_smoke.jsonl"


def default_split_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "splits"


def load_split_task_ids(split: str, split_dir: str | Path | None = None) -> list[str]:
    root = Path(split_dir) if split_dir is not None else default_split_dir()
    path = root / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing AgentMemoryGym split file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def select_tasks_by_split(
    tasks: list[ShoppingTask],
    split: str,
    split_dir: str | Path | None = None,
) -> list[ShoppingTask]:
    task_by_id = {task.task_id: task for task in tasks}
    split_task_ids = load_split_task_ids(split, split_dir=split_dir)
    missing = [task_id for task_id in split_task_ids if task_id not in task_by_id]
    if missing:
        raise ValueError(f"Split '{split}' references unknown AgentMemoryGym task ids: {missing}.")
    selected = [task_by_id[task_id] for task_id in split_task_ids]
    mismatched = [task.task_id for task in selected if task.split != split]
    if mismatched:
        raise ValueError(f"Split '{split}' contains tasks whose record split differs: {mismatched}.")
    return selected


def load_tasks_from_jsonl(path: str | Path) -> list[ShoppingTask]:
    tasks: list[ShoppingTask] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            tasks.append(task_from_record(record, source_path=str(path), line_number=line_number))
    if not tasks:
        raise ValueError(f"No AgentMemoryGym tasks found in {path}.")
    return tasks


def task_from_record(record: dict[str, Any], *, source_path: str, line_number: int) -> ShoppingTask:
    for key in ["task_id", "title", "subtasks"]:
        if key not in record:
            raise ValueError(f"{source_path}:{line_number} missing required field '{key}'.")
    subtasks = tuple(subtask_from_record(item, source_path=source_path, line_number=line_number) for item in record["subtasks"])
    if not subtasks:
        raise ValueError(f"{source_path}:{line_number} must contain at least one subtask.")
    return ShoppingTask(
        task_id=str(record["task_id"]),
        title=str(record["title"]),
        subtasks=subtasks,
        split=str(record.get("split", "train")),
        source=str(record.get("source", "jsonl")),
        difficulty=str(record.get("difficulty", "unknown")),
        memory_dependency=str(record.get("memory_dependency", "cross_session_product_attribute")),
    )


def subtask_from_record(record: dict[str, Any], *, source_path: str, line_number: int) -> ShoppingSubtask:
    for key in ["instruction", "candidate_products", "target_product_id"]:
        if key not in record:
            raise ValueError(f"{source_path}:{line_number} subtask missing required field '{key}'.")
    products = tuple(product_from_record(item, source_path=source_path, line_number=line_number) for item in record["candidate_products"])
    if not products:
        raise ValueError(f"{source_path}:{line_number} subtask must contain candidate_products.")
    product_ids = {product.product_id for product in products}
    target_product_id = str(record["target_product_id"])
    if target_product_id not in product_ids:
        raise ValueError(
            f"{source_path}:{line_number} target_product_id '{target_product_id}' "
            f"is not in candidate product ids {sorted(product_ids)}."
        )
    return ShoppingSubtask(
        instruction=str(record["instruction"]),
        candidate_products=products,
        target_product_id=target_product_id,
    )


def product_from_record(record: dict[str, Any], *, source_path: str, line_number: int) -> Product:
    for key in ["product_id", "title", "attributes"]:
        if key not in record:
            raise ValueError(f"{source_path}:{line_number} product missing required field '{key}'.")
    attributes = record["attributes"]
    if not isinstance(attributes, dict):
        raise ValueError(f"{source_path}:{line_number} product attributes must be an object.")
    return Product(
        product_id=str(record["product_id"]),
        title=str(record["title"]),
        attributes=attributes,
    )


def build_tv_bundle_task() -> ShoppingTask:
    tv_a = Product("tv_a", "Orion 4K TV", {"tv_size_in": 65, "tv_weight_kg": 24, "vesa": "300x300"})
    tv_b = Product("tv_b", "Nebula 4K TV", {"tv_size_in": 75, "tv_weight_kg": 32, "vesa": "400x400"})
    mount_a = Product(
        "mount_a",
        "Tilt wall mount A",
        {"compatible_tv_min": 40, "compatible_tv_max": 60, "max_weight_kg": 28, "supported_vesa": ("200x200", "300x300")},
    )
    mount_b = Product(
        "mount_b",
        "Articulating wall mount B",
        {"compatible_tv_min": 65, "compatible_tv_max": 85, "max_weight_kg": 45, "supported_vesa": ("300x300", "400x400")},
    )
    console_a = Product("console_a", "Media console A", {"compatible_tv_min": 45, "compatible_tv_max": 65, "width_in": 58})
    console_b = Product("console_b", "Media console B", {"compatible_tv_min": 70, "compatible_tv_max": 85, "width_in": 78})
    return ShoppingTask(
        task_id="tv_bundle_75",
        title="Bundle a living-room TV setup across multiple shopping sessions",
        subtasks=(
            ShoppingSubtask(
                instruction="Buy the user's chosen larger-screen TV for a large living room. Remember product attributes that later accessories may need.",
                candidate_products=(tv_a, tv_b),
                target_product_id="tv_b",
            ),
            ShoppingSubtask(
                instruction="Buy a wall mount compatible with the previously purchased TV. The current request does not restate the TV size.",
                candidate_products=(mount_a, mount_b),
                target_product_id="mount_b",
            ),
            ShoppingSubtask(
                instruction="Buy a media console compatible with the same previously purchased TV.",
                candidate_products=(console_a, console_b),
                target_product_id="console_b",
            ),
        ),
        split="train",
        source="memoryarena_webshop_style_handcrafted_v0",
        difficulty="smoke_dependency_distance_2",
        memory_dependency="tv_size_weight_vesa_reused_across_sessions",
    )


def build_laptop_bundle_task() -> ShoppingTask:
    laptop_a = Product("laptop_a", "Atlas developer laptop", {"laptop_size_in": 14, "laptop_ports": "usb-c"})
    laptop_b = Product("laptop_b", "Forge mobile workstation", {"laptop_size_in": 16, "laptop_ports": "usb-c"})
    sleeve_a = Product("sleeve_a", "Slim sleeve A", {"compatible_laptop_min": 12, "compatible_laptop_max": 13})
    sleeve_b = Product("sleeve_b", "Protective sleeve B", {"compatible_laptop_min": 14, "compatible_laptop_max": 14})
    dock_a = Product("dock_a", "Dual-display dock A", {"required_port": "usb-c"})
    dock_b = Product("dock_b", "Legacy dock B", {"required_port": "barrel"})
    return ShoppingTask(
        task_id="laptop_bundle_14",
        title="Bundle laptop accessories across multiple shopping sessions",
        subtasks=(
            ShoppingSubtask(
                instruction="Buy the 14-inch laptop the user selected. Remember attributes needed for accessories.",
                candidate_products=(laptop_a, laptop_b),
                target_product_id="laptop_a",
            ),
            ShoppingSubtask(
                instruction="Buy a sleeve compatible with the previously purchased laptop size.",
                candidate_products=(sleeve_a, sleeve_b),
                target_product_id="sleeve_b",
            ),
            ShoppingSubtask(
                instruction="Buy the dock that matches the previously purchased laptop port type.",
                candidate_products=(dock_a, dock_b),
                target_product_id="dock_a",
            ),
        ),
        split="dev",
        source="memoryarena_webshop_style_handcrafted_v0",
        difficulty="smoke_dependency_distance_2",
        memory_dependency="laptop_size_port_reused_across_sessions",
    )


def require_str(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise InvalidAction(f"Missing required field '{key}'.")
    value = payload[key]
    if not isinstance(value, str):
        raise InvalidAction(f"Field '{key}' must be a string.")
    return value


def empty_memory_diff() -> dict[str, list[dict[str, Any]]]:
    return {"added": [], "updated": [], "deleted": []}


def memory_entry_dict(entry: MemoryEntry) -> dict[str, Any]:
    return {
        "memory_id": entry.memory_id,
        "key": entry.key,
        "value": entry.value,
        "created_step": entry.created_step,
        "updated_step": entry.updated_step,
        "access_count": entry.access_count,
    }


def render_attr_value(value: Any) -> str:
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieval_score(query: str, text: str) -> int:
    return len(tokenize(query) & tokenize(text))


def has_compatibility_constraints(product: Product) -> bool:
    return any(key.startswith("compatible_") or key == "required_port" for key in product.attributes)
