"""Episode-private Mem0 v2.0.19 adapter for frozen CAMG evaluation.

The policy never receives a Mem0 action.  At a native ``replace_messages``
boundary the adapter sends the outgoing conversation through Mem0's official
``Memory.add(..., infer=True)`` pipeline, searches the same episode-private
store, and injects the retrieved memories into the replacement context.  The
native environment remains the sole owner of task state and reward.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA,
    PolicyActionBudget,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_action_budget_receipt,
)

from .agemem import _copy_messages


MEM0_SOURCE_REVISION = "71fba8d46436f88569d600f81a55208c38ad30b5"
MEM0_VERSION = "2.0.19"
MEM0_ADAPTER_SCHEMA = "camg_mem0_adapter_v1"
MEM0_PROMPT_MARKER = "[CAMG_MEM0_AUTOMATIC_MEMORY_V2.0.19]"

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_CUSTOM_INSTRUCTIONS = (
    "Extract only facts, constraints, evidence, decisions, and intermediate "
    "results that can help complete the current long-horizon task after the "
    "active context is replaced. Preserve exact names, paths, commands, "
    "numbers, dates, identifiers, and error messages. Do not invent facts."
)


def _loopback_openai_base_url(value: Any, *, field: str) -> str:
    endpoint = str(value or "").rstrip("/")
    parsed = urlparse(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} has an invalid port") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or port is None
        or parsed.path not in {"/v1", "/v1/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            f"{field} must be a same-Pod loopback OpenAI-compatible /v1 URL"
        )
    return endpoint


@dataclass(frozen=True)
class Mem0AdapterConfig:
    runtime_root: str = "/tmp/camg-mem0"
    llm_base_url: str = "http://127.0.0.1:65201/v1"
    llm_model: str = "Qwen3.5-4B"
    llm_api_key: str = "camg-local"
    embedding_base_url: str = "http://127.0.0.1:65202/v1"
    embedding_model: str = "BAAI/bge-m3"
    embedding_dims: int = 1024
    top_k: int = 5
    threshold: float = 0.1
    max_query_bytes: int = 24576
    max_injected_bytes: int = 24576

    def __post_init__(self) -> None:
        root = Path(self.runtime_root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("Mem0 runtime_root must be an absolute safe path")
        object.__setattr__(
            self,
            "llm_base_url",
            _loopback_openai_base_url(self.llm_base_url, field="Mem0 llm_base_url"),
        )
        object.__setattr__(
            self,
            "embedding_base_url",
            _loopback_openai_base_url(
                self.embedding_base_url, field="Mem0 embedding_base_url"
            ),
        )
        for field in ("llm_model", "llm_api_key", "embedding_model"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Mem0 {field} must be non-empty text")
        for field in (
            "embedding_dims",
            "top_k",
            "max_query_bytes",
            "max_injected_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Mem0 {field} must be a positive integer")
        if isinstance(self.threshold, bool):
            raise TypeError("Mem0 threshold must be numeric")
        try:
            threshold = float(self.threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Mem0 threshold must be finite and in [0, 1]") from exc
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("Mem0 threshold must be finite and in [0, 1]")
        object.__setattr__(self, "threshold", threshold)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "Mem0AdapterConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("Mem0 adapter config must be a mapping")
        normalized = {str(key): item for key, item in value.items()}
        unknown = sorted(set(normalized) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("unknown Mem0 adapter config fields: " + ", ".join(unknown))
        return cls(**normalized)


def _official_memory_factory(config: dict[str, Any]):
    # Mem0 reads this flag at import time.  The matched evaluator is local-only
    # and must not emit analytics traffic.
    os.environ["MEM0_TELEMETRY"] = "false"
    from mem0 import Memory

    return Memory.from_config(config)


class Mem0EnvClientAdapter(BaseEnvClient):
    """Run Mem0 automatically at native context-replacement boundaries."""

    def __init__(
        self,
        native_client: BaseEnvClient,
        config: Mem0AdapterConfig | Mapping[str, Any] | None = None,
        *,
        memory_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if not isinstance(native_client, BaseEnvClient):
            raise TypeError("Mem0 adapter requires a BaseEnvClient")
        super().__init__(native_client.action_format)
        self.native_client = native_client
        self.config = (
            config
            if isinstance(config, Mem0AdapterConfig)
            else Mem0AdapterConfig.from_mapping(config)
        )
        self._memory_factory = memory_factory or _official_memory_factory
        self._memory: Any = None
        self._episode_dir: Path | None = None
        self._run_id: str | None = None
        self._episode_source_identity: dict[str, Any] | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        self._boundary_count = 0
        self._pending_action_budget: PolicyActionBudget | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_client, name)

    def __len__(self) -> int:
        return len(self.native_client)

    @property
    def sample_excluded(self) -> bool:
        return bool(getattr(self.native_client, "sample_excluded", False))

    def observe(self) -> str:
        return str(self.native_client.observe())

    def policy_framing(self) -> list[dict[str, str]]:
        method = getattr(self.native_client, "policy_framing", None)
        return _copy_messages(method() if callable(method) else ())

    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        return _copy_messages(
            self.native_client.normalize_initial_policy_context(
                self._strip_retrieval(_copy_messages(messages))
            )
        )

    def bind_policy_context(
        self, messages: Sequence[Mapping[str, str]], *, initial: bool = False
    ) -> None:
        normalized = _copy_messages(messages)
        self._current_policy_context = deepcopy(normalized)
        self.native_client.bind_policy_context(
            self._strip_retrieval(normalized), initial=initial
        )

    def policy_turn_candidate(self) -> str | None:
        return self.native_client.policy_turn_candidate()

    def prepare_policy_turn(self, pressure: PolicyContextPressure | None) -> str | None:
        return self.native_client.prepare_policy_turn(pressure)

    def bind_policy_action_budget(self, budget: PolicyActionBudget) -> None:
        if not isinstance(budget, PolicyActionBudget):
            raise TypeError("Mem0 adapter action budget must be PolicyActionBudget")
        if self._pending_action_budget is not None:
            raise RuntimeError("Mem0 adapter action budget was rebound before use")
        self._pending_action_budget = budget
        self.native_client.bind_policy_action_budget(budget)

    def step(self, action: str) -> StepOutput:
        if not isinstance(action, str):
            raise TypeError("Mem0-adapted policy action must be text")
        budget = self._take_action_budget()
        output = self.native_client.step(action)
        return self._wrap_native(
            output,
            action=action,
            allow_pipeline=True,
            budget=budget,
        )

    def reset(self, idx: int = 0) -> Any:
        self._cleanup_memory()
        self._current_policy_context = None
        self._episode_source_identity = None
        self._usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._boundary_count = 0
        self._pending_action_budget = None
        response = self.native_client.reset(idx)
        identity = getattr(self.native_client, "episode_source_identity", None)
        if not isinstance(identity, Mapping) or not identity:
            raise RuntimeError("Mem0 native reset lacks episode source identity")
        self._episode_source_identity = deepcopy(dict(identity))
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        root = Path(self.config.runtime_root)
        if root.exists() and root.is_symlink():
            raise RuntimeError("Mem0 runtime_root must not be a symlink")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        owner = os.environ.get("AGENTMEMORY_PROCESS_OWNER", "local")
        self._episode_dir = Path(
            tempfile.mkdtemp(prefix=f"{owner}.{digest[:16]}.", dir=root)
        )
        self._run_id = f"camg-{digest}"
        try:
            self._memory = self._memory_factory(self._memory_config())
        except Exception:
            self._cleanup_memory()
            raise
        return response

    def finalize_policy_horizon(self) -> StepOutput | None:
        output = self.native_client.finalize_policy_horizon()
        return (
            None
            if output is None
            else self._wrap_native(
                output,
                action="",
                allow_pipeline=False,
                budget=None,
            )
        )

    def close(self) -> Any:
        try:
            return self.native_client.close()
        finally:
            self._cleanup_memory()

    def _memory_config(self) -> dict[str, Any]:
        episode_dir = self._require_episode_dir()
        return {
            "version": "v1.1",
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(episode_dir / "qdrant"),
                    "collection_name": "camg_mem0",
                    "embedding_model_dims": self.config.embedding_dims,
                    "on_disk": True,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.config.llm_model,
                    "api_key": self.config.llm_api_key,
                    "openai_base_url": self.config.llm_base_url,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 2000,
                    "is_reasoning_model": False,
                    "response_callback": self._record_response,
                },
            },
            "embedder": {
                # The endpoint serves the pinned BGE-M3 weights through the
                # OpenAI-compatible embeddings API.  Using Mem0's OpenAI
                # transport keeps one shared sidecar per Pod instead of
                # importing one SentenceTransformer in every evaluator worker.
                "provider": "openai",
                "config": {
                    "model": self.config.embedding_model,
                    "api_key": "camg-local",
                    "openai_base_url": self.config.embedding_base_url,
                },
            },
            "history_db_path": str(episode_dir / "history.db"),
            "custom_instructions": _CUSTOM_INSTRUCTIONS,
        }

    def _record_response(self, _llm: Any, response: Any, _params: dict[str, Any]) -> None:
        self._usage["calls"] += 1
        usage = getattr(response, "usage", None)
        for target, names in (
            ("input_tokens", ("prompt_tokens", "input_tokens")),
            ("output_tokens", ("completion_tokens", "output_tokens")),
        ):
            value = 0
            for name in names:
                candidate = getattr(usage, name, None) if usage is not None else None
                if candidate is not None:
                    value = candidate
                    break
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(f"Mem0 hidden response has invalid {target}")
            self._usage[target] += value

    def _wrap_native(
        self,
        output: StepOutput,
        *,
        action: str,
        allow_pipeline: bool,
        budget: PolicyActionBudget | None,
    ) -> StepOutput:
        if not isinstance(output, StepOutput):
            raise TypeError("native client step must return StepOutput")
        info = deepcopy(dict(output.info)) if isinstance(output.info, Mapping) else {}
        transition = info.get("context_transition")
        operation_counts: dict[str, int] = {}
        retrieved: list[dict[str, Any]] = []
        before = dict(self._usage)
        started = time.monotonic()
        boundary_requested = False
        boundary_pipeline = False
        atomic_operation_blocked = False
        if isinstance(transition, Mapping):
            transition = deepcopy(dict(transition))
            if (
                allow_pipeline
                and transition.get("schema") == TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA
                and transition.get("operation") == CONTEXT_OPERATION_REPLACE
            ):
                boundary_requested = True
                self._boundary_count += 1
                if budget is None:
                    raise RuntimeError("Mem0 boundary lacks a bound action budget")
                if budget.remaining_steps < 3:
                    atomic_operation_blocked = True
                else:
                    boundary_pipeline = True
                    retrieved, operation_counts = self._add_then_search(
                        action=action,
                        observation=str(output.state),
                        replacement_messages=_copy_messages(
                            transition.get("messages", ())
                        ),
                    )
                    transition["messages"] = self._inject_retrieval(
                        _copy_messages(transition.get("messages", ())), retrieved
                    )
            info["context_transition"] = transition
        elapsed_ms = (
            int(round((time.monotonic() - started) * 1000))
            if boundary_pipeline
            else 0
        )
        usage_delta = {
            key: self._usage[key] - before[key]
            for key in ("calls", "input_tokens", "output_tokens")
        }
        if boundary_pipeline and usage_delta["calls"] <= 0:
            raise RuntimeError(
                "Mem0 boundary pipeline completed without an observed LLM call"
            )

        evidence = deepcopy(dict(info.get("wrapper_evidence") or {}))
        evidence["mem0_adapter"] = {
            "schema": MEM0_ADAPTER_SCHEMA,
            "event": "native_action_passthrough",
            "episode_private": True,
            "official_pipeline": True,
            "source_revision": MEM0_SOURCE_REVISION,
            "version": MEM0_VERSION,
            "boundary_requested": boundary_requested,
            "boundary_pipeline": boundary_pipeline,
            "boundary_index": self._boundary_count,
            "operation_counts": operation_counts,
            "retrieved_memory_count": len(retrieved),
            "hidden_model_calls": usage_delta["calls"],
            "hidden_input_tokens": usage_delta["input_tokens"],
            "hidden_output_tokens": usage_delta["output_tokens"],
            "hidden_latency_ms": elapsed_ms,
        }
        info["wrapper_evidence"] = evidence
        env_info = info.get("env_info")
        if env_info is None:
            env_info = {}
        if not isinstance(env_info, Mapping):
            raise TypeError("native client env_info must be a mapping")
        env_info = deepcopy(dict(env_info))
        identity = self._identity()
        observed_identity = env_info.get("episode_source_identity")
        if observed_identity is not None and observed_identity != identity:
            raise RuntimeError("native client episode source identity drifted")
        env_info["episode_source_identity"] = identity
        if atomic_operation_blocked:
            env_info["truncated"] = True
            env_info["terminal_reason"] = "combined_step_budget_exhausted"
            evidence["outcome"] = "terminal_failure"
            evidence["terminal_reason"] = "combined_step_budget_exhausted"
        info["env_info"] = env_info
        if allow_pipeline:
            if budget is None:
                raise RuntimeError("Mem0 policy step lacks a bound action budget")
            info["action_budget"] = build_task_neutral_action_budget_receipt(
                budget,
                auxiliary_steps=2 if boundary_pipeline else 0,
                required_auxiliary_steps=2 if boundary_requested else 0,
                atomic_operation_blocked=atomic_operation_blocked,
            )
        reward = output.reward
        if reward is None:
            if not bool(output.done) or not self.sample_excluded:
                raise RuntimeError(
                    "native client returned a null reward without a terminal "
                    "sample-exclusion receipt"
                )
        else:
            reward = float(reward)
        return StepOutput(
            state=str(output.state),
            reward=reward,
            done=bool(output.done) or atomic_operation_blocked,
            info=info,
        )

    def _take_action_budget(self) -> PolicyActionBudget:
        budget = self._pending_action_budget
        self._pending_action_budget = None
        if budget is None:
            raise RuntimeError("Mem0 policy step requires a fresh action budget binding")
        return budget

    def _add_then_search(
        self,
        *,
        action: str,
        observation: str,
        replacement_messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        memory = self._require_memory()
        run_id = self._require_run_id()
        outgoing = self._strip_retrieval(self._current_policy_context or [])
        messages = [
            message for message in outgoing if message["role"] in {"user", "assistant"}
        ]
        messages.extend(
            [
                {"role": "assistant", "content": action},
                {"role": "user", "content": observation},
            ]
        )
        add_result = memory.add(messages, run_id=run_id, infer=True)
        if not isinstance(add_result, Mapping) or not isinstance(
            add_result.get("results"), list
        ):
            raise RuntimeError("Mem0 add returned an invalid official result")
        add_rows = add_result["results"]
        event_counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NONE": 0}
        for row in add_rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("Mem0 add result contains a non-object row")
            event = str(row.get("event") or "NONE").upper()
            if event not in event_counts:
                raise RuntimeError(f"Mem0 add returned unsupported event {event!r}")
            event_counts[event] += 1

        query = self._search_query(replacement_messages, observation)
        search_result = memory.search(
            query,
            filters={"run_id": run_id},
            top_k=self.config.top_k,
            threshold=self.config.threshold,
        )
        if not isinstance(search_result, Mapping) or not isinstance(
            search_result.get("results"), list
        ):
            raise RuntimeError("Mem0 search returned an invalid official result")
        normalized: list[dict[str, Any]] = []
        for row in search_result["results"]:
            if not isinstance(row, Mapping):
                raise RuntimeError("Mem0 search result contains a non-object row")
            memory_text = row.get("memory")
            if not isinstance(memory_text, str) or not memory_text.strip():
                raise RuntimeError("Mem0 search result lacks memory text")
            normalized.append(
                {
                    "memory": memory_text.strip(),
                    "score": row.get("score"),
                }
            )
        return normalized, {
            "add": 1,
            "search": 1,
            "added": event_counts["ADD"],
            "updated": event_counts["UPDATE"],
            "deleted": event_counts["DELETE"],
            "unchanged": event_counts["NONE"],
            "retrieved": len(normalized),
        }

    def _search_query(
        self, replacement_messages: Sequence[Mapping[str, str]], observation: str
    ) -> str:
        parts = [
            message["content"]
            for message in self._strip_retrieval(replacement_messages)
            if message["role"] == "user" and message["content"].strip()
        ]
        if not parts:
            parts = [observation]
        return _truncate_utf8("\n\n".join(parts), self.config.max_query_bytes)

    def _inject_retrieval(
        self,
        messages: Sequence[Mapping[str, str]],
        memories: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        normalized = self._strip_retrieval(messages)
        if not memories:
            return normalized
        bullets = "\n".join(
            f"- {str(memory['memory']).strip()}" for memory in memories
        )
        payload = _truncate_utf8(
            f"{MEM0_PROMPT_MARKER}\nRelevant episode memories retrieved automatically:\n{bullets}",
            self.config.max_injected_bytes,
        )
        for message in normalized:
            if message["role"] == "system":
                message["content"] = message["content"] + "\n\n" + payload
                return normalized
        return [{"role": "system", "content": payload}, *normalized]

    @staticmethod
    def _strip_retrieval(
        messages: Sequence[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        normalized = _copy_messages(messages)
        stripped: list[dict[str, str]] = []
        for message in normalized:
            if message["role"] == "system":
                index = message["content"].find(MEM0_PROMPT_MARKER)
                if index >= 0:
                    message["content"] = message["content"][:index].rstrip("\n")
            if message["role"] != "system" or message["content"]:
                stripped.append(message)
        return stripped

    def _identity(self) -> dict[str, Any]:
        if not isinstance(self._episode_source_identity, Mapping):
            raise RuntimeError("Mem0 episode source identity is unavailable")
        return deepcopy(dict(self._episode_source_identity))

    def _require_memory(self) -> Any:
        if self._memory is None:
            raise RuntimeError("Mem0 episode store is unavailable")
        return self._memory

    def _require_episode_dir(self) -> Path:
        if self._episode_dir is None:
            raise RuntimeError("Mem0 episode directory is unavailable")
        return self._episode_dir

    def _require_run_id(self) -> str:
        if self._run_id is None:
            raise RuntimeError("Mem0 episode run_id is unavailable")
        return self._run_id

    def _cleanup_memory(self) -> None:
        memory = self._memory
        episode_dir = self._episode_dir
        self._memory = None
        self._episode_dir = None
        self._run_id = None
        self._pending_action_budget = None
        if memory is not None:
            clients: list[Any] = []
            for store_name in ("vector_store", "_entity_store"):
                store = getattr(memory, store_name, None)
                client = getattr(store, "client", None)
                if client is not None and all(client is not item for item in clients):
                    clients.append(client)
            close = getattr(memory, "close", None)
            if callable(close):
                close()
            for client in clients:
                close_client = getattr(client, "close", None)
                if callable(close_client):
                    close_client()
        gc.collect()
        if episode_dir is not None and episode_dir.exists():
            shutil.rmtree(episode_dir)


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    marker = "\n...[truncated]"
    budget = max(0, max_bytes - len(marker.encode("utf-8")))
    prefix = encoded[:budget]
    while prefix:
        try:
            return prefix.decode("utf-8") + marker
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return marker.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


__all__ = [
    "MEM0_ADAPTER_SCHEMA",
    "MEM0_PROMPT_MARKER",
    "MEM0_SOURCE_REVISION",
    "MEM0_VERSION",
    "Mem0AdapterConfig",
    "Mem0EnvClientAdapter",
]
