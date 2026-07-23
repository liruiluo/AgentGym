from __future__ import annotations

import glob
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

from ..runtime.domain import DomainContract, DomainTransition
from .memoryarena_dataset import (
    MemoryArenaDatasetProvenance,
    verify_memoryarena_dataset_provenance,
)


BROWSECOMP_DOMAIN_ID = "progressive_search"
BROWSECOMP_CONTRACT_MODES = ("paper_eval", "failfast")
BROWSECOMP_SURFACES = {
    "paper_eval": (
        "memoryarena_progressive_search_paper_eval_public221_one_action_v3"
    ),
    "failfast": (
        "memoryarena_progressive_search_failfast_public221_one_action_v3"
    ),
}
BROWSECOMP_PUBLIC_TASK_COUNT = 221
BROWSECOMP_PUBLIC_PHASE_COUNT = 1641
BROWSECOMP_PAPER_TASK_COUNT = 256
BROWSECOMP_PAPER_MISSING_TASK_COUNT = 35
BROWSECOMP_FROZEN_MEMORYARENA_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
BROWSECOMP_SUBQUERY_MAX_ITERATIONS = 35
BROWSECOMP_FINAL_MAX_ITERATIONS = 30
BROWSECOMP_MEMORY_ACTION_ALLOWANCE_PER_PHASE = 16
BROWSECOMP_FROZEN_EMBEDDING_MODEL = "text-embedding-3-small"
BROWSECOMP_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
BROWSECOMP_FROZEN_SNIPPET_TOKENS = 512
BROWSECOMP_FROZEN_SEARCH_K = 5
BROWSECOMP_FROZEN_TOKENIZER_REPOSITORY = "Qwen/Qwen3-0.6B"
BROWSECOMP_FROZEN_TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BROWSECOMP_FROZEN_TOKENIZER_FILES_SHA256 = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
BROWSECOMP_FROZEN_TOKENIZER_BUNDLE_SHA256 = (
    "62851e5e39395f893633e2283ace53d5b223896d0058c751fa086f81c7a4f187"
)
BROWSECOMP_FROZEN_INDEX_DIMENSION = 1536
BROWSECOMP_FROZEN_DOCUMENT_COUNT = 100195
BROWSECOMP_FROZEN_INDEX_REPOSITORY = "Joanna690/websearch-embeddings"
BROWSECOMP_FROZEN_INDEX_REVISION = "7a784780b46d16ddc926aed9b63c34def2014c47"
BROWSECOMP_FROZEN_CORPUS_REPOSITORY = "Tevatron/browsecomp-plus-corpus"
BROWSECOMP_FROZEN_CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
BROWSECOMP_JOANNA_REFERENCE = "Joanna690/browsecomp_all_jsons"
BROWSECOMP_JOANNA_REFERENCE_SHA256 = (
    "6f6b1f6c40ae37196e23fe4053747568ed2031bffd3da3733748f99c6631b46f"
)

BROWSECOMP_FROZEN_MATERIALIZED_CORPUS_SHA256 = (
    "6b306573f6194367d5e2a7daaae12d9cb4242409413f261ea6d81a19d7cf4b26"
)

BROWSECOMP_FROZEN_INDEX_SHARDS = (
    {
        "name": "shard0.index",
        "index_sha256": "57c7b94af14d6d84445da99d207f52dd044f275c6d6142353c88b19e0d938956",
        "id_map_sha256": "1d65ae1fa019cc8b61ffc29d0a72adf1567ebef1b01a735697eb61019719d1e4",
        "vector_count": 25049,
    },
    {
        "name": "shard1.index",
        "index_sha256": "d253728c07e641e0d9ec1dbf992a519db2c1cf4808cbf8167bf3bdc40df5c41f",
        "id_map_sha256": "4d1c2272ebb9d359cfab6cb07f7affd8e78a3823335b42aa2bf3cd39e64e4150",
        "vector_count": 25049,
    },
    {
        "name": "shard2.index",
        "index_sha256": "22726188a49f92bbcddf14bee9d00b3bb69d11eb01e5995b52c9c79a8d111ca9",
        "id_map_sha256": "dd4e3014335e41c979984bdbbb79bc20ee554ad5c6e70a59209a395080deffbd",
        "vector_count": 25049,
    },
    {
        "name": "shard3.index",
        "index_sha256": "652eb57c1c0ab498d7b53253ebdd21591be394e5d546650ca6027774ddc4845c",
        "id_map_sha256": "334aa558fbe525b9be2db23a9338446fdb07c8713ab033d8fe7871ed19233d9c",
        "vector_count": 25048,
    },
)
BROWSECOMP_FROZEN_CORPUS_SHARDS = (
    "7c07f9e23b1ca548110fd831714cadc67d44db5223bace6e45fcaa795d3153d0",
    "e92d8202e0f656a85b262153dbcd22ecf80ea2d0c96d9884f9c8e25480b869ab",
    "0e4113a4503342527258d8f2c49877747435f3e65bfe1f7306b4f488c8d225fe",
    "0ceea5e703332a2e3ce700f641273400d84583fad84b659d3248ed06d3a9fef3",
    "15b62914ddc3de6946893c770f07d5d84d29646e833ca1447955668f2b57940c",
    "a9a75708ad37c522e93a774e5a968a3129e12b0559971c8f950a5628e0201df0",
    "290062b60c1a6ebba7d5469a37a431f0a2596e68788295284b1b2d35db07b62c",
)
BROWSECOMP_JUDGE_PROMPT_TEMPLATE_SHA256 = (
    "0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835"
)
BROWSECOMP_UPSTREAM_REQUIRED_FILES = (
    "env/__init__.py",
    "env/env_systems/__init__.py",
    "env/env_systems/base_env.py",
    "agent/search.py",
    "run_search.py",
    "env/env_systems/browsecomp_plus_env.py",
)
BROWSECOMP_UPSTREAM_SOURCE_ROOTS = (
    "agent",
    "env/env_systems/web_search_env",
)
_ACTION_RE = re.compile(
    r"\A(search|get_document|SUBMIT_ANSWER)\s+(\{.*\})\Z",
    flags=re.DOTALL,
)
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def _contract(mode: str) -> DomainContract:
    if mode == "paper_eval":
        system_prompt = (
            "You are operating the public-221 MemoryArena Progressive Web Search "
            "paper-evaluation adapter. Every phase is privately judged and then "
            "advances, including after an incorrect answer. Online reward is always "
            "zero because this surface is evaluation-only. Task PS is the fraction "
            "of correct phases, SR@k is correctness at phase depth k, and task SR is "
            "the final-phase verdict."
        )
    elif mode == "failfast":
        system_prompt = (
            "You are operating the public-221 MemoryArena Progressive Web Search "
            "fail-fast training variant. Every submitted phase answer is privately "
            "judged. A correct answer earns +1 and advances exactly once; an incorrect "
            "answer earns 0 and ends the episode immediately while preserving rewards "
            "already earned on earlier phases."
        )
    else:  # pragma: no cover - callers validate before construction.
        raise ValueError(f"unsupported BrowseComp contract mode: {mode}")
    return DomainContract(
        contract_id=(
            f"memoryarena_progressive_search_{mode}_public221_"
            "one_action_v3_20260723"
        ),
        system_prompt=(
            system_prompt
            + " Use lowercase search to retrieve five 512-token snippets from the "
            "frozen corpus, get_document to inspect a document by docid, and "
            "SUBMIT_ANSWER for the current question. Private reference answers are "
            "never exposed in observations. This one-action adapter executes exactly "
            "one policy action per turn. Its 35-action subquery and 30-action final "
            "phase limits count native action attempts, including invalid attempts, "
            "but not memory actions. They are not parity with upstream model turns "
            "that may batch multiple tool calls."
        ),
        native_action_descriptions=(
            'search {"query": "..."}',
            'get_document {"docid": "..."}',
            (
                'SUBMIT_ANSWER {"answer": "Explanation: ...\\nExact Answer: '
                '...\\nConfidence: ...%"}'
            ),
        ),
        max_steps=560,
    )


BROWSECOMP_CONTRACTS = {mode: _contract(mode) for mode in BROWSECOMP_CONTRACT_MODES}


@dataclass(frozen=True)
class BrowseCompPhase:
    kind: str
    question: str
    answer: str


@dataclass(frozen=True)
class BrowseCompTask:
    query_id: str
    phases: tuple[BrowseCompPhase, ...]


BrowseSearch = Callable[[str, dict[str, Any]], str]
BrowseJudge = Callable[[str, str, str], dict[str, Any]]


class BrowseCompPlusFactory:
    domain_id = BROWSECOMP_DOMAIN_ID

    def __init__(
        self,
        *,
        contract_mode: str,
        tasks_path: str | Path,
        dataset_provenance: MemoryArenaDatasetProvenance,
        memoryarena_root: str | Path | None = None,
        index_path: str | Path | None = None,
        corpus_path: str | Path | None = None,
        corpus_manifest_path: str | Path | None = None,
        embedding_model: str = BROWSECOMP_FROZEN_EMBEDDING_MODEL,
        provider: str = "openai",
        embedding_endpoint: str | None = None,
        judge_config: dict[str, Any] | None = None,
        search_asset_provenance: dict[str, Any] | None = None,
        search_tool: BrowseSearch | None = None,
        judge: BrowseJudge | None = None,
        expected_memoryarena_commit: str | None = None,
        test_mode: bool = False,
    ) -> None:
        if contract_mode not in BROWSECOMP_CONTRACT_MODES:
            raise ValueError(
                "BrowseComp contract_mode must be one of: "
                + ", ".join(BROWSECOMP_CONTRACT_MODES)
            )
        self.contract_mode = contract_mode
        if provider not in {"openai", "openrouter"}:
            raise ValueError(
                "BrowseComp embedding provider must be openai or openrouter"
            )
        if contract_mode == "paper_eval" and provider != "openai":
            raise RuntimeError(
                "Progressive Search paper_eval requires the OpenAI embedding provider"
            )
        self.surface = BROWSECOMP_SURFACES[contract_mode]
        self.contract = BROWSECOMP_CONTRACTS[contract_mode]
        self.tasks_path = Path(tasks_path).expanduser().resolve()
        verify_memoryarena_dataset_provenance(
            self.tasks_path,
            expected_config="progressive_search",
            provenance=dataset_provenance,
        )
        if dataset_provenance.mode != "frozen_public_hf_dataset" and not test_mode:
            raise RuntimeError(
                "Production Progressive Search requires the frozen public "
                "MemoryArena progressive_search dataset"
            )
        self.dataset_provenance = dataset_provenance
        self.tasks = load_browsecomp_tasks(self.tasks_path)
        self.task_count = len(self.tasks)
        self.phase_count = sum(len(task.phases) for task in self.tasks)
        if (
            self.task_count != dataset_provenance.record_count
            or self.phase_count != dataset_provenance.phase_count
        ):
            raise RuntimeError(
                "Loaded Progressive Search tasks differ from dataset provenance"
            )
        self.embedding_route_provenance = (
            _embedding_route_provenance(
                provider=provider,
                model=embedding_model,
                endpoint=embedding_endpoint,
                contract_mode=contract_mode,
            )
            if embedding_endpoint is not None
            else {"mode": "injected_test_double"}
            if test_mode
            else None
        )
        if self.embedding_route_provenance is None:
            raise RuntimeError(
                "Production Progressive Search requires an explicit embedding endpoint"
            )
        max_phase_count = max(len(task.phases) for task in self.tasks)
        native_action_limit = (
            (max_phase_count - 1) * BROWSECOMP_SUBQUERY_MAX_ITERATIONS
            + BROWSECOMP_FINAL_MAX_ITERATIONS
        )
        memory_action_allowance = (
            max_phase_count * BROWSECOMP_MEMORY_ACTION_ALLOWANCE_PER_PHASE
        )
        total_action_limit = native_action_limit + memory_action_allowance
        self.native_action_limit = native_action_limit
        self.memory_action_allowance = memory_action_allowance
        self.contract = DomainContract(
            contract_id=self.contract.contract_id,
            system_prompt=(
                self.contract.system_prompt
                + " The global episode action cap is separate from the native "
                "per-phase limits and reserves up to "
                f"{BROWSECOMP_MEMORY_ACTION_ALLOWANCE_PER_PHASE} memory actions "
                f"per dataset phase ({memory_action_allowance} total at the "
                f"longest task), so using memory does not consume the "
                f"{native_action_limit} native-action allowance. The exact global "
                f"cap for this dataset is {total_action_limit}."
            ),
            native_action_descriptions=self.contract.native_action_descriptions,
            max_steps=total_action_limit,
        )
        self.memoryarena_root = (
            Path(memoryarena_root).expanduser().resolve()
            if memoryarena_root is not None
            else None
        )

        if test_mode:
            if search_tool is None or judge is None:
                raise RuntimeError(
                    "BrowseComp test_mode requires explicit search and judge doubles"
                )
            self.upstream_provenance = {"mode": "injected_test_double"}
            self.search_asset_provenance = (
                dict(search_asset_provenance)
                if search_asset_provenance is not None
                else {"mode": "injected_test_double"}
            )
            self.judge_provenance = _injected_judge_provenance()
        else:
            if search_tool is not None or judge is not None:
                raise RuntimeError(
                    "Production BrowseComp refuses injected search or judge doubles"
                )
            if search_asset_provenance is not None:
                raise RuntimeError(
                    "Production BrowseComp computes search asset provenance "
                    "internally and refuses injected provenance"
                )
            if self.memoryarena_root is None:
                raise RuntimeError("memoryarena_root is required for production Search")
            if expected_memoryarena_commit != BROWSECOMP_FROZEN_MEMORYARENA_COMMIT:
                raise RuntimeError(
                    "Production Search requires frozen MemoryArena commit "
                    f"{BROWSECOMP_FROZEN_MEMORYARENA_COMMIT}"
                )
            if index_path is None or corpus_path is None or corpus_manifest_path is None:
                raise RuntimeError(
                    "Production Search requires explicit index, corpus, and corpus "
                    "manifest paths"
                )
            resolved_index_path = _resolve_search_path(
                self.memoryarena_root,
                index_path,
            )
            resolved_corpus_path = Path(
                _resolve_search_path(self.memoryarena_root, corpus_path)
            )
            resolved_manifest_path = Path(corpus_manifest_path).expanduser().resolve()
            normalized_judge_config = _normalize_judge_config(judge_config)
            self.upstream_provenance = attest_browsecomp_upstream(
                self.memoryarena_root,
                expected_commit=BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
            )
            self.search_asset_provenance = attest_browsecomp_search_assets(
                index_pattern=resolved_index_path,
                corpus_path=resolved_corpus_path,
                corpus_manifest_path=resolved_manifest_path,
                embedding_model=embedding_model,
            )
            search_tool = _build_upstream_search(
                self.memoryarena_root,
                index_path=resolved_index_path,
                corpus_path=resolved_corpus_path,
                embedding_model=embedding_model,
                provider=provider,
                embedding_endpoint=embedding_endpoint,
            )
            judge = _build_upstream_judge(
                self.memoryarena_root,
                config=normalized_judge_config,
            )
            self.judge_provenance = _judge_provenance(
                normalized_judge_config,
                mode="upstream_memoryarena_judge",
            )
        assert search_tool is not None and judge is not None
        self.search_tool = search_tool
        self.judge = judge

    def create(self, env_uid: str):
        return BrowseCompPlusDriver(
            contract_mode=self.contract_mode,
            tasks=self.tasks,
            search_tool=self.search_tool,
            judge=self.judge,
            env_uid=env_uid,
            contract=self.contract,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "MemoryArena",
            "dataset_config": "progressive_search",
            "dataset_sha256": self.dataset_provenance.sha256,
            "task_count": self.task_count,
            "phase_count": self.phase_count,
            "dataset_provenance": self.dataset_provenance.metadata(),
            "release_scope": {
                "name": "public221",
                "public_task_count": BROWSECOMP_PUBLIC_TASK_COUNT,
                "paper_task_count": BROWSECOMP_PAPER_TASK_COUNT,
                "missing_private_task_count": BROWSECOMP_PAPER_MISSING_TASK_COUNT,
                "paper_panel_complete": False,
            },
            "cross_source_parity": {
                "reference": BROWSECOMP_JOANNA_REFERENCE,
                "reference_sha256": BROWSECOMP_JOANNA_REFERENCE_SHA256,
                "relation": "questions_and_answers_equal_in_order_ignoring_id",
                "task_count": BROWSECOMP_PUBLIC_TASK_COUNT,
            },
            "contract_mode": self.contract_mode,
            "semantic_variant": (
                "paper_metric_evaluation_continue_on_incorrect_one_action_v1"
                if self.contract_mode == "paper_eval"
                else "ordered_phase_failfast_training_one_action_v1"
            ),
            "action_granularity": {
                "variant": "one_action_v3",
                "policy_actions_per_turn": 1,
                "upstream_batched_model_turn_parity": False,
                "upstream_model_turn_may_batch_tool_calls": True,
            },
            "max_total_actions": self.contract.max_steps,
            "total_action_budget": {
                "limit": self.contract.max_steps,
                "enforced_by": "agentmemory_runtime_wrapper",
                "counts": ["native", "memory", "invalid"],
                "legacy_max_steps_field_is_same_limit": True,
                "native_action_allowance": self.native_action_limit,
                "memory_action_allowance": self.memory_action_allowance,
                "memory_action_allowance_per_phase": (
                    BROWSECOMP_MEMORY_ACTION_ALLOWANCE_PER_PHASE
                ),
            },
            "native_iteration_budget": {
                "subquery_per_phase": BROWSECOMP_SUBQUERY_MAX_ITERATIONS,
                "final_phase": BROWSECOMP_FINAL_MAX_ITERATIONS,
                "counts": ["native", "invalid"],
                "memory_actions_consume_budget": False,
                "separately_tracked_from_total_action_budget": True,
                "upstream_batched_model_turn_parity": False,
            },
            "native_tool_ops": ["search", "get_document"],
            "native_search_k": BROWSECOMP_FROZEN_SEARCH_K,
            "native_snippet_max_tokens": BROWSECOMP_FROZEN_SNIPPET_TOKENS,
            "snippet_tokenizer": {
                "repository": BROWSECOMP_FROZEN_TOKENIZER_REPOSITORY,
                "revision": BROWSECOMP_FROZEN_TOKENIZER_REVISION,
                "local_files_only": True,
                "files_sha256": dict(BROWSECOMP_FROZEN_TOKENIZER_FILES_SHA256),
                "bundle_sha256": BROWSECOMP_FROZEN_TOKENIZER_BUNDLE_SHA256,
            },
            "native_subquery_max_iterations": BROWSECOMP_SUBQUERY_MAX_ITERATIONS,
            "native_final_max_iterations": BROWSECOMP_FINAL_MAX_ITERATIONS,
            "judge": "memoryarena_browsecomp_gpt_judge_v1",
            "judge_provenance": self.judge_provenance,
            "reward_contract": (
                "evaluation_only_zero_reward_not_for_training"
                if self.contract_mode == "paper_eval"
                else "correct_phase_plus_1; incorrect_phase_0_terminal"
            ),
            "paper_evaluation": {
                "id": "memoryarena_progressive_search_ps_sr_at_k_final_sr_v1",
                "dataset_scope": "public221_of_paper256",
                "available": self.contract_mode == "paper_eval",
                "metrics": ["PS", "SR@k", "SR"],
                "metric_scale": "unit_interval",
                "paper_panel_complete": False,
                "public_task_count": BROWSECOMP_PUBLIC_TASK_COUNT,
                "paper_task_count": BROWSECOMP_PAPER_TASK_COUNT,
                "separate_from_online_reward": True,
            },
            "memory_semantics": "agentmemory_explicit_tools_between_native_phases",
            "upstream_provenance": self.upstream_provenance,
            "search_asset_provenance": self.search_asset_provenance,
            "embedding_route_provenance": self.embedding_route_provenance,
        }


class BrowseCompPlusDriver:
    domain_id = BROWSECOMP_DOMAIN_ID

    def __init__(
        self,
        *,
        contract_mode: str,
        tasks: Sequence[BrowseCompTask],
        search_tool: BrowseSearch,
        judge: BrowseJudge,
        env_uid: str,
        contract: DomainContract,
    ) -> None:
        if contract_mode not in BROWSECOMP_CONTRACT_MODES:
            raise ValueError(f"unsupported BrowseComp contract mode: {contract_mode}")
        if not tasks:
            raise ValueError("BrowseCompPlusDriver requires tasks")
        self.contract_mode = contract_mode
        self.surface = BROWSECOMP_SURFACES[contract_mode]
        self.contract = contract
        self.tasks = tuple(tasks)
        self.search_tool = search_tool
        self.judge = judge
        self.env_uid = env_uid
        self.task: BrowseCompTask | None = None
        self.data_idx = 0
        self.phase_index = 0
        self.phase_native_iteration_count = 0
        self.phase_verdict_ledger: list[dict[str, Any]] = []
        self.retrieved_docids: list[str] = []
        self.done = False
        self.status = "idle"

    def reset(self, data_idx: int) -> DomainTransition:
        index = int(data_idx)
        if index < 0 or index >= len(self.tasks):
            raise IndexError(
                f"BrowseComp data index {index} is outside [0, {len(self.tasks)})"
            )
        self.data_idx = index
        self.task = self.tasks[index]
        self.phase_index = 0
        self.phase_native_iteration_count = 0
        self.phase_verdict_ledger = []
        self.retrieved_docids = []
        self.done = False
        self.status = "active"
        return self._transition(self._render_phase())

    def step(self, action: str, env_step: int) -> DomainTransition:
        if self.task is None:
            raise RuntimeError("BrowseComp driver must be reset before step")
        if self.done:
            return self._transition(
                "The Progressive Search episode is complete.", done=True
            )
        phase_index_before = self.phase_index
        self.phase_native_iteration_count += 1
        parsed = _parse_action(action)
        if parsed is None:
            transition = self._invalid(
                action,
                env_step,
                "invalid Progressive Search action grammar",
            )
        else:
            op, payload = parsed
            transition = (
                self._submit_answer(payload, action, env_step)
                if op == "SUBMIT_ANSWER"
                else self._search_tool(op, payload, action, env_step)
            )
        return self._apply_phase_limit(
            transition,
            env_step,
            phase_index_before=phase_index_before,
        )

    def close(self) -> None:
        self.status = "closed"
        self.done = True

    def _search_tool(
        self,
        op: str,
        payload: dict[str, Any],
        raw_action: str,
        env_step: int,
    ) -> DomainTransition:
        expected_key = "query" if op == "search" else "docid"
        if set(payload) != {expected_key}:
            return self._invalid(
                raw_action,
                env_step,
                f"{op} expects only {expected_key}",
            )
        value = payload.get(expected_key)
        if not isinstance(value, str) or not value.strip():
            return self._invalid(
                raw_action,
                env_step,
                f"{expected_key} must be a non-empty string",
            )
        arguments = {expected_key: value}
        try:
            result = self.search_tool(op, arguments)
        except Exception as exc:
            return self._infra_error(op, env_step, exc)
        if not isinstance(result, str):
            result = str(result)
        docids = _extract_docids(result)
        for docid in docids:
            if docid not in self.retrieved_docids:
                self.retrieved_docids.append(docid)
        return self._transition(
            f"Tool result ({op}):\n{result}\n\n{self._render_phase()}",
            action_execution={
                "op": op,
                "status": "executed",
                "step": env_step,
                "arguments_sha256": _sha256_json(arguments),
            },
            tool_ops=(
                {
                    "op": op,
                    "step": env_step,
                    "retrieved_docids": docids,
                    "result_length": len(result),
                },
            ),
            reward_components=(
                {
                    "name": "browsecomp_search_transition",
                    "value": 0.0,
                    "op": op,
                    "step": env_step,
                },
            ),
            domain_evidence={
                **self._base_evidence(),
                "search_result_sha256": hashlib.sha256(
                    result.encode("utf-8")
                ).hexdigest(),
            },
        )

    def _submit_answer(
        self,
        payload: dict[str, Any],
        raw_action: str,
        env_step: int,
    ) -> DomainTransition:
        if set(payload) != {"answer"}:
            return self._invalid(
                raw_action,
                env_step,
                "SUBMIT_ANSWER expects only answer",
            )
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return self._invalid(
                raw_action,
                env_step,
                "answer must be a non-empty string",
            )
        phase = self._current_phase()
        try:
            judgement = self.judge(phase.question, answer, phase.answer)
            passed = _validate_judgement(judgement)
        except Exception as exc:
            return self._infra_error("SUBMIT_ANSWER", env_step, exc)

        answer_sha256 = hashlib.sha256(answer.encode("utf-8")).hexdigest()
        phase_before = self.phase_index
        submitted_iteration = self.phase_native_iteration_count
        verdict = {
            "phase_index": phase_before,
            "phase_kind": phase.kind,
            "correct": passed,
            "verdict_source": "memoryarena_llm_judge",
            "answer_sha256": answer_sha256,
            "judge_response_sha256": _sha256_json(judgement),
            "judge_confidence": judgement.get("confidence"),
            "judge_parse_error": False,
            "retrieved_docids": list(self.retrieved_docids),
        }
        self.phase_verdict_ledger.append(verdict)
        final_phase = phase.kind == "final"
        self.phase_native_iteration_count = 0

        if self.contract_mode == "paper_eval":
            self.phase_index += 1
            self.done = final_phase
            self.status = (
                "success"
                if final_phase and passed
                else "completed_incorrect"
                if final_phase
                else "active"
            )
            reward = 0.0
            phase_advanced = True
            terminal = final_phase
            component_name = "browsecomp_paper_eval_non_training"
            observation = (
                "The final Progressive Search response was privately evaluated."
                if terminal
                else "The response was privately evaluated. The next phase is ready."
                "\n\n" + self._render_phase()
            )
        else:
            phase_advanced = passed
            if passed:
                self.phase_index += 1
            terminal = (not passed) or (passed and final_phase)
            self.done = terminal
            self.status = (
                "success"
                if passed and final_phase
                else "failed_on_incorrect_answer"
                if not passed
                else "active"
            )
            reward = 1.0 if passed else 0.0
            component_name = (
                "browsecomp_failfast_phase_correct"
                if passed
                else "browsecomp_failfast_phase_incorrect"
            )
            observation = (
                "The final Progressive Search response was correct."
                if passed and final_phase
                else "The phase response was incorrect and the episode ended."
                if not passed
                else "The phase response was correct. The next phase is ready.\n\n"
                + self._render_phase()
            )
        if phase_advanced and not terminal:
            self.retrieved_docids = []
        evidence = self._base_evidence()
        evidence["phase_verdict_ledger"] = [
            dict(item) for item in self.phase_verdict_ledger
        ]
        if self.contract_mode == "paper_eval":
            evidence["paper_evaluation"] = self._paper_metrics_snapshot()
        return self._transition(
            observation,
            reward=reward,
            done=terminal,
            status=self.status,
            episode_success=terminal and final_phase and passed,
            action_execution={
                "op": "SUBMIT_ANSWER",
                "status": "committed_correct" if passed else "committed_incorrect",
                "step": env_step,
            },
            tool_ops=(
                {
                    "op": "SUBMIT_ANSWER",
                    "step": env_step,
                    "committed": True,
                    "phase_advanced": phase_advanced,
                    "phase_index": phase_before,
                    "phase_kind": phase.kind,
                    "native_iteration_index": submitted_iteration,
                    "terminal": terminal,
                    "submission_correct": passed,
                    "answer_sha256": answer_sha256,
                },
            ),
            reward_components=(
                {
                    "name": component_name,
                    "value": reward,
                    "op": "SUBMIT_ANSWER",
                    "step": env_step,
                },
            ),
            domain_evidence=evidence,
        )

    def _apply_phase_limit(
        self,
        transition: DomainTransition,
        env_step: int,
        *,
        phase_index_before: int,
    ) -> DomainTransition:
        if transition.done or self.phase_index != phase_index_before:
            return transition
        limit = self._current_phase_iteration_limit()
        if self.phase_native_iteration_count < limit:
            return self._with_iteration_evidence(transition, limit=limit)
        return self._phase_budget_exhausted(transition, env_step, limit=limit)

    def _phase_budget_exhausted(
        self,
        transition: DomainTransition,
        env_step: int,
        *,
        limit: int,
    ) -> DomainTransition:
        phase = self._current_phase()
        phase_before = self.phase_index
        verdict = {
            "phase_index": phase_before,
            "phase_kind": phase.kind,
            "correct": False,
            "verdict_source": "phase_budget_exhausted_without_submission",
            "answer_sha256": None,
            "judge_response_sha256": None,
            "judge_confidence": None,
            "judge_parse_error": False,
            "retrieved_docids": list(self.retrieved_docids),
        }
        self.phase_verdict_ledger.append(verdict)
        final_phase = phase.kind == "final"
        self.phase_native_iteration_count = 0
        if self.contract_mode == "paper_eval":
            self.phase_index += 1
            self.done = final_phase
            self.status = "completed_incorrect" if final_phase else "active"
            phase_advanced = True
            if not final_phase:
                self.retrieved_docids = []
            observation = (
                "The final phase exhausted its native action budget without a "
                "submission and was recorded incorrect."
                if final_phase
                else "The phase exhausted its native action budget without a "
                "submission and was recorded incorrect. The next phase is ready.\n\n"
                + self._render_phase()
            )
        else:
            self.done = True
            self.status = "failed_on_missing_answer"
            phase_advanced = False
            observation = (
                "The phase exhausted its native action budget without a submission; "
                "the fail-fast episode ended."
            )
        evidence = dict(transition.domain_evidence)
        evidence.update(self._base_evidence())
        evidence["phase_verdict_ledger"] = [
            dict(item) for item in self.phase_verdict_ledger
        ]
        if self.contract_mode == "paper_eval":
            evidence["paper_evaluation"] = self._paper_metrics_snapshot()
        action_execution = dict(transition.action_execution)
        action_execution.update(
            {
                "phase_budget_exhausted": True,
                "phase_advanced": phase_advanced,
                "terminal": self.done,
                "native_iteration_index": limit,
                "native_iteration_limit": limit,
            }
        )
        return self._transition(
            observation,
            reward=transition.reward,
            done=self.done,
            status=self.status,
            episode_success=False,
            action_execution=action_execution,
            tool_ops=tuple(transition.tool_ops)
            + (
                {
                    "op": "PHASE_BUDGET_EXHAUSTED",
                    "step": env_step,
                    "phase_index": phase_before,
                    "phase_kind": phase.kind,
                    "native_iteration_count": limit,
                    "native_iteration_limit": limit,
                    "phase_advanced": phase_advanced,
                    "terminal": self.done,
                    "submission_present": False,
                },
            ),
            reward_components=tuple(transition.reward_components)
            + (
                {
                    "name": "browsecomp_phase_budget_exhausted",
                    "value": 0.0,
                    "op": "PHASE_BUDGET_EXHAUSTED",
                    "step": env_step,
                },
            ),
            domain_evidence=evidence,
            sample_excluded=transition.sample_excluded,
        )

    def _with_iteration_evidence(
        self,
        transition: DomainTransition,
        *,
        limit: int,
    ) -> DomainTransition:
        action_execution = dict(transition.action_execution)
        action_execution.update(
            {
                "native_iteration_index": self.phase_native_iteration_count,
                "native_iteration_limit": limit,
            }
        )
        evidence = dict(transition.domain_evidence)
        evidence.update(
            {
                "native_iteration_count": self.phase_native_iteration_count,
                "native_iteration_limit": limit,
            }
        )
        return self._transition(
            transition.observation,
            reward=transition.reward,
            done=transition.done,
            status=transition.status,
            episode_success=transition.episode_success,
            action_execution=action_execution,
            tool_ops=transition.tool_ops,
            reward_components=transition.reward_components,
            domain_evidence=evidence,
            sample_excluded=transition.sample_excluded,
        )

    def _current_phase_iteration_limit(self) -> int:
        return (
            BROWSECOMP_FINAL_MAX_ITERATIONS
            if self._current_phase().kind == "final"
            else BROWSECOMP_SUBQUERY_MAX_ITERATIONS
        )

    def _invalid(
        self, raw_action: str, env_step: int, message: str
    ) -> DomainTransition:
        return self._transition(
            f"Invalid action: {message}\n\n{self._render_phase()}",
            action_execution={
                "op": "INVALID",
                "status": "invalid",
                "step": env_step,
                "attempted_action_sha256": hashlib.sha256(
                    raw_action.encode("utf-8")
                ).hexdigest(),
            },
            reward_components=(
                {
                    "name": "invalid_action",
                    "value": 0.0,
                    "op": "INVALID",
                    "step": env_step,
                },
            ),
            domain_evidence=self._base_evidence(),
        )

    def _infra_error(self, op: str, env_step: int, exc: Exception) -> DomainTransition:
        self.done = True
        self.status = "infrastructure_error"
        return self._transition(
            "Progressive Search encountered an infrastructure error while executing "
            f"{op}; this sample is excluded.",
            done=True,
            status=self.status,
            action_execution={"op": op, "status": "error", "step": env_step},
            tool_ops=(
                {
                    "op": "INFRA_ERROR",
                    "attempted_op": op,
                    "step": env_step,
                    "sample_excluded": True,
                    "error_type": type(exc).__name__,
                },
            ),
            reward_components=(
                {
                    "name": "infrastructure_error_excluded",
                    "value": 0.0,
                    "op": op,
                    "step": env_step,
                    "error_type": type(exc).__name__,
                },
            ),
            domain_evidence=self._base_evidence(),
            sample_excluded=True,
        )

    def _render_phase(self) -> str:
        task = self._require_task()
        phase = self._current_phase()
        label = (
            f"Dependent subquery {self.phase_index + 1}/{len(task.phases) - 1}"
            if phase.kind == "subquery"
            else "Final combined query"
        )
        return "\n\n".join(
            [
                (
                    "Task family: progressive_search_public221\n"
                    f"Contract: {self.contract_mode}\n"
                    f"Workflow progress: {self.phase_index}/{len(task.phases)}\n"
                    f"Phase: {label}\n"
                    "Action granularity: one policy action per turn; no upstream "
                    "batched-turn parity\n"
                    "Native iteration budget: "
                    f"{self.phase_native_iteration_count}/"
                    f"{self._current_phase_iteration_limit()}"
                ),
                f"Question: {phase.question}",
                (
                    "Response format for SUBMIT_ANSWER:\n"
                    "Explanation: evidence-backed explanation with [docid] citations\n"
                    "Exact Answer: succinct final answer\n"
                    "Confidence: 0% to 100%"
                ),
            ]
        )

    def _paper_metrics_snapshot(self) -> dict[str, Any]:
        task = self._require_task()
        verdicts = [dict(item) for item in self.phase_verdict_ledger]
        correct = sum(item["correct"] is True for item in verdicts)
        complete = self.done and len(verdicts) == len(task.phases)
        final_success = verdicts[-1]["correct"] is True if complete else None
        return {
            "metric_contract": (
                "memoryarena_progressive_search_ps_sr_at_k_final_sr_v1"
            ),
            "dataset_scope": "public221_of_paper256",
            "query_id": task.query_id,
            "complete": complete,
            "metric_scale": "unit_interval",
            "phase_verdicts": verdicts,
            "completed_phase_count": len(verdicts),
            "process_score_numerator": correct,
            "process_score_denominator": len(task.phases),
            "process_score": correct / len(task.phases),
            "sr_at_k": {
                str(index + 1): {
                    "correct": item["correct"] is True,
                    "numerator": int(item["correct"] is True),
                    "denominator": 1,
                }
                for index, item in enumerate(verdicts)
            },
            "final_sr_numerator": int(final_success) if complete else None,
            "final_sr_denominator": 1 if complete else None,
            "final_success": final_success,
            "online_reward_is_separate": True,
        }

    def _base_evidence(self) -> dict[str, Any]:
        return {
            "query_id": self._require_task().query_id,
            "contract_mode": self.contract_mode,
            "retrieved_docids": list(self.retrieved_docids),
        }

    def _current_phase(self) -> BrowseCompPhase:
        task = self._require_task()
        if self.phase_index >= len(task.phases):
            return task.phases[-1]
        return task.phases[self.phase_index]

    def _require_task(self) -> BrowseCompTask:
        if self.task is None:
            raise RuntimeError("BrowseComp driver must be reset before use")
        return self.task

    def _transition(
        self,
        observation: str,
        *,
        reward: float = 0.0,
        done: bool | None = None,
        status: str | None = None,
        episode_success: bool = False,
        action_execution=None,
        tool_ops=(),
        reward_components=(),
        domain_evidence=None,
        sample_excluded: bool = False,
    ) -> DomainTransition:
        task = self._require_task()
        return DomainTransition(
            observation=observation,
            reward=reward,
            done=self.done if done is None else done,
            status=self.status if status is None else status,
            phase_index=self.phase_index,
            phase_count=len(task.phases),
            episode_success=episode_success,
            action_execution=action_execution or {},
            tool_ops=tool_ops,
            reward_components=reward_components,
            domain_evidence=domain_evidence or self._base_evidence(),
            sample_excluded=sample_excluded,
        )


def load_browsecomp_tasks(tasks_path: Path) -> tuple[BrowseCompTask, ...]:
    """Load only the frozen MemoryArena progressive_search schema."""

    if not tasks_path.is_file():
        raise FileNotFoundError(f"Progressive Search dataset not found: {tasks_path}")
    rows = _read_jsonl_rows(tasks_path, label="Progressive Search dataset")
    if not rows:
        raise ValueError(f"no Progressive Search tasks loaded from {tasks_path}")
    tasks: list[BrowseCompTask] = []
    seen: set[str] = set()
    for line_number, payload in rows:
        if not {"id", "questions", "answers"}.issubset(payload):
            raise ValueError(
                f"Progressive Search row {line_number} requires id/questions/answers"
            )
        query_id = str(payload["id"]).strip()
        questions = payload["questions"]
        answers = payload["answers"]
        if not query_id or query_id in seen:
            raise ValueError(
                f"Progressive Search row {line_number} has empty or duplicate id"
            )
        if not isinstance(questions, list) or not isinstance(answers, list):
            raise ValueError(
                f"Progressive Search row {line_number} requires list questions/answers"
            )
        if not questions or len(questions) != len(answers):
            raise ValueError(
                f"Progressive Search row {line_number} has misaligned questions/answers"
            )
        if any(not isinstance(value, str) or not value.strip() for value in questions):
            raise ValueError(f"Progressive Search row {query_id} has invalid question")
        if any(not isinstance(value, str) or not value.strip() for value in answers):
            raise ValueError(f"Progressive Search row {query_id} has invalid answer")
        phases = tuple(
            BrowseCompPhase(
                kind="final" if index == len(questions) - 1 else "subquery",
                question=question,
                answer=answer,
            )
            for index, (question, answer) in enumerate(zip(questions, answers))
        )
        tasks.append(BrowseCompTask(query_id=query_id, phases=phases))
        seen.add(query_id)
    return tuple(tasks)


def aggregate_browsecomp_paper_metrics(
    task_snapshots: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the paper's task-macro PS, SR@k, and final-phase SR."""

    if not task_snapshots:
        raise ValueError("at least one completed paper-eval snapshot is required")
    process_scores: list[float] = []
    final_successes = 0
    depth_correct: dict[int, int] = {}
    depth_eligible: dict[int, int] = {}
    for task_index, snapshot in enumerate(task_snapshots):
        if not isinstance(snapshot, dict) or snapshot.get("complete") is not True:
            raise ValueError(f"paper-eval snapshot {task_index} is incomplete")
        if (
            snapshot.get("metric_contract")
            != "memoryarena_progressive_search_ps_sr_at_k_final_sr_v1"
            or snapshot.get("dataset_scope") != "public221_of_paper256"
            or snapshot.get("metric_scale") != "unit_interval"
        ):
            raise ValueError(
                f"paper-eval snapshot {task_index} has the wrong metric contract"
            )
        verdicts = snapshot.get("phase_verdicts")
        phase_count = snapshot.get("process_score_denominator")
        if (
            not isinstance(verdicts, list)
            or isinstance(phase_count, bool)
            or not isinstance(phase_count, int)
            or phase_count < 1
            or len(verdicts) != phase_count
        ):
            raise ValueError(f"paper-eval snapshot {task_index} has invalid ledger")
        correct_flags: list[bool] = []
        for phase_index, verdict in enumerate(verdicts):
            if (
                not isinstance(verdict, dict)
                or verdict.get("phase_index") != phase_index
                or type(verdict.get("correct")) is not bool
            ):
                raise ValueError(
                    f"paper-eval snapshot {task_index} has invalid phase verdict"
                )
            correct_flags.append(verdict["correct"])
        process_scores.append(sum(correct_flags) / phase_count)
        final_successes += int(correct_flags[-1])
        for depth, correct in enumerate(correct_flags, start=1):
            depth_eligible[depth] = depth_eligible.get(depth, 0) + 1
            depth_correct[depth] = depth_correct.get(depth, 0) + int(correct)
    process_score_sum = sum(process_scores)
    return {
        "metric_contract": "memoryarena_progressive_search_ps_sr_at_k_final_sr_v1",
        "dataset_scope": "public221_of_paper256",
        "metric_scale": "unit_interval",
        "task_count": len(task_snapshots),
        "process_score_numerator": process_score_sum,
        "process_score_denominator": len(process_scores),
        "process_score": process_score_sum / len(process_scores),
        "sr_at_k": {
            str(depth): {
                "correct_tasks": depth_correct[depth],
                "eligible_tasks": depth_eligible[depth],
                "rate": depth_correct[depth] / depth_eligible[depth],
            }
            for depth in sorted(depth_eligible)
        },
        "final_sr_numerator": final_successes,
        "final_sr_denominator": len(task_snapshots),
        "final_success_rate": final_successes / len(task_snapshots),
        "paper_panel_complete": False,
    }


def attest_browsecomp_cross_source_parity(
    tasks_path: Path,
    reference_path: Path,
    *,
    expected_reference_sha256: str = BROWSECOMP_JOANNA_REFERENCE_SHA256,
) -> dict[str, Any]:
    """Audit public221 against Joanna690 questions/answers, ignoring row IDs."""

    tasks = load_browsecomp_tasks(tasks_path)
    observed_reference_sha256 = _sha256_file(reference_path)
    if observed_reference_sha256 != expected_reference_sha256:
        raise RuntimeError("Joanna690 BrowseComp reference hash mismatch")
    reference_rows = _read_jsonl_rows(
        reference_path,
        label="Joanna690 BrowseComp reference",
    )
    if len(reference_rows) != len(tasks):
        raise RuntimeError("BrowseComp cross-source task count mismatch")
    for row_index, ((_, row), task) in enumerate(zip(reference_rows, tasks)):
        questions = row.get("question", row.get("questions"))
        answers = row.get("answer", row.get("answers"))
        if (
            not isinstance(questions, list)
            or not isinstance(answers, list)
            or questions != [phase.question for phase in task.phases]
            or answers != [phase.answer for phase in task.phases]
        ):
            raise RuntimeError(
                "BrowseComp cross-source questions/answers mismatch at public row "
                f"{row_index}"
            )
    return {
        "reference": BROWSECOMP_JOANNA_REFERENCE,
        "reference_sha256": observed_reference_sha256,
        "relation": "questions_and_answers_equal_in_order_ignoring_id",
        "task_count": len(tasks),
    }


def _read_jsonl_rows(path: Path, *, label: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank {label} row on line {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {label} JSON on line {line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{label} line {line_number} must be a JSON object")
            rows.append((line_number, payload))
    return rows


def _parse_action(action: str) -> tuple[str, dict[str, Any]] | None:
    match = _ACTION_RE.fullmatch(action.strip())
    if match is None:
        return None
    try:
        payload = json.loads(match.group(2))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return match.group(1), payload


def _extract_docids(result: str) -> list[str]:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else [payload]
    docids = []
    for item in items:
        if isinstance(item, dict) and item.get("docid") is not None:
            docids.append(str(item["docid"]))
    return docids


def _validate_judgement(judgement: Any) -> bool:
    if not isinstance(judgement, dict):
        raise RuntimeError("BrowseComp judge returned a non-object response")
    if judgement.get("parse_error") is True:
        raise RuntimeError("BrowseComp judge response could not be parsed")
    if type(judgement.get("correct")) is not bool:
        raise RuntimeError("BrowseComp judge response lacks a boolean verdict")
    return judgement["correct"]


def _build_upstream_search(
    memoryarena_root: Path,
    *,
    index_path: str | Path | None,
    corpus_path: str | Path | None,
    embedding_model: str,
    provider: str,
    embedding_endpoint: str | None,
) -> BrowseSearch:
    normalized_endpoint = _normalize_embedding_endpoint(embedding_endpoint)
    if provider == "openai":
        observed_endpoint = _normalize_embedding_endpoint(
            os.environ.get("OPENAI_BASE_URL")
        )
        if observed_endpoint != normalized_endpoint:
            raise RuntimeError(
                "BrowseComp embedding endpoint differs from OPENAI_BASE_URL"
            )
    elif normalized_endpoint != BROWSECOMP_OPENROUTER_ENDPOINT:
        raise RuntimeError(
            "BrowseComp OpenRouter endpoint differs from the frozen upstream route"
        )
    search_agent_dir = memoryarena_root / "env/env_systems/web_search_env/search_agent"
    for path in (memoryarena_root, search_agent_dir):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    client_module = importlib.import_module(
        "env.env_systems.web_search_env.search_agent.openai_client"
    )
    searcher_module = importlib.import_module(
        "env.env_systems.web_search_env.searcher.searchers.openai_searcher"
    )
    _require_module_under_root(client_module, memoryarena_root)
    _require_module_under_root(searcher_module, memoryarena_root)
    resolved_index = _resolve_search_path(
        memoryarena_root,
        index_path or "web_search_env/embeddings/shard*.index",
    )
    resolved_corpus = _resolve_search_path(
        memoryarena_root,
        corpus_path or "web_search_env/data/corpus.jsonl",
    )
    args = SimpleNamespace(
        index_path=resolved_index,
        id_map_path=resolved_index.replace(".index", "_id_map.json"),
        corpus_path=resolved_corpus,
        openai_model=embedding_model,
        provider=provider,
    )
    searcher = searcher_module.OpenAISearcher(args)
    validate_loaded_browsecomp_searcher(searcher, embedding_model=embedding_model)

    # Construct without truncation so upstream cannot trigger its unpinned
    # AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B") call.
    handler = client_module.SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=None,
        k=BROWSECOMP_FROZEN_SEARCH_K,
        include_get_document=True,
        allowed_docids=None,
        step_memory_client=None,
    )
    handler.snippet_max_tokens = BROWSECOMP_FROZEN_SNIPPET_TOKENS
    handler.tokenizer = _load_frozen_snippet_tokenizer()

    def execute(tool_name: str, arguments: dict[str, Any]) -> str:
        return handler.execute_tool(tool_name, arguments)

    return execute


def _normalize_embedding_endpoint(endpoint: str | None) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise RuntimeError("BrowseComp embedding endpoint must be explicit")
    normalized = endpoint.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("BrowseComp embedding endpoint must be an HTTP(S) URL")
    return normalized


def _embedding_route_provenance(
    *,
    provider: str,
    model: str,
    endpoint: str | None,
    contract_mode: str,
) -> dict[str, Any]:
    normalized_endpoint = _normalize_embedding_endpoint(endpoint)
    if (
        provider == "openrouter"
        and normalized_endpoint != BROWSECOMP_OPENROUTER_ENDPOINT
    ):
        raise RuntimeError(
            "BrowseComp OpenRouter endpoint differs from the frozen upstream route"
        )
    route_variant = (
        "paper_eval_openai_embedding_v1"
        if contract_mode == "paper_eval"
        else "failfast_openai_embedding_v1"
        if provider == "openai"
        else "failfast_openrouter_nonpaper_embedding_v1"
    )
    public_config = {
        "provider": provider,
        "model": model,
        "endpoint_sha256": hashlib.sha256(
            normalized_endpoint.encode("utf-8")
        ).hexdigest(),
        "route_variant": route_variant,
    }
    return {
        "mode": "explicit_hashed_embedding_route",
        **public_config,
        "config_sha256": _sha256_json(public_config),
    }


def _load_frozen_snippet_tokenizer():
    snapshot_path = _resolve_frozen_tokenizer_snapshot()
    attest_frozen_snippet_tokenizer(snapshot_path)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Frozen BrowseComp snippet tokenizer is not available in the local "
            "Transformers cache; refusing an unpinned network download"
        ) from exc


def _resolve_frozen_tokenizer_snapshot() -> Path:
    explicit = os.environ.get("AGENTMEMORY_BROWSECOMP_TOKENIZER_SNAPSHOT_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        hub_root = Path(hub_cache).expanduser()
    else:
        hf_home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        ).expanduser()
        hub_root = hf_home / "hub"
    return (
        hub_root
        / "models--Qwen--Qwen3-0.6B"
        / "snapshots"
        / BROWSECOMP_FROZEN_TOKENIZER_REVISION
    ).resolve()


def attest_frozen_snippet_tokenizer(
    snapshot_path: Path,
    *,
    expected_files_sha256: dict[str, str] | None = None,
    expected_bundle_sha256: str | None = None,
) -> dict[str, Any]:
    expected_files = dict(
        expected_files_sha256 or BROWSECOMP_FROZEN_TOKENIZER_FILES_SHA256
    )
    expected_bundle = (
        expected_bundle_sha256
        if expected_bundle_sha256 is not None
        else BROWSECOMP_FROZEN_TOKENIZER_BUNDLE_SHA256
    )
    root = snapshot_path.expanduser().resolve()
    observed = {}
    for relative_path, expected_sha256 in expected_files.items():
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(
                f"Frozen BrowseComp tokenizer file is missing: {relative_path}"
            )
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(
                f"Frozen BrowseComp tokenizer hash mismatch: {relative_path}"
            )
        observed[relative_path] = observed_sha256
    bundle_sha256 = _sha256_json(observed)
    if bundle_sha256 != expected_bundle:
        raise RuntimeError("Frozen BrowseComp tokenizer bundle hash mismatch")
    return {
        "repository": BROWSECOMP_FROZEN_TOKENIZER_REPOSITORY,
        "revision": BROWSECOMP_FROZEN_TOKENIZER_REVISION,
        "files_sha256": observed,
        "bundle_sha256": bundle_sha256,
    }


def attest_browsecomp_search_assets(
    *,
    index_pattern: str,
    corpus_path: Path,
    corpus_manifest_path: Path,
    embedding_model: str,
    expected_index_shards: Sequence[dict[str, Any]] | None = None,
    expected_corpus_shards: Sequence[str] | None = None,
    expected_document_count: int = BROWSECOMP_FROZEN_DOCUMENT_COUNT,
    expected_corpus_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify exact FAISS/id-map bytes and a canonically pinned corpus."""

    if embedding_model != BROWSECOMP_FROZEN_EMBEDDING_MODEL:
        raise RuntimeError(
            "Frozen BrowseComp index requires embedding model "
            f"{BROWSECOMP_FROZEN_EMBEDDING_MODEL!r}, got {embedding_model!r}"
        )
    canonical_corpus_sha256 = (
        expected_corpus_sha256
        if expected_corpus_sha256 is not None
        else BROWSECOMP_FROZEN_MATERIALIZED_CORPUS_SHA256
    )
    if (
        not isinstance(canonical_corpus_sha256, str)
        or _SHA256_RE.fullmatch(canonical_corpus_sha256) is None
    ):
        raise RuntimeError(
            "Frozen materialized BrowseComp corpus SHA256 is not configured"
        )
    shard_specs = tuple(expected_index_shards or BROWSECOMP_FROZEN_INDEX_SHARDS)
    corpus_shas = tuple(expected_corpus_shards or BROWSECOMP_FROZEN_CORPUS_SHARDS)
    index_paths = tuple(
        Path(value).resolve() for value in sorted(glob.glob(index_pattern))
    )
    expected_names = tuple(str(spec["name"]) for spec in shard_specs)
    observed_names = tuple(path.name for path in index_paths)
    if observed_names != expected_names:
        raise RuntimeError(
            "BrowseComp FAISS shard set is incomplete or unexpected: "
            f"expected={expected_names} observed={observed_names}"
        )

    all_ids: list[str] = []
    shard_evidence = []
    for path, spec in zip(index_paths, shard_specs):
        id_map_path = path.with_name(path.stem + "_id_map.json")
        if not id_map_path.is_file():
            raise RuntimeError(f"BrowseComp FAISS index lacks id map: {id_map_path}")
        index_sha256 = _sha256_file(path)
        id_map_sha256 = _sha256_file(id_map_path)
        if index_sha256 != spec["index_sha256"]:
            raise RuntimeError(f"BrowseComp FAISS index hash mismatch: {path}")
        if id_map_sha256 != spec["id_map_sha256"]:
            raise RuntimeError(f"BrowseComp FAISS id-map hash mismatch: {id_map_path}")
        try:
            with id_map_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"BrowseComp FAISS id map is unreadable: {id_map_path}"
            ) from exc
        ids = payload.get("ids") if isinstance(payload, dict) else payload
        if not isinstance(ids, list) or any(
            not isinstance(docid, str) or not docid for docid in ids
        ):
            raise RuntimeError(f"BrowseComp FAISS id map is malformed: {id_map_path}")
        if len(ids) != int(spec["vector_count"]):
            raise RuntimeError(f"BrowseComp FAISS id-map count mismatch: {id_map_path}")
        all_ids.extend(ids)
        shard_evidence.append(
            {
                "name": path.name,
                "index_sha256": index_sha256,
                "id_map_sha256": id_map_sha256,
                "vector_count": len(ids),
            }
        )
    expected_ids = {str(index) for index in range(expected_document_count)}
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != expected_ids:
        raise RuntimeError("BrowseComp FAISS id maps do not uniquely cover the corpus")

    try:
        with corpus_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("BrowseComp corpus manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("BrowseComp corpus manifest must be a JSON object")
    if manifest.get("format") != "agentmemory_browsecomp_corpus_manifest_v2":
        raise RuntimeError("BrowseComp corpus manifest has an unsupported format")
    if manifest.get("path_base") != "manifest_directory":
        raise RuntimeError("BrowseComp corpus manifest has a non-portable path base")
    source = manifest.get("source")
    projection = manifest.get("projection")
    output = manifest.get("output")
    if not all(isinstance(item, dict) for item in (source, projection, output)):
        raise RuntimeError("BrowseComp corpus manifest is incomplete")
    assert isinstance(source, dict) and isinstance(projection, dict)
    assert isinstance(output, dict)
    if (
        source.get("repository") != BROWSECOMP_FROZEN_CORPUS_REPOSITORY
        or source.get("revision") != BROWSECOMP_FROZEN_CORPUS_REVISION
    ):
        raise RuntimeError(
            "BrowseComp corpus manifest repository/revision does not match the freeze"
        )
    if source.get("columns") != ["docid", "text", "url"]:
        raise RuntimeError("BrowseComp corpus manifest source schema does not match")
    if projection.get("output_columns") != ["docid", "text"]:
        raise RuntimeError("BrowseComp corpus manifest projection does not match")
    source_files = source.get("files")
    if (
        not isinstance(source_files, list)
        or len(source_files) != len(corpus_shas)
        or any(not isinstance(item, dict) for item in source_files)
    ):
        raise RuntimeError("BrowseComp corpus manifest lacks source files")
    if source.get("file_count") != len(source_files):
        raise RuntimeError("BrowseComp corpus manifest source file count is invalid")

    source_evidence = []
    observed_source_shas = []
    observed_source_paths: set[Path] = set()
    for item in source_files:
        assert isinstance(item, dict)
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise RuntimeError("BrowseComp corpus manifest source path is invalid")
        source_path = Path(raw_path).expanduser()
        if source_path.is_absolute():
            raise RuntimeError("BrowseComp corpus source paths must be relative")
        source_path = corpus_manifest_path.parent / source_path
        source_path = source_path.resolve()
        if source_path in observed_source_paths or not source_path.is_file():
            raise RuntimeError(
                "BrowseComp corpus source shard path is missing or duplicate"
            )
        observed_source_paths.add(source_path)
        actual_sha256 = _sha256_file(source_path)
        actual_size = source_path.stat().st_size
        if item.get("sha256") != actual_sha256:
            raise RuntimeError(
                "BrowseComp corpus source shard hash is self-inconsistent"
            )
        if item.get("size_bytes") != actual_size:
            raise RuntimeError(
                "BrowseComp corpus source shard byte count is inconsistent"
            )
        if item.get("columns") != ["docid", "text", "url"]:
            raise RuntimeError("BrowseComp corpus source shard schema is inconsistent")
        row_count = item.get("row_count")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
        ):
            raise RuntimeError("BrowseComp corpus source shard row count is invalid")
        observed_source_shas.append(actual_sha256)
        source_evidence.append(
            {
                "name": source_path.name,
                "sha256": actual_sha256,
                "size_bytes": actual_size,
                "row_count": row_count,
            }
        )
    if tuple(sorted(observed_source_shas)) != tuple(sorted(corpus_shas)):
        raise RuntimeError(
            "BrowseComp corpus source shard hashes do not match the freeze"
        )

    corpus_path = corpus_path.expanduser().resolve()
    output_path = output.get("path")
    if not isinstance(output_path, str) or not output_path.strip():
        raise RuntimeError("BrowseComp corpus manifest output path is invalid")
    manifest_output_path = Path(output_path).expanduser()
    if manifest_output_path.is_absolute():
        raise RuntimeError("BrowseComp corpus output path must be relative")
    manifest_output_path = (
        corpus_manifest_path.parent / manifest_output_path
    ).resolve()
    if manifest_output_path != corpus_path:
        raise RuntimeError(
            "BrowseComp corpus manifest output path does not match runtime"
        )
    corpus_sha256, corpus_docids = _inspect_materialized_corpus(corpus_path)
    if output.get("sha256") != corpus_sha256:
        raise RuntimeError("BrowseComp corpus hash does not match its manifest")
    if corpus_sha256 != canonical_corpus_sha256:
        raise RuntimeError("BrowseComp corpus hash does not match the canonical freeze")
    if output.get("row_count") != expected_document_count:
        raise RuntimeError(
            "BrowseComp corpus manifest row count does not match the freeze"
        )
    if corpus_docids != expected_ids:
        raise RuntimeError(
            "BrowseComp corpus does not exactly cover indexed document IDs"
        )

    return {
        "mode": "frozen_public_assets",
        "embedding_model": embedding_model,
        "embedding_dimension": BROWSECOMP_FROZEN_INDEX_DIMENSION,
        "document_count": expected_document_count,
        "index_repository": BROWSECOMP_FROZEN_INDEX_REPOSITORY,
        "index_revision": BROWSECOMP_FROZEN_INDEX_REVISION,
        "index_shards": shard_evidence,
        "corpus_repository": BROWSECOMP_FROZEN_CORPUS_REPOSITORY,
        "corpus_revision": BROWSECOMP_FROZEN_CORPUS_REVISION,
        "corpus_source_shards": source_evidence,
        "corpus_sha256": corpus_sha256,
        "corpus_manifest_sha256": _sha256_file(corpus_manifest_path),
    }


def _inspect_materialized_corpus(path: Path) -> tuple[str, set[str]]:
    if not path.is_file():
        raise RuntimeError(f"BrowseComp corpus does not exist: {path}")
    digest = hashlib.sha256()
    docids: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                raise RuntimeError(f"BrowseComp corpus has blank row {line_number}")
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"BrowseComp corpus has invalid JSON row {line_number}"
                ) from exc
            if not isinstance(row, dict) or set(row) != {"docid", "text"}:
                raise RuntimeError(
                    f"BrowseComp corpus row {line_number} has invalid projection"
                )
            docid = row["docid"]
            text = row["text"]
            if (
                not isinstance(docid, str)
                or not docid
                or not isinstance(text, str)
                or not text
                or docid in docids
            ):
                raise RuntimeError(
                    f"BrowseComp corpus row {line_number} is malformed or duplicate"
                )
            docids.add(docid)
    return digest.hexdigest(), docids


def validate_loaded_browsecomp_searcher(
    searcher: Any,
    *,
    embedding_model: str,
    expected_index_shards: Sequence[dict[str, Any]] | None = None,
    expected_document_count: int = BROWSECOMP_FROZEN_DOCUMENT_COUNT,
) -> None:
    if embedding_model != BROWSECOMP_FROZEN_EMBEDDING_MODEL:
        raise RuntimeError("BrowseComp searcher uses an incompatible embedding model")
    shard_specs = tuple(expected_index_shards or BROWSECOMP_FROZEN_INDEX_SHARDS)
    indexes = tuple(getattr(searcher, "indexes", ()))
    id_maps = tuple(getattr(searcher, "id_maps", ()))
    corpus = getattr(searcher, "docid_to_text", None)
    if len(indexes) != len(shard_specs) or len(id_maps) != len(shard_specs):
        raise RuntimeError("BrowseComp searcher did not load the complete shard set")
    all_ids: list[str] = []
    for index, ids, spec in zip(indexes, id_maps, shard_specs):
        if int(getattr(index, "d", -1)) != BROWSECOMP_FROZEN_INDEX_DIMENSION:
            raise RuntimeError("BrowseComp FAISS dimension does not match the freeze")
        vector_count = int(spec["vector_count"])
        if (
            int(getattr(index, "ntotal", -1)) != vector_count
            or len(ids) != vector_count
        ):
            raise RuntimeError("BrowseComp FAISS vectors and id map are misaligned")
        all_ids.extend(str(docid) for docid in ids)
    expected_ids = {str(index) for index in range(expected_document_count)}
    if len(all_ids) != expected_document_count or set(all_ids) != expected_ids:
        raise RuntimeError("BrowseComp loaded id maps do not exactly cover documents")
    if not isinstance(corpus, dict) or set(corpus) != expected_ids:
        raise RuntimeError(
            "BrowseComp loaded corpus does not exactly cover indexed documents"
        )


def _normalize_judge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise RuntimeError("BrowseComp judge_config must be an explicit mapping")
    required = ("backend", "model_name", "base_url", "max_tokens")
    missing = [key for key in required if key not in config]
    if missing:
        raise RuntimeError(
            "BrowseComp judge_config lacks fields: " + ", ".join(missing)
        )
    backend = str(config["backend"]).strip().lower()
    model_name = str(config["model_name"]).strip()
    base_url = str(config["base_url"]).strip().rstrip("/")
    if backend != "openai_responses":
        raise RuntimeError("BrowseComp judge backend must be openai_responses")
    if not model_name or not base_url:
        raise RuntimeError("BrowseComp judge configuration cannot contain blanks")
    try:
        max_tokens = int(config["max_tokens"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("BrowseComp judge max_tokens is invalid") from exc
    if max_tokens < 1:
        raise RuntimeError("BrowseComp judge max_tokens must be positive")
    return {
        "backend": backend,
        "model_name": model_name,
        "base_url": base_url,
        "max_tokens": max_tokens,
    }


def _judge_provenance(config: dict[str, Any], *, mode: str) -> dict[str, Any]:
    payload = {
        "mode": mode,
        "backend": config["backend"],
        "model": config["model_name"],
        "max_tokens": config["max_tokens"],
        "endpoint_sha256": hashlib.sha256(
            config["base_url"].encode("utf-8")
        ).hexdigest(),
        "prompt_template_sha256": BROWSECOMP_JUDGE_PROMPT_TEMPLATE_SHA256,
    }
    payload["config_sha256"] = _sha256_json(payload)
    return payload


def _injected_judge_provenance() -> dict[str, Any]:
    payload = {
        "mode": "injected_test_double",
        "backend": "injected_test_double",
        "model": "injected_test_double",
        "max_tokens": None,
        "endpoint_sha256": None,
        "prompt_template_sha256": BROWSECOMP_JUDGE_PROMPT_TEMPLATE_SHA256,
    }
    payload["config_sha256"] = _sha256_json(payload)
    return payload


def _build_upstream_judge(
    memoryarena_root: Path,
    *,
    config: dict[str, Any],
) -> BrowseJudge:
    web_search_root = memoryarena_root / "env/env_systems/web_search_env"
    for path in (memoryarena_root, web_search_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module("env.env_systems.browsecomp_plus_env")
    prompts_module = importlib.import_module("search_agent.prompts")
    _require_module_under_root(module, memoryarena_root)
    _require_module_under_root(prompts_module, memoryarena_root)
    prompt_sha256 = hashlib.sha256(
        prompts_module.GRADER_TEMPLATE.encode("utf-8")
    ).hexdigest()
    if prompt_sha256 != BROWSECOMP_JUDGE_PROMPT_TEMPLATE_SHA256:
        raise RuntimeError(
            "MemoryArena BrowseComp judge prompt does not match the freeze"
        )
    import openai

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=config["base_url"],
    )

    def judge(question: str, predicted_answer: str, correct_answer: str):
        return module.BrowseCompPlusEnvironment.evaluate_answer_with_judge(
            client,
            question,
            predicted_answer,
            correct_answer,
            model=config["model_name"],
            max_output_tokens=config["max_tokens"],
        )

    return judge


def _resolve_search_path(memoryarena_root: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    path_text = str(path)
    if path_text.startswith("env/") or path_text.startswith("env\\"):
        return str((memoryarena_root / path).resolve())
    return str((memoryarena_root / "env/env_systems" / path).resolve())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("BrowseComp evidence is not JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def attest_browsecomp_upstream(
    memoryarena_root: Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    root = memoryarena_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"MemoryArena root is not a git worktree: {root}")
    commit = _git(root, "rev-parse", "HEAD").strip()
    if commit != expected_commit:
        raise RuntimeError(
            "MemoryArena commit mismatch for BrowseComp: "
            f"expected {expected_commit}, observed {commit}"
        )
    relative_paths = _browsecomp_executable_source_paths(root, expected_commit)
    source_sha256 = {}
    for relative_path in relative_paths:
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing MemoryArena BrowseComp source file: {path}")
        observed_sha256 = _sha256_file(path)
        committed_sha256 = hashlib.sha256(
            _git_bytes(root, "show", f"{expected_commit}:{relative_path}")
        ).hexdigest()
        if observed_sha256 != committed_sha256:
            raise RuntimeError(
                "MemoryArena BrowseComp executable source is not pristine: "
                f"{relative_path}"
            )
        source_sha256[relative_path] = observed_sha256
    return {
        "mode": "pinned_pristine_upstream",
        "memoryarena_commit": commit,
        "source_files_sha256": source_sha256,
        "source_bundle_sha256": _sha256_json(source_sha256),
    }


def _browsecomp_executable_source_paths(
    root: Path,
    expected_commit: str,
) -> tuple[str, ...]:
    pathspecs = (*BROWSECOMP_UPSTREAM_REQUIRED_FILES, *BROWSECOMP_UPSTREAM_SOURCE_ROOTS)
    tracked = {
        line.strip()
        for line in _git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            expected_commit,
            "--",
            *pathspecs,
        ).splitlines()
        if line.strip().endswith(".py")
    }
    required = set(BROWSECOMP_UPSTREAM_REQUIRED_FILES)
    if not required <= tracked:
        raise RuntimeError(
            "Pinned MemoryArena commit lacks required BrowseComp executable files: "
            + ", ".join(sorted(required - tracked))
        )

    filesystem = set(required)
    for relative_root in BROWSECOMP_UPSTREAM_SOURCE_ROOTS:
        source_root = root / relative_root
        if not source_root.is_dir():
            raise RuntimeError(
                f"Missing MemoryArena BrowseComp source directory: {source_root}"
            )
        filesystem.update(
            path.relative_to(root).as_posix()
            for path in source_root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    if filesystem != tracked:
        raise RuntimeError(
            "MemoryArena BrowseComp executable source set is not pristine: "
            f"untracked={sorted(filesystem - tracked)} "
            f"missing={sorted(tracked - filesystem)}"
        )
    return tuple(sorted(tracked))


def _git(root: Path, *args: str) -> str:
    command = ["git", "-c", f"safe.directory={root}", "-C", str(root), *args]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise RuntimeError(
            f"Cannot attest MemoryArena BrowseComp source at {root}: {stderr.strip()}"
        ) from exc


def _git_bytes(root: Path, *args: str) -> bytes:
    command = ["git", "-c", f"safe.directory={root}", "-C", str(root), *args]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        raise RuntimeError(
            "Cannot read pinned MemoryArena BrowseComp source at "
            f"{root}: {stderr.decode('utf-8', errors='replace').strip()}"
        ) from exc


def _require_module_under_root(module: Any, root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"Imported module {module.__name__!r} has no source path")
    path = Path(module_file).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Imported {module.__name__} from the wrong MemoryArena root: {path}"
        ) from exc
