from __future__ import annotations

import hashlib
import importlib
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from ..runtime.domain import DomainContract, DomainTransition


BROWSECOMP_SURFACE = "memoryarena_browsecomp_plus_v3"
BROWSECOMP_DOMAIN_ID = "browsecomp_plus"
BROWSECOMP_SUBQUERY_MAX_ITERATIONS = 35
BROWSECOMP_FINAL_MAX_ITERATIONS = 30
BROWSECOMP_FROZEN_EMBEDDING_MODEL = "text-embedding-3-small"
BROWSECOMP_FROZEN_INDEX_DIMENSION = 1536
BROWSECOMP_FROZEN_DOCUMENT_COUNT = 100195
BROWSECOMP_FROZEN_INDEX_REPOSITORY = "Joanna690/websearch-embeddings"
BROWSECOMP_FROZEN_INDEX_REVISION = "7a784780b46d16ddc926aed9b63c34def2014c47"
BROWSECOMP_FROZEN_CORPUS_REPOSITORY = "Tevatron/browsecomp-plus-corpus"
BROWSECOMP_FROZEN_CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
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
BROWSECOMP_UPSTREAM_RELATIVE_PATHS = (
    "agent/search.py",
    "run_search.py",
    "env/env_systems/browsecomp_plus_env.py",
    "env/env_systems/web_search_env/evaluate_with_openai.py",
    "env/env_systems/web_search_env/search_agent/openai_client.py",
    "env/env_systems/web_search_env/search_agent/prompts.py",
    "env/env_systems/web_search_env/search_agent/utils.py",
    "env/env_systems/web_search_env/searcher/searchers/base.py",
    "env/env_systems/web_search_env/searcher/searchers/openai_searcher.py",
)
_ACTION_RE = re.compile(
    r"\A(search|SUBMIT_ANSWER)\s+(\{.*\})\Z",
    flags=re.DOTALL,
)


BROWSECOMP_CONTRACT = DomainContract(
    contract_id="memoryarena_browsecomp_plus_v3_20260721",
    system_prompt=(
        "You are operating the MemoryArena BrowseComp-Plus domain. Each episode "
        "contains sequential research subqueries followed by the original final "
        "query. Use the native lowercase search JSON action to inspect the frozen "
        "BrowseComp-Plus corpus, then submit the current phase response with "
        "SUBMIT_ANSWER. A submitted subquery advances without label feedback or "
        "task reward; only the final response is scored by the original "
        "MemoryArena judge. Short-term search context is cleared on phase advance, "
        "so use the AgentMemoryGym memory actions when later phases need prior "
        "evidence."
    ),
    native_action_descriptions=(
        'search {"query": "..."}',
        (
            'SUBMIT_ANSWER {"answer": "Explanation: ...\\nExact Answer: '
            '...\\nConfidence: ...%"}'
        ),
    ),
    # The factory replaces this conservative default with the exact dataset-wide
    # maximum while the driver enforces each phase's native iteration budget.
    max_steps=560,
)


@dataclass(frozen=True)
class BrowseCompPhase:
    kind: str
    query: str


@dataclass(frozen=True)
class BrowseCompTask:
    query_id: str
    phases: tuple[BrowseCompPhase, ...]
    final_answer: str


BrowseSearch = Callable[[str], str]
BrowseJudge = Callable[[str, str, str], dict[str, Any]]


class BrowseCompPlusFactory:
    domain_id = BROWSECOMP_DOMAIN_ID
    surface = BROWSECOMP_SURFACE
    contract = BROWSECOMP_CONTRACT

    def __init__(
        self,
        *,
        ground_truth_path: str | Path,
        decomposition_path: str | Path | None = None,
        memoryarena_root: str | Path | None = None,
        index_path: str | Path | None = None,
        corpus_path: str | Path | None = None,
        embedding_model: str = "text-embedding-3-small",
        provider: str = "openai",
        judge_model: str = "gpt-4.1",
        search_asset_provenance: dict[str, Any] | None = None,
        search_tool: BrowseSearch | None = None,
        judge: BrowseJudge | None = None,
        expected_memoryarena_commit: str | None = None,
    ) -> None:
        self.ground_truth_path = Path(ground_truth_path).expanduser().resolve()
        self.decomposition_path = (
            Path(decomposition_path).expanduser().resolve()
            if decomposition_path is not None
            else None
        )
        self.tasks = load_browsecomp_tasks(
            self.ground_truth_path,
            self.decomposition_path,
        )
        self.dataset_schema = _detect_dataset_schema(self.ground_truth_path)
        self.task_count = len(self.tasks)
        max_phase_count = max(len(task.phases) for task in self.tasks)
        self.contract = DomainContract(
            contract_id=BROWSECOMP_CONTRACT.contract_id,
            system_prompt=BROWSECOMP_CONTRACT.system_prompt,
            native_action_descriptions=BROWSECOMP_CONTRACT.native_action_descriptions,
            max_steps=(
                (max_phase_count - 1) * BROWSECOMP_SUBQUERY_MAX_ITERATIONS
                + BROWSECOMP_FINAL_MAX_ITERATIONS
            ),
        )
        self.dataset_sha256 = _sha256_file(self.ground_truth_path)
        self.decomposition_sha256 = (
            _sha256_file(self.decomposition_path)
            if self.decomposition_path is not None and self.decomposition_path.is_file()
            else self.dataset_sha256 if self.dataset_schema == "progressive_search" else None
        )
        self.memoryarena_root = (
            Path(memoryarena_root).expanduser().resolve()
            if memoryarena_root is not None
            else None
        )
        self.upstream_provenance = (
            attest_browsecomp_upstream(
                self.memoryarena_root,
                expected_commit=expected_memoryarena_commit,
            )
            if self.memoryarena_root is not None
            else {"mode": "injected_test_double"}
        )
        self.search_asset_provenance = (
            dict(search_asset_provenance)
            if search_asset_provenance is not None
            else {"mode": "injected_test_double"}
        )
        if search_tool is None or judge is None:
            if self.memoryarena_root is None:
                raise RuntimeError(
                    "memoryarena_root is required when BrowseComp search or judge is not injected"
                )
            if search_tool is None:
                if search_asset_provenance is None:
                    raise RuntimeError(
                        "production BrowseComp search requires frozen asset provenance"
                    )
                search_tool = _build_upstream_search(
                    self.memoryarena_root,
                    index_path=index_path,
                    corpus_path=corpus_path,
                    embedding_model=embedding_model,
                    provider=provider,
                )
            if judge is None:
                judge = _build_upstream_judge(
                    self.memoryarena_root,
                    judge_model=judge_model,
                )
        self.search_tool = search_tool
        self.judge = judge
        self.judge_model = judge_model

    def create(self, env_uid: str):
        return BrowseCompPlusDriver(
            tasks=self.tasks,
            search_tool=self.search_tool,
            judge=self.judge,
            env_uid=env_uid,
            contract=self.contract,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "source": "MemoryArena",
            "dataset_schema": self.dataset_schema,
            "dataset_sha256": self.dataset_sha256,
            "decomposition_sha256": self.decomposition_sha256,
            "decomposition_mode": (
                "progressive_search_direct"
                if self.dataset_schema == "progressive_search"
                else "browsecomp_all_jsons"
                if self.decomposition_sha256 is not None
                else "original_query_fallback"
            ),
            "native_tool_ops": ["search"],
            "native_search_k": 10,
            "native_subquery_max_iterations": BROWSECOMP_SUBQUERY_MAX_ITERATIONS,
            "native_final_max_iterations": BROWSECOMP_FINAL_MAX_ITERATIONS,
            "native_iteration_budget_semantics": (
                "memory_actions_do_not_count; subquery_exhaustion_advances_unjudged; "
                "final_exhaustion_without_extractable_answer_ends_incorrect"
            ),
            "judge": "memoryarena_browsecomp_gpt_judge_v1",
            "judge_model": self.judge_model,
            "intermediate_submission_reward": 0.0,
            "final_reward": "1_if_original_judge_correct_else_0",
            "memory_semantics": "agentmemory_explicit_tools_between_native_phases",
            "upstream_provenance": self.upstream_provenance,
            "search_asset_provenance": self.search_asset_provenance,
        }


class BrowseCompPlusDriver:
    domain_id = BROWSECOMP_DOMAIN_ID
    surface = BROWSECOMP_SURFACE
    contract = BROWSECOMP_CONTRACT

    def __init__(
        self,
        *,
        tasks: Sequence[BrowseCompTask],
        search_tool: BrowseSearch,
        judge: BrowseJudge,
        env_uid: str,
        contract: DomainContract = BROWSECOMP_CONTRACT,
    ) -> None:
        if not tasks:
            raise ValueError("BrowseCompPlusDriver requires tasks")
        self.tasks = tuple(tasks)
        self.search_tool = search_tool
        self.judge = judge
        self.env_uid = env_uid
        self.contract = contract
        self.task: BrowseCompTask | None = None
        self.data_idx = 0
        self.phase_index = 0
        self.phase_native_iteration_count = 0
        self.done = False
        self.status = "idle"
        self.retrieved_docids: list[str] = []

    def reset(self, data_idx: int) -> DomainTransition:
        index = int(data_idx)
        if index < 0 or index >= len(self.tasks):
            raise IndexError(
                f"BrowseComp data index {index} is outside "
                f"[0, {len(self.tasks)})"
            )
        self.data_idx = index
        self.task = self.tasks[self.data_idx]
        self.phase_index = 0
        self.phase_native_iteration_count = 0
        self.done = False
        self.status = "active"
        self.retrieved_docids = []
        return self._transition(self._render_phase())

    def step(self, action: str, env_step: int) -> DomainTransition:
        if self.task is None:
            raise RuntimeError("BrowseComp driver must be reset before step")
        if self.done:
            return self._transition("The BrowseComp episode is already complete.", done=True)
        phase_index_before = self.phase_index
        self.phase_native_iteration_count += 1
        parsed = _parse_action(action)
        if parsed is None:
            transition = self._invalid(
                action,
                env_step,
                "invalid BrowseComp action grammar",
            )
            return self._apply_phase_limit(
                transition,
                env_step,
                phase_index_before=phase_index_before,
            )
        op, payload = parsed
        if op == "search":
            transition = self._search(payload, action, env_step)
            return self._apply_phase_limit(
                transition,
                env_step,
                phase_index_before=phase_index_before,
            )
        transition = self._submit_answer(payload, action, env_step)
        return self._apply_phase_limit(
            transition,
            env_step,
            phase_index_before=phase_index_before,
        )

    def close(self) -> None:
        self.status = "closed"
        self.done = True

    def _search(
        self,
        payload: dict[str, Any],
        raw_action: str,
        env_step: int,
    ) -> DomainTransition:
        if set(payload) != {"query"}:
            return self._invalid(raw_action, env_step, "search expects only query")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            return self._invalid(raw_action, env_step, "query must be a non-empty string")
        try:
            result = self.search_tool(query)
        except Exception as exc:
            # The native OpenAI tool loop returns search failures as tool messages and
            # lets the model continue, so this is not an excluded terminal failure.
            message = f"Error executing search: {exc}"
            return self._transition(
                f"Tool result (search):\n{message}\n\n{self._render_phase()}",
                action_execution={
                    "op": "search",
                    "status": "error",
                    "step": env_step,
                    "arguments": {"query": query},
                },
                tool_ops=(
                    {
                        "op": "search",
                        "step": env_step,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    },
                ),
                reward_components=(
                    {
                        "name": "browsecomp_search_tool_error",
                        "value": 0.0,
                        "op": "search",
                        "step": env_step,
                    },
                ),
                domain_evidence={
                    "query_id": self._require_task().query_id,
                    "search_error_type": type(exc).__name__,
                },
            )
        if not isinstance(result, str):
            result = str(result)
        docids = _extract_docids(result)
        for docid in docids:
            if docid not in self.retrieved_docids:
                self.retrieved_docids.append(docid)
        return self._transition(
            f"Tool result (search):\n{result}\n\n{self._render_phase()}",
            action_execution={
                "op": "search",
                "status": "executed",
                "step": env_step,
                "arguments": {"query": query},
            },
            tool_ops=(
                {
                    "op": "search",
                    "step": env_step,
                    "arguments": {"query": query},
                    "retrieved_docids": docids,
                    "result_length": len(result),
                },
            ),
            reward_components=(
                {
                    "name": "browsecomp_search_transition",
                    "value": 0.0,
                    "op": "search",
                    "step": env_step,
                },
            ),
            domain_evidence={
                "query_id": self._require_task().query_id,
                "search_result_sha256": hashlib.sha256(result.encode("utf-8")).hexdigest(),
                "retrieved_docids": list(self.retrieved_docids),
            },
        )

    def _submit_answer(
        self,
        payload: dict[str, Any],
        raw_action: str,
        env_step: int,
    ) -> DomainTransition:
        if set(payload) != {"answer"}:
            return self._invalid(raw_action, env_step, "SUBMIT_ANSWER expects only answer")
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return self._invalid(raw_action, env_step, "answer must be a non-empty string")
        task = self._require_task()
        phase = self._current_phase()
        final = phase.kind == "final"
        judgement: dict[str, Any] | None = None
        if final:
            try:
                judgement = self.judge(phase.query, answer, task.final_answer)
            except Exception as exc:
                return self._infra_error("SUBMIT_ANSWER", env_step, exc)
            if not isinstance(judgement, dict):
                return self._infra_error(
                    "SUBMIT_ANSWER",
                    env_step,
                    TypeError("BrowseComp judge must return a dictionary"),
                )
        passed = bool(judgement is not None and judgement.get("correct") is True)
        prior_phase = self.phase_index
        submitted_iteration = self.phase_native_iteration_count
        self.phase_index += 1
        self.phase_native_iteration_count = 0
        self.done = final
        self.status = (
            "success"
            if final and passed
            else "completed_incorrect"
            if final
            else "active"
        )
        reward = 1.0 if passed else 0.0
        component_name = (
            "browsecomp_final_answer_correct"
            if passed
            else "browsecomp_final_answer_incorrect"
            if final
            else "browsecomp_subquery_submission"
        )
        observation = (
            "The final BrowseComp response was evaluated."
            if final
            else "The subquery response was submitted. The next phase is ready.\n\n"
            + self._render_phase()
        )
        evidence = {
            "query_id": task.query_id,
            "submitted_phase_index": prior_phase,
            "submitted_phase_kind": phase.kind,
            "retrieved_docids": list(self.retrieved_docids),
        }
        if judgement is not None:
            evidence.update(
                {
                    "judge_id": "memoryarena_browsecomp_gpt_judge_v1",
                    "judge_correct": passed,
                    "judge_confidence": judgement.get("confidence"),
                    "judge_parse_error": bool(judgement.get("parse_error", False)),
                }
            )
        return self._transition(
            observation,
            reward=reward,
            done=final,
            status=self.status,
            episode_success=final and passed,
            action_execution={
                "op": "SUBMIT_ANSWER",
                "status": (
                    "committed_correct"
                    if passed
                    else "committed_incorrect"
                    if final
                    else "committed_unjudged"
                ),
                "step": env_step,
            },
            tool_ops=(
                {
                    "op": "SUBMIT_ANSWER",
                    "step": env_step,
                    "committed": True,
                    "phase_advanced": True,
                    "phase_index": prior_phase,
                    "phase_kind": phase.kind,
                    "native_iteration_index": submitted_iteration,
                    "terminal": final,
                    "submission_correct": passed if final else None,
                    "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
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
        phase = self._current_phase()
        limit = self._current_phase_iteration_limit()
        if self.phase_native_iteration_count < limit:
            return self._with_iteration_evidence(transition, limit=limit)

        task = self._require_task()
        exhausted_phase_index = self.phase_index
        exhausted_iteration_count = self.phase_native_iteration_count
        final = phase.kind == "final"
        self.phase_index += 1
        self.phase_native_iteration_count = 0
        self.done = final
        self.status = "completed_incorrect" if final else "active"

        action_execution = dict(transition.action_execution)
        action_execution.update(
            {
                "phase_budget_exhausted": True,
                "phase_advanced": True,
                "terminal": final,
                "native_iteration_index": exhausted_iteration_count,
                "native_iteration_limit": limit,
            }
        )
        tool_ops = tuple(transition.tool_ops) + (
            {
                "op": "PHASE_BUDGET_EXHAUSTED",
                "step": env_step,
                "phase_index": exhausted_phase_index,
                "phase_kind": phase.kind,
                "native_iteration_count": exhausted_iteration_count,
                "native_iteration_limit": limit,
                "phase_advanced": True,
                "terminal": final,
                "extractable_answer": False,
            },
        )
        reward_components = tuple(transition.reward_components) + (
            {
                "name": (
                    "browsecomp_final_budget_exhausted_without_answer"
                    if final
                    else "browsecomp_subquery_budget_exhausted"
                ),
                "value": 0.0,
                "op": "PHASE_BUDGET_EXHAUSTED",
                "step": env_step,
            },
        )
        evidence = dict(transition.domain_evidence)
        evidence.update(
            {
                "query_id": task.query_id,
                "exhausted_phase_index": exhausted_phase_index,
                "exhausted_phase_kind": phase.kind,
                "native_iteration_count": exhausted_iteration_count,
                "native_iteration_limit": limit,
                "extractable_answer": False,
            }
        )
        observation = (
            "The final BrowseComp phase reached its native iteration limit "
            "without an extractable final answer. The episode ended incorrect."
            if final
            else "The BrowseComp subquery reached its native iteration limit and "
            "advanced unjudged with zero reward. The next phase is ready.\n\n"
            + self._render_phase()
        )
        return self._transition(
            observation,
            reward=transition.reward,
            done=final,
            status=self.status,
            episode_success=False,
            action_execution=action_execution,
            tool_ops=tool_ops,
            reward_components=reward_components,
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

    def _invalid(self, raw_action: str, env_step: int, message: str) -> DomainTransition:
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
            domain_evidence={"query_id": self._require_task().query_id},
        )

    def _infra_error(self, op: str, env_step: int, exc: Exception) -> DomainTransition:
        self.done = True
        self.status = "infra_error"
        return self._transition(
            "The BrowseComp environment encountered an infrastructure error.",
            done=True,
            status=self.status,
            action_execution={
                "op": op,
                "status": "error",
                "step": env_step,
            },
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
            domain_evidence={"query_id": self._require_task().query_id},
            sample_excluded=True,
        )

    def _render_phase(self) -> str:
        task = self._require_task()
        phase = self._current_phase()
        label = (
            f"Research subquery {self.phase_index + 1}/{len(task.phases) - 1}"
            if phase.kind == "subquery"
            else "Final combined query"
        )
        return "\n\n".join(
            [
                (
                    "Task family: browsecomp_plus\n"
                    f"Progress: {self.phase_index}/{len(task.phases)}\n"
                    f"Phase: {label}\n"
                    "Native iteration budget: "
                    f"{self.phase_native_iteration_count}/"
                    f"{self._current_phase_iteration_limit()}"
                ),
                f"Question: {phase.query}",
                (
                    "Response format for SUBMIT_ANSWER:\n"
                    "Explanation: evidence-backed explanation with [docid] citations\n"
                    "Exact Answer: succinct final answer\n"
                    "Confidence: 0% to 100%"
                ),
            ]
        )

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
            domain_evidence=domain_evidence or {"query_id": task.query_id},
            sample_excluded=sample_excluded,
        )


def load_browsecomp_tasks(
    ground_truth_path: Path,
    decomposition_path: Path | None = None,
) -> tuple[BrowseCompTask, ...]:
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"BrowseComp ground truth not found: {ground_truth_path}")
    ground_truth_rows = _read_jsonl_rows(
        ground_truth_path,
        label="BrowseComp ground truth",
    )
    if not ground_truth_rows:
        raise ValueError(f"no BrowseComp tasks loaded from {ground_truth_path}")

    progressive_flags = {
        _is_progressive_row(payload) for _, payload in ground_truth_rows
    }
    if len(progressive_flags) > 1:
        raise ValueError(
            "BrowseComp ground truth mixes progressive_search and legacy schemas"
        )
    if True in progressive_flags:
        tasks = _load_progressive_tasks(ground_truth_rows)
        if decomposition_path is not None and decomposition_path.resolve() == ground_truth_path.resolve():
            decompositions = _load_decompositions(decomposition_path)
            for task in tasks:
                expected = tuple(
                    phase.query for phase in task.phases if phase.kind == "subquery"
                )
                actual = decompositions.get(task.query_id, ())
                if actual != expected:
                    raise ValueError(
                        "progressive BrowseComp source does not reproduce its own "
                        f"subqueries for query_id {task.query_id}"
                    )
        return tasks

    decompositions = _load_decompositions(decomposition_path)
    tasks = []
    seen = set()
    for line_number, payload in ground_truth_rows:
        query_id = str(payload.get("query_id", "")).strip()
        final_query = payload.get("query")
        final_answer = payload.get("answer")
        if not query_id or query_id in seen:
            raise ValueError(
                f"missing or duplicate BrowseComp query_id on line {line_number}"
            )
        if not isinstance(final_query, str) or not final_query.strip():
            raise ValueError(f"BrowseComp query {query_id} has no final query")
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise ValueError(f"BrowseComp query {query_id} has no final answer")
        subqueries = decompositions.get(query_id, ())
        if not subqueries:
            # run_search.py uses the original query as a subquery when the
            # optional decomposition file is absent or has no usable row.
            subqueries = (final_query,)
        phases = tuple(
            [BrowseCompPhase(kind="subquery", query=query) for query in subqueries]
            + [BrowseCompPhase(kind="final", query=final_query)]
        )
        tasks.append(
            BrowseCompTask(
                query_id=query_id,
                phases=phases,
                final_answer=final_answer,
            )
        )
        seen.add(query_id)
    return tuple(tasks)


def _read_jsonl_rows(path: Path, *, label: str) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid {label} JSON on line {line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{label} line {line_number} must be a JSON object")
            rows.append((line_number, payload))
    return rows


def _is_progressive_row(payload: dict[str, Any]) -> bool:
    # Treat a partial progressive row as progressive so it fails closed below,
    # rather than silently falling through to the legacy query_id/query schema.
    return "questions" in payload or "answers" in payload


def _detect_dataset_schema(path: Path) -> str:
    rows = _read_jsonl_rows(path, label="BrowseComp ground truth")
    flags = {_is_progressive_row(payload) for _, payload in rows}
    if len(flags) > 1:
        raise ValueError(
            "BrowseComp ground truth mixes progressive_search and legacy schemas"
        )
    return "progressive_search" if True in flags else "legacy_query_answer"


def _load_progressive_tasks(
    rows: Sequence[tuple[int, dict[str, Any]]],
) -> tuple[BrowseCompTask, ...]:
    tasks: list[BrowseCompTask] = []
    seen: set[str] = set()
    for line_number, payload in rows:
        if not {"id", "questions", "answers"}.issubset(payload):
            raise ValueError(
                "progressive BrowseComp row on line "
                f"{line_number} must contain id/questions/answers"
            )
        query_id = str(payload["id"]).strip()
        questions = payload["questions"]
        answers = payload["answers"]
        if not query_id:
            raise ValueError(
                f"progressive BrowseComp row on line {line_number} has empty id"
            )
        if query_id in seen:
            raise ValueError(
                f"duplicate progressive BrowseComp id {query_id!r} on line {line_number}"
            )
        if not isinstance(questions, list) or not isinstance(answers, list):
            raise ValueError(
                "progressive BrowseComp row on line "
                f"{line_number} requires list questions and answers"
            )
        if not questions or len(questions) != len(answers):
            raise ValueError(
                "progressive BrowseComp row on line "
                f"{line_number} has misaligned questions/answers"
            )
        if any(not isinstance(value, str) or not value.strip() for value in questions):
            raise ValueError(
                f"progressive BrowseComp row {query_id} has an invalid question"
            )
        if any(not isinstance(value, str) or not value.strip() for value in answers):
            raise ValueError(
                f"progressive BrowseComp row {query_id} has an invalid answer"
            )
        phases = tuple(
            [BrowseCompPhase(kind="subquery", query=query) for query in questions[:-1]]
            + [BrowseCompPhase(kind="final", query=questions[-1])]
        )
        tasks.append(
            BrowseCompTask(
                query_id=query_id,
                phases=phases,
                final_answer=answers[-1],
            )
        )
        seen.add(query_id)
    return tuple(tasks)


def _load_decompositions(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None or not path.is_file():
        return {}
    decompositions = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid BrowseComp decomposition JSON on line {line_number}"
                ) from exc
            query_id = str(payload.get("id", "")).strip()
            questions = payload.get("question", payload.get("questions", []))
            answers = payload.get("answer", payload.get("answers", []))
            if not query_id or not isinstance(questions, list) or not isinstance(answers, list):
                continue
            # This is the exact alignment used by agent.search.load_correct_answers.
            aligned = questions[: min(len(questions), len(answers))]
            subqueries = aligned[:-1] if len(aligned) > 1 else []
            if any(
                not isinstance(question, str) or not question.strip()
                for question in subqueries
            ):
                raise ValueError(
                    f"BrowseComp decomposition {query_id} has an invalid subquery"
                )
            normalized = tuple(subqueries)
            # agent.search.load_correct_answers stops at the first matching row.
            if normalized and query_id not in decompositions:
                decompositions[query_id] = normalized
    return decompositions


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
    if not isinstance(payload, list):
        return []
    docids = []
    for item in payload:
        if isinstance(item, dict) and item.get("docid") is not None:
            docids.append(str(item["docid"]))
    return docids


def _build_upstream_search(
    memoryarena_root: Path,
    *,
    index_path: str | Path | None,
    corpus_path: str | Path | None,
    embedding_model: str,
    provider: str,
) -> BrowseSearch:
    search_agent_dir = (
        memoryarena_root / "env/env_systems/web_search_env/search_agent"
    )
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
        provider=provider or "openai",
    )
    searcher = searcher_module.OpenAISearcher(args)
    validate_loaded_browsecomp_searcher(searcher, embedding_model=embedding_model)
    handler = client_module.SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=512,
        k=10,
        include_get_document=False,
        allowed_docids=None,
        step_memory_client=None,
    )

    def search(query: str) -> str:
        return handler.execute_tool("search", {"query": query})

    return search


def attest_browsecomp_search_assets(
    *,
    index_pattern: str,
    corpus_path: Path,
    corpus_manifest_path: Path,
    embedding_model: str,
    expected_index_shards: Sequence[dict[str, Any]] | None = None,
    expected_corpus_shards: Sequence[str] | None = None,
    expected_document_count: int = BROWSECOMP_FROZEN_DOCUMENT_COUNT,
) -> dict[str, Any]:
    """Verify the exact frozen FAISS/id-map bytes and materialized corpus."""

    if embedding_model != BROWSECOMP_FROZEN_EMBEDDING_MODEL:
        raise RuntimeError(
            "Frozen BrowseComp index requires embedding model "
            f"{BROWSECOMP_FROZEN_EMBEDDING_MODEL!r}, got {embedding_model!r}"
        )
    shard_specs = tuple(expected_index_shards or BROWSECOMP_FROZEN_INDEX_SHARDS)
    corpus_shas = tuple(expected_corpus_shards or BROWSECOMP_FROZEN_CORPUS_SHARDS)
    index_paths = tuple(Path(value).resolve() for value in sorted(glob.glob(index_pattern)))
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
            raise RuntimeError(
                f"BrowseComp FAISS id-map count mismatch: {id_map_path}"
            )
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
        raise RuntimeError(
            "BrowseComp FAISS id maps do not uniquely cover the frozen corpus"
        )

    try:
        with corpus_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("BrowseComp corpus manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("BrowseComp corpus manifest must be a JSON object")
    if manifest.get("format") != "agentmemory_browsecomp_corpus_manifest_v1":
        raise RuntimeError("BrowseComp corpus manifest has an unsupported format")
    source = manifest.get("source")
    projection = manifest.get("projection")
    output = manifest.get("output")
    if (
        not isinstance(source, dict)
        or not isinstance(projection, dict)
        or not isinstance(output, dict)
    ):
        raise RuntimeError("BrowseComp corpus manifest is incomplete")
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
    file_count = source.get("file_count")
    if isinstance(file_count, bool) or file_count != len(source_files):
        raise RuntimeError("BrowseComp corpus manifest source file count is invalid")
    observed_source_shas = tuple(
        sorted(str(item.get("sha256", "")) for item in source_files)
    )
    if observed_source_shas != tuple(sorted(corpus_shas)):
        raise RuntimeError("BrowseComp corpus source shard hashes do not match the freeze")
    corpus_path = corpus_path.expanduser().resolve()
    output_path = output.get("path")
    if (
        not isinstance(output_path, str)
        or not output_path.strip()
        or Path(output_path).expanduser().resolve() != corpus_path
    ):
        raise RuntimeError("BrowseComp corpus manifest output path does not match runtime")
    row_count = output.get("row_count")
    if (
        isinstance(row_count, bool)
        or not isinstance(row_count, int)
        or row_count != expected_document_count
    ):
        raise RuntimeError("BrowseComp corpus manifest row count does not match the freeze")
    corpus_sha256 = _sha256_file(corpus_path)
    if not isinstance(output.get("sha256"), str) or output["sha256"] != corpus_sha256:
        raise RuntimeError("BrowseComp corpus hash does not match its manifest")

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
        "corpus_sha256": corpus_sha256,
        "corpus_manifest_sha256": _sha256_file(corpus_manifest_path),
    }


def validate_loaded_browsecomp_searcher(
    searcher: Any,
    *,
    embedding_model: str,
    expected_index_shards: Sequence[dict[str, Any]] | None = None,
    expected_document_count: int = BROWSECOMP_FROZEN_DOCUMENT_COUNT,
) -> None:
    """Cross-check FAISS dimensions/counts and exact corpus coverage after load."""

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
        if int(getattr(index, "ntotal", -1)) != vector_count or len(ids) != vector_count:
            raise RuntimeError("BrowseComp FAISS vectors and id map are misaligned")
        all_ids.extend(str(docid) for docid in ids)
    if len(all_ids) != expected_document_count or len(set(all_ids)) != expected_document_count:
        raise RuntimeError("BrowseComp loaded id maps contain missing or duplicate documents")
    if not isinstance(corpus, dict) or set(corpus) != set(all_ids):
        raise RuntimeError("BrowseComp loaded corpus does not exactly cover indexed documents")


def _build_upstream_judge(
    memoryarena_root: Path,
    *,
    judge_model: str,
) -> BrowseJudge:
    root_text = str(memoryarena_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("env.env_systems.browsecomp_plus_env")
    _require_module_under_root(module, memoryarena_root)
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    client = openai.OpenAI(api_key=api_key, base_url=base_url if base_url else None)

    def judge(question: str, predicted_answer: str, correct_answer: str):
        return module.BrowseCompPlusEnvironment.evaluate_answer_with_judge(
            client,
            question,
            predicted_answer,
            correct_answer,
            model=judge_model,
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


def attest_browsecomp_upstream(
    memoryarena_root: Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Fail closed if the BrowseComp implementation differs from its git commit."""

    root = memoryarena_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"MemoryArena root is not a git worktree: {root}")
    commit = _git(root, "rev-parse", "HEAD").strip()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError(
            "MemoryArena commit mismatch for BrowseComp: "
            f"expected {expected_commit}, observed {commit}"
        )
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *BROWSECOMP_UPSTREAM_RELATIVE_PATHS,
    )
    if status.strip():
        raise RuntimeError(
            "MemoryArena BrowseComp source is not pristine at the pinned commit:\n"
            + status.rstrip()
        )
    source_sha256 = {}
    for relative_path in BROWSECOMP_UPSTREAM_RELATIVE_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise RuntimeError(f"Missing MemoryArena BrowseComp source file: {path}")
        source_sha256[relative_path] = _sha256_file(path)
    digest = hashlib.sha256(
        json.dumps(
            source_sha256,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "mode": "pinned_pristine_upstream",
        "memoryarena_commit": commit,
        "source_files_sha256": source_sha256,
        "source_bundle_sha256": digest,
    }


def _git(root: Path, *args: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={root}",
        "-C",
        str(root),
        *args,
    ]
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
