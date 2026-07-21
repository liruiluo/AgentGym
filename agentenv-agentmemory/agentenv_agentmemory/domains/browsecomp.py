from __future__ import annotations

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

from ..runtime.domain import DomainContract, DomainTransition


BROWSECOMP_SURFACE = "memoryarena_browsecomp_plus_v3"
BROWSECOMP_DOMAIN_ID = "browsecomp_plus"
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
        "evidence. Reply with brief reasoning followed by exactly one Action line."
    ),
    native_action_descriptions=(
        'search {"query": "..."}',
        (
            'SUBMIT_ANSWER {"answer": "Explanation: ...\\nExact Answer: '
            '...\\nConfidence: ...%"}'
        ),
    ),
    # The factory replaces this conservative default with the exact dataset-wide
    # maximum: 35 iterations per subquery plus 30 for the final query.
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
        self.task_count = len(self.tasks)
        max_phase_count = max(len(task.phases) for task in self.tasks)
        self.contract = DomainContract(
            contract_id=BROWSECOMP_CONTRACT.contract_id,
            system_prompt=BROWSECOMP_CONTRACT.system_prompt,
            native_action_descriptions=BROWSECOMP_CONTRACT.native_action_descriptions,
            max_steps=(max_phase_count - 1) * 35 + 30,
        )
        self.dataset_sha256 = _sha256_file(self.ground_truth_path)
        self.decomposition_sha256 = (
            _sha256_file(self.decomposition_path)
            if self.decomposition_path is not None and self.decomposition_path.is_file()
            else None
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
        if search_tool is None or judge is None:
            if self.memoryarena_root is None:
                raise RuntimeError(
                    "memoryarena_root is required when BrowseComp search or judge is not injected"
                )
            if search_tool is None:
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
            "dataset_sha256": self.dataset_sha256,
            "decomposition_sha256": self.decomposition_sha256,
            "decomposition_mode": (
                "browsecomp_all_jsons"
                if self.decomposition_sha256 is not None
                else "original_query_fallback"
            ),
            "native_tool_ops": ["search"],
            "native_search_k": 10,
            "native_subquery_max_iterations": 35,
            "native_final_max_iterations": 30,
            "judge": "memoryarena_browsecomp_gpt_judge_v1",
            "judge_model": self.judge_model,
            "intermediate_submission_reward": 0.0,
            "final_reward": "1_if_original_judge_correct_else_0",
            "memory_semantics": "agentmemory_explicit_tools_between_native_phases",
            "upstream_provenance": self.upstream_provenance,
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
        self.done = False
        self.status = "idle"
        self.retrieved_docids: list[str] = []

    def reset(self, data_idx: int) -> DomainTransition:
        self.data_idx = int(data_idx) % len(self.tasks)
        self.task = self.tasks[self.data_idx]
        self.phase_index = 0
        self.done = False
        self.status = "active"
        self.retrieved_docids = []
        return self._transition(self._render_phase())

    def step(self, action: str, env_step: int) -> DomainTransition:
        if self.task is None:
            raise RuntimeError("BrowseComp driver must be reset before step")
        if self.done:
            return self._transition("The BrowseComp episode is already complete.", done=True)
        parsed = _parse_action(action)
        if parsed is None:
            return self._invalid(action, env_step, "invalid BrowseComp action grammar")
        op, payload = parsed
        if op == "search":
            return self._search(payload, action, env_step)
        return self._submit_answer(payload, action, env_step)

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
        self.phase_index += 1
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
                    f"Phase: {label}"
                ),
                f"Question: {phase.query}",
                (
                    "Response format for SUBMIT_ANSWER:\n"
                    "Explanation: evidence-backed explanation with [docid] citations\n"
                    "Exact Answer: succinct final answer\n"
                    "Confidence: 0% to 100%"
                ),
                "Native BrowseComp actions:\n"
                + "\n".join(
                    f"- {item}" for item in self.contract.native_action_descriptions
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
    decompositions = _load_decompositions(decomposition_path)
    tasks = []
    seen = set()
    with ground_truth_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid BrowseComp ground truth JSON on line {line_number}"
                ) from exc
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
    if not tasks:
        raise ValueError(f"no BrowseComp tasks loaded from {ground_truth_path}")
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
