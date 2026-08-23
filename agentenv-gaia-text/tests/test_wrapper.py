from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from agentenv_gaia_text.backend import BackendError, FixtureBackend
from agentenv_gaia_text.contracts import EvaluationArm, ProtocolContract
from agentenv_gaia_text.dataset import GaiaTextDataset
from agentenv_gaia_text.submission import SubmissionStore
from agentenv_gaia_text.wrapper import GaiaTextEpisodeManager
from support import FileWorkspace, protocol_kwargs, write_runtime_fixture


def _components(tmp_path: Path, arm: EvaluationArm):
    runtime = write_runtime_fixture(tmp_path / arm.value)
    contract = ProtocolContract(**protocol_kwargs(runtime.rows))
    dataset = GaiaTextDataset.load(
        runtime.manifest,
        runtime.questions,
        expected_questions_sha256=runtime.questions_sha256,
        contract=contract,
    )
    backend = FixtureBackend.load(
        runtime.backend, runtime.backend_sha256, page_chars=80
    )
    store = SubmissionStore(dataset.task_ids, runtime.predictions)
    workspaces: list[FileWorkspace] = []

    def factory(env_id: int, task_id: str, episode_index: int) -> FileWorkspace:
        workspace = FileWorkspace(
            tmp_path / "workspaces",
            f"env-{env_id}-episode-{episode_index}-{task_id}",
        )
        workspaces.append(workspace)
        return workspace

    manager = GaiaTextEpisodeManager(
        dataset,
        backend,
        store,
        arm=arm,
        workspace_factory=factory if arm is EvaluationArm.AMG_MEMORY else None,
        max_policy_steps=12,
    )
    return runtime, backend, store, workspaces, manager


def test_evaluation_arms_are_exactly_the_frozen_triad() -> None:
    assert tuple(arm.value for arm in EvaluationArm) == (
        "native",
        "amg_compaction_only",
        "amg_memory",
    )


def test_create_is_unbound_and_native_never_constructs_a_workspace(
    tmp_path: Path,
) -> None:
    runtime, _, _, workspaces, manager = _components(tmp_path, EvaluationArm.NATIVE)
    created = manager.create()
    assert created["info"]["status"] == "unbound"
    assert "question" not in created["observation"].casefold()
    assert workspaces == []

    reset = manager.reset(created["id"], 0)
    assert json.loads(reset["observation"]) == {
        "task_id": runtime.rows[0]["task_id"],
        "level": 1,
        "question": "Synthetic research question for synthetic-task-000?",
    }
    invalid = manager.step(
        created["id"],
        'shell_command {"command":"printf forbidden > note.txt","workdir":"."}',
    )
    assert invalid["done"] is False
    assert invalid["info"]["status"] == "invalid_action"
    assert invalid["info"]["step"] == 1
    assert invalid["info"]["workspace_action_count"] == 0
    assert workspaces == []
    assert manager.metadata()["workspace_available"] is False
    assert manager.metadata()["compaction_available"] is False


def test_compaction_only_has_compaction_but_no_workspace_lifecycle(
    tmp_path: Path,
) -> None:
    _, _, _, workspaces, manager = _components(
        tmp_path, EvaluationArm.AMG_COMPACTION_ONLY
    )
    metadata = manager.metadata()
    assert metadata["arm"] == "amg_compaction_only"
    assert metadata["compaction_available"] is True
    assert (
        metadata["compaction_contract"]
        == "task_neutral_client_replace_messages_v1"
    )
    assert metadata["workspace_available"] is False
    assert metadata["workspace_contract"] == "disabled"
    assert metadata["workspace_lifetime"] == "none"
    assert metadata["workspace_runtime"] == {}
    assert metadata["active_workspace_count"] == 0

    env_id = manager.create()["id"]
    manager.reset(env_id, 0)
    attempted = manager.step(
        env_id,
        'shell_command {"command":"printf forbidden > note.txt","workdir":"."}',
    )
    assert attempted["info"]["status"] == "invalid_action"
    assert attempted["info"]["domain_action"] == "invalid"
    assert attempted["info"]["workspace_action_count"] == 0
    assert "workspace" not in attempted["observation"].casefold()
    assert str(tmp_path) not in str(attempted)
    assert workspaces == []
    assert not (tmp_path / "workspaces").exists()

    manager.finalize_horizon(env_id)
    assert manager.close(env_id) is True
    assert manager.metadata()["active_workspace_count"] == 0


def test_memory_disabled_arms_reject_workspace_construction_inputs(
    tmp_path: Path,
) -> None:
    runtime, backend, _, _, native = _components(
        tmp_path / "base", EvaluationArm.NATIVE
    )
    dataset = native.dataset

    def factory(env_id: int, task_id: str, episode_index: int) -> FileWorkspace:
        return FileWorkspace(
            tmp_path / "forbidden",
            f"{env_id}-{task_id}-{episode_index}",
        )

    for arm in (EvaluationArm.NATIVE, EvaluationArm.AMG_COMPACTION_ONLY):
        with pytest.raises(ValueError, match="must not receive a workspace factory"):
            GaiaTextEpisodeManager(
                dataset,
                backend,
                SubmissionStore(
                    dataset.task_ids,
                    runtime.root / f"{arm.value}-factory.jsonl",
                ),
                arm=arm,
                workspace_factory=factory,
            )
        with pytest.raises(ValueError, match="must not receive workspace runtime"):
            GaiaTextEpisodeManager(
                dataset,
                backend,
                SubmissionStore(
                    dataset.task_ids,
                    runtime.root / f"{arm.value}-runtime.jsonl",
                ),
                arm=arm,
                workspace_runtime={"hidden_root": str(tmp_path / "forbidden")},
            )


def test_active_episode_cannot_be_reset_and_horizon_records_null(
    tmp_path: Path,
) -> None:
    runtime, _, _, _, manager = _components(tmp_path, EvaluationArm.NATIVE)
    env_id = manager.create()["id"]
    manager.reset(env_id, 0)
    with pytest.raises(RuntimeError, match="unfinished"):
        manager.reset(env_id, 1)

    terminal = manager.finalize_horizon(env_id)
    assert terminal["done"] is True
    assert terminal["reward"] == 0.0
    assert terminal["info"]["status"] == "policy_horizon"
    partial = Path(str(runtime.predictions) + ".partial").read_text()
    assert json.loads(partial)["task_id"] == runtime.rows[0]["task_id"]
    assert json.loads(partial)["model_answer"] is None


def test_three_arms_share_domain_path_and_answer_extraction(tmp_path: Path) -> None:
    components = {
        arm: _components(tmp_path / arm.value, arm) for arm in EvaluationArm
    }
    managers = {arm: value[-1] for arm, value in components.items()}
    env_ids = {arm: manager.create()["id"] for arm, manager in managers.items()}
    metadata = [manager.metadata() for manager in managers.values()]
    assert (
        len({json.dumps(item["backend"], sort_keys=True) for item in metadata})
        == 1
    )
    assert len(
        {
            json.dumps(item["paired_runtime_contract"], sort_keys=True)
            for item in metadata
        }
    ) == 1
    assert len({item["paired_runtime_contract_sha256"] for item in metadata}) == 1
    reset_observations = {
        manager.reset(env_ids[arm], 0)["observation"]
        for arm, manager in managers.items()
    }
    assert len(reset_observations) == 1

    actions = (
        '<tool_call>{"name":"search","arguments":{"query":["alpha evidence"]}}</tool_call>',
        '<tool_call>{"name":"visit","arguments":{"url":"gaia-text://fixture/alpha","goal":"synthetic result","page":1}}</tool_call>',
    )
    for action in actions:
        results = [
            manager.step(env_ids[arm], action) for arm, manager in managers.items()
        ]
        assert len({result["observation"] for result in results}) == 1
        assert len({result["info"]["domain_action"] for result in results}) == 1

    answers = [
        manager.step(
            env_ids[arm], "prefix <answer> forty two </answer> suffix"
        )
        for arm, manager in managers.items()
    ]
    assert all(answer["done"] is True for answer in answers)
    assert {answer["reward"] for answer in answers} == {0.0}
    assert len({answer["observation"] for answer in answers}) == 1
    assert len(
        {
            json.dumps(value[1].call_trace, sort_keys=True)
            for value in components.values()
        }
    ) == 1
    for runtime, _, _, _, _ in components.values():
        first = json.loads(Path(str(runtime.predictions) + ".partial").read_text())
        assert first == {
            "task_id": "synthetic-task-000",
            "model_answer": "forty two",
        }


def test_paired_runtime_digest_changes_with_arm_neutral_runtime_inputs(
    tmp_path: Path,
) -> None:
    runtime, _, _, _, manager = _components(tmp_path, EvaluationArm.NATIVE)
    baseline = manager.metadata()["paired_runtime_contract_sha256"]
    contract = ProtocolContract(**protocol_kwargs(runtime.rows))
    dataset = GaiaTextDataset.load(
        runtime.manifest,
        runtime.questions,
        expected_questions_sha256=runtime.questions_sha256,
        contract=contract,
    )

    changed_page_manager = GaiaTextEpisodeManager(
        dataset,
        FixtureBackend.load(runtime.backend, runtime.backend_sha256, page_chars=81),
        SubmissionStore(dataset.task_ids, tmp_path / "changed-page.jsonl"),
        arm=EvaluationArm.NATIVE,
        max_policy_steps=12,
    )
    assert changed_page_manager.metadata()["paired_runtime_contract_sha256"] != baseline

    changed_backend_path = tmp_path / "changed-backend.json"
    changed_backend = json.loads(runtime.backend.read_text())
    changed_backend["documents"][0]["content"] += " changed"
    changed_backend_bytes = json.dumps(
        changed_backend,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    changed_backend_path.write_bytes(changed_backend_bytes)
    changed_backend_manager = GaiaTextEpisodeManager(
        dataset,
        FixtureBackend.load(
            changed_backend_path,
            hashlib.sha256(changed_backend_bytes).hexdigest(),
            page_chars=80,
        ),
        SubmissionStore(dataset.task_ids, tmp_path / "changed-backend.jsonl"),
        arm=EvaluationArm.NATIVE,
        max_policy_steps=12,
    )
    assert (
        changed_backend_manager.metadata()["paired_runtime_contract_sha256"] != baseline
    )

    changed_questions = tmp_path / "changed-questions.jsonl"
    changed_question_bytes = runtime.questions.read_bytes().replace(
        b"Synthetic research question for synthetic-task-000?",
        b"Changed synthetic research question for synthetic-task-000?",
        1,
    )
    changed_questions.write_bytes(changed_question_bytes)
    changed_dataset = GaiaTextDataset.load(
        runtime.manifest,
        changed_questions,
        expected_questions_sha256=hashlib.sha256(changed_question_bytes).hexdigest(),
        contract=contract,
    )
    changed_questions_manager = GaiaTextEpisodeManager(
        changed_dataset,
        FixtureBackend.load(runtime.backend, runtime.backend_sha256, page_chars=80),
        SubmissionStore(dataset.task_ids, tmp_path / "changed-questions-output.jsonl"),
        arm=EvaluationArm.NATIVE,
        max_policy_steps=12,
    )
    assert (
        changed_questions_manager.metadata()["paired_runtime_contract_sha256"]
        != baseline
    )


def test_memory_workspace_is_clean_per_task_and_not_seeded_with_inputs(
    tmp_path: Path,
) -> None:
    _, _, _, workspaces, manager = _components(tmp_path, EvaluationArm.AMG_MEMORY)
    env_id = manager.create()["id"]
    manager.reset(env_id, 0)
    write = manager.step(
        env_id,
        'shell_command {"command":"printf memory-one > note.txt","workdir":"."}',
    )
    assert write["info"]["workspace_action_count"] == 1
    first_root = workspaces[0].root
    assert (first_root / "note.txt").read_text() == "memory-one"
    assert sorted(path.name for path in first_root.iterdir()) == ["note.txt"]

    manager.step(env_id, "<answer>done</answer>")
    manager.reset(env_id, 1)
    assert workspaces[0].closed is True
    assert len(workspaces) == 2
    assert list(workspaces[1].root.iterdir()) == []
    read = manager.step(
        env_id,
        'shell_command {"command":"test ! -e note.txt && printf clean","workdir":"."}',
    )
    assert "clean" in read["observation"]


def test_duplicate_task_reset_is_rejected_before_workspace_mutation(
    tmp_path: Path,
) -> None:
    _, _, _, workspaces, manager = _components(tmp_path, EvaluationArm.AMG_MEMORY)
    first_id = manager.create()["id"]
    second_id = manager.create()["id"]
    manager.reset(first_id, 0)
    manager.reset(second_id, 1)
    manager.step(
        second_id,
        'shell_command {"command":"printf keep > note.txt","workdir":"."}',
    )
    manager.step(second_id, "<answer>done</answer>")
    prior_workspace = workspaces[1]

    with pytest.raises(RuntimeError, match="already claimed"):
        manager.reset(second_id, 0)

    assert len(workspaces) == 2
    assert prior_workspace.closed is False
    assert (prior_workspace.root / "note.txt").read_text() == "keep"


def test_failed_workspace_construction_releases_the_task_claim(tmp_path: Path) -> None:
    runtime = write_runtime_fixture(tmp_path / "runtime")
    contract = ProtocolContract(**protocol_kwargs(runtime.rows))
    dataset = GaiaTextDataset.load(
        runtime.manifest,
        runtime.questions,
        expected_questions_sha256=runtime.questions_sha256,
        contract=contract,
    )
    workspaces: list[FileWorkspace] = []

    class FailingWorkspace(FileWorkspace):
        def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None:
            raise RuntimeError("synthetic workspace reset failure")

    def factory(env_id: int, task_id: str, episode_index: int) -> FileWorkspace:
        workspace_type = FailingWorkspace if not workspaces else FileWorkspace
        workspace = workspace_type(
            tmp_path / "workspaces",
            f"env-{env_id}-episode-{episode_index}-{task_id}",
        )
        workspaces.append(workspace)
        return workspace

    manager = GaiaTextEpisodeManager(
        dataset,
        FixtureBackend.load(runtime.backend, runtime.backend_sha256),
        SubmissionStore(dataset.task_ids, runtime.predictions),
        arm=EvaluationArm.AMG_MEMORY,
        workspace_factory=factory,
    )
    failed_id = manager.create()["id"]
    retry_id = manager.create()["id"]
    with pytest.raises(RuntimeError, match="synthetic workspace reset failure"):
        manager.reset(failed_id, 0)

    assert workspaces[0].closed is True
    reset = manager.reset(retry_id, 0)
    assert reset["info"]["task_id"] == "synthetic-task-000"
    assert list(workspaces[1].root.iterdir()) == []


def test_long_backend_call_does_not_block_another_episode(tmp_path: Path) -> None:
    _, backend, _, _, manager = _components(tmp_path, EvaluationArm.NATIVE)
    first_id = manager.create()["id"]
    second_id = manager.create()["id"]
    manager.reset(first_id, 0)
    manager.reset(second_id, 1)
    started = threading.Event()
    release = threading.Event()
    real_search = backend.search

    def blocking_search(query, *, top_k=5):
        if query == "blocked":
            started.set()
            if not release.wait(timeout=5):
                raise AssertionError("test did not release the blocked backend call")
        return real_search(query, top_k=top_k)

    backend.search = blocking_search  # type: ignore[method-assign]
    blocked_action = (
        '<tool_call>{"name":"search","arguments":{"query":"blocked"}}</tool_call>'
    )
    free_action = (
        '<tool_call>{"name":"search","arguments":{"query":"alpha"}}</tool_call>'
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        blocked = executor.submit(manager.step, first_id, blocked_action)
        assert started.wait(timeout=2)
        free = executor.submit(manager.step, second_id, free_action)
        try:
            free_result = free.result(timeout=2)
        finally:
            release.set()
        blocked_result = blocked.result(timeout=2)

    assert free_result["info"]["domain_action"] == "search"
    assert blocked_result["info"]["domain_action"] == "search"


def test_gold_scorer_and_private_paths_are_absent_from_public_surfaces(
    tmp_path: Path,
) -> None:
    _, _, _, workspaces, manager = _components(tmp_path, EvaluationArm.AMG_MEMORY)
    env_id = manager.create()["id"]
    reset = manager.reset(env_id, 0)
    public = json.dumps(
        {"metadata": manager.metadata(), "reset": reset},
        ensure_ascii=False,
    ).casefold()
    assert "final_answer" not in public
    assert "annotator" not in public
    assert "scorer" not in public
    assert "gold" not in public
    assert str(tmp_path).casefold() not in public
    assert list(workspaces[0].root.iterdir()) == []


def test_one_policy_output_cannot_mix_tool_and_answer(tmp_path: Path) -> None:
    _, backend, _, _, manager = _components(tmp_path, EvaluationArm.NATIVE)
    env_id = manager.create()["id"]
    manager.reset(env_id, 0)
    result = manager.step(
        env_id,
        '<tool_call>{"name":"search","arguments":{"query":"alpha"}}</tool_call>'
        "<answer>forty two</answer>",
    )
    assert result["done"] is False
    assert result["info"]["status"] == "invalid_action"
    assert result["info"]["step"] == 1
    assert backend.call_trace == ()


def test_backend_failure_propagates_for_infrastructure_exclusion(
    tmp_path: Path,
) -> None:
    runtime, backend, _, _, manager = _components(tmp_path, EvaluationArm.NATIVE)

    def fail_search(*args, **kwargs):
        raise BackendError("synthetic backend failure")

    backend.search = fail_search  # type: ignore[method-assign]
    env_id = manager.create()["id"]
    manager.reset(env_id, 0)
    with pytest.raises(BackendError, match="synthetic backend failure"):
        manager.step(
            env_id,
            '<tool_call>{"name":"search","arguments":{"query":"alpha"}}</tool_call>',
        )
    assert not Path(str(runtime.predictions) + ".partial").exists()
    assert not runtime.predictions.exists()


def test_parser_implementation_error_is_not_reclassified_as_model_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _, _, _, manager = _components(tmp_path, EvaluationArm.NATIVE)
    env_id = manager.create()["id"]
    manager.reset(env_id, 0)

    def broken_parser(_raw: str):
        raise ValueError("synthetic parser implementation failure")

    monkeypatch.setattr(
        "agentenv_gaia_text.wrapper._parse_tool_call",
        broken_parser,
    )
    with pytest.raises(ValueError, match="synthetic parser implementation failure"):
        manager.step(env_id, "any policy output")
    assert not Path(str(runtime.predictions) + ".partial").exists()
    assert not runtime.predictions.exists()
