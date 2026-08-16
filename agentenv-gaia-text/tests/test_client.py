from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlsplit

import pytest
from agentenv.controller.policy_turn import (
    bind_initial_policy_context,
    complete_policy_turn,
    prepare_policy_turn,
)
from agentenv_gaia_text.backend import FixtureBackend
from agentenv_gaia_text.client import (
    GAIA_TEXT_DOMAIN_PROMPT,
    GAIA_TEXT_MEMORY_AFFORDANCE,
    GAIA_TEXT_POLICY_CONTINUATION_MARKER,
    GaiaTextEnvClient,
)
from agentenv_gaia_text.contracts import EvaluationArm, ProtocolContract
from agentenv_gaia_text.dataset import GaiaTextDataset
from agentenv_gaia_text.server import create_app
from agentenv_gaia_text.submission import SubmissionStore
from agentenv_gaia_text.wrapper import GaiaTextEpisodeManager
from fastapi.testclient import TestClient
from support import FileWorkspace, protocol_kwargs, write_runtime_fixture


def _app_and_contract(tmp_path: Path, arm: EvaluationArm):
    runtime = write_runtime_fixture(tmp_path)
    contract = ProtocolContract(**protocol_kwargs(runtime.rows))
    dataset = GaiaTextDataset.load(
        runtime.manifest,
        runtime.questions,
        expected_questions_sha256=runtime.questions_sha256,
        contract=contract,
    )
    workspaces: list[FileWorkspace] = []

    def factory(env_id: int, task_id: str, episode_index: int):
        workspace = FileWorkspace(
            tmp_path / "workspaces",
            f"{env_id}-{episode_index}-{task_id}",
        )
        workspaces.append(workspace)
        return workspace

    manager = GaiaTextEpisodeManager(
        dataset,
        FixtureBackend.load(runtime.backend, runtime.backend_sha256),
        SubmissionStore(dataset.task_ids, runtime.predictions),
        arm=arm,
        workspace_factory=factory if arm is EvaluationArm.AMG_MEMORY else None,
        max_policy_steps=12,
    )
    return create_app(manager), dataset.public_metadata(), manager, workspaces


def _requests_adapter(test_client: TestClient):
    def request(method: str, url: str, **kwargs):
        kwargs.pop("timeout", None)
        parsed = urlsplit(url)
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query
        return test_client.request(method, path, **kwargs)

    return request


def test_native_prompt_and_client_have_zero_memory_or_compaction(
    tmp_path: Path,
) -> None:
    app, contract, _, _ = _app_and_contract(tmp_path, EvaluationArm.NATIVE)
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
    ):
        client = GaiaTextEnvClient(
            "http://gaia.test",
            arm="native",
            expected_protocol=contract,
        )
        assert client.policy_framing() == [
            {"role": "system", "content": GAIA_TEXT_DOMAIN_PROMPT}
        ]
        assert client.conversation_start[0]["value"] == GAIA_TEXT_DOMAIN_PROMPT
        client.reset(0)
        initial = bind_initial_policy_context(
            client,
            [
                {"role": "system", "content": GAIA_TEXT_DOMAIN_PROMPT},
                {"role": "user", "content": client.observe()},
            ],
        )
        assert client.policy_turn_candidate() is None
        prepared = prepare_policy_turn(
            client,
            initial,
            count_prompt_tokens=lambda messages: 80,
            max_prompt_tokens=150,
            max_model_tokens=200,
            max_response_tokens=40,
            max_observation_tokens=30,
            action_observation_envelope_tokens=10,
        )
        assert prepared.control_request is None
        invalid = client.step(
            'shell_command {"command":"printf forbidden > note.txt","workdir":"."}'
        )
        assert invalid.done is False
        assert invalid.info["policy_step_after"] == 1
        assert invalid.info["native_call_count_after"] == 1
        assert invalid.info["env_info"]["workspace_action_count"] == 0
        assert client.context_epoch == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observation", {"not": "text"}),
        ("reward", "0.0"),
        ("reward", True),
        ("reward", float("inf")),
        ("done", "false"),
        ("info", []),
    ),
)
def test_client_rejects_malformed_step_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    app, contract, _, _ = _app_and_contract(tmp_path, EvaluationArm.NATIVE)
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
    ):
        client = GaiaTextEnvClient(
            "http://gaia.test",
            arm="native",
            expected_protocol=contract,
        )
        client.reset(0)
        request = client._request

        def malformed(method: str, path: str, **kwargs):
            response = request(method, path, **kwargs)
            if method == "POST" and path == "step":
                response[field] = value
            return response

        client._request = malformed  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="step response types drifted"):
            client.step('<answer>fixture</answer>')


def test_memory_prompt_diff_is_only_memory_and_compaction_affordance(
    tmp_path: Path,
) -> None:
    app, contract, _, _ = _app_and_contract(tmp_path, EvaluationArm.AMG_MEMORY)
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
    ):
        client = GaiaTextEnvClient(
            "http://gaia.test",
            arm="amg_memory",
            expected_protocol=contract,
        )
        memory_prompt = client.policy_framing()[0]["content"]
        assert memory_prompt == GAIA_TEXT_DOMAIN_PROMPT + GAIA_TEXT_MEMORY_AFFORDANCE
        assert client.conversation_start[0]["value"] == memory_prompt


def _exercise_compaction(tmp_path: Path, arm: EvaluationArm) -> dict[str, object]:
    app, contract, manager, workspaces = _app_and_contract(tmp_path, arm)
    with TestClient(app) as http:
        request = Mock(wraps=_requests_adapter(http))
        with patch("agentenv.envs.gaia_text.requests.request", request):
            client = GaiaTextEnvClient(
                "http://gaia.test",
                arm=arm.value,
                expected_protocol=contract,
            )
            client.reset(0)
            initial = bind_initial_policy_context(
                client,
                [
                    {
                        "role": "system",
                        "content": client.policy_framing()[0]["content"],
                    },
                    {"role": "user", "content": client.observe()},
                ],
            )
            candidate = client.policy_turn_candidate()
            prepared = prepare_policy_turn(
                client,
                initial,
                count_prompt_tokens=lambda messages: (
                    100 if messages[-1]["content"] == client.compaction_request else 80
                ),
                max_prompt_tokens=150,
                max_model_tokens=200,
                max_response_tokens=40,
                max_observation_tokens=30,
                action_observation_envelope_tokens=10,
            )
            before_compaction_requests = request.call_count
            compacted, replaced_messages = complete_policy_turn(
                client,
                prepared,
                "Preserve the same concise unresolved state.",
            )
            assert request.call_count == before_compaction_requests
            assert replaced_messages == compacted.info["context_transition"]["messages"]
            client.finalize_policy_horizon()
            assert client.close() is True
    return {
        "candidate": candidate,
        "control_request": prepared.control_request,
        "prompt_token_count": prepared.prompt_token_count,
        "compaction_request": client.compaction_request,
        "info": compacted.info,
        "metadata": manager.metadata(),
        "workspace_count": len(workspaces),
    }


def test_compacting_arms_share_trigger_transition_and_action_accounting(
    tmp_path: Path,
) -> None:
    compaction_only = _exercise_compaction(
        tmp_path / "compaction-only",
        EvaluationArm.AMG_COMPACTION_ONLY,
    )
    full_memory = _exercise_compaction(
        tmp_path / "full-memory",
        EvaluationArm.AMG_MEMORY,
    )

    for key in (
        "candidate",
        "control_request",
        "prompt_token_count",
        "compaction_request",
    ):
        assert compaction_only[key] == full_memory[key]
    compaction_info = deepcopy(compaction_only["info"])
    memory_info = deepcopy(full_memory["info"])
    assert isinstance(compaction_info, dict)
    assert isinstance(memory_info, dict)
    compaction_messages = compaction_info["context_transition"]["messages"]
    memory_messages = memory_info["context_transition"]["messages"]
    assert memory_messages[0]["content"] == (
        compaction_messages[0]["content"] + GAIA_TEXT_MEMORY_AFFORDANCE
    )
    memory_messages[0]["content"] = compaction_messages[0]["content"]
    assert compaction_info == memory_info
    info = compaction_info
    assert info["policy_step_before"] == 0
    assert info["policy_step_after"] == 1
    assert info["native_call_count_before"] == 0
    assert info["native_call_count_after"] == 0
    assert info["context_epoch_before"] == 0
    assert info["context_epoch_after"] == 1
    assert info["context_transition"]["operation"] == "replace_messages"
    assert info["context_transition"]["messages"][-1] == {
        "role": "user",
        "content": GAIA_TEXT_POLICY_CONTINUATION_MARKER,
    }
    assert "workspace" not in json.dumps(
        info["context_transition"], sort_keys=True
    ).casefold()
    assert compaction_only["workspace_count"] == 0
    assert full_memory["workspace_count"] == 1
    assert compaction_only["metadata"]["active_workspace_count"] == 0


def test_compaction_only_prompt_and_receipts_expose_no_memory_capability(
    tmp_path: Path,
) -> None:
    app, contract, manager, workspaces = _app_and_contract(
        tmp_path, EvaluationArm.AMG_COMPACTION_ONLY
    )
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
    ):
        client = GaiaTextEnvClient(
            "http://gaia.test",
            arm="amg_compaction_only",
            expected_protocol=contract,
        )
        assert client.policy_framing() == [
            {"role": "system", "content": GAIA_TEXT_DOMAIN_PROMPT}
        ]
        assert GAIA_TEXT_MEMORY_AFFORDANCE not in client.conversation_start[0]["value"]
        assert "workspace" not in client.compaction_request.casefold()
        client.reset(0)
        attempted = client.step(
            'shell_command {"command":"printf forbidden > note.txt","workdir":"."}'
        )
        serialized = json.dumps(attempted.info, sort_keys=True)
        assert attempted.done is False
        assert attempted.info["env_info"]["status"] == "invalid_action"
        assert attempted.info["env_info"]["workspace_action_count"] == 0
        assert attempted.info["env_info"]["domain_action"] == "invalid"
        assert "workspace_op" not in serialized
        assert '"kind": "workspace"' not in serialized
        assert str(tmp_path) not in serialized
        assert workspaces == []
        assert not (tmp_path / "workspaces").exists()
        assert manager.metadata()["active_workspace_count"] == 0


def test_compaction_only_client_rejects_forged_memory_runtime_without_path_leak(
    tmp_path: Path,
) -> None:
    app, contract, manager, _ = _app_and_contract(
        tmp_path, EvaluationArm.AMG_COMPACTION_ONLY
    )
    hidden_path = str(tmp_path / "hidden-memory-root")
    forged = manager.metadata()
    forged["workspace_runtime"] = {"root": hidden_path}
    manager.metadata = lambda: deepcopy(forged)  # type: ignore[method-assign]
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
        pytest.raises(RuntimeError, match="exposed runtime state") as error,
    ):
        GaiaTextEnvClient(
            "http://gaia.test",
            arm="amg_compaction_only",
            expected_protocol=contract,
        )
    assert hidden_path not in str(error.value)


def test_memory_write_survives_client_owned_replace_messages_compaction(
    tmp_path: Path,
) -> None:
    app, contract, manager, workspaces = _app_and_contract(
        tmp_path, EvaluationArm.AMG_MEMORY
    )
    with TestClient(app) as http:
        request = Mock(wraps=_requests_adapter(http))
        with patch("agentenv.envs.gaia_text.requests.request", request):
            client = GaiaTextEnvClient(
                "http://gaia.test",
                arm="amg_memory",
                expected_protocol=contract,
            )
            client.reset(0)
            initial = bind_initial_policy_context(
                client,
                [
                    {
                        "role": "system",
                        "content": client.policy_framing()[0]["content"],
                    },
                    {"role": "user", "content": client.observe()},
                ],
            )

            written = client.step(
                'shell_command {"command":"printf memory-survives > notes.txt","workdir":"."}'
            )
            assert written.info["policy_step_after"] == 1
            assert written.info["native_call_count_after"] == 1
            assert (workspaces[0].root / "notes.txt").read_text() == "memory-survives"

            messages = initial + [
                {"role": "assistant", "content": "workspace write"},
                {"role": "user", "content": written.state},
            ]
            prepared = prepare_policy_turn(
                client,
                messages,
                count_prompt_tokens=lambda candidate: (
                    100 if candidate[-1]["content"] == client.compaction_request else 80
                ),
                max_prompt_tokens=150,
                max_model_tokens=200,
                max_response_tokens=40,
                max_observation_tokens=30,
                action_observation_envelope_tokens=10,
            )
            assert prepared.control_request == client.compaction_request
            assert prepared.prompt_token_count == 100
            before_compaction_requests = request.call_count
            compacted, replaced_messages = complete_policy_turn(
                client,
                prepared,
                "Continue from notes.txt; preserve the unresolved state.",
            )
            assert request.call_count == before_compaction_requests
            assert compacted.done is False
            assert compacted.info["policy_step_before"] == 1
            assert compacted.info["policy_step_after"] == 2
            assert compacted.info["native_call_count_before"] == 1
            assert compacted.info["native_call_count_after"] == 1
            assert compacted.info["context_epoch_after"] == 1
            transition = compacted.info["context_transition"]
            assert transition["operation"] == "replace_messages"
            assert replaced_messages == transition["messages"]
            assert transition["messages"][-2:] == [
                {
                    "role": "assistant",
                    "content": "Continue from notes.txt; preserve the unresolved state.",
                },
                {
                    "role": "user",
                    "content": GAIA_TEXT_POLICY_CONTINUATION_MARKER,
                },
            ]
            assert compacted.info["wrapper_evidence"] == {
                "event": "context_compaction",
                "native_environment_call_count": 0,
            }

            read = client.step(
                'shell_command {"command":"cat notes.txt","workdir":"."}'
            )
            assert "memory-survives" in read.state
            assert read.info["policy_step_after"] == 3
            assert read.info["native_call_count_after"] == 2
            assert manager.metadata()["active_workspace_count"] == 1


def test_http_server_client_round_trip_and_external_horizon(tmp_path: Path) -> None:
    app, contract, _, _ = _app_and_contract(tmp_path, EvaluationArm.NATIVE)
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
    ):
        client = GaiaTextEnvClient(
            "http://gaia.test",
            arm="native",
            expected_protocol=contract,
        )
        reset = client.reset(0)
        assert reset["done"] is False
        search = client.step(
            '<tool_call>{"name":"search","arguments":{"query":"alpha evidence"}}</tool_call>'
        )
        assert "gaia-text://fixture/alpha" in search.state
        terminal = client.finalize_policy_horizon()
        assert terminal.done is True
        assert terminal.reward == 0.0
        assert terminal.info["env_info"]["status"] == "policy_horizon"
        assert client.close() is True


def test_client_validates_and_optionally_pins_paired_runtime_digest(
    tmp_path: Path,
) -> None:
    app, contract, manager, _ = _app_and_contract(tmp_path, EvaluationArm.NATIVE)
    expected_digest = manager.metadata()["paired_runtime_contract_sha256"]
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
    ):
        client = GaiaTextEnvClient(
            "http://gaia.test",
            arm="native",
            expected_protocol=contract,
            expected_paired_runtime_sha256=expected_digest,
        )
        assert client.paired_runtime_contract_sha256 == expected_digest
        assert (
            client.paired_runtime_contract
            == manager.metadata()["paired_runtime_contract"]
        )
        assert client.close() is True
        with pytest.raises(RuntimeError, match="paired-runtime digest mismatch"):
            GaiaTextEnvClient(
                "http://gaia.test",
                arm="native",
                expected_protocol=contract,
                expected_paired_runtime_sha256="0" * 64,
            )


def test_client_rejects_forged_paired_runtime_digest(tmp_path: Path) -> None:
    app, contract, manager, _ = _app_and_contract(tmp_path, EvaluationArm.NATIVE)
    forged = manager.metadata()
    forged["paired_runtime_contract_sha256"] = "0" * 64
    manager.metadata = lambda: deepcopy(forged)  # type: ignore[method-assign]
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
        pytest.raises(RuntimeError, match="canonical digest"),
    ):
        GaiaTextEnvClient(
            "http://gaia.test",
            arm="native",
            expected_protocol=contract,
        )


def test_client_rejects_extra_paired_runtime_contract_fields(tmp_path: Path) -> None:
    app, contract, manager, _ = _app_and_contract(tmp_path, EvaluationArm.NATIVE)
    forged = manager.metadata()
    forged_contract = forged["paired_runtime_contract"]
    forged_contract["unexpected"] = True
    canonical = json.dumps(
        forged_contract,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    forged["paired_runtime_contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    manager.metadata = lambda: deepcopy(forged)  # type: ignore[method-assign]
    with (
        TestClient(app) as http,
        patch(
            "agentenv.envs.gaia_text.requests.request",
            side_effect=_requests_adapter(http),
        ),
        pytest.raises(RuntimeError, match="contract schema"),
    ):
        GaiaTextEnvClient(
            "http://gaia.test",
            arm="native",
            expected_protocol=contract,
        )
