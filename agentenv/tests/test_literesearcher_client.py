from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from agentenv.controller import complete_policy_turn, prepare_policy_turn
from agentenv.controller.types import PolicyContextPressure
from agentenv.envs.literesearcher import (
    LITERESEARCHER_CONTEXT_COMPACTION_EXAMPLE,
    LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
    LITERESEARCHER_CONTINUATION_PATH,
    LITERESEARCHER_CONTINUATION_READ_ACTION,
    LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
    LITERESEARCHER_POLICY_CONTINUATION_MARKER,
    LITERESEARCHER_RESEARCH_NOTE_PATH,
    LITERESEARCHER_RESEARCH_NOTE_READ_ACTION,
    LITERESEARCHER_RESEARCH_NOTE_WRITE_EXAMPLE,
    LITERESEARCHER_TOOL_SERIALIZATION_CONTRACT,
    LiteResearcherEnvClient,
    _is_workspace_action_candidate,
    _parse_legacy_tool_call,
    _parse_qwen35_tool_call,
    _render_qwen35_tool_call,
)


class LiteResearcherClientTests(unittest.TestCase):
    @staticmethod
    def _client() -> LiteResearcherEnvClient:
        client = object.__new__(LiteResearcherEnvClient)
        client.env_server_base = "http://literesearcher.example"
        client.timeout = 30
        client.env_id = 7
        client.info = {"observation": "Which source answers this question?"}
        client.max_policy_steps = 40
        client.invalid_action_reward = -0.01
        client._reset_policy_transition_state()
        client._current_policy_context = [
            {"role": "user", "content": "Which source answers this question?"}
        ]
        return client

    @staticmethod
    def _workspace_envelope_arguments(action: str) -> dict:
        parsed = _parse_qwen35_tool_call(action)
        if parsed is None:
            raise AssertionError("workspace action is not one native tool_call")
        name, arguments, _prefix_chars = parsed
        if name != "shell_command":
            raise AssertionError("workspace envelope is not shell_command")
        return arguments

    def test_policy_framing_exposes_normalized_conversation_start(self) -> None:
        framing = self._client().policy_framing()
        self.assertEqual(
            [message["role"] for message in framing], ["user", "assistant"]
        )
        self.assertIn("deep-research agent", framing[0]["content"])
        self.assertEqual(
            framing[1], {"role": "assistant", "content": "Understood."}
        )

    def test_policy_framing_exposes_selective_literal_workspace_actions(self) -> None:
        prompt = self._client().policy_framing()[0]["content"]
        self.assertIn(LITERESEARCHER_RESEARCH_NOTE_WRITE_EXAMPLE, prompt)
        self.assertIn(LITERESEARCHER_RESEARCH_NOTE_READ_ACTION, prompt)
        self.assertIn(f"{LITERESEARCHER_RESEARCH_NOTE_PATH} is optional", prompt)
        self.assertIn("Do not write it after every useful Visit", prompt)
        self.assertIn("answer directly instead of staging", prompt)
        self.assertIn("use the Qwen3.5 native tool-call format", prompt)
        self.assertIn("uses function shell_command", prompt)
        self.assertNotIn("use raw Codex syntax", prompt)
        self.assertIn("create or replace the note", prompt)
        self.assertIn("distinct from the optional research note", prompt)
        self.assertIn("At an explicit context-checkpoint request", prompt)
        self.assertIn("executable write", prompt)
        self.assertIn("not free-form continuation text", prompt)
        self.assertNotIn("After the first useful Visit", prompt)

    def test_research_note_write_example_is_parseable_and_idempotent(self) -> None:
        payload = self._workspace_envelope_arguments(
            LITERESEARCHER_RESEARCH_NOTE_WRITE_EXAMPLE
        )
        with tempfile.TemporaryDirectory() as td:
            note = Path(td) / LITERESEARCHER_RESEARCH_NOTE_PATH
            for stale in (None, "stale content\n"):
                if stale is not None:
                    note.parent.mkdir(parents=True, exist_ok=True)
                    note.write_text(stale, encoding="utf-8")
                result = subprocess.run(
                    ["bash", "-lc", payload["command"]],
                    cwd=td,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                content = note.read_text(encoding="utf-8")
                self.assertIn("question: ...", content)
                self.assertIn("evidence_with_urls: ...", content)
                self.assertNotIn("stale content", content)

    def test_native_tool_parser_is_strict_about_structure_and_parameters(self) -> None:
        action = _render_qwen35_tool_call(
            "visit",
            url="https://literesearcher.local/page/00001",
            goal="extract evidence",
            page=2,
        )
        self.assertEqual(
            _parse_qwen35_tool_call(action),
            (
                "visit",
                {
                    "url": "https://literesearcher.local/page/00001",
                    "goal": "extract evidence",
                    "page": 2,
                },
                0,
            ),
        )
        self.assertIsNone(
            _parse_qwen35_tool_call(
                "<tool_call><function=visit>"
                "<parameter=page>one</parameter>"
                "</function></tool_call>"
            )
        )
        self.assertIsNone(
            _parse_qwen35_tool_call(
                "<tool_call><function=search>"
                "<parameter=query>first</parameter>"
                "<parameter=query>second</parameter>"
                "</function></tool_call>"
            )
        )
        self.assertIsNone(
            _parse_qwen35_tool_call(
                "<tool_call><function=search>"
                "<parameter=query>incomplete"
                "</function></tool_call>"
            )
        )

    def test_valid_legacy_checkpoint_envelope_remains_compatible_without_repair(self) -> None:
        valid = (
            '<tool_call>{"name":"shell_command","arguments":'
            '{"command":"cat .agent_memory/CONTINUATION.md",'
            '"workdir":".","timeout_ms":10000}}</tool_call>'
        )
        malformed = valid.replace("}}</tool_call>", "}</tool_call>")

        self.assertEqual(_parse_legacy_tool_call(valid)[0], "shell_command")
        self.assertTrue(_is_workspace_action_candidate(valid))
        self.assertIsNone(_parse_legacy_tool_call(malformed))
        self.assertFalse(_is_workspace_action_candidate(malformed))

    def test_tool_parser_allows_one_bounded_prefix_and_rejects_suffix(self) -> None:
        action = _render_qwen35_tool_call(
            "shell_command",
            command="cat .agent_memory/CONTINUATION.md",
            workdir=".",
            timeout_ms=10000,
        )
        prefix = "I will recover the verified checkpoint now.\n"
        parsed = _parse_qwen35_tool_call(prefix + action)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed[2], len(prefix))
        self.assertTrue(_is_workspace_action_candidate(prefix + action))
        self.assertIsNone(_parse_qwen35_tool_call(action + " trailing prose"))
        self.assertFalse(_is_workspace_action_candidate(action + " trailing prose"))
        raw_workspace_action = 'shell_command {"command":"pwd"}'
        for fenced_prefix in ("```analysis```\n", "~~~xml\n"):
            with self.subTest(fenced_prefix=fenced_prefix):
                self.assertIsNone(_parse_qwen35_tool_call(fenced_prefix + action))
                self.assertFalse(
                    _is_workspace_action_candidate(fenced_prefix + action)
                )
                self.assertFalse(
                    _is_workspace_action_candidate(
                        fenced_prefix + raw_workspace_action
                    )
                )
        self.assertIsNone(_parse_qwen35_tool_call(("x" * 2049) + action))

    def test_legacy_tool_parser_rejects_duplicate_keys(self) -> None:
        duplicate_top = (
            '<tool_call>{"name":"shell_command","name":"search",'
            '"arguments":{"command":"true"}}</tool_call>'
        )
        duplicate_argument = (
            '<tool_call>{"name":"shell_command","arguments":'
            '{"command":"true","command":"false"}}</tool_call>'
        )
        self.assertIsNone(_parse_legacy_tool_call(duplicate_top))
        self.assertIsNone(_parse_legacy_tool_call(duplicate_argument))
        self.assertFalse(_is_workspace_action_candidate(duplicate_top))
        self.assertFalse(_is_workspace_action_candidate(duplicate_argument))


    def test_compaction_request_requires_one_real_bounded_workspace_write(self) -> None:
        request = LITERESEARCHER_CONTEXT_COMPACTION_REQUEST
        self.assertIn("CHECKPOINT WRITE PHASE", request)
        self.assertIn(LITERESEARCHER_CONTINUATION_PATH, request)
        self.assertIn("overwrite", request.lower())
        self.assertIn("write-only phase", request)
        self.assertIn("mkdir -p .agent_memory &&", request)
        self.assertIn("Qwen3.5-native <tool_call>/<function=shell_command>", request)
        self.assertIn("Do not emit raw shell_command syntax", request)
        self.assertIn("cat > .agent_memory/CONTINUATION.md <<'AMG_CHECKPOINT'", request)
        self.assertIn("8192", request)
        self.assertNotIn("`", request)
        self.assertNotIn("will not call", request)
        self.assertTrue(request.endswith(LITERESEARCHER_CONTEXT_COMPACTION_EXAMPLE))
        self.assertIn("CHECKPOINT READ PHASE", LITERESEARCHER_POLICY_CONTINUATION_MARKER)
        self.assertNotIn("`", LITERESEARCHER_POLICY_CONTINUATION_MARKER)
        self.assertTrue(
            LITERESEARCHER_POLICY_CONTINUATION_MARKER.endswith(
                LITERESEARCHER_CONTINUATION_READ_ACTION
            )
        )

    def test_compaction_example_is_native_parseable_and_shell_executable(self) -> None:
        payload = self._workspace_envelope_arguments(
            LITERESEARCHER_CONTEXT_COMPACTION_EXAMPLE
        )
        with tempfile.TemporaryDirectory() as td:
            result = subprocess.run(
                ["bash", "-lc", payload["command"]],
                cwd=td,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            checkpoint = Path(td) / LITERESEARCHER_CONTINUATION_PATH
            self.assertTrue(checkpoint.is_file())
            self.assertGreater(checkpoint.stat().st_size, 0)
        self.assertEqual(payload["workdir"], ".")
        self.assertEqual(payload["timeout_ms"], 10_000)
        read_payload = self._workspace_envelope_arguments(
            LITERESEARCHER_CONTINUATION_READ_ACTION
        )
        self.assertEqual(
            read_payload,
            {
                "command": "cat .agent_memory/CONTINUATION.md",
                "workdir": ".",
                "timeout_ms": 10_000,
            },
        )

    def test_checkpoint_retry_keeps_unquoted_example_as_final_text(self) -> None:
        client = self._bound_client()
        client._checkpoint_retry_reason = "missing_receipt"
        retry = client.policy_turn_candidate()
        self.assertIsNotNone(retry)
        self.assertNotIn("`", retry)
        self.assertTrue(retry.endswith(LITERESEARCHER_CONTEXT_COMPACTION_EXAMPLE))

    @staticmethod
    def _bound_client(*, selected: bool = True) -> LiteResearcherEnvClient:
        client = LiteResearcherClientTests._client()
        client.metadata = {
            "max_policy_steps": 40,
            "compaction_contract": "task_neutral_filesystem_checkpoint_v1",
        }
        client._reset_policy_transition_state()
        client._immutable_policy_context = [
            {"role": "system", "content": "system framing"},
            {"role": "user", "content": "original question"},
        ]
        client._current_policy_context = [
            dict(message) for message in client._immutable_policy_context
        ]
        client._policy_context_bound = True
        client._selected_policy_control = (
            "context_compaction" if selected else None
        )
        client._checkpoint_retry_context = (
            [dict(message) for message in client._current_policy_context]
            if selected
            else None
        )
        return client

    @staticmethod
    def _checkpoint_response(*, valid: bool, done: bool = False) -> dict:
        receipt = {
            "schema": "agentmemory_continuation_checkpoint_v2",
            "path": ".agent_memory/CONTINUATION.md",
            "action_kind": "SHELL_COMMAND",
            "action_execution_succeeded": valid,
            "change_kind": "added" if valid else None,
            "before_sha256": None,
            "content_changed": valid,
            "changed_in_action": valid,
            "nonempty": valid,
            "within_size_limit": valid,
            "bytes": 128 if valid else None,
            "sha256": "a" * 64 if valid else None,
            "valid": valid,
            "rejection_reason": None if valid else "not_changed_in_action",
        }
        return {
            "observation": "Done!",
            "reward": 0.0,
            "done": done,
            "info": {
                "status": "active" if not done else "success",
                "native_environment_call_count": 0,
                "action_submission": {
                    "kind": "workspace",
                    "op": "SHELL_COMMAND",
                },
                "wrapper_evidence": {
                    "continuation_checkpoint": receipt,
                    "native_environment_call_count": 0,
                },
            },
        }

    def test_verified_checkpoint_write_replaces_without_leaking_write_content(self) -> None:
        client = self._bound_client()
        client._request = Mock(return_value=self._checkpoint_response(valid=True))
        client._checkpoint_retry_reason = "previous_rejection"
        raw_action = _render_qwen35_tool_call(
            "shell_command",
            command="printf secret-evidence > .agent_memory/CONTINUATION.md",
            workdir=".",
            timeout_ms=10000,
        )

        output = client.step(raw_action)

        transition = output.info["context_transition"]
        self.assertEqual(transition["operation"], "replace_messages")
        self.assertEqual(
            transition["messages"],
            [
                {"role": "system", "content": "system framing"},
                {"role": "user", "content": "original question"},
                {
                    "role": "user",
                    "content": LITERESEARCHER_POLICY_CONTINUATION_MARKER,
                },
            ],
        )
        rendered = repr(transition["messages"])
        self.assertNotIn("secret-evidence", rendered)
        self.assertNotIn(raw_action, rendered)
        self.assertEqual(output.info["native_call_count_after"], 1)
        self.assertEqual(output.info["policy_step_after"], 1)
        self.assertEqual(output.info["context_epoch_after"], 1)
        self.assertEqual(
            output.info["wrapper_evidence"]["event"],
            "forced_checkpoint_write",
        )
        self.assertTrue(
            output.info["wrapper_evidence"]["continuation_checkpoint"]["valid"]
        )
        self.assertIsNone(client._checkpoint_retry_reason)

    def test_rejected_checkpoint_retries_restore_pre_attempt_context_without_growth(
        self,
    ) -> None:
        client = self._bound_client(selected=False)
        client.max_policy_steps = 40
        client._request = Mock()
        research_context = [
            {"role": "system", "content": "system framing"},
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "evidence " + ("x" * 16_800)},
            {"role": "user", "content": "bounded visit result"},
        ]
        messages = [dict(message) for message in research_context]
        prompt_sizes: list[int] = []
        invalid_action = _render_qwen35_tool_call("search", query=["source"])

        def count_prompt_tokens(candidate: list[dict[str, str]]) -> int:
            return sum(len(message["content"]) for message in candidate)

        for attempt in range(30):
            prepared = prepare_policy_turn(
                client,
                messages,
                count_prompt_tokens=count_prompt_tokens,
                max_prompt_tokens=30_720,
                max_model_tokens=32_768,
                max_response_tokens=2_048,
                max_observation_tokens=(
                    LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE
                ),
            )
            if attempt == 0:
                self.assertEqual(
                    prepared.control_request,
                    LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
                )
            else:
                self.assertIn("CHECKPOINT WRITE RETRY", prepared.control_request)
                self.assertIn("workspace_action_required", prepared.control_request)
                self.assertIn("Do not read", prepared.control_request)
                self.assertNotIn('query":["source"]', prepared.control_request)
            prompt_sizes.append(prepared.prompt_token_count)
            output, messages = complete_policy_turn(
                client, prepared, invalid_action
            )
            self.assertEqual(output.reward, -0.01)
            self.assertFalse(output.done)
            self.assertEqual(messages, research_context)
            self.assertEqual(
                output.info["context_transition"]["operation"],
                "replace_messages",
            )
            self.assertTrue(
                output.info["wrapper_evidence"]["retry_context_restored"]
            )
            self.assertEqual(client._policy_step_count, attempt + 1)

        self.assertEqual(len(set(prompt_sizes[1:])), 1)
        self.assertEqual(client._policy_step_count, 30)
        client._request.assert_not_called()

    def test_rejected_checkpoint_remains_pending_below_pressure_threshold(
        self,
    ) -> None:
        client = self._bound_client(selected=False)
        client._request = Mock()
        research_context = [
            {"role": "system", "content": "system framing"},
            {"role": "user", "content": "original question"},
            {"role": "assistant", "content": "evidence"},
            {"role": "user", "content": "bounded visit result"},
        ]
        messages = [dict(message) for message in research_context]

        over_threshold = PolicyContextPressure(
            action_prompt_tokens=16_257,
            candidate_prompt_tokens=16_415,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
            action_observation_envelope_tokens=127,
        )
        self.assertEqual(
            client.prepare_policy_turn(over_threshold),
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )
        output = client.step(
            _render_qwen35_tool_call("search", query=["source"])
        )
        self.assertFalse(output.done)
        self.assertTrue(
            output.info["wrapper_evidence"]["retry_context_restored"]
        )

        # A full chat-template rerender can shorten an otherwise identical
        # message list by one token.  An already-pending checkpoint must not be
        # dropped merely because the recomputed pressure is now just below the
        # threshold.
        below_threshold_after_rerender = PolicyContextPressure(
            action_prompt_tokens=16_256,
            candidate_prompt_tokens=16_414,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
            action_observation_envelope_tokens=127,
        )
        retry = client.prepare_policy_turn(below_threshold_after_rerender)
        self.assertIn("CHECKPOINT WRITE RETRY", retry)
        self.assertIn("workspace_action_required", retry)
        self.assertEqual(client.policy_turn_candidate(), retry)
        self.assertEqual(client._selected_policy_control, "context_compaction")
        client._request.assert_not_called()

    def test_forced_checkpoint_rejects_research_action_without_endpoint_dispatch(self) -> None:
        client = self._bound_client()
        client._request = Mock()
        search_action = _render_qwen35_tool_call("search", query=["source"])

        output = client.step(search_action)

        client._request.assert_not_called()
        self.assertEqual(output.reward, -0.01)
        self.assertFalse(output.done)
        self.assertEqual(
            output.info["wrapper_evidence"]["reward_overlay"],
            {
                "schema": "literesearcher_invalid_action_reward_v1",
                "native_reward": 0.0,
                "penalty": -0.01,
                "total_reward": -0.01,
                "terminal": False,
            },
        )
        self.assertEqual(
            output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertEqual(
            output.info["context_transition"]["messages"],
            client._immutable_policy_context,
        )
        self.assertTrue(
            output.info["wrapper_evidence"]["retry_context_restored"]
        )
        self.assertEqual(output.info["native_call_count_after"], 0)
        self.assertEqual(output.info["policy_step_after"], 1)
        self.assertEqual(
            output.info["action_submission"]["raw_policy_output"],
            search_action,
        )
        self.assertFalse(
            output.info["wrapper_evidence"]["endpoint_step_dispatched"]
        )
        self.assertEqual(
            output.info["wrapper_evidence"]["checkpoint_rejection_reason"],
            "workspace_action_required",
        )

    def test_valid_diff_from_failed_checkpoint_action_restores_retry_context(self) -> None:
        client = self._bound_client()
        response = self._checkpoint_response(valid=True)
        receipt = response["info"]["wrapper_evidence"][
            "continuation_checkpoint"
        ]
        receipt["action_execution_succeeded"] = False
        receipt["valid"] = False
        receipt["rejection_reason"] = "action_execution_failed"
        client._request = Mock(return_value=response)

        output = client.step(
            'shell_command {"command":"write-then-exit-7"}'
        )

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertEqual(
            output.info["context_transition"]["messages"],
            client._immutable_policy_context,
        )
        self.assertTrue(
            output.info["wrapper_evidence"]["retry_context_restored"]
        )
        self.assertEqual(output.info["context_epoch_after"], 0)
        self.assertEqual(
            output.info["wrapper_evidence"]["checkpoint_rejection_reason"],
            "action_execution_failed",
        )

    def test_failed_checkpoint_write_restores_retry_context(self) -> None:
        client = self._bound_client()
        client._request = Mock(return_value=self._checkpoint_response(valid=False))

        output = client.step('shell_command {"command":"true"}')

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertEqual(
            output.info["context_transition"]["messages"],
            client._immutable_policy_context,
        )
        self.assertTrue(
            output.info["wrapper_evidence"]["retry_context_restored"]
        )
        self.assertEqual(output.info["context_epoch_after"], 0)
        self.assertEqual(output.info["native_call_count_after"], 1)
        self.assertEqual(output.info["policy_step_after"], 1)
        self.assertIn("not accepted", output.state.lower())
        self.assertIn(LITERESEARCHER_CONTINUATION_PATH, output.state)
        self.assertEqual(
            output.info["wrapper_evidence"]["event"],
            "forced_checkpoint_rejected",
        )
        self.assertEqual(output.reward, -0.01)
        self.assertEqual(
            output.info["wrapper_evidence"]["reward_overlay"]["penalty"],
            -0.01,
        )


    def test_endpoint_rejection_reason_is_visible_on_next_retry_prompt(self) -> None:
        client = self._bound_client()
        response = self._checkpoint_response(valid=False)
        response["info"]["wrapper_evidence"]["continuation_checkpoint"][
            "rejection_reason"
        ] = "action_execution_failed"
        client._request = Mock(return_value=response)

        output = client.step('shell_command {"command":"false","workdir":"."}')
        self.assertFalse(output.done)
        pressure = PolicyContextPressure(
            action_prompt_tokens=100,
            candidate_prompt_tokens=200,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
            action_observation_envelope_tokens=4,
        )
        retry = client.prepare_policy_turn(pressure)
        self.assertEqual(client.policy_turn_candidate(), retry)
        self.assertIn("CHECKPOINT WRITE RETRY", retry)
        self.assertIn("action_execution_failed", retry)
        self.assertIn("Do not read", retry)
        self.assertNotIn('shell_command {"command":"false"', retry)

    def test_stale_modified_receipt_with_identical_digest_does_not_replace(self) -> None:
        client = self._bound_client()
        response = self._checkpoint_response(valid=True)
        receipt = response["info"]["wrapper_evidence"]["continuation_checkpoint"]
        receipt.update(
            {
                "change_kind": "modified",
                "before_sha256": receipt["sha256"],
                "content_changed": True,
            }
        )
        client._request = Mock(return_value=response)

        output = client.step('shell_command {"command":"rewrite-identically"}')

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "replace_messages",
        )
        self.assertEqual(
            output.info["context_transition"]["messages"],
            client._immutable_policy_context,
        )
        self.assertTrue(
            output.info["wrapper_evidence"]["retry_context_restored"]
        )
        self.assertEqual(output.info["context_epoch_after"], 0)
        self.assertEqual(
            output.info["wrapper_evidence"]["checkpoint_rejection_reason"],
            "inconsistent_valid_receipt",
        )
        self.assertEqual(output.reward, -0.01)
        self.assertEqual(
            output.info["wrapper_evidence"]["reward_overlay"]["penalty"],
            -0.01,
        )

    def test_checkpoint_rejection_fails_closed_when_write_read_answer_no_longer_fit(
        self,
    ) -> None:
        client = self._bound_client()
        client._policy_step_count = 37
        client._request = Mock()

        output = client.step(
            _render_qwen35_tool_call("search", query=["source"])
        )

        self.assertTrue(output.done)
        self.assertEqual(output.reward, -0.01)
        self.assertEqual(
            output.info["env_info"]["status"],
            "checkpoint_retry_budget_exhausted",
        )
        self.assertEqual(
            output.info["wrapper_evidence"]["event"],
            "forced_checkpoint_retry_budget_exhausted",
        )
        self.assertFalse(
            output.info["wrapper_evidence"]["retry_context_restored"]
        )
        self.assertEqual(
            output.info["wrapper_evidence"][
                "checkpoint_retry_remaining_actions"
            ],
            2,
        )
        client._request.assert_not_called()

    def test_horizon_finalization_clears_pending_checkpoint_retry(self) -> None:
        client = self._bound_client()
        client._checkpoint_retry_reason = "workspace_action_required"

        output = client.finalize_policy_horizon()

        self.assertTrue(output.done)
        self.assertEqual(
            output.info["env_info"]["status"], "max_policy_steps_exhausted"
        )
        self.assertIsNone(client._checkpoint_retry_context)
        self.assertIsNone(client._checkpoint_retry_reason)
        self.assertIsNone(client._selected_policy_control)
        self.assertEqual(
            client.policy_turn_candidate(),
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )

    def test_checkpoint_backend_fault_is_excluded_without_policy_penalty(self) -> None:
        client = self._bound_client()
        client._request = Mock(
            return_value={
                "observation": "Workspace backend failed; episode excluded.",
                "reward": 0.0,
                "done": True,
                "info": {
                    "status": "environment_error",
                    "sample_excluded": True,
                    "action_submission": {"raw_policy_output": "checkpoint"},
                    "wrapper_evidence": {
                        "native_environment_call_count": 0,
                        "backend_error": "WorkspaceError",
                    },
                },
            }
        )

        output = client.step(
            'shell_command {"command":"printf state > '
            '.agent_memory/CONTINUATION.md","workdir":"."}'
        )

        self.assertEqual(output.reward, 0.0)
        self.assertTrue(output.done)
        self.assertTrue(output.info["env_info"]["sample_excluded"])
        self.assertNotIn("reward_overlay", output.info["wrapper_evidence"])
        self.assertEqual(
            output.info["wrapper_evidence"]["event"],
            "forced_checkpoint_terminal",
        )

    def test_checkpoint_accepts_prior_cumulative_research_calls(self) -> None:
        client = self._bound_client()
        response = self._checkpoint_response(valid=True)
        response["info"]["native_environment_call_count"] = 7
        client._request = Mock(return_value=response)

        output = client.step(
            'shell_command {"command":"printf state > '
            '.agent_memory/CONTINUATION.md","workdir":"."}'
        )

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "replace_messages",
        )

    def test_checkpoint_reward_or_backend_call_fails_closed(self) -> None:
        for mutation, message in (
            (("reward", 0.25), "changed reward"),
            (("native_environment_call_count", 1), "research backend"),
        ):
            with self.subTest(mutation=mutation):
                client = self._bound_client()
                response = self._checkpoint_response(valid=True)
                key, value = mutation
                if key == "reward":
                    response[key] = value
                else:
                    response["info"]["wrapper_evidence"][key] = value
                client._request = Mock(return_value=response)
                with self.assertRaisesRegex(RuntimeError, message):
                    client.step(
                        'shell_command {"command":"printf state > '
                        '.agent_memory/CONTINUATION.md","workdir":"."}'
                    )

    def test_compaction_is_not_forced_when_fewer_than_three_actions_remain(self) -> None:
        pressure = PolicyContextPressure(
            action_prompt_tokens=18_000,
            candidate_prompt_tokens=17_900,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
            action_observation_envelope_tokens=4,
        )
        client = self._bound_client(selected=False)
        client._policy_step_count = 38
        self.assertIsNone(client.prepare_policy_turn(pressure))
        self.assertIsNone(client._selected_policy_control)

        client._policy_step_count = 37
        self.assertEqual(
            client.prepare_policy_turn(pressure),
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )

    def test_candidate_exactly_at_capacity_still_selects_checkpoint(self) -> None:
        client = self._bound_client(selected=False)
        pressure = PolicyContextPressure(
            action_prompt_tokens=18_432,
            candidate_prompt_tokens=30_720,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
            action_observation_envelope_tokens=12_288,
        )

        self.assertEqual(
            client.prepare_policy_turn(pressure),
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertEqual(client._selected_policy_control, "context_compaction")

    def test_shorter_rendered_candidate_does_not_fail_without_pressure(self) -> None:
        client = self._client()
        client._policy_context_bound = True
        client._selected_policy_control = None
        pressure = PolicyContextPressure(
            action_prompt_tokens=140,
            candidate_prompt_tokens=130,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
            action_observation_envelope_tokens=4,
        )

        self.assertIsNone(client.prepare_policy_turn(pressure))
        self.assertIsNone(client._selected_policy_control)

    def test_shorter_rendered_candidate_compacts_when_append_would_overflow(self) -> None:
        client = self._client()
        client._policy_context_bound = True
        client._selected_policy_control = None
        pressure = PolicyContextPressure(
            action_prompt_tokens=18_000,
            candidate_prompt_tokens=17_900,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
            action_observation_envelope_tokens=4,
        )

        self.assertEqual(
            client.prepare_policy_turn(pressure),
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertEqual(client._selected_policy_control, "context_compaction")


    def test_underreported_observation_envelope_fails_before_sampling(self) -> None:
        client = self._client()
        client._policy_context_bound = True
        client._selected_policy_control = None
        pressure = PolicyContextPressure(
            action_prompt_tokens=18_000,
            candidate_prompt_tokens=17_900,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=4_096,
            action_observation_envelope_tokens=4,
        )

        with self.assertRaisesRegex(
            RuntimeError, "observation-token envelope is too small"
        ):
            client.prepare_policy_turn(pressure)

    def test_close_accepts_server_boolean_true(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = True
        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=response,
        ) as request:
            self.assertTrue(self._client().close())
        request.assert_called_once_with(
            "POST",
            "http://literesearcher.example/close",
            timeout=30,
            json={"id": 7},
        )

    def test_close_rejects_false_acknowledgement(self) -> None:
        response = Mock(status_code=200)
        response.json.return_value = False
        with (
            patch(
                "agentenv.envs.literesearcher.requests.request",
                return_value=response,
            ),
            self.assertRaisesRegex(requests.RequestException, "did not return true"),
        ):
            self._client().close()


class LiteResearcherInvalidActionRewardTests(unittest.TestCase):
    @staticmethod
    def _step_client(invalid_action_reward: float = -0.01) -> LiteResearcherEnvClient:
        client = object.__new__(LiteResearcherEnvClient)
        client.env_server_base = "http://literesearcher.example"
        client.timeout = 30
        client.env_id = 7
        client.invalid_action_reward = invalid_action_reward
        client.info = {"observation": "question", "info": {}}
        client._reset_policy_transition_state()
        return client

    def test_invalid_action_penalty_is_nonterminal_and_does_not_leak(self) -> None:
        client = self._step_client()
        client._request = Mock(
            side_effect=[
                {
                    "observation": "Invalid policy action: malformed",
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "status": "invalid_action",
                        "sample_excluded": False,
                        "action_submission": {"raw_policy_output": "bad"},
                        "wrapper_evidence": {"invalid_action": True},
                    },
                },
                {
                    "observation": "valid search result",
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "status": "active",
                        "sample_excluded": False,
                        "action_submission": {"raw_policy_output": "good"},
                        "wrapper_evidence": {"backend_call": "search"},
                    },
                },
            ]
        )

        invalid = client._step_native_policy_action("bad")
        valid = client._step_native_policy_action("good")

        self.assertEqual(invalid.reward, -0.01)
        self.assertFalse(invalid.done)
        self.assertEqual(
            invalid.info["wrapper_evidence"]["reward_overlay"],
            {
                "schema": "literesearcher_invalid_action_reward_v1",
                "native_reward": 0.0,
                "penalty": -0.01,
                "total_reward": -0.01,
                "terminal": False,
            },
        )
        self.assertEqual(valid.reward, 0.0)
        self.assertFalse(valid.done)
        self.assertNotIn("reward_overlay", valid.info["wrapper_evidence"])

    def test_backend_fault_is_not_penalized(self) -> None:
        client = self._step_client()
        client._request = Mock(
            return_value={
                "observation": "Frozen research backend failed; episode excluded.",
                "reward": 0.0,
                "done": True,
                "info": {
                    "status": "environment_error",
                    "sample_excluded": True,
                    "action_submission": {"raw_policy_output": "search"},
                    "wrapper_evidence": {"backend_error": "Timeout"},
                },
            }
        )

        result = client._step_native_policy_action("search")

        self.assertEqual(result.reward, 0.0)
        self.assertTrue(result.done)
        self.assertTrue(result.info["env_info"]["sample_excluded"])
        self.assertNotIn("reward_overlay", result.info["wrapper_evidence"])

    def test_constructor_rejects_positive_nonfinite_or_boolean_penalty(self) -> None:
        metadata = {
            "domain_id": "literesearcher",
            "compaction_contract": "task_neutral_filesystem_checkpoint_v2",
            "continuation_checkpoint_receipt_schema": (
                "agentmemory_continuation_checkpoint_v2"
            ),
            "workspace_action_envelope_contract": (
                "literesearcher_qwen35_native_xml_v1"
            ),
            "workspace_action_envelope_tools": ["shell_command"],
            "tool_serialization": LITERESEARCHER_TOOL_SERIALIZATION_CONTRACT,
            "raw_workspace_action_compatibility": True,
            "workspace_memory_reward": 0.0,
            "compaction_calls_endpoint_step": True,
            "compaction_calls_research_backend": False,
            "continuation_checkpoint_path": ".agent_memory/CONTINUATION.md",
            "continuation_checkpoint_max_bytes": 8192,
            "max_policy_steps": 40,
            "task_count": 1,
        }
        created = {"id": 7, "observation": "question", "info": {}}
        for value in (0.01, float("inf"), float("nan"), "not-a-number", True):
            with self.subTest(value=value):
                with (
                    patch.object(
                        LiteResearcherEnvClient,
                        "_request",
                        side_effect=[metadata, created],
                    ),
                    self.assertRaises(ValueError),
                ):
                    LiteResearcherEnvClient(
                        "http://literesearcher.example",
                        invalid_action_reward=value,
                    )


    def test_constructor_rejects_stale_or_rewarded_checkpoint_endpoint(self) -> None:
        base_metadata = {
            "domain_id": "literesearcher",
            "compaction_contract": "task_neutral_filesystem_checkpoint_v2",
            "continuation_checkpoint_receipt_schema": (
                "agentmemory_continuation_checkpoint_v2"
            ),
            "workspace_action_envelope_contract": (
                "literesearcher_qwen35_native_xml_v1"
            ),
            "workspace_action_envelope_tools": ["shell_command"],
            "tool_serialization": LITERESEARCHER_TOOL_SERIALIZATION_CONTRACT,
            "raw_workspace_action_compatibility": True,
            "workspace_memory_reward": 0.0,
            "compaction_calls_endpoint_step": True,
            "compaction_calls_research_backend": False,
            "continuation_checkpoint_path": ".agent_memory/CONTINUATION.md",
            "continuation_checkpoint_max_bytes": 8192,
            "max_policy_steps": 40,
            "task_count": 1,
        }
        cases = (
            (
                "continuation_checkpoint_receipt_schema",
                "agentmemory_continuation_checkpoint_v1",
                "receipt schema",
            ),
            (
                "workspace_action_envelope_contract",
                "literesearcher_raw_workspace_v0",
                "workspace action envelope",
            ),
            (
                "workspace_action_envelope_tools",
                ["search"],
                "enveloped workspace tools",
            ),
            (
                "tool_serialization",
                {"contract": "stale"},
                "tool serialization contract",
            ),
            (
                "raw_workspace_action_compatibility",
                False,
                "removed raw workspace action compatibility",
            ),
            ("workspace_memory_reward", 0.01, "changes workspace memory reward"),
            ("workspace_memory_reward", False, "changes workspace memory reward"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                metadata = dict(base_metadata)
                metadata[field] = value
                with (
                    patch.object(
                        LiteResearcherEnvClient,
                        "_request",
                        return_value=metadata,
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    LiteResearcherEnvClient(
                        "http://literesearcher.example",
                        invalid_action_reward=-0.01,
                    )


if __name__ == "__main__":
    unittest.main()
