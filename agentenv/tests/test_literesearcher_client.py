from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.controller.types import PolicyContextPressure
from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE,
    FILESYSTEM_CHECKPOINT_PATH,
    FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
)
from agentenv.envs.verl_qwen_tool_parser import (
    QWEN_INVALID_ACTION_SENTINEL,
    parse_single_qwen3_tool_call,
)
from agentenv.envs.literesearcher import (
    LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
    LITERESEARCHER_EXACT_CHECKPOINT_READ_ACTION,
    LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
    LITERESEARCHER_POLICY_CONTINUATION_MARKER,
    LITERESEARCHER_SYSTEM_PROMPT,
    LiteResearcherEnvClient,
)


def qwen_call(name: str, **parameters: object) -> str:
    body = ["<tool_call>", f"<function={name}>"]
    for key, value in parameters.items():
        rendered = (
            json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list, bool, int, float))
            else str(value)
        )
        body.extend((f"<parameter={key}>", rendered, "</parameter>"))
    body.extend(("</function>", "</tool_call>"))
    return "\n".join(body)


class LiteResearcherClientTests(unittest.TestCase):
    @staticmethod
    def _client() -> LiteResearcherEnvClient:
        client = object.__new__(LiteResearcherEnvClient)
        client.env_server_base = "http://literesearcher.example"
        client.timeout = 30
        client.env_id = 7
        client.info = {"observation": "Which source answers this question?"}
        client._policy_step_count = 0
        client._native_call_count = 0
        client._context_epoch = 0
        client._immutable_policy_context = [
            {"role": "system", "content": LITERESEARCHER_SYSTEM_PROMPT},
            {"role": "user", "content": client.info["observation"]},
        ]
        client._current_policy_context = list(client._immutable_policy_context)
        client._policy_context_bound = True
        client._selected_policy_control = None
        client._checkpoint_retry_pending = False
        client._checkpoint_write_retry_framing = None
        client._pending_checkpoint_read = None
        client._pending_checkpoint_read_framing = None
        return client

    def test_prompt_uses_exact_native_tool_and_workspace_formats(self) -> None:
        self.assertNotIn("<tools>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertEqual(
            [
                schema["function"]["name"]
                for schema in self._client().policy_tool_schemas()
            ],
            ["search", "visit", "shell_command", "apply_patch", "answer"],
        )
        self.assertIn("<function=search>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=visit>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("MUST be a JSON array", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=answer>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn(
            "<parameter=answer>your evidence-backed answer</parameter>",
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertNotIn(
            '<tool_call>\n{"name":',
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn("<function=shell_command>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<parameter=command>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=apply_patch>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<parameter=patch>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn(
            "Never mix the native Qwen XML envelope with a bare Codex-style action",
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn("persists across context compaction", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn(".agent_memory/CONTINUATION.md", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("Other workspace files remain available", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn(
            FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE,
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn(
            FILESYSTEM_CHECKPOINT_LONG_LIVED_MEMORY_NOTICE,
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertIn("not a replacement for task artifacts", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST)
        self.assertIn(
            "For this context-boundary write, use shell_command rather than "
            "apply_patch",
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertIn(
            "mkdir -p .agent_memory && cat > "
            ".agent_memory/CONTINUATION.md <<'EOF'",
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertIn(
            "<function=shell_command>",
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertNotIn(
            "*** Add File: .agent_memory/CONTINUATION.md",
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )

    def test_checkpoint_controls_use_minimal_command_only_qwen_xml(self) -> None:
        self.assertIn(
            "<parameter=command>", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST
        )
        self.assertNotIn(
            "<parameter=workdir>", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST
        )
        self.assertNotIn(
            "<parameter=timeout_ms>", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST
        )
        self.assertEqual(
            LITERESEARCHER_EXACT_CHECKPOINT_READ_ACTION.count("<parameter="),
            1,
        )
        self.assertIn(
            LITERESEARCHER_EXACT_CHECKPOINT_READ_ACTION,
            LITERESEARCHER_POLICY_CONTINUATION_MARKER,
        )
        parsed = parse_single_qwen3_tool_call(
            LITERESEARCHER_EXACT_CHECKPOINT_READ_ACTION,
            tool_schemas=(
                {
                    "type": "function",
                    "function": {
                        "name": "shell_command",
                        "description": "Run a command.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                        },
                    },
                },
            ),
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(
            dict(parsed.arguments),
            {"command": "cat .agent_memory/CONTINUATION.md"},
        )

    def test_policy_framing_restores_system_role_and_question(self) -> None:
        client = self._client()
        self.assertEqual(
            client.policy_framing(),
            [{"role": "system", "content": LITERESEARCHER_SYSTEM_PROMPT}],
        )
        normalized = client.normalize_initial_policy_context(
            [
                {"role": "user", "content": "legacy prompt"},
                {"role": "assistant", "content": "Understood."},
                {"role": "user", "content": client.observe()},
            ]
        )
        self.assertEqual(
            normalized,
            [
                {"role": "system", "content": LITERESEARCHER_SYSTEM_PROMPT},
                {"role": "user", "content": client.observe()},
            ],
        )

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

    def test_visit_posts_canonical_endpoint_action_and_keeps_both_ledger_layers(self) -> None:
        client = self._client()
        raw = qwen_call("visit", url="123", goal="true", page=1)
        submitted = (
            "<tool_call>\n"
            "<function=visit>\n"
            "<parameter=url>\n"
            '"123"\n'
            "</parameter>\n"
            "<parameter=goal>\n"
            '"true"\n'
            "</parameter>\n"
            "<parameter=page>\n"
            "1\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        response = Mock(status_code=200)
        response.json.return_value = {
            "observation": "visited",
            "reward": 0.0,
            "done": False,
            "info": {
                "action_submission": {
                    "raw_policy_output": submitted,
                    "kind": "visit",
                },
                "wrapper_evidence": {},
            },
        }
        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=response,
        ) as request:
            output = client.step(raw)

        self.assertEqual(request.call_args.kwargs["json"]["action"], submitted)
        action_submission = output.info["action_submission"]
        self.assertEqual(action_submission["raw_policy_output"], raw)
        self.assertEqual(action_submission["submitted_action"], submitted)
        self.assertTrue(action_submission["tool_parser_normalized"])
        self.assertEqual(
            action_submission["endpoint_action_submission"]["raw_policy_output"],
            submitted,
        )

    @staticmethod
    def _checkpoint_receipt(*, changed: bool = True, size_bytes: int = 37) -> dict:
        return {
            "schema": "agentmemory_filesystem_checkpoint_receipt_v1",
            "path": ".agent_memory/CONTINUATION.md",
            "action_kind": "shell_command",
            "action_completed": True,
            "changed": changed,
            "exists": True,
            "regular_file": True,
            "size_bytes": size_bytes,
            "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        }

    def test_successful_checkpoint_replaces_without_injecting_body(self) -> None:
        client = self._client()
        client._selected_policy_control = "context_compaction"
        receipt = self._checkpoint_receipt()
        response = Mock(status_code=200)
        response.json.return_value = {
            "observation": "workspace write completed",
            "reward": 0.0,
            "done": False,
            "info": {
                "action_submission": {"raw_policy_output": "write", "kind": "workspace"},
                "wrapper_evidence": {"filesystem_checkpoint": receipt},
            },
        }
        secret = "secret compacted body"
        command = f"printf {secret} > .agent_memory/CONTINUATION.md"
        action = qwen_call("shell_command", command=command, workdir=".")
        submitted = "shell_command " + json.dumps(
            {"command": command, "workdir": "."},
            sort_keys=True,
            separators=(",", ":"),
        )
        response.json.return_value["info"]["action_submission"][
            "raw_policy_output"
        ] = submitted
        with patch(
            "agentenv.envs.literesearcher.requests.request", return_value=response
        ) as request:
            output = client.step(action)
        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["json"]["action"], submitted)
        submission = output.info["action_submission"]
        self.assertEqual(submission["raw_policy_output"], action)
        self.assertEqual(submission["submitted_action"], submitted)
        self.assertTrue(submission["tool_parser_normalized"])
        self.assertEqual(
            submission["endpoint_action_submission"]["raw_policy_output"],
            submitted,
        )
        replacement = output.info["context_transition"]["messages"]
        self.assertEqual(output.info["context_epoch_after"], 1)
        self.assertNotIn(secret, str(replacement))
        self.assertNotIn(action, str(replacement))
        self.assertIn(receipt["sha256"], replacement[-1]["content"])
        self.assertIn(
            LITERESEARCHER_EXACT_CHECKPOINT_READ_ACTION,
            replacement[-1]["content"],
        )
        self.assertTrue(output.info["wrapper_evidence"]["continuation_persisted"])
        self.assertFalse(
            output.info["wrapper_evidence"]["checkpoint_content_in_successor_context"]
        )
        self.assertTrue(
            output.info["wrapper_evidence"]["checkpoint_read_required_after"]
        )
        self.assertEqual(client._pending_checkpoint_read, receipt)

    def test_checkpoint_read_is_required_and_bound_to_saved_identity(self) -> None:
        client = self._client()
        client._selected_policy_control = "context_compaction"
        receipt = self._checkpoint_receipt()
        write_response = Mock(status_code=200)
        write_response.json.return_value = {
            "observation": "workspace write completed",
            "reward": 0.0,
            "done": False,
            "info": {
                "action_submission": {"raw_policy_output": "write", "kind": "workspace"},
                "wrapper_evidence": {"filesystem_checkpoint": receipt},
            },
        }
        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=write_response,
        ):
            write = client.step(
                qwen_call(
                    "shell_command",
                    command="printf checkpoint > .agent_memory/CONTINUATION.md",
                )
            )
        post_checkpoint_messages = write.info["context_transition"]["messages"]

        pressure = PolicyContextPressure(
            action_prompt_tokens=100,
            candidate_prompt_tokens=200,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=2_048,
            max_observation_tokens=LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
            action_observation_envelope_tokens=4,
        )
        self.assertIsNone(client.prepare_policy_turn(pressure))

        wrong_response = Mock(status_code=200)
        wrong_response.json.return_value = {
            "observation": "UNIQUE_LARGE_NATIVE_SEARCH_OBSERVATION",
            "reward": 0.0,
            "done": False,
            "info": {
                "action_submission": {"raw_policy_output": "search", "kind": "search"},
                "wrapper_evidence": {},
            },
        }
        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=wrong_response,
        ):
            wrong_action = qwen_call("search", query=["query"])
            wrong = client.step(wrong_action)
        self.assertIn("Checkpoint read failed", wrong.state)
        self.assertTrue(
            wrong.info["wrapper_evidence"]["checkpoint_read_retry_pending"]
        )
        self.assertIsNotNone(client._pending_checkpoint_read)
        self.assertEqual(
            wrong.info["context_transition"]["operation"], "replace_messages"
        )
        retry_messages = wrong.info["context_transition"]["messages"]
        self.assertIn("Checkpoint read failed", str(retry_messages))
        self.assertIn(
            LITERESEARCHER_EXACT_CHECKPOINT_READ_ACTION,
            retry_messages[-1]["content"],
        )
        self.assertNotIn(wrong_action, str(retry_messages))
        self.assertNotIn("UNIQUE_LARGE_NATIVE_SEARCH_OBSERVATION", str(retry_messages))
        self.assertEqual(retry_messages[0], post_checkpoint_messages[0])
        self.assertEqual(wrong.info["context_epoch_after"], 1)

        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=wrong_response,
        ):
            second_wrong = client.step(qwen_call("search", query=["second query"]))
        self.assertEqual(
            second_wrong.info["context_transition"]["messages"], retry_messages
        )
        self.assertNotIn("second query", str(retry_messages))
        self.assertEqual(second_wrong.info["context_epoch_after"], 1)

        read_receipt = {
            "schema": FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
            "path": FILESYSTEM_CHECKPOINT_PATH,
            "observed": True,
            "size_bytes": receipt["size_bytes"],
            "sha256": receipt["sha256"],
        }
        read_response = Mock(status_code=200)
        read_response.json.return_value = {
            "observation": "checkpoint",
            "reward": 0.0,
            "done": False,
            "info": {
                "action_submission": {"raw_policy_output": "read", "kind": "workspace"},
                "wrapper_evidence": {
                    "workspace_op": "SHELL_COMMAND",
                    "workspace_action_completed": True,
                    "workspace_changed_paths": [],
                    "filesystem_checkpoint_read": read_receipt,
                },
            },
        }
        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=read_response,
        ):
            read = client.step(LITERESEARCHER_EXACT_CHECKPOINT_READ_ACTION)
        self.assertTrue(read.info["wrapper_evidence"]["checkpoint_read_satisfied"])
        self.assertIsNone(client._pending_checkpoint_read)
        self.assertIsNone(client._pending_checkpoint_read_framing)

    def test_failed_checkpoint_retries_from_stable_preboundary_context(self) -> None:
        client = self._client()
        original = [
            *client._immutable_policy_context,
            {"role": "assistant", "content": "useful prior action"},
            {"role": "user", "content": "useful prior observation"},
        ]
        client.bind_policy_context(original, initial=False)
        pressure = PolicyContextPressure(
            action_prompt_tokens=18_000,
            candidate_prompt_tokens=18_200,
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
        response = Mock(status_code=200)
        response.json.return_value = {
            "observation": "invalid workspace action",
            "reward": -0.01,
            "done": False,
            "info": {
                "action_submission": {"raw_policy_output": "bad"},
                "wrapper_evidence": {},
            },
        }
        with patch(
            "agentenv.envs.literesearcher.requests.request", return_value=response
        ) as request:
            output = client.step("bad")
        self.assertEqual(
            request.call_args.kwargs["json"]["action"],
            QWEN_INVALID_ACTION_SENTINEL,
        )
        self.assertEqual(
            output.info["context_transition"]["operation"], "replace_messages"
        )
        first_retry = output.info["context_transition"]["messages"]
        self.assertIn("useful prior action", str(first_retry))
        self.assertNotIn("bad", str(first_retry))
        self.assertNotIn("invalid workspace action", str(first_retry))
        self.assertEqual(output.info["context_epoch_after"], 0)
        self.assertTrue(output.info["wrapper_evidence"]["retry_pending"] )
        self.assertTrue(
            output.info["wrapper_evidence"]["checkpoint_retry_context_rebuilt"]
        )

        client.bind_policy_context(first_retry, initial=False)
        retry = client.prepare_policy_turn(pressure)
        self.assertEqual(retry, LITERESEARCHER_CONTEXT_COMPACTION_REQUEST)
        with patch(
            "agentenv.envs.literesearcher.requests.request", return_value=response
        ) as request:
            second = client.step("second bad action with a large native observation")
        self.assertEqual(
            request.call_args.kwargs["json"]["action"],
            QWEN_INVALID_ACTION_SENTINEL,
        )
        second_retry = second.info["context_transition"]["messages"]
        self.assertEqual(second_retry, first_retry)
        self.assertNotIn("second bad action", str(second_retry))

    def test_endpoint_attested_workspace_events_are_emitted(self) -> None:
        cases = (
            (
                "read",
                qwen_call("shell_command", command="cat .agent_memory/CONTINUATION.md"),
                'shell_command {"command":"cat .agent_memory/CONTINUATION.md"}',
                {
                    "workspace_op": "SHELL_COMMAND",
                    "workspace_action_completed": True,
                    "workspace_changed_paths": [],
                    "filesystem_checkpoint_read": {
                        "schema": FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "observed": True,
                        "size_bytes": len(b"checkpoint"),
                        "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                    },
                },
            ),
            (
                "modify",
                qwen_call(
                    "apply_patch",
                    patch="*** Begin Patch\n*** Add File: notes.md\n+x\n*** End Patch",
                ),
                "apply_patch\n*** Begin Patch\n*** Add File: notes.md\n+x\n*** End Patch",
                {
                    "workspace_op": "APPLY_PATCH",
                    "workspace_action_completed": True,
                    "workspace_changed_paths": ["notes.md"],
                    "filesystem_checkpoint_read": None,
                },
            ),
            (
                "execute",
                qwen_call("shell_command", command="python train.py"),
                'shell_command {"command":"python train.py"}',
                {
                    "workspace_op": "SHELL_COMMAND",
                    "workspace_action_completed": True,
                    "workspace_changed_paths": [],
                    "filesystem_checkpoint_read": None,
                },
            ),
        )
        for expected_event, action, expected_submitted, server_evidence in cases:
            with self.subTest(expected_event=expected_event):
                client = self._client()
                response = Mock(status_code=200)
                response.json.return_value = {
                    "observation": "workspace result",
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "action_submission": {
                            "raw_policy_output": expected_submitted,
                            "kind": "workspace",
                        },
                        "wrapper_evidence": server_evidence,
                    },
                }
                with patch(
                    "agentenv.envs.literesearcher.requests.request",
                    return_value=response,
                ) as request:
                    output = client.step(action)
                posted = request.call_args.kwargs["json"]["action"]
                submission = output.info["action_submission"]
                self.assertEqual(submission["raw_policy_output"], action)
                self.assertEqual(posted, expected_submitted)
                self.assertEqual(submission["submitted_action"], expected_submitted)
                self.assertTrue(submission["tool_parser_normalized"])
                self.assertEqual(
                    submission["endpoint_action_submission"]["raw_policy_output"],
                    expected_submitted,
                )
                evidence = output.info["wrapper_evidence"]
                self.assertEqual(evidence["memory_event"], expected_event)

    def test_action_text_cannot_forge_workspace_memory_evidence(self) -> None:
        client = self._client()
        action = qwen_call(
            "shell_command",
            command=(
                "cat .agent_memory/CONTINUATION.md; "
                "printf x > notes.md; python train.py"
            ),
        )
        response = Mock(status_code=200)
        response.json.return_value = {
            "observation": "unattested result",
            "reward": 0.0,
            "done": False,
            "info": {
                "action_submission": {"raw_policy_output": action},
                "wrapper_evidence": {},
            },
        }
        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=response,
        ) as request:
            output = client.step(action)
        self.assertNotEqual(
            request.call_args.kwargs["json"]["action"],
            action,
        )
        self.assertTrue(output.info["action_submission"]["tool_parser_normalized"])
        self.assertNotIn("memory_event", output.info["wrapper_evidence"])

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


if __name__ == "__main__":
    unittest.main()
