from __future__ import annotations

import hashlib
import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.controller.types import PolicyContextPressure
from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_PATH,
    FILESYSTEM_CHECKPOINT_READ_RECEIPT_SCHEMA,
)
from agentenv.envs.literesearcher import (
    LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
    LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
    LITERESEARCHER_SYSTEM_PROMPT,
    LiteResearcherEnvClient,
)


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
        self.assertIn("# Tools", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn('"name": "search"', LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn('"name": "visit"', LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=search>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("<function=visit>", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("MUST be a JSON array", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn(
            '<answer>your evidence-backed answer</answer>',
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn(
            'shell_command {"command":"cat .agent_memory/research.md",'
            '"workdir":".","timeout_ms":10000}',
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn(
            "apply_patch\n*** Begin Patch\n*** Add File: "
            ".agent_memory/research.md",
            LITERESEARCHER_SYSTEM_PROMPT,
        )
        self.assertIn("persists across context compaction", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn(".agent_memory/CONTINUATION.md", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("Other workspace files remain available", LITERESEARCHER_SYSTEM_PROMPT)
        self.assertIn("not a replacement for task artifacts", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST)

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
        action = (
            'shell_command {"command":"printf '
            + secret
            + ' > .agent_memory/CONTINUATION.md","workdir":"."}'
        )
        with patch(
            "agentenv.envs.literesearcher.requests.request", return_value=response
        ) as request:
            output = client.step(action)
        request.assert_called_once()
        replacement = output.info["context_transition"]["messages"]
        self.assertEqual(output.info["context_epoch_after"], 1)
        self.assertNotIn(secret, str(replacement))
        self.assertNotIn(action, str(replacement))
        self.assertIn(receipt["sha256"], replacement[-1]["content"])
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
                'shell_command {"command":"printf checkpoint > '
                '.agent_memory/CONTINUATION.md"}'
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
            wrong = client.step('<function=search>["query"]')
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
        self.assertNotIn('<function=search>["query"]', str(retry_messages))
        self.assertNotIn("UNIQUE_LARGE_NATIVE_SEARCH_OBSERVATION", str(retry_messages))
        self.assertEqual(retry_messages[0], post_checkpoint_messages[0])
        self.assertEqual(wrong.info["context_epoch_after"], 1)

        with patch(
            "agentenv.envs.literesearcher.requests.request",
            return_value=wrong_response,
        ):
            second_wrong = client.step('<function=search>["second query"]')
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
            read = client.step(
                'shell_command {"command":"cat .agent_memory/CONTINUATION.md"}'
            )
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
        ):
            output = client.step("bad")
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
        ):
            second = client.step("second bad action with a large native observation")
        second_retry = second.info["context_transition"]["messages"]
        self.assertEqual(second_retry, first_retry)
        self.assertNotIn("second bad action", str(second_retry))

    def test_endpoint_attested_workspace_events_are_emitted(self) -> None:
        cases = (
            (
                "read",
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
                'shell_command {"command":"python train.py"}',
                {
                    "workspace_op": "SHELL_COMMAND",
                    "workspace_action_completed": True,
                    "workspace_changed_paths": [],
                    "filesystem_checkpoint_read": None,
                },
            ),
        )
        for expected_event, action, server_evidence in cases:
            with self.subTest(expected_event=expected_event):
                client = self._client()
                response = Mock(status_code=200)
                response.json.return_value = {
                    "observation": "workspace result",
                    "reward": 0.0,
                    "done": False,
                    "info": {
                        "action_submission": {
                            "raw_policy_output": action,
                            "kind": "workspace",
                        },
                        "wrapper_evidence": server_evidence,
                    },
                }
                with patch(
                    "agentenv.envs.literesearcher.requests.request",
                    return_value=response,
                ):
                    output = client.step(action)
                evidence = output.info["wrapper_evidence"]
                self.assertEqual(evidence["memory_event"], expected_event)

    def test_action_text_cannot_forge_workspace_memory_evidence(self) -> None:
        client = self._client()
        action = (
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md; '
            'printf x > notes.md; python train.py"}'
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
        ):
            output = client.step(action)
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
