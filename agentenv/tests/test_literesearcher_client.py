from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.controller.types import PolicyContextPressure
from agentenv.envs.literesearcher import (
    LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
    LITERESEARCHER_CONTINUATION_PATH,
    LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
    LITERESEARCHER_POLICY_CONTINUATION_MARKER,
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
        client.max_policy_steps = 40
        client._policy_step_count = 0
        return client

    def test_policy_framing_exposes_normalized_conversation_start(self) -> None:
        framing = self._client().policy_framing()
        self.assertEqual(
            [message["role"] for message in framing], ["user", "assistant"]
        )
        self.assertIn("deep-research agent", framing[0]["content"])
        self.assertEqual(
            framing[1], {"role": "assistant", "content": "Understood."}
        )

    def test_policy_framing_exposes_literal_workspace_actions(self) -> None:
        prompt = self._client().policy_framing()[0]["content"]
        self.assertIn(
            'shell_command {"command":"cat .agent_memory/research.md",'
            '"workdir":".","timeout_ms":10000}',
            prompt,
        )
        self.assertIn(
            "apply_patch\n*** Begin Patch\n*** Add File: "
            ".agent_memory/research.md\n+question: ...\n+evidence: ...\n"
            "+next_step: ...\n*** End Patch",
            prompt,
        )
        self.assertIn("not <tool_call> objects", prompt)
        self.assertIn("After the first useful Visit", prompt)
        self.assertIn("source URL, extracted evidence, and next step", prompt)
        self.assertIn("read it with shell_command after", prompt)
        self.assertIn("context compaction before continuing", prompt)
        self.assertIn("Except when an explicit context-compaction request", prompt)


    def test_compaction_request_requires_one_real_bounded_workspace_write(self) -> None:
        self.assertIn("shell_command", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST)
        self.assertIn(
            LITERESEARCHER_CONTINUATION_PATH,
            LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertIn("overwrite", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST.lower())
        self.assertIn("8192", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST)
        self.assertNotIn("will not call", LITERESEARCHER_CONTEXT_COMPACTION_REQUEST)

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
        client._policy_context_bound = True
        client._selected_policy_control = (
            "context_compaction" if selected else None
        )
        return client

    @staticmethod
    def _checkpoint_response(*, valid: bool, done: bool = False) -> dict:
        receipt = {
            "schema": "agentmemory_continuation_checkpoint_v1",
            "path": ".agent_memory/CONTINUATION.md",
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
                "action_submission": {
                    "kind": "workspace",
                    "op": "SHELL_COMMAND",
                },
                "wrapper_evidence": {
                    "continuation_checkpoint": receipt,
                },
            },
        }

    def test_verified_checkpoint_write_replaces_without_leaking_write_content(self) -> None:
        client = self._bound_client()
        client._request = Mock(return_value=self._checkpoint_response(valid=True))
        raw_action = (
            'shell_command {"command":"printf secret-evidence > '
            '.agent_memory/CONTINUATION.md","workdir":"."}'
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

    def test_failed_checkpoint_write_does_not_replace_context(self) -> None:
        client = self._bound_client()
        client._request = Mock(return_value=self._checkpoint_response(valid=False))

        output = client.step('shell_command {"command":"true"}')

        self.assertEqual(
            output.info["context_transition"]["operation"],
            "append_observation",
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
            "compaction_contract": "task_neutral_client_replace_messages_v1",
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


if __name__ == "__main__":
    unittest.main()
