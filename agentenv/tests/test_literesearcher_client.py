from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.controller.types import PolicyContextPressure
from agentenv.envs.literesearcher import (
    LITERESEARCHER_CONTEXT_COMPACTION_REQUEST,
    LITERESEARCHER_MIN_OBSERVATION_TOKEN_ENVELOPE,
    LiteResearcherEnvClient,
)


class LiteResearcherClientTests(unittest.TestCase):
    @staticmethod
    def _client() -> LiteResearcherEnvClient:
        client = object.__new__(LiteResearcherEnvClient)
        client.env_server_base = "http://literesearcher.example"
        client.timeout = 30
        client.env_id = 7
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
