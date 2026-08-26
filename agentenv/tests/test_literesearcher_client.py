from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from agentenv.controller.types import PolicyContextPressure
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
