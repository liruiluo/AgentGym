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
        self.assertIn("At an explicit context-checkpoint request", prompt)
        self.assertIn("executable workspace write", prompt)
        self.assertIn("not free-form continuation text", prompt)
        self.assertNotIn("asks for continuation text", prompt)


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

    def test_forced_checkpoint_rejects_research_action_without_endpoint_dispatch(self) -> None:
        client = self._bound_client()
        client._request = Mock()
        search_action = (
            '<tool_call>{"name":"search","arguments":'
            '{"query":["source"]}}</tool_call>'
        )

        output = client.step(search_action)

        client._request.assert_not_called()
        self.assertEqual(output.reward, 0.0)
        self.assertFalse(output.done)
        self.assertEqual(
            output.info["context_transition"]["operation"],
            "append_observation",
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

    def test_valid_diff_from_failed_checkpoint_action_does_not_replace(self) -> None:
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
            "append_observation",
        )
        self.assertEqual(output.info["context_epoch_after"], 0)
        self.assertEqual(
            output.info["wrapper_evidence"]["checkpoint_rejection_reason"],
            "action_execution_failed",
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
            "append_observation",
        )
        self.assertEqual(output.info["context_epoch_after"], 0)
        self.assertEqual(
            output.info["wrapper_evidence"]["checkpoint_rejection_reason"],
            "inconsistent_valid_receipt",
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
                    response["info"][key] = value
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
