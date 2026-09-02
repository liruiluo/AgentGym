from __future__ import annotations

import unittest

from agentenv.controller.policy_turn import (
    bind_initial_policy_context,
    complete_policy_turn,
    prepare_policy_turn,
)
from agentenv.controller.types import PolicyContextPressure
from agentenv.envs.agentmemory import (
    AgentMemoryEnvClient,
    strip_filesystem_webshop_session_trace,
)
from agentenv.envs.context_compaction import (
    COMPACTIONRL_POLICY_SUFFIX,
    COMPACTIONRL_REQUEST,
    COMPACTIONRL_RESUME_PREFIX,
    CompactionRLController,
    build_compactionrl_request,
    compactionrl_policy_prompt,
    configure_compactionrl_controller,
    context_compaction_controller,
)
from agentenv.envs.literesearcher import (
    LITERESEARCHER_SYSTEM_PROMPT,
    LiteResearcherEnvClient,
)
from agentenv.envs.openmle_fast import (
    OPENMLE_FAST_POLICY_SYSTEM_PROMPT,
    OpenMLEFastEnvClient,
)
from agentenv.envs.swesmith import SWE_POLICY_SYSTEM_PROMPT, SwesmithEnvClient


def count_characters(messages) -> int:
    return sum(len(message["content"]) for message in messages)


def pressure(*, capacity: int, action: int = 100, candidate: int = 120):
    return PolicyContextPressure(
        action_prompt_tokens=action,
        candidate_prompt_tokens=candidate,
        max_prompt_tokens=capacity,
        max_model_tokens=capacity + 64,
        max_response_tokens=64,
        max_observation_tokens=128,
        action_observation_envelope_tokens=4,
    )


class CompactionRLControllerTests(unittest.TestCase):
    def _controller(self, **kwargs) -> CompactionRLController:
        controller = CompactionRLController(mode="compactionrl", **kwargs)
        framing = [{"role": "system", "content": "system"}]
        history = framing + [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "action-0"},
            {"role": "user", "content": "observation-0"},
            {"role": "assistant", "content": "action-1"},
            {"role": "user", "content": "observation-1"},
            {"role": "assistant", "content": "action-2"},
            {"role": "user", "content": "observation-2"},
        ]
        controller.bind_policy_context(
            history,
            immutable_framing=framing,
            initial=True,
        )
        controller.bind_prompt_counter(count_characters)
        return controller

    def test_trigger_uses_capacity_projection_and_not_native_state(self) -> None:
        controller = self._controller()
        self.assertIsNone(controller.prepare_policy_turn(pressure(capacity=4096)))
        self.assertFalse(controller.selected)

        request = controller.prepare_policy_turn(pressure(capacity=512))
        self.assertEqual(request, COMPACTIONRL_REQUEST)
        self.assertTrue(controller.selected)

    def test_valid_summary_keeps_latest_complete_pairs_and_zero_reward(self) -> None:
        controller = self._controller(recent_steps=2)
        self.assertEqual(
            controller.prepare_policy_turn(
                pressure(capacity=900, action=200, candidate=220)
            ),
            COMPACTIONRL_REQUEST,
        )
        completion = controller.complete(
            "compact state",
            native_call_count=7,
            context_epoch=2,
            session_epoch=1,
            policy_step_count=9,
            workspace_continuity_id="episode-1",
        )

        output = completion.step_output
        replacement = output.info["context_transition"]["messages"]
        self.assertEqual(output.reward, 0.0)
        self.assertFalse(output.done)
        self.assertEqual(completion.retained_recent_steps, 2)
        self.assertEqual(replacement[0], {"role": "system", "content": "system"})
        self.assertEqual(
            replacement[1],
            {
                "role": "user",
                "content": COMPACTIONRL_RESUME_PREFIX + "compact state",
            },
        )
        self.assertEqual(
            replacement[2:],
            [
                {"role": "assistant", "content": "action-1"},
                {"role": "user", "content": "observation-1"},
                {"role": "assistant", "content": "action-2"},
                {"role": "user", "content": "observation-2"},
            ],
        )
        evidence = output.info["wrapper_evidence"]
        self.assertEqual(evidence["native_environment_call_count"], 0)
        self.assertFalse(evidence["summary_sent_to_native_environment"])
        self.assertTrue(evidence["summary_valid"])
        self.assertTrue(evidence["context_replaced"])

    def test_adaptive_tail_reduces_only_complete_pairs(self) -> None:
        controller = self._controller(recent_steps=2)
        summary = "compact state"
        framing_and_summary = [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": COMPACTIONRL_RESUME_PREFIX + summary,
            },
        ]
        one_pair = framing_and_summary + [
            {"role": "assistant", "content": "action-2"},
            {"role": "user", "content": "observation-2"},
        ]
        two_pairs = framing_and_summary + [
            {"role": "assistant", "content": "action-1"},
            {"role": "user", "content": "observation-1"},
            {"role": "assistant", "content": "action-2"},
            {"role": "user", "content": "observation-2"},
        ]
        next_request = [{"role": "user", "content": COMPACTIONRL_REQUEST}]
        capacity = (
            count_characters(one_pair + next_request)
            + count_characters(two_pairs + next_request)
        ) // 2
        self.assertEqual(
            controller.prepare_policy_turn(
                pressure(capacity=capacity, action=200, candidate=220)
            ),
            COMPACTIONRL_REQUEST,
        )
        completion = controller.complete(
            summary,
            native_call_count=0,
            context_epoch=0,
            session_epoch=0,
            policy_step_count=0,
            workspace_continuity_id="episode",
        )
        replacement = completion.step_output.info["context_transition"]["messages"]
        self.assertEqual(completion.retained_recent_steps, 1)
        self.assertEqual(replacement[-2:], one_pair[-2:])
        self.assertNotIn({"role": "assistant", "content": "action-1"}, replacement)

    def test_replacement_reserves_room_for_the_next_control_request(self) -> None:
        controller = self._controller(recent_steps=0)
        summary = "compact state"
        replacement = [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": COMPACTIONRL_RESUME_PREFIX + summary,
            },
        ]
        capacity = count_characters(replacement)
        self.assertLess(
            capacity,
            count_characters(
                replacement + [{"role": "user", "content": COMPACTIONRL_REQUEST}]
            ),
        )
        self.assertEqual(
            controller.prepare_policy_turn(
                pressure(capacity=capacity, action=100, candidate=120)
            ),
            COMPACTIONRL_REQUEST,
        )

        completion = controller.complete(
            summary,
            native_call_count=0,
            context_epoch=0,
            session_epoch=0,
            policy_step_count=0,
            workspace_continuity_id="episode",
        )

        evidence = completion.step_output.info["wrapper_evidence"]
        self.assertFalse(evidence["summary_valid"])
        self.assertEqual(
            evidence["summary_failure_reason"],
            "summary_prompt_overflow",
        )
        self.assertFalse(completion.context_replaced)

    def test_invalid_and_prompt_overflow_summaries_retry_without_native_call(self) -> None:
        controller = self._controller(summary_max_bytes=16)
        self.assertEqual(
            controller.prepare_policy_turn(pressure(capacity=512)),
            build_compactionrl_request(16),
        )
        rejected = controller.complete(
            "",
            native_call_count=3,
            context_epoch=4,
            session_epoch=5,
            policy_step_count=6,
            workspace_continuity_id="episode",
        )
        evidence = rejected.step_output.info["wrapper_evidence"]
        self.assertFalse(evidence["summary_valid"])
        self.assertEqual(evidence["summary_failure_reason"], "empty_summary")
        self.assertEqual(rejected.step_output.info["native_call_count_after"], 3)
        self.assertIn("Retry now", controller.policy_turn_candidate())

        controller.bind_policy_context(
            rejected.step_output.info["context_transition"]["messages"],
            immutable_framing=[{"role": "system", "content": "system"}],
        )
        controller.bind_prompt_counter(
            lambda messages: 513
            if any(COMPACTIONRL_RESUME_PREFIX in item["content"] for item in messages)
            else 100
        )
        self.assertIsNotNone(controller.prepare_policy_turn(pressure(capacity=512)))
        overflow = controller.complete(
            "short summary",
            native_call_count=3,
            context_epoch=4,
            session_epoch=5,
            policy_step_count=7,
            workspace_continuity_id="episode",
        )
        overflow_evidence = overflow.step_output.info["wrapper_evidence"]
        self.assertFalse(overflow_evidence["summary_valid"])
        self.assertEqual(
            overflow_evidence["summary_failure_reason"],
            "summary_prompt_overflow",
        )
        self.assertFalse(overflow.context_replaced)


class CompactionRLWrapperIntegrationTests(unittest.TestCase):
    @staticmethod
    def _client(client_cls):
        client = object.__new__(client_cls)
        client.info = {"observation": "current task"}
        client.env_id = 17
        configure_compactionrl_controller(client, mode="compactionrl")
        client._reset_policy_transition_state()
        return client

    def test_two_wrappers_share_the_same_policy_turn_contract(self) -> None:
        for client_cls in (SwesmithEnvClient, LiteResearcherEnvClient):
            with self.subTest(client=client_cls.__name__):
                client = self._client(client_cls)
                messages = bind_initial_policy_context(
                    client,
                    [{"role": "user", "content": client.observe()}],
                )
                messages.extend(
                    [
                        {"role": "assistant", "content": "task action"},
                        {"role": "user", "content": "task observation"},
                    ]
                )
                candidate_tokens = count_characters(
                    messages + [{"role": "user", "content": COMPACTIONRL_REQUEST}]
                )
                expected_replacement = [
                    client.policy_framing()[0],
                    {
                        "role": "user",
                        "content": (
                            COMPACTIONRL_RESUME_PREFIX + "same shared summary"
                        ),
                    },
                    {"role": "assistant", "content": "task action"},
                    {"role": "user", "content": "task observation"},
                    {"role": "user", "content": COMPACTIONRL_REQUEST},
                ]
                capacity = max(
                    candidate_tokens,
                    count_characters(expected_replacement),
                ) + 128
                prepared = prepare_policy_turn(
                    client,
                    messages,
                    count_prompt_tokens=count_characters,
                    max_prompt_tokens=capacity,
                    max_model_tokens=capacity + 64,
                    max_response_tokens=64,
                    max_observation_tokens=capacity,
                    action_observation_envelope_tokens=4,
                )
                self.assertEqual(prepared.control_request, COMPACTIONRL_REQUEST)
                output, replacement = complete_policy_turn(
                    client,
                    prepared,
                    "same shared summary",
                )
                self.assertEqual(output.reward, 0.0)
                self.assertEqual(output.info["native_call_count_before"], 0)
                self.assertEqual(output.info["native_call_count_after"], 0)
                self.assertIn(
                    COMPACTIONRL_RESUME_PREFIX + "same shared summary",
                    replacement[1]["content"],
                )

    def test_prompt_override_is_explicit_and_default_is_byte_identical(self) -> None:
        base = "normal action-only prompt"
        self.assertEqual(
            compactionrl_policy_prompt(base, mode="filesystem"),
            base,
        )
        override = compactionrl_policy_prompt(base, mode="compactionrl")
        self.assertIn(base, override)
        self.assertIn("plain-text continuation summary", override)
        self.assertIn("does not provide a voluntary `.agent_memory/**`", override)

    def test_all_four_wrappers_expose_the_same_compaction_framing_contract(self) -> None:
        clients_and_bases = []

        webshop = object.__new__(AgentMemoryEnvClient)
        webshop.info = {
            "observation": (
                "webshop task\n\nCurrent-session action trace:\n<empty>\n\n"
                "Persistent workspace tools:\nfixture contract"
            )
        }
        webshop.env_id = 1
        webshop.is_filesystem = True
        webshop._policy_system_prompt = "webshop system prompt"
        configure_compactionrl_controller(webshop, mode="compactionrl")
        webshop._reset_policy_transition_state({})
        clients_and_bases.append((webshop, "webshop system prompt"))

        for client_cls, base_prompt in (
            (SwesmithEnvClient, SWE_POLICY_SYSTEM_PROMPT),
            (LiteResearcherEnvClient, LITERESEARCHER_SYSTEM_PROMPT),
            (OpenMLEFastEnvClient, OPENMLE_FAST_POLICY_SYSTEM_PROMPT),
        ):
            client = object.__new__(client_cls)
            client.info = {"observation": f"{client_cls.__name__} task"}
            client.env_id = client_cls.__name__
            configure_compactionrl_controller(client, mode="compactionrl")
            if isinstance(client, OpenMLEFastEnvClient):
                client._reset_transition_state()
            else:
                client._reset_policy_transition_state()
            clients_and_bases.append((client, base_prompt))

        for client, base_prompt in clients_and_bases:
            with self.subTest(client=type(client).__name__):
                framing = client.policy_framing()
                self.assertEqual(len(framing), 1)
                self.assertEqual(
                    framing[0],
                    {
                        "role": "system",
                        "content": base_prompt.rstrip() + COMPACTIONRL_POLICY_SUFFIX,
                    },
                )
                messages = bind_initial_policy_context(
                    client,
                    [{"role": "user", "content": client.observe()}],
                )
                self.assertEqual(messages[0], framing[0])
                self.assertEqual(
                    context_compaction_controller(client).policy_turn_candidate(),
                    COMPACTIONRL_REQUEST,
                )


class WebShopCompactionRenderingTests(unittest.TestCase):
    def test_cumulative_native_trace_is_removed_without_touching_other_sections(self) -> None:
        observation = (
            "Task family: bundled_shopping\nProgress: 0/6\n\n"
            "page\n\n"
            "Current-session action trace:\n"
            "- S0: Action: search[x]\nResult: result x\n"
            "- S1: Action: click[y]\nResult: result y\n\n"
            "Persistent workspace tools:\ncontract"
        )
        normalized = strip_filesystem_webshop_session_trace(observation)
        self.assertEqual(
            normalized,
            "Task family: bundled_shopping\nProgress: 0/6\n\n"
            "page\n\nPersistent workspace tools:\ncontract",
        )

    def test_webshop_compaction_mode_uses_trace_free_policy_observation(self) -> None:
        client = object.__new__(AgentMemoryEnvClient)
        client.is_filesystem = True
        configure_compactionrl_controller(client, mode="compactionrl")
        observation = (
            "page\n\nCurrent-session action trace:\n<empty>\n\n"
            "Persistent workspace: unavailable in this intervention."
        )
        self.assertEqual(
            client._policy_observation(observation),
            "page\n\nPersistent workspace: unavailable in this intervention.",
        )


if __name__ == "__main__":
    unittest.main()
