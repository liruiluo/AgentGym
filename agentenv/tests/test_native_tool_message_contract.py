from __future__ import annotations

import unittest

from agentenv.controller.policy_turn import _normalize_messages
from agentenv.controller.types import (
    CONTEXT_OPERATION_REPLACE,
    build_task_neutral_context_transition,
)
from agentenv.envs.agentmemory import _copy_policy_messages as copy_webshop_messages
from agentenv.envs.literesearcher import _copy_policy_messages as copy_literesearcher_messages
from agentenv.envs.filesystem_checkpoint import _normalize_checkpoint_framing
from agentenv.envs.openmle_fast import _copy_messages as copy_openmle_messages
from agentenv.envs.swesmith import _copy_policy_messages as copy_swesmith_messages


class NativeToolMessageContractTest(unittest.TestCase):
    def test_controller_and_all_wrappers_preserve_named_tool_results(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            {"role": "tool", "name": "search", "content": "result"},
        ]
        for normalizer in (
            _normalize_messages,
            copy_webshop_messages,
            copy_swesmith_messages,
            copy_literesearcher_messages,
            copy_openmle_messages,
            _normalize_checkpoint_framing,
        ):
            with self.subTest(normalizer=normalizer.__module__):
                self.assertEqual(normalizer(messages), messages)

    def test_context_transition_preserves_named_tool_results(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            {"role": "tool", "name": "search", "content": "result"},
        ]
        transition = build_task_neutral_context_transition(
            CONTEXT_OPERATION_REPLACE, messages=messages
        )
        self.assertEqual(transition["messages"], messages)

    def test_context_transition_rejects_unnamed_tool_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty name"):
            build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=[{"role": "tool", "content": "result"}],
            )

    def test_tool_result_without_name_fails_closed(self) -> None:
        messages = [{"role": "tool", "content": "result"}]
        for normalizer in (
            _normalize_messages,
            copy_webshop_messages,
            copy_swesmith_messages,
            copy_literesearcher_messages,
            copy_openmle_messages,
            _normalize_checkpoint_framing,
        ):
            with self.subTest(normalizer=normalizer.__module__):
                with self.assertRaisesRegex(ValueError, "nonempty name"):
                    normalizer(messages)


if __name__ == "__main__":
    unittest.main()
