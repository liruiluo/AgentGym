from __future__ import annotations

import hashlib
import unittest
from unittest.mock import Mock

from agentenv.controller.types import StepOutput
from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_PATH,
    build_filesystem_checkpoint_receipt,
)
from agentenv.envs.literesearcher import LiteResearcherEnvClient
from agentenv.envs.swesmith import SwesmithEnvClient


def _receipt(*, changed: bool) -> dict:
    payload = b"objective: preserve evidence\nnext: continue\n"
    entry = {
        "path": FILESYSTEM_CHECKPOINT_PATH,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "kind": "file",
    }
    return build_filesystem_checkpoint_receipt(
        action_kind="shell_command",
        action_completed=True,
        workspace_diff={
            "added": [entry] if changed else [],
            "modified": [],
            "deleted": [],
        },
        workspace_snapshot={"files": [entry] if changed else []},
    )


def _swe_client() -> SwesmithEnvClient:
    client = object.__new__(SwesmithEnvClient)
    client.env_id = 1
    client.checkpoint_contract_penalty = -0.1
    client._selected_policy_control = "context_compaction"
    client._checkpoint_retry_pending = False
    client._checkpoint_write_retry_framing = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    client._pending_checkpoint_read = None
    client._pending_checkpoint_read_framing = None
    client._immutable_policy_context = list(client._checkpoint_write_retry_framing)
    client._zero_progress_shell_receipts = set()
    client._context_epoch = 0
    return client


def _lr_client() -> LiteResearcherEnvClient:
    client = object.__new__(LiteResearcherEnvClient)
    client.env_id = 2
    client.invalid_action_reward = -0.01
    client.info = {"observation": "question"}
    client._policy_step_count = 0
    client.max_policy_steps = 40
    client._native_call_count = 0
    client._context_epoch = 0
    client._immutable_policy_context = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]
    client._current_policy_context = list(client._immutable_policy_context)
    client._policy_context_bound = True
    client._selected_policy_control = None
    client._checkpoint_retry_pending = False
    client._checkpoint_write_retry_framing = None
    client._pending_checkpoint_read = None
    client._pending_checkpoint_read_framing = None
    return client


class AcceptedBehaviorIntakeTests(unittest.TestCase):
    def test_swe_failed_checkpoint_uses_penalty_as_ceiling(self) -> None:
        client = _swe_client()
        client._step_native_policy_action = Mock(
            return_value=StepOutput(
                state="not written",
                reward=0.0,
                done=False,
                info={
                    "env_info": {"filesystem_checkpoint": _receipt(changed=False)},
                    "wrapper_evidence": {},
                },
            )
        )
        output = client._complete_context_compaction("bad checkpoint")
        self.assertEqual(output.reward, -0.1)
        overlay = output.info["wrapper_evidence"]["reward_overlay"]
        self.assertEqual(overlay["applied_delta"], -0.1)
        self.assertFalse(overlay["deduplicated"])

        client = _swe_client()
        client._step_native_policy_action = Mock(
            return_value=StepOutput(
                state="parser rejected",
                reward=-0.2,
                done=False,
                info={
                    "env_info": {"filesystem_checkpoint": _receipt(changed=False)},
                    "wrapper_evidence": {},
                },
            )
        )
        output = client._complete_context_compaction("bad checkpoint")
        self.assertEqual(output.reward, -0.2)
        overlay = output.info["wrapper_evidence"]["reward_overlay"]
        self.assertEqual(overlay["applied_delta"], 0.0)
        self.assertTrue(overlay["deduplicated"])

    def test_swe_successful_checkpoint_and_terminal_submit_are_unchanged(self) -> None:
        client = _swe_client()
        client._step_native_policy_action = Mock(
            return_value=StepOutput(
                state="written",
                reward=0.0,
                done=False,
                info={
                    "env_info": {"filesystem_checkpoint": _receipt(changed=True)},
                    "wrapper_evidence": {},
                },
            )
        )
        output = client._complete_context_compaction("write checkpoint")
        self.assertEqual(output.reward, 0.0)
        self.assertNotIn("reward_overlay", output.info["wrapper_evidence"])

        client = _swe_client()
        client._step_native_policy_action = Mock(
            return_value=StepOutput(
                state="submitted",
                reward=1.0,
                done=True,
                info={"env_info": {"filesystem_checkpoint": None}, "wrapper_evidence": {}},
            )
        )
        output = client._complete_context_compaction("submit")
        self.assertEqual(output.reward, 1.0)
        self.assertTrue(output.done)
        self.assertNotIn("reward_overlay", output.info["wrapper_evidence"])

    def test_literesearcher_invalid_nonterminal_is_minus_point_zero_one(self) -> None:
        client = _lr_client()
        response = Mock(status_code=200)
        response.json.return_value = {
            "observation": "invalid action; retry",
            "reward": 0.0,
            "done": False,
            "info": {
                "status": "invalid_action",
                "sample_excluded": False,
                "action_submission": {"kind": "parser_error"},
                "wrapper_evidence": {"invalid_action": True},
            },
        }
        client._request = Mock(return_value=response.json.return_value)
        output = client.step("bad")
        self.assertEqual(output.reward, -0.01)
        self.assertFalse(output.done)
        self.assertEqual(
            output.info["wrapper_evidence"]["reward_overlay"]["schema"],
            "literesearcher_invalid_action_reward_v1",
        )

    def test_literesearcher_valid_action_reward_is_unchanged(self) -> None:
        client = _lr_client()
        client._request = Mock(
            return_value={
                "observation": "search results",
                "reward": 0.25,
                "done": False,
                "info": {
                    "status": "ok",
                    "sample_excluded": False,
                    "action_submission": {"kind": "search"},
                    "wrapper_evidence": {},
                },
            }
        )
        output = client.step("search")
        self.assertEqual(output.reward, 0.25)
        self.assertNotIn("reward_overlay", output.info["wrapper_evidence"])

    def test_literesearcher_failed_checkpoint_is_penalized_once(self) -> None:
        client = _lr_client()
        client._selected_policy_control = "context_compaction"
        client._checkpoint_write_retry_framing = list(client._immutable_policy_context)
        client._step_native_policy_action = Mock(
            return_value=StepOutput(
                state="not written",
                reward=0.0,
                done=False,
                info={
                    "env_info": {
                        "sample_excluded": False,
                        "wrapper_evidence": {
                            "filesystem_checkpoint": _receipt(changed=False)
                        },
                    },
                    "wrapper_evidence": {},
                },
            )
        )
        output = client._complete_context_compaction("wrong file")
        self.assertEqual(output.reward, -0.01)
        self.assertEqual(
            output.info["wrapper_evidence"]["reward_overlay"]["basis"],
            "checkpoint_contract_unsatisfied",
        )


if __name__ == "__main__":
    unittest.main()
