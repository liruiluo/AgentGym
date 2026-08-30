from __future__ import annotations

import hashlib
import json
import unittest

from agentenv.controller.types import PolicyContextPressure
from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_BARE_CHECKPOINT_CONTINUATION_MARKER,
    FILESYSTEM_BARE_CHECKPOINT_READ_ACTION,
    FILESYSTEM_BARE_CHECKPOINT_WRITE_ACTION,
    FILESYSTEM_BARE_CHECKPOINT_WRITE_GUIDANCE,
    FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER,
    FILESYSTEM_CHECKPOINT_MAX_BYTES,
    FILESYSTEM_CHECKPOINT_PATH,
    FILESYSTEM_CHECKPOINT_REQUEST_TOKEN_SLACK,
    FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS,
    build_filesystem_checkpoint_read_retry_observation,
    build_filesystem_checkpoint_read_receipt,
    build_filesystem_checkpoint_receipt,
    build_filesystem_checkpoint_retry_observation,
    build_filesystem_checkpoint_write_retry_context,
    build_post_checkpoint_context,
    build_post_checkpoint_read_retry_context,
    checkpoint_retry_ceiling_tokens,
    checkpoint_retry_trigger_tokens,
    filesystem_checkpoint_action_completed,
    filesystem_checkpoint_read_failure_reason,
    filesystem_checkpoint_read_matches,
    filesystem_checkpoint_read_observed,
    filesystem_checkpoint_failure_reason,
    filesystem_checkpoint_write_succeeded,
    normalize_filesystem_checkpoint_receipt,
    filesystem_workspace_action_request_sha256,
)


class FilesystemCheckpointContractTest(unittest.TestCase):

    def test_bare_checkpoint_control_actions_are_minimal_and_exact(self) -> None:
        write_name, write_payload = FILESYSTEM_BARE_CHECKPOINT_WRITE_ACTION.split(
            " ", 1
        )
        read_name, read_payload = FILESYSTEM_BARE_CHECKPOINT_READ_ACTION.split(
            " ", 1
        )
        self.assertEqual(write_name, "shell_command")
        self.assertEqual(read_name, "shell_command")
        write_args = json.loads(write_payload)
        read_args = json.loads(read_payload)
        self.assertEqual(set(write_args), {"command"})
        self.assertEqual(
            read_args, {"command": "cat .agent_memory/CONTINUATION.md"}
        )
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, write_args["command"])
        self.assertIn(
            FILESYSTEM_BARE_CHECKPOINT_WRITE_ACTION,
            FILESYSTEM_BARE_CHECKPOINT_WRITE_GUIDANCE,
        )
        self.assertIn(
            FILESYSTEM_BARE_CHECKPOINT_READ_ACTION,
            FILESYSTEM_BARE_CHECKPOINT_CONTINUATION_MARKER,
        )
        self.assertIn(
            "before any dependent task action",
            FILESYSTEM_BARE_CHECKPOINT_CONTINUATION_MARKER,
        )
        for value in (
            FILESYSTEM_BARE_CHECKPOINT_WRITE_ACTION,
            FILESYSTEM_BARE_CHECKPOINT_READ_ACTION,
        ):
            self.assertNotIn('"workdir"', value)
            self.assertNotIn('"timeout_ms"', value)

    def test_failed_write_retry_context_is_bounded_stable_and_preserves_history(
        self,
    ) -> None:
        framing = [
            {"role": "system", "content": "trusted task contract"},
            {"role": "user", "content": "original evidence and current state"},
            {"role": "assistant", "content": "earlier useful action"},
            {"role": "user", "content": "earlier useful observation"},
        ]
        reason = "missing_receipt"

        first = build_filesystem_checkpoint_write_retry_context(framing, reason)
        second = build_filesystem_checkpoint_write_retry_context(framing, reason)

        self.assertEqual(first, second)
        self.assertEqual(first[:3], framing[:3])
        self.assertIn(framing[-1]["content"], first[-1]["content"])
        self.assertIn("Filesystem checkpoint was not accepted", str(first))
        self.assertNotIn("failed policy action", str(first))
        self.assertNotIn("large native observation", str(first))
        self.assertLess(
            len(str(first).encode("utf-8")),
            len(str(framing).encode("utf-8")) + 1024,
        )

    def test_failed_read_retry_context_is_bounded_stable_and_nonleaking(self) -> None:
        receipt = {
            "schema": "agentmemory_filesystem_checkpoint_receipt_v1",
            "path": FILESYSTEM_CHECKPOINT_PATH,
            "action_kind": "shell_command",
            "action_completed": True,
            "changed": True,
            "exists": True,
            "regular_file": True,
            "size_bytes": len(b"secret checkpoint body"),
            "sha256": hashlib.sha256(b"secret checkpoint body").hexdigest(),
        }
        framing = [
            {"role": "system", "content": "trusted task contract"},
            {"role": "user", "content": "immutable task observation"},
        ]
        reason = "checkpoint_read_not_observed"

        first = build_post_checkpoint_read_retry_context(
            framing,
            receipt,
            reason,
        )
        second = build_post_checkpoint_read_retry_context(
            framing,
            receipt,
            reason,
        )

        self.assertEqual(first, second)
        self.assertEqual(first[0], framing[0])
        self.assertIn(receipt["sha256"], first[-1]["content"])
        self.assertIn("Checkpoint read failed", first[-1]["content"])
        self.assertNotIn("secret checkpoint body", str(first))
        self.assertNotIn("failed policy action", str(first))
        self.assertNotIn("large native observation", str(first))
        self.assertLess(len(str(first).encode("utf-8")), 4096)

    def test_shell_checkpoint_requires_zero_exit_without_timeout(self) -> None:
        self.assertTrue(
            filesystem_checkpoint_action_completed(
                "shell_command",
                {"status": "executed", "exit_code": 0, "timed_out": False},
            )
        )
        for execution in (
            {"status": "executed", "exit_code": 1, "timed_out": False},
            {"status": "executed", "exit_code": 0, "timed_out": True},
            {"status": "executed"},
        ):
            with self.subTest(execution=execution):
                self.assertFalse(
                    filesystem_checkpoint_action_completed(
                        "shell_command", execution
                    )
                )
        self.assertTrue(
            filesystem_checkpoint_action_completed(
                "apply_patch", {"status": "executed"}
            )
        )

    def test_shell_checkpoint_rejects_noninteger_zero_exit_codes(self) -> None:
        for exit_code in (False, 0.0, "0"):
            with self.subTest(exit_code=exit_code):
                self.assertFalse(
                    filesystem_checkpoint_action_completed(
                        "shell_command",
                        {
                            "status": "executed",
                            "exit_code": exit_code,
                            "timed_out": False,
                        },
                    )
                )

    def test_due_checkpoint_reserves_one_failed_attempt_and_retry(self) -> None:
        pressure = PolicyContextPressure(
            action_prompt_tokens=1_000,
            candidate_prompt_tokens=1_040,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=200,
            max_observation_tokens=300,
            action_observation_envelope_tokens=4,
        )
        request = "save-state"
        expected = (
            1_040
            + 200
            + FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS
            + 4
            + len(request.encode("utf-8"))
            + FILESYSTEM_CHECKPOINT_REQUEST_TOKEN_SLACK
        )
        self.assertEqual(
            checkpoint_retry_ceiling_tokens(pressure, control_request=request),
            expected,
        )

    def test_trigger_projection_uses_route_bound_once_and_bounded_retry(self) -> None:
        pressure = PolicyContextPressure(
            action_prompt_tokens=1_000,
            candidate_prompt_tokens=1_040,
            max_prompt_tokens=30_720,
            max_model_tokens=32_768,
            max_response_tokens=200,
            max_observation_tokens=300,
            action_observation_envelope_tokens=4,
        )
        request = "save-state"
        expected = (
            1_000
            + (200 + 300 + 4)
            + (200 + FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS + 4)
            + 2 * (
                len(request.encode("utf-8"))
                + FILESYSTEM_CHECKPOINT_REQUEST_TOKEN_SLACK
            )
        )
        self.assertEqual(
            checkpoint_retry_trigger_tokens(
                pressure, control_request=request
            ),
            expected,
        )

    def test_retry_observation_is_bounded_and_names_reason(self) -> None:
        observation = build_filesystem_checkpoint_retry_observation(
            "checkpoint_not_changed"
        )
        self.assertIn("checkpoint_not_changed", observation)
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, observation)
        self.assertLessEqual(
            len(observation.encode("utf-8")),
            FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS,
        )

        read_observation = build_filesystem_checkpoint_read_retry_observation(
            "checkpoint_read_not_observed"
        )
        self.assertIn("checkpoint_read_not_observed", read_observation)
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, read_observation)
        self.assertLessEqual(
            len(read_observation.encode("utf-8")),
            FILESYSTEM_CHECKPOINT_RETRY_OBSERVATION_MAX_TOKENS,
        )

    def test_workspace_action_request_digest_matches_endpoint_canonicalization(
        self,
    ) -> None:
        shell_action = (
            ' shell_command {"workdir":".", "command":"cat x"}  '
        )
        canonical_shell = '{"command":"cat x","workdir":"."}'
        self.assertEqual(
            filesystem_workspace_action_request_sha256(shell_action),
            hashlib.sha256(canonical_shell.encode("utf-8")).hexdigest(),
        )
        patch = "*** Begin Patch\n*** Add File: x\n+y\n*** End Patch"
        self.assertEqual(
            filesystem_workspace_action_request_sha256(f"apply_patch\n{patch}"),
            hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        )
        self.assertIsNone(filesystem_workspace_action_request_sha256("search[x]"))

    def test_added_checkpoint_is_accepted(self) -> None:
        payload = b"objective: repair\nnext_action: run test\n"
        receipt = build_filesystem_checkpoint_receipt(
            action_kind="SHELL_COMMAND",
            action_completed=True,
            workspace_diff={
                "added": [
                    {
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "kind": "file",
                    }
                ],
                "modified": [],
                "deleted": [],
            },
            workspace_snapshot={
                "files": [
                    {
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "kind": "file",
                    }
                ]
            },
        )
        self.assertTrue(filesystem_checkpoint_write_succeeded(receipt))
        self.assertIsNone(filesystem_checkpoint_failure_reason(receipt))
        self.assertEqual(
            normalize_filesystem_checkpoint_receipt(receipt), receipt
        )

    def test_existing_but_unchanged_checkpoint_is_rejected(self) -> None:
        receipt = build_filesystem_checkpoint_receipt(
            action_kind="shell_command",
            action_completed=True,
            workspace_diff={"added": [], "modified": [], "deleted": []},
            workspace_snapshot={
                "files": [
                    {
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "bytes": 12,
                        "sha256": "a" * 64,
                        "kind": "file",
                    }
                ]
            },
        )
        self.assertFalse(filesystem_checkpoint_write_succeeded(receipt))
        self.assertEqual(
            filesystem_checkpoint_failure_reason(receipt),
            "checkpoint_not_changed",
        )

    def test_empty_and_oversized_checkpoints_are_rejected(self) -> None:
        for size, reason in (
            (0, "checkpoint_empty"),
            (FILESYSTEM_CHECKPOINT_MAX_BYTES + 1, "checkpoint_too_large"),
        ):
            receipt = build_filesystem_checkpoint_receipt(
                action_kind="apply_patch",
                action_completed=True,
                workspace_diff={
                    "added": [],
                    "modified": [
                        {
                            "before": {
                                "path": FILESYSTEM_CHECKPOINT_PATH,
                                "size": 1,
                                "sha256": "a" * 64,
                                "kind": "file",
                            },
                            "after": {
                                "path": FILESYSTEM_CHECKPOINT_PATH,
                                "size": size,
                                "sha256": "b" * 64,
                                "kind": "file",
                            }
                        }
                    ],
                    "deleted": [],
                },
                workspace_snapshot={
                    "files": [
                        {
                            "path": FILESYSTEM_CHECKPOINT_PATH,
                            "size": size,
                            "sha256": "b" * 64,
                            "kind": "file",
                        }
                    ]
                },
            )
            self.assertFalse(filesystem_checkpoint_write_succeeded(receipt))
            self.assertEqual(filesystem_checkpoint_failure_reason(receipt), reason)

    def test_post_checkpoint_context_exposes_receipt_but_not_written_body(self) -> None:
        secret_body = "private continuation body that must be read from disk"
        receipt = {
            "schema": "agentmemory_filesystem_checkpoint_receipt_v1",
            "path": FILESYSTEM_CHECKPOINT_PATH,
            "action_kind": "apply_patch",
            "action_completed": True,
            "changed": True,
            "exists": True,
            "regular_file": True,
            "size_bytes": len(secret_body.encode("utf-8")),
            "sha256": hashlib.sha256(secret_body.encode("utf-8")).hexdigest(),
        }
        framing = [
            {"role": "system", "content": "system contract"},
            {"role": "user", "content": "task observation"},
        ]

        replacement = build_post_checkpoint_context(framing, receipt)

        self.assertEqual(framing[-1]["content"], "task observation")
        self.assertEqual(len(replacement), 2)
        self.assertIn(FILESYSTEM_CHECKPOINT_PATH, replacement[-1]["content"])
        self.assertIn(receipt["sha256"], replacement[-1]["content"])
        self.assertIn(f"size_bytes={receipt['size_bytes']}", replacement[-1]["content"])
        self.assertNotIn(secret_body, str(replacement))
        self.assertNotIn("apply_patch", str(replacement))

    def test_post_checkpoint_context_accepts_route_specific_marker(self) -> None:
        body = b"checkpoint state"
        receipt = {
            "schema": "agentmemory_filesystem_checkpoint_receipt_v1",
            "path": FILESYSTEM_CHECKPOINT_PATH,
            "action_kind": "shell_command",
            "action_completed": True,
            "changed": True,
            "exists": True,
            "regular_file": True,
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        marker = "Route-specific mandatory read marker."
        replacement = build_post_checkpoint_context(
            [{"role": "user", "content": "task framing"}],
            receipt,
            continuation_marker=marker,
        )
        self.assertIn(marker, replacement[-1]["content"])
        self.assertNotIn(
            FILESYSTEM_CHECKPOINT_CONTINUATION_MARKER,
            replacement[-1]["content"],
        )
        retry = build_post_checkpoint_read_retry_context(
            [{"role": "user", "content": "task framing"}],
            receipt,
            "checkpoint_read_not_observed",
            continuation_marker=marker,
        )
        self.assertIn(marker, retry[-1]["content"])
        self.assertIn("Checkpoint read failed", retry[-1]["content"])

    def test_post_checkpoint_context_rejects_unverified_receipt(self) -> None:
        with self.assertRaisesRegex(ValueError, "successful receipt"):
            build_post_checkpoint_context(
                [{"role": "user", "content": "task"}],
                None,
            )

    def test_wrong_path_or_wrong_action_does_not_authorize_replacement(self) -> None:
        receipt = build_filesystem_checkpoint_receipt(
            action_kind="visit",
            action_completed=True,
            workspace_diff={
                "added": [
                    {"path": "notes.md", "size": 4, "sha256": "c" * 64}
                ],
                "modified": [],
                "deleted": [],
            },
        )
        self.assertFalse(filesystem_checkpoint_write_succeeded(receipt))
        self.assertEqual(
            filesystem_checkpoint_failure_reason(receipt), "wrong_action_kind"
        )

    def test_exact_successful_stdout_attests_checkpoint_read(self) -> None:
        payload = b"objective: repair\nnext_action: run test\n"
        checkpoint = build_filesystem_checkpoint_receipt(
            action_kind="shell_command",
            action_completed=True,
            workspace_diff={"added": [], "modified": [], "deleted": []},
            workspace_snapshot={
                "files": [
                    {
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "kind": "file",
                    }
                ]
            },
        )

        read = build_filesystem_checkpoint_read_receipt(
            checkpoint,
            action_kind="shell_command",
            action_completed=True,
            stdout=payload,
        )
        saved_checkpoint = dict(checkpoint, changed=True)

        self.assertTrue(filesystem_checkpoint_read_observed(read))
        self.assertTrue(filesystem_checkpoint_read_matches(read, saved_checkpoint))
        self.assertIsNone(
            filesystem_checkpoint_read_failure_reason(read, saved_checkpoint)
        )

        wrong_checkpoint = dict(
            saved_checkpoint,
            size_bytes=checkpoint["size_bytes"] + 1,
            sha256="f" * 64,
        )
        self.assertFalse(
            filesystem_checkpoint_read_matches(read, wrong_checkpoint)
        )
        self.assertEqual(
            filesystem_checkpoint_read_failure_reason(read, wrong_checkpoint),
            "checkpoint_read_identity_mismatch",
        )

    def test_echo_partial_failed_or_mutating_output_is_not_a_read(self) -> None:
        payload = b"objective: repair\nnext_action: run test\n"
        base = build_filesystem_checkpoint_receipt(
            action_kind="shell_command",
            action_completed=True,
            workspace_diff={"added": [], "modified": [], "deleted": []},
            workspace_snapshot={
                "files": [
                    {
                        "path": FILESYSTEM_CHECKPOINT_PATH,
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "kind": "file",
                    }
                ]
            },
        )
        changed = dict(base, changed=True)
        cases = (
            {"checkpoint_receipt": base, "action_completed": False, "stdout": payload},
            {"checkpoint_receipt": base, "action_completed": True, "stdout": payload[:8]},
            {"checkpoint_receipt": changed, "action_completed": True, "stdout": payload},
        )
        for case in cases:
            with self.subTest(case=case):
                read = build_filesystem_checkpoint_read_receipt(
                    case["checkpoint_receipt"],
                    action_kind="shell_command",
                    action_completed=case["action_completed"],
                    stdout=case["stdout"],
                )
                self.assertFalse(filesystem_checkpoint_read_observed(read))

    def test_modified_checkpoint_requires_a_real_content_change_and_matching_snapshot(
        self,
    ) -> None:
        path = FILESYSTEM_CHECKPOINT_PATH
        before = {
            "path": path,
            "bytes": 4,
            "sha256": "a" * 64,
            "kind": "file",
        }
        after = {
            "path": path,
            "bytes": 4,
            "sha256": "b" * 64,
            "kind": "file",
        }
        cases = (
            (
                "metadata-only",
                {"added": [], "modified": [{"before": before, "after": before}], "deleted": []},
                {"files": [before]},
            ),
            (
                "stale-diff",
                {"added": [], "modified": [{"before": before, "after": after}], "deleted": []},
                {"files": [dict(after, sha256="c" * 64)]},
            ),
            (
                "missing-kind",
                {"added": [], "modified": [{"before": before, "after": after}], "deleted": []},
                {"files": [{key: value for key, value in after.items() if key != "kind"}]},
            ),
            (
                "symlink",
                {"added": [], "modified": [{"before": before, "after": after}], "deleted": []},
                {"files": [dict(after, kind="symlink")]},
            ),
        )
        for label, diff, snapshot in cases:
            with self.subTest(case=label):
                receipt = build_filesystem_checkpoint_receipt(
                    action_kind="apply_patch",
                    action_completed=True,
                    workspace_diff=diff,
                    workspace_snapshot=snapshot,
                )
                self.assertFalse(filesystem_checkpoint_write_succeeded(receipt))

    def test_added_checkpoint_requires_matching_explicit_regular_snapshot(self) -> None:
        entry = {
            "path": FILESYSTEM_CHECKPOINT_PATH,
            "bytes": 8,
            "sha256": "a" * 64,
            "kind": "file",
        }
        receipt = build_filesystem_checkpoint_receipt(
            action_kind="shell_command",
            action_completed=True,
            workspace_diff={"added": [entry], "modified": [], "deleted": []},
            workspace_snapshot={"files": [dict(entry, bytes=9)]},
        )
        self.assertFalse(filesystem_checkpoint_write_succeeded(receipt))


if __name__ == "__main__":
    unittest.main()
