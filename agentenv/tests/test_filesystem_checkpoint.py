from __future__ import annotations

import hashlib
import unittest

from agentenv.envs.filesystem_checkpoint import (
    FILESYSTEM_CHECKPOINT_MAX_BYTES,
    FILESYSTEM_CHECKPOINT_PATH,
    build_filesystem_checkpoint_receipt,
    filesystem_checkpoint_failure_reason,
    filesystem_checkpoint_write_succeeded,
    normalize_filesystem_checkpoint_receipt,
)


class FilesystemCheckpointContractTest(unittest.TestCase):
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
            )
            self.assertFalse(filesystem_checkpoint_write_succeeded(receipt))
            self.assertEqual(filesystem_checkpoint_failure_reason(receipt), reason)

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


if __name__ == "__main__":
    unittest.main()
