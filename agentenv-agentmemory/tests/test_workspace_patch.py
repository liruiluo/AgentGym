from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path, PurePosixPath

from agentenv_agentmemory.workspace_patch import (
    WorkspacePatchError,
    apply_workspace_patch_touched_transaction,
    apply_workspace_patch_transaction,
    parse_workspace_patch,
)


def normalize(value: str) -> str:
    path = PurePosixPath(value)
    if value.startswith("/") or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkspacePatchError("invalid relative path")
    return path.as_posix()


def validate(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise WorkspacePatchError("symlink forbidden")


class WorkspacePatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "workspace"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def apply(self, patch: str):
        return apply_workspace_patch_transaction(
            self.root,
            parse_workspace_patch(patch),
            normalize_path=normalize,
            validate_tree=validate,
        )

    def apply_touched(self, patch: str, *, validator=validate):
        return apply_workspace_patch_touched_transaction(
            self.root,
            parse_workspace_patch(patch),
            normalize_path=normalize,
            validate_tree=validator,
        )

    def test_add_empty_and_newline_terminated_files(self) -> None:
        result = self.apply(
            "*** Begin Patch\n"
            "*** Add File: empty.md\n"
            "*** Add File: note.md\n"
            "+line one\n"
            "+line two\n"
            "*** End Patch"
        )
        self.assertEqual(result.added_paths, ("empty.md", "note.md"))
        self.assertEqual((self.root / "empty.md").read_bytes(), b"")
        self.assertEqual((self.root / "note.md").read_bytes(), b"line one\nline two\n")

    def test_update_preserves_missing_final_newline(self) -> None:
        (self.root / "note.md").write_bytes(b"old")
        self.apply(
            "*** Begin Patch\n"
            "*** Update File: note.md\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )
        self.assertEqual((self.root / "note.md").read_bytes(), b"new")

    def test_section_hint_and_eof_context(self) -> None:
        (self.root / "note.md").write_text(
            "header\nsection\nold\ntail\n",
            encoding="utf-8",
        )
        self.apply(
            "*** Begin Patch\n"
            "*** Update File: note.md\n"
            "@@ section\n"
            "-old\n"
            "+new\n"
            "@@\n"
            " tail\n"
            "+end\n"
            "*** End of File\n"
            "*** End Patch"
        )
        self.assertEqual(
            (self.root / "note.md").read_text(encoding="utf-8"),
            "header\nsection\nnew\ntail\nend\n",
        )

    def test_move_updates_content_and_removes_empty_parent(self) -> None:
        source_dir = self.root / "old"
        source_dir.mkdir()
        (source_dir / "note.md").write_text("old\n", encoding="utf-8")
        result = self.apply(
            "*** Begin Patch\n"
            "*** Update File: old/note.md\n"
            "*** Move to: new/note.md\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch"
        )
        self.assertEqual(result.deleted_paths, ("old/note.md",))
        self.assertEqual(result.added_paths, ("new/note.md",))
        self.assertFalse(source_dir.exists())
        self.assertEqual((self.root / "new/note.md").read_text(), "new\n")

    def test_failure_rolls_back_prior_operations(self) -> None:
        (self.root / "one.md").write_text("one\n", encoding="utf-8")
        before = (self.root / "one.md").read_bytes()
        with self.assertRaisesRegex(WorkspacePatchError, "context was not found"):
            self.apply(
                "*** Begin Patch\n"
                "*** Add File: two.md\n"
                "+two\n"
                "*** Update File: one.md\n"
                "@@\n"
                "-missing\n"
                "+changed\n"
                "*** End Patch"
            )
        self.assertEqual((self.root / "one.md").read_bytes(), before)
        self.assertFalse((self.root / "two.md").exists())

    def test_touched_transaction_updates_without_replacing_unrelated_files(self) -> None:
        note = self.root / "note.md"
        untouched = self.root / "untouched.bin"
        note.write_text("old\n", encoding="utf-8")
        untouched.write_bytes(b"keep")
        untouched_inode = os.stat(untouched).st_ino
        result = self.apply_touched(
            "*** Begin Patch\n"
            "*** Update File: note.md\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** Add File: nested/new.md\n"
            "+created\n"
            "*** End Patch"
        )
        self.assertEqual(result.updated_paths, ("note.md",))
        self.assertEqual(result.added_paths, ("nested/new.md",))
        self.assertEqual(note.read_text(encoding="utf-8"), "new\n")
        self.assertEqual((self.root / "nested/new.md").read_text(), "created\n")
        self.assertEqual(os.stat(untouched).st_ino, untouched_inode)

    def test_touched_transaction_preserves_mode_and_rolls_back_all_paths(self) -> None:
        source = self.root / "script.sh"
        source.write_text("old\n", encoding="utf-8")
        source.chmod(0o751)
        before = source.read_bytes()

        def reject_changed_tree(root: Path) -> None:
            validate(root)
            if (root / "added.txt").exists():
                raise WorkspacePatchError("forced validation failure")

        with self.assertRaisesRegex(WorkspacePatchError, "forced validation failure"):
            self.apply_touched(
                "*** Begin Patch\n"
                "*** Update File: script.sh\n"
                "@@\n"
                "-old\n"
                "+new\n"
                "*** Add File: added.txt\n"
                "+temporary\n"
                "*** End Patch",
                validator=reject_changed_tree,
            )
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(os.stat(source).st_mode & 0o777, 0o751)
        self.assertFalse((self.root / "added.txt").exists())

    def test_parser_rejects_duplicate_paths_and_unsupported_headers(self) -> None:
        with self.assertRaisesRegex(WorkspacePatchError, "same path"):
            parse_workspace_patch(
                "*** Begin Patch\n"
                "*** Add File: note.md\n"
                "+one\n"
                "*** Delete File: note.md\n"
                "*** End Patch"
            )
        with self.assertRaisesRegex(WorkspacePatchError, "unsupported"):
            parse_workspace_patch(
                "*** Begin Patch\n*** Rename File: a\n*** End Patch"
            )


if __name__ == "__main__":
    unittest.main()
