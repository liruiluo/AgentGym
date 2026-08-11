from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentenv_agentmemory.workspace_patch import (
    apply_workspace_patch_touched_transaction,
    parse_workspace_patch,
)
from agentenv_swesmith.patch_conversion import unified_to_codex_patch


class SwesmithPatchConversionTests(unittest.TestCase):
    def test_reverses_update_and_preserves_unified_section_description(self) -> None:
        patch = (
            "diff --git a/src/value.py b/src/value.py\n"
            "--- a/src/value.py\n"
            "+++ b/src/value.py\n"
            "@@ -1,3 +1,3 @@ def value():\n"
            " keep\n"
            "-old\n"
            "+new\n"
            " tail\n"
        )
        self.assertEqual(
            unified_to_codex_patch(patch, reverse=True).splitlines(),
            [
                "*** Begin Patch",
                "*** Update File: src/value.py",
                "@@ def value():",
                " keep",
                "+old",
                "-new",
                " tail",
                "*** End Patch",
            ],
        )

    def test_section_hint_disambiguates_repeated_context(self) -> None:
        patch = (
            "diff --git a/retry.py b/retry.py\n"
            "--- a/retry.py\n"
            "+++ b/retry.py\n"
            "@@ -10,4 +10,4 @@ class retry_all(retry_base):\n"
            "         self.retries = retries\n"
            " \n"
            "     def __call__(self, retry_state):\n"
            "-        return all(r(retry_state) for r in self.retries)\n"
            "+        return any(r(retry_state) for r in self.retries)\n"
        )
        buggy = (
            "class retry_any(retry_base):\n"
            "    def __init__(self, *retries):\n"
            "        self.retries = retries\n"
            "\n"
            "    def __call__(self, retry_state):\n"
            "        return any(r(retry_state) for r in self.retries)\n"
            "\n"
            "\n"
            "class retry_all(retry_base):\n"
            "    def __init__(self, *retries):\n"
            "        self.retries = retries\n"
            "\n"
            "    def __call__(self, retry_state):\n"
            "        return any(r(retry_state) for r in self.retries)\n"
        )
        converted = unified_to_codex_patch(patch, reverse=True)
        self.assertIn("@@ class retry_all(retry_base):", converted)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "retry.py"
            target.write_text(buggy, encoding="utf-8")
            apply_workspace_patch_touched_transaction(
                root,
                parse_workspace_patch(converted),
                normalize_path=lambda value: value,
                validate_tree=lambda _: None,
            )
            fixed = target.read_text(encoding="utf-8")

        self.assertIn(
            "class retry_any(retry_base):\n"
            "    def __init__(self, *retries):\n"
            "        self.retries = retries\n\n"
            "    def __call__(self, retry_state):\n"
            "        return any(r(retry_state) for r in self.retries)",
            fixed,
        )
        self.assertIn(
            "class retry_all(retry_base):\n"
            "    def __init__(self, *retries):\n"
            "        self.retries = retries\n\n"
            "    def __call__(self, retry_state):\n"
            "        return all(r(retry_state) for r in self.retries)",
            fixed,
        )

    def test_reverses_add_and_delete_files(self) -> None:
        patch = (
            "diff --git a/new.txt b/new.txt\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new.txt\n"
            "@@ -0,0 +1 @@\n"
            "+new\n"
            "diff --git a/old.txt b/old.txt\n"
            "deleted file mode 100644\n"
            "--- a/old.txt\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-old\n"
        )
        self.assertEqual(
            unified_to_codex_patch(patch, reverse=True).splitlines(),
            [
                "*** Begin Patch",
                "*** Delete File: new.txt",
                "*** Add File: old.txt",
                "+old",
                "*** End Patch",
            ],
        )

    def test_reverses_move_and_hunk_signs(self) -> None:
        patch = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 80%\n"
            "rename from old.py\n"
            "rename to new.py\n"
            "--- a/old.py\n"
            "+++ b/new.py\n"
            "@@ -1,3 +1,3 @@\n"
            " keep\n"
            "-old\n"
            "+new\n"
            " tail\n"
        )
        self.assertEqual(
            unified_to_codex_patch(patch, reverse=True).splitlines(),
            [
                "*** Begin Patch",
                "*** Update File: new.py",
                "*** Move to: old.py",
                "@@",
                " keep",
                "+old",
                "-new",
                " tail",
                "*** End Patch",
            ],
        )

    def test_empty_diff_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            unified_to_codex_patch("", reverse=True)


if __name__ == "__main__":
    unittest.main()
