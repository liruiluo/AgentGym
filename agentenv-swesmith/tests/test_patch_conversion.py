from __future__ import annotations

import unittest

from agentenv_swesmith.patch_conversion import unified_to_codex_patch


class SwesmithPatchConversionTests(unittest.TestCase):
    def test_reverses_update_and_drops_unified_section_description(self) -> None:
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
                "@@",
                " keep",
                "+old",
                "-new",
                " tail",
                "*** End Patch",
            ],
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
