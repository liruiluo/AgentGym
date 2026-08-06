from __future__ import annotations

import unittest

from agentenv_swesmith.provenance import (
    SwesmithProvenanceError,
    validate_revision_binding,
)


class SwesmithProvenanceTests(unittest.TestCase):
    def args(self, **overrides: str | None) -> dict[str, str | None]:
        values: dict[str, str | None] = {
            "dataset_revision": "d" * 40,
            "source_revision": "s" * 40,
            "image_dataset_revision": "d" * 40,
            "image_source_revision": "s" * 40,
        }
        values.update(overrides)
        return values

    def test_accepts_separate_dataset_and_source_revisions(self) -> None:
        validate_revision_binding(**self.args())

    def test_rejects_dataset_mismatch(self) -> None:
        with self.assertRaisesRegex(SwesmithProvenanceError, "dataset revisions"):
            validate_revision_binding(**self.args(image_dataset_revision="x" * 40))

    def test_rejects_missing_or_wrong_source_attestation(self) -> None:
        with self.assertRaisesRegex(SwesmithProvenanceError, "must attest"):
            validate_revision_binding(**self.args(image_source_revision=None))
        with self.assertRaisesRegex(SwesmithProvenanceError, "source revisions"):
            validate_revision_binding(**self.args(image_source_revision="x" * 40))


if __name__ == "__main__":
    unittest.main()
