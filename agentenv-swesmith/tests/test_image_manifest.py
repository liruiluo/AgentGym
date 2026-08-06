from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentenv_swesmith.image_manifest import (
    SwesmithImageManifest,
    SwesmithImageManifestError,
)


class ImageManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, images: list[dict]) -> Path:
        path = self.root / "images.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "swesmith_oci_image_manifest_v1",
                    "upstream": {
                        "repository": "SWE-bench/SWE-smith",
                        "dataset_revision": "e" * 40,
                        "source_revision": "f" * 40,
                    },
                    "images": images,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_resolves_exact_image_to_digest(self) -> None:
        manifest = SwesmithImageManifest(
            self.write(
                [
                    {
                        "image": "swebench/swesmith.example",
                        "digest": "sha256:" + "a" * 64,
                    }
                ]
            )
        )
        binding = manifest.resolve("swebench/swesmith.example")
        self.assertEqual(binding.digest, "sha256:" + "a" * 64)
        self.assertEqual(manifest.dataset_revision, "e" * 40)
        self.assertEqual(manifest.source_revision, "f" * 40)
        self.assertEqual(manifest.public_metadata()["image_count"], 1)

    def test_legacy_revision_field_is_accepted_as_dataset_revision(self) -> None:
        path = self.write(
            [{"image": "one", "digest": "sha256:" + "a" * 64}]
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["upstream"].pop("dataset_revision")
        payload["upstream"]["revision"] = "e" * 40
        path.write_text(json.dumps(payload), encoding="utf-8")
        manifest = SwesmithImageManifest(path)
        self.assertEqual(manifest.dataset_revision, "e" * 40)

    def test_unknown_or_duplicate_images_fail_closed(self) -> None:
        manifest = SwesmithImageManifest(
            self.write(
                [
                    {"image": "one", "digest": "sha256:" + "a" * 64},
                ]
            )
        )
        with self.assertRaises(SwesmithImageManifestError):
            manifest.resolve("two")
        with self.assertRaisesRegex(SwesmithImageManifestError, "duplicate image"):
            SwesmithImageManifest(
                self.write(
                    [
                        {"image": "one", "digest": "sha256:" + "a" * 64},
                        {"image": "one", "digest": "sha256:" + "b" * 64},
                    ]
                )
            )


if __name__ == "__main__":
    unittest.main()
