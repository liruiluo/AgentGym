from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_agentmemory.domains.memoryarena_dataset import (
    FROZEN_MEMORYARENA_DATASETS,
    FrozenMemoryArenaDatasetSpec,
    attest_frozen_memoryarena_dataset,
    attest_injected_test_dataset,
    verify_memoryarena_dataset_provenance,
)


def _write_fixture(path: Path, *, phases: int = 2) -> None:
    row = {
        "id": 1,
        "questions": [f"q{index}" for index in range(phases)],
        "answers": [f"a{index}" for index in range(phases)],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


class MemoryArenaDatasetProvenanceTest(unittest.TestCase):
    def test_frozen_specs_bind_public_revision_and_formal_counts(self):
        math = FROZEN_MEMORYARENA_DATASETS["formal_reasoning_math"]
        phys = FROZEN_MEMORYARENA_DATASETS["formal_reasoning_phys"]
        self.assertEqual((math.record_count, math.phase_count), (40, 354))
        self.assertEqual((phys.record_count, phys.phase_count), (20, 86))
        self.assertNotEqual(math.sha256, phys.sha256)
        search = FROZEN_MEMORYARENA_DATASETS["progressive_search"]
        self.assertEqual((search.record_count, search.phase_count), (221, 1641))

    def test_attestation_rejects_math_bytes_as_physics(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "math.jsonl"
            _write_fixture(path)
            observed = attest_injected_test_dataset(
                path,
                config="formal_reasoning_math",
            )
            math_spec = FrozenMemoryArenaDatasetSpec(
                config="formal_reasoning_math",
                repo_path="formal_reasoning_math/data.jsonl",
                sha256=observed.sha256,
                record_count=observed.record_count,
                phase_count=observed.phase_count,
                phase_field="questions",
            )
            phys_spec = FrozenMemoryArenaDatasetSpec(
                config="formal_reasoning_phys",
                repo_path="formal_reasoning_phys/data.jsonl",
                sha256="0" * 64,
                record_count=20,
                phase_count=86,
                phase_field="questions",
            )
            with patch.dict(
                FROZEN_MEMORYARENA_DATASETS,
                {
                    "formal_reasoning_math": math_spec,
                    "formal_reasoning_phys": phys_spec,
                },
                clear=True,
            ):
                math = attest_frozen_memoryarena_dataset(
                    path,
                    config="formal_reasoning_math",
                )
                self.assertEqual(math.dataset_config, "formal_reasoning_math")
                with self.assertRaisesRegex(RuntimeError, "formal_reasoning_phys"):
                    attest_frozen_memoryarena_dataset(
                        path,
                        config="formal_reasoning_phys",
                    )

    def test_provenance_is_rechecked_after_file_changes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "fixture.jsonl"
            _write_fixture(path)
            provenance = attest_injected_test_dataset(
                path,
                config="formal_reasoning_math",
            )
            _write_fixture(path, phases=3)
            with self.assertRaisesRegex(RuntimeError, "changed after"):
                verify_memoryarena_dataset_provenance(
                    path,
                    expected_config="formal_reasoning_math",
                    provenance=provenance,
                )

    def test_search_frozen_attestation_rejects_wrong_config_and_tampering(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "progressive-search.jsonl"
            _write_fixture(path, phases=4)
            observed = attest_injected_test_dataset(
                path,
                config="progressive_search",
            )
            search_spec = FrozenMemoryArenaDatasetSpec(
                config="progressive_search",
                repo_path="progressive_search/data.jsonl",
                sha256=observed.sha256,
                record_count=observed.record_count,
                phase_count=observed.phase_count,
                phase_field="questions",
            )
            math_spec = FrozenMemoryArenaDatasetSpec(
                config="formal_reasoning_math",
                repo_path="formal_reasoning_math/data.jsonl",
                sha256="0" * 64,
                record_count=40,
                phase_count=354,
                phase_field="questions",
            )
            with patch.dict(
                FROZEN_MEMORYARENA_DATASETS,
                {
                    "progressive_search": search_spec,
                    "formal_reasoning_math": math_spec,
                },
                clear=True,
            ):
                search = attest_frozen_memoryarena_dataset(
                    path,
                    config="progressive_search",
                )
                with self.assertRaisesRegex(RuntimeError, "formal_reasoning_math"):
                    attest_frozen_memoryarena_dataset(
                        path,
                        config="formal_reasoning_math",
                    )
                _write_fixture(path, phases=5)
                with self.assertRaisesRegex(RuntimeError, "changed after"):
                    verify_memoryarena_dataset_provenance(
                        path,
                        expected_config="progressive_search",
                        provenance=search,
                    )

    def test_injected_fixture_mode_must_still_match_surface_config(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "fixture.jsonl"
            _write_fixture(path)
            provenance = attest_injected_test_dataset(
                path,
                config="formal_reasoning_math",
            )
            with self.assertRaisesRegex(RuntimeError, "config mismatch"):
                verify_memoryarena_dataset_provenance(
                    path,
                    expected_config="formal_reasoning_phys",
                    provenance=provenance,
                )


if __name__ == "__main__":
    unittest.main()
