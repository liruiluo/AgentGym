from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

from agentenv_agentmemory.service_identity import decorate_service_metadata


def fixture_metadata() -> dict:
    return {
        "surface": "agentmemory_webshop_latent_preference_train_v1",
        "provider": {
            "provider_mode": "fixed_window",
            "generator_base_seed": 233,
            "product_pool_sha256": "1" * 64,
        },
        "runtime_inputs": {
            "catalog_sha256": "2" * 64,
            "attributes_sha256": "3" * 64,
            "lucene_manifest_sha256": "4" * 64,
        },
        "reward_contract": {"first_valid_add_reward": 0.0},
        "ltm_inventory_mode": "hidden",
        "ltm_transition_notice_mode": "none",
        "action_listing_mode": "separate",
        "memory_prompt_mode": "latent_preference_sop",
        "active_environment_count": 0,
        "backend": {
            "surface": "memoryarena_webshop_native_v1",
            "price_seed": 233,
            "product_count": 1_181_436,
            "price_table_sha256": "5" * 64,
            "upstream_provenance": {
                "memoryarena_commit": "6" * 40,
                "source_bundle_sha256": "7" * 64,
            },
        },
    }


class ServiceIdentityTest(unittest.TestCase):
    def _decorate(self, metadata: dict, *, run_id: str = "run-a") -> dict:
        with patch.dict(
            os.environ,
            {
                "AGENTMEMORY_SERVICE_ROLE": "smoke",
                "AGENTMEMORY_RUNTIME_SOURCE_ID": "8" * 40,
                "AGENTMEMORY_RUN_ID": run_id,
                "MEMORYARENA_BASE_COMMIT": "6" * 40,
            },
            clear=True,
        ):
            return decorate_service_metadata(metadata)

    def test_fingerprint_ignores_instance_and_active_session_count(self):
        first = self._decorate(fixture_metadata(), run_id="run-a")
        second_metadata = fixture_metadata()
        second_metadata["active_environment_count"] = 17
        second = self._decorate(second_metadata, run_id="run-b")

        self.assertEqual(
            first["service"]["fingerprint_sha256"],
            second["service"]["fingerprint_sha256"],
        )
        self.assertNotEqual(
            first["service"]["instance_run_id"],
            second["service"]["instance_run_id"],
        )

    def test_fingerprint_changes_with_provider_or_source(self):
        first = self._decorate(fixture_metadata())
        changed = copy.deepcopy(fixture_metadata())
        changed["provider"]["generator_base_seed"] = 234
        second = self._decorate(changed)
        self.assertNotEqual(
            first["service"]["fingerprint_sha256"],
            second["service"]["fingerprint_sha256"],
        )

        with patch.dict(
            os.environ,
            {
                "AGENTMEMORY_SERVICE_ROLE": "smoke",
                "AGENTMEMORY_RUNTIME_SOURCE_ID": "9" * 40,
                "MEMORYARENA_BASE_COMMIT": "6" * 40,
            },
            clear=True,
        ):
            third = decorate_service_metadata(fixture_metadata())
        self.assertNotEqual(
            first["service"]["fingerprint_sha256"],
            third["service"]["fingerprint_sha256"],
        )

    def test_smoke_role_requires_source_identity(self):
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_SERVICE_ROLE": "smoke"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "RUNTIME_SOURCE_ID"):
                decorate_service_metadata(fixture_metadata())


if __name__ == "__main__":
    unittest.main()
