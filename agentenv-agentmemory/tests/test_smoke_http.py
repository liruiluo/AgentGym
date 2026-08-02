from __future__ import annotations

import copy
import unittest

from agentenv_agentmemory.smoke_http import (
    AgentMemorySmokeHttpClient,
    SmokeHttpError,
    SmokeServiceExpectation,
    assert_clean_reset,
    require_correct_buy,
    validate_smoke_service,
)


SOURCE_ID = "1" * 40
MEMORYARENA_COMMIT = "2" * 40
POOL_FILE_SHA = "3" * 64
POOL_SEMANTIC_SHA = "4" * 64
CATALOG_SHA = "5" * 64
ATTRIBUTES_SHA = "6" * 64
LUCENE_SHA = "7" * 64


def fixture_metadata(*, active_count: int = 0) -> dict:
    return {
        "surface": "agentmemory_webshop_latent_preference_train_v1",
        "provider": {
            "provider_mode": "fixed_window",
            "split": "train",
            "task_count": 10_000,
            "generator_base_seed": 233,
            "product_pool_sha256": POOL_SEMANTIC_SHA,
            "fixed_window": {"start_orbit": 0, "end_orbit_exclusive": 5000},
        },
        "runtime_inputs": {
            "product_pool_file_sha256": POOL_FILE_SHA,
            "product_pool_semantic_sha256": POOL_SEMANTIC_SHA,
            "catalog_sha256": CATALOG_SHA,
            "attributes_sha256": ATTRIBUTES_SHA,
            "lucene_manifest_sha256": LUCENE_SHA,
        },
        "reward_contract": {
            "first_valid_add_reward": 0.0,
            "first_valid_later_session_retrieve_reward": 0.0,
        },
        "memory_prompt_mode": "latent_preference_sop",
        "active_environment_count": active_count,
        "backend": {
            "price_seed": 233,
            "upstream_provenance": {
                "memoryarena_commit": MEMORYARENA_COMMIT,
            },
        },
        "service": {
            "role": "smoke",
            "runtime_source_id": SOURCE_ID,
            "fingerprint_sha256": "8" * 64,
        },
    }


def expectation() -> SmokeServiceExpectation:
    return SmokeServiceExpectation(
        surface="agentmemory_webshop_latent_preference_train_v1",
        runtime_source_id=SOURCE_ID,
        memoryarena_base_commit=MEMORYARENA_COMMIT,
        product_pool_file_sha256=POOL_FILE_SHA,
        product_pool_semantic_sha256=POOL_SEMANTIC_SHA,
        catalog_sha256=CATALOG_SHA,
        attributes_sha256=ATTRIBUTES_SHA,
        lucene_manifest_sha256=LUCENE_SHA,
        generator_seed=233,
        split="train",
        price_seed=233,
        memory_prompt_mode="latent_preference_sop",
        minimum_task_count=24,
    )


def clean_reset_payload() -> dict:
    return {
        "id": 4,
        "observation": "fresh task",
        "reward": 0.0,
        "done": False,
        "info": {
            "current_subtask_index": 0,
            "ltm_inventory_count": 0,
            "session_trace": [],
            "tool_ops": [],
            "memory_state_diff": {"added": [], "updated": [], "deleted": []},
        },
    }


class FakeTransport:
    def __init__(self) -> None:
        self.active_count = 0
        self.calls = []

    def __call__(self, method, url, body, timeout):
        del timeout
        path = url.split("test.local", 1)[1]
        self.calls.append((method, path, body))
        if path == "/metadata":
            return fixture_metadata(active_count=self.active_count)
        if path == "/create":
            self.active_count += 1
            return {"id": 4, "observation": "bootstrap"}
        if path == "/reset":
            return clean_reset_payload()
        if path == "/close":
            self.active_count -= 1
            return True
        raise AssertionError((method, path, body))


class SmokeHttpTest(unittest.TestCase):
    def test_validates_all_static_fingerprint_inputs(self):
        fingerprint = validate_smoke_service(fixture_metadata(), expectation())
        self.assertEqual(fingerprint, "8" * 64)

        stale = copy.deepcopy(fixture_metadata())
        stale["provider"]["generator_base_seed"] = 999
        with self.assertRaisesRegex(SmokeHttpError, "generator_base_seed"):
            validate_smoke_service(stale, expectation())

    def test_open_resets_and_close_restores_active_environment_count(self):
        transport = FakeTransport()
        client = AgentMemorySmokeHttpClient(
            "http://test.local",
            request_json=transport,
        )
        with client.open(23) as session:
            self.assertEqual(session.env_id, 4)
            self.assertEqual(transport.active_count, 1)
        self.assertEqual(transport.active_count, 0)
        self.assertIn(("POST", "/reset", {"id": 4, "data_idx": 23}), transport.calls)
        self.assertEqual(client.request_trace[0]["path"], "/metadata")
        self.assertEqual(client.request_trace[-1]["path"], "/metadata")
        self.assertEqual(client.request_trace[1]["response"], {"id": 4, "observation": "bootstrap"})

    def test_clean_reset_rejects_ltm_or_trace_residue(self):
        leaked = clean_reset_payload()
        leaked["info"]["ltm_inventory_count"] = 1
        leaked["info"]["session_trace"] = ["old action"]
        with self.assertRaisesRegex(SmokeHttpError, "ltm_inventory_count"):
            assert_clean_reset(leaked)

    def test_correct_buy_receipt_is_sanitized_but_complete(self):
        info = {
            "tool_ops": [
                {
                    "op": "BUY",
                    "raw_action": "click[Buy Now]",
                    "committed": True,
                    "purchase_correct": True,
                    "session_advanced": True,
                    "session_index": 2,
                }
            ]
        }
        require_correct_buy(info, session_index=2)
        info["tool_ops"][0]["actual_asin"] = "B000000001"
        with self.assertRaisesRegex(SmokeHttpError, "identity leaked"):
            require_correct_buy(info, session_index=2)


if __name__ == "__main__":
    unittest.main()
