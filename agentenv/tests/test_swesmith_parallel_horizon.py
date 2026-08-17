from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from agentenv.envs.swesmith import SwesmithEnvClient


class _ConcurrencyTracker:
    def __init__(self) -> None:
        self.active = 0
        self.calls: list[int] = []
        self.lock = threading.Lock()
        self.peak = 0

    def run(self, item_id: int) -> None:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append(item_id)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1


class _FrozenClient(SwesmithEnvClient):
    def __init__(self, item_id: int, tracker: _ConcurrencyTracker) -> None:
        self.item_id = item_id
        self.tracker = tracker
        self.env_server_base = "fixture"
        self.timeout = 1
        self.env_id = item_id
        self.data_len = 8
        self.metadata = {"configured_max_policy_turns": 30}
        self._max_policy_turns = 30
        self.info = {
            "observation": f"item-{item_id}-before",
            "info": {
                "action_kind": "shell_command",
                "episode_success": False,
                "step": 29,
                "terminal": False,
            },
        }
        self._reset_policy_transition_state()
        self._policy_step_count = 29
        self._native_call_count = 29
        self._policy_context_bound = True
        self._immutable_policy_context = [
            {"role": "system", "content": "frozen fixture"}
        ]
        self._selected_policy_control = "context_compaction"

    def _request(self, method: str, path: str, **kwargs):
        self.assert_request(method, path, kwargs)
        self.tracker.run(self.item_id)
        reward = 0.0
        return {
            "observation": f"item-{self.item_id}-horizon-exhausted",
            "reward": reward,
            "done": True,
            "info": {
                "action_kind": "policy_turn_horizon",
                "episode_success": False,
                "step": 29,
                "terminal": True,
            },
        }

    def assert_request(self, method: str, path: str, kwargs) -> None:
        if method != "POST" or path != "horizon":
            raise AssertionError((method, path))
        if kwargs != {"json": {"id": self.item_id}}:
            raise AssertionError(kwargs)


def _receipt_projection(output) -> dict:
    info = deepcopy(dict(output.info))
    embedded = info["wrapper_evidence"]["horizon_finalization"]
    terminal = embedded["info"]["env_info"]
    return {
        "action_submission": info["action_submission"],
        "context_transition": info["context_transition"],
        "done": output.done,
        "action_kind": terminal["action_kind"],
        "episode_success": terminal["episode_success"],
        "terminal": terminal["terminal"],
        "reward": output.reward,
        "state": output.state,
        "wrapper_event": info["wrapper_evidence"]["event"],
    }


def _serial_projection(client: _FrozenClient, action: str) -> dict:
    ordinary = client._complete_context_compaction(action)
    horizon = client.finalize_policy_horizon()
    merged = client._finalize_policy_boundary(ordinary)
    embedded = merged.info["wrapper_evidence"]["horizon_finalization"]
    if embedded["info"] != horizon.info:
        raise AssertionError("cached horizon receipt drifted")
    return _receipt_projection(merged)


class SwesmithParallelHorizonTest(unittest.TestCase):
    def test_frozen_serial_and_parallel_batches_are_receipt_identical(self) -> None:
        actions = [f"compact-item-{item_id}" for item_id in range(8)]

        serial_tracker = _ConcurrencyTracker()
        serial_clients = [
            _FrozenClient(item_id, serial_tracker) for item_id in range(8)
        ]
        serial_started = time.perf_counter()
        serial = [
            _serial_projection(client, action)
            for client, action in zip(serial_clients, actions)
        ]
        serial_seconds = time.perf_counter() - serial_started
        self.assertEqual(serial_tracker.peak, 1)

        parallel_tracker = _ConcurrencyTracker()
        parallel_clients = [
            _FrozenClient(item_id, parallel_tracker) for item_id in range(8)
        ]
        parallel_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as executor:
            parallel_outputs = list(
                executor.map(
                    lambda pair: pair[0].step(pair[1]),
                    zip(parallel_clients, actions),
                )
            )
        parallel_seconds = time.perf_counter() - parallel_started
        parallel = [_receipt_projection(output) for output in parallel_outputs]

        self.assertEqual(serial, parallel)
        self.assertEqual(parallel_tracker.peak, 8)
        self.assertEqual(parallel_tracker.active, 0)
        self.assertEqual(sorted(parallel_tracker.calls), list(range(8)))
        self.assertEqual(len(parallel_tracker.calls), 8)
        self.assertEqual(
            [record["state"] for record in parallel],
            [f"item-{item_id}-horizon-exhausted" for item_id in range(8)],
        )
        serial_digests = [
            hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for record in serial
        ]
        parallel_digests = [
            hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for record in parallel
        ]
        self.assertEqual(serial_digests, parallel_digests)

        evidence_path = os.environ.get("SWESMITH_PARALLEL_HORIZON_EVIDENCE")
        if evidence_path:
            report = {
                "item_count": len(parallel),
                "parallel": {
                    "active_after": parallel_tracker.active,
                    "call_count": len(parallel_tracker.calls),
                    "peak_concurrency": parallel_tracker.peak,
                    "seconds": parallel_seconds,
                    "worker_limit": 8,
                },
                "per_item_receipt_sha256": parallel_digests,
                "reward_values": [record["reward"] for record in parallel],
                "schema": "swesmith_parallel_horizon_exhaustion_equality_gate_v2",
                "serial": {
                    "call_count": len(serial_tracker.calls),
                    "peak_concurrency": serial_tracker.peak,
                    "seconds": serial_seconds,
                },
                "status": "pass",
                "episode_success_values": [
                    record["episode_success"] for record in parallel
                ],
            }
            destination = Path(evidence_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(destination)


if __name__ == "__main__":
    unittest.main()
