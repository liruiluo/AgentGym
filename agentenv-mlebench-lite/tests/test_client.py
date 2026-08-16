from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
import unittest
import uuid
from pathlib import Path

import requests
from agentenv_mlebench_lite.identity import (
    LITE_COMPETITION_IDS,
    SPLIT_SHA256,
    UPSTREAM_COMMIT,
)

from tests.support import FAKE_RUNNER_SHA256, FAKE_RUNTIME_DIGEST

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_PATH = REPO_ROOT / "agentenv" / "agentenv" / "envs" / "mlebench_lite.py"
TYPES_PATH = REPO_ROOT / "agentenv" / "agentenv" / "controller" / "types.py"
PUBLIC_MANIFEST_SHA256 = "4" * 64


def _load_client_module():
    saved = {
        name: sys.modules.get(name)
        for name in (
            "agentenv",
            "agentenv.controller",
            "agentenv.controller.types",
            "mlebench_lite_client_under_test",
        )
    }
    agentenv_module = types.ModuleType("agentenv")
    agentenv_module.__path__ = []
    controller_module = types.ModuleType("agentenv.controller")
    controller_module.__path__ = []

    class BaseEnvClient:
        def __init__(self, action_format="react") -> None:
            self.action_format = action_format

    class BaseTask:
        def __init__(self, client_args, n_clients=1, *args, **kwargs) -> None:
            del args, kwargs
            self.clients = [
                self.env_client_cls(**client_args) for _ in range(n_clients)
            ]

    sys.modules["agentenv"] = agentenv_module
    sys.modules["agentenv.controller"] = controller_module
    types_spec = importlib.util.spec_from_file_location(
        "agentenv.controller.types", TYPES_PATH
    )
    assert types_spec is not None and types_spec.loader is not None
    types_module = importlib.util.module_from_spec(types_spec)
    sys.modules["agentenv.controller.types"] = types_module
    types_spec.loader.exec_module(types_module)
    controller_module.BaseEnvClient = BaseEnvClient
    controller_module.BaseTask = BaseTask
    controller_module.StepOutput = types_module.StepOutput

    client_spec = importlib.util.spec_from_file_location(
        "mlebench_lite_client_under_test", CLIENT_PATH
    )
    assert client_spec is not None and client_spec.loader is not None
    client_module = importlib.util.module_from_spec(client_spec)
    sys.modules["mlebench_lite_client_under_test"] = client_module
    client_spec.loader.exec_module(client_module)

    def restore() -> None:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    return client_module, types_module, restore


_CLIENT, _TYPES, _RESTORE_MODULES = _load_client_module()
CONTEXT_OPERATION_REPLACE = _TYPES.CONTEXT_OPERATION_REPLACE
PolicyContextPressure = _TYPES.PolicyContextPressure
CLIENT_LITE_COMPETITION_IDS = _CLIENT.LITE_COMPETITION_IDS
CLIENT_MODES = _CLIENT.MODES
MLEBenchLiteEnvClient = _CLIENT.MLEBenchLiteEnvClient
_resource_contract = _CLIENT._resource_contract
_resource_contract_sha256 = _CLIENT._resource_contract_sha256
_zero_counters = _CLIENT._zero_counters


def tearDownModule() -> None:
    _RESTORE_MODULES()


class _Response:
    status_code = 200
    text = ""

    def __init__(self, value):
        self.value = value

    def json(self):
        return self.value


class _FakeRequester:
    def __init__(self) -> None:
        self.calls = []
        self.capability_token = "a" * 64
        self.mode = "native"
        self.counters = _zero_counters()
        self.competition_id = LITE_COMPETITION_IDS[0]
        self.action_cache = {}
        self.compaction_counter_drift = False
        self.compaction_action_count_as_float = False
        self.compaction_receipt_overrides = {}
        self.compaction_terminal = False
        self.precontrol_infrastructure_terminal = False
        self.cumulative_counter_drift = False
        self.lose_next_step_response = False
        self.omit_terminal_receipt = False
        self.reward_override = 0.0
        self.info_extra = {}
        self.close_response = {"closed": True}

    def metadata(self):
        resource_contract = _resource_contract(
            max_actions=30,
            max_submission_bytes=100_000_000,
            max_shell_timeout_ms=3_600_000,
        )
        return {
            "schema": "mlebench_lite_metadata_v2",
            "upstream_commit": UPSTREAM_COMMIT,
            "split_sha256": SPLIT_SHA256,
            "competition_ids": list(LITE_COMPETITION_IDS),
            "task_count": 22,
            "public_manifest_sha256": PUBLIC_MANIFEST_SHA256,
            "runner_sha256": FAKE_RUNNER_SHA256,
            "runtime_digest": FAKE_RUNTIME_DIGEST,
            "submission_path": "/home/submission/submission.csv",
            "modes": ["native", "amg_compaction_only", "amg_memory"],
            "resource_contract": resource_contract,
            "resource_contract_sha256": _resource_contract_sha256(resource_contract),
        }

    def __call__(self, method, url, *, timeout, **kwargs):
        path = url.rsplit("/", 1)[-1]
        self.calls.append(
            {"method": method, "path": path, "timeout": timeout, **kwargs}
        )
        if path == "metadata":
            return _Response(self.metadata())
        if path == "create":
            self.mode = kwargs.get("json", {}).get("mode")
            if self.mode not in {
                "native",
                "amg_compaction_only",
                "amg_memory",
            }:
                raise AssertionError("client sent an invalid mode")
            return _Response({"id": 7, "capability_token": self.capability_token})
        if path == "reset":
            request = kwargs["json"]
            self._assert_capability(request)
            self.counters = _zero_counters()
            self.action_cache.clear()
            self.competition_id = LITE_COMPETITION_IDS[request["data_idx"]]
            return _Response(
                {
                    "observation": "competition task",
                    "reward": self.reward_override,
                    "done": False,
                    "info": {
                        "counters": dict(self.counters),
                        **self.info_extra,
                    },
                }
            )
        if path == "close":
            self._assert_capability(kwargs["json"])
            return _Response(self.close_response)
        if path != "step":
            raise AssertionError(path)

        request = kwargs["json"]
        self._assert_capability(request)
        action_id = request["action_id"]
        if str(uuid.UUID(action_id)) != action_id:
            raise AssertionError("client sent a non-canonical action UUID")
        canonical_payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
        cached = self.action_cache.get(action_id)
        if cached is not None:
            if cached[0] != canonical_payload:
                raise AssertionError("client changed a replayed action payload")
            return _Response(copy.deepcopy(cached[1]))
        if request["expected_action_count"] != self.counters["action_count"]:
            raise AssertionError("client sent a stale action sequence")
        before = self.counters["action_count"]
        delta = _zero_counters()
        delta["action_count"] = 1
        self.counters["action_count"] += 1
        if request.get("control") == "compaction":
            if self.precontrol_infrastructure_terminal:
                value = {
                    "observation": "Episode terminated.",
                    "reward": self.reward_override,
                    "done": True,
                    "info": {
                        "action_kind": "infrastructure_terminal",
                        "terminal_reason": "infrastructure_failure",
                        "counters": dict(self.counters),
                        "counter_delta": delta,
                        **self.info_extra,
                    },
                }
            else:
                after = self.counters["action_count"] + int(
                    self.compaction_counter_drift
                )
                receipt = {
                    "schema": "mlebench_lite_compaction_receipt_v2",
                    "action_count_before": before,
                    "action_count_after": after,
                    "counter_delta": delta,
                    "accepted": True,
                }
                receipt.update(self.compaction_receipt_overrides)
                value = {
                    "observation": (
                        "Action budget exhausted."
                        if self.compaction_terminal
                        else "competition task"
                    ),
                    "reward": self.reward_override,
                    "done": self.compaction_terminal,
                    "info": {
                        "counters": dict(self.counters),
                        "counter_delta": delta,
                        "control_receipt": receipt,
                        **(
                            {"terminal_reason": "action_budget_exhausted"}
                            if self.compaction_terminal
                            else {}
                        ),
                        **self.info_extra,
                    },
                }
                if self.compaction_action_count_as_float:
                    value["info"]["counters"]["action_count"] = float(
                        value["info"]["counters"]["action_count"]
                    )
        else:
            delta["native_action_count"] = 1
            self.counters["native_action_count"] += 1
            done = request["action"] == "submit"
            value = {
                "observation": ("Submission handed off." if done else "sandbox stdout"),
                "reward": self.reward_override,
                "done": done,
                "info": {
                    "counters": dict(self.counters),
                    "counter_delta": delta,
                    **(
                        {
                            "action_kind": "submit",
                            "terminal_reason": "submission_handoff",
                            **(
                                {}
                                if self.omit_terminal_receipt
                                else {
                                    "terminal_receipt": {
                                        "competition_id": self.competition_id,
                                        "submission_path": (
                                            "/home/submission/submission.csv"
                                        ),
                                        "submission_sha256": "b" * 64,
                                    }
                                }
                            ),
                        }
                        if done
                        else {}
                    ),
                    **self.info_extra,
                },
            }
            if self.mode == "amg_memory" and request["action"].startswith(
                'inspect {"path":"/run/amg_memory/'
            ):
                value["info"]["external_memory_operation"] = "read"
            elif self.mode == "amg_memory" and request["action"].startswith(
                'edit {"path":"/run/amg_memory/'
            ):
                value["info"]["external_memory_operation"] = "write"
        if self.cumulative_counter_drift:
            value["info"]["counters"]["cpu_time_ms"] += 1
        self.action_cache[action_id] = (canonical_payload, copy.deepcopy(value))
        if self.lose_next_step_response:
            self.lose_next_step_response = False
            raise requests.ConnectionError("injected lost step response")
        return _Response(value)

    def _assert_capability(self, request):
        if (
            request.get("id") != 7
            or request.get("capability_token") != self.capability_token
        ):
            raise AssertionError("client capability binding drifted")


class MLEBenchLiteClientTest(unittest.TestCase):
    def make_client(self, mode: str, requester=None):
        requester = requester or _FakeRequester()
        client = MLEBenchLiteEnvClient(
            env_server_base="http://mlebench-lite.invalid",
            mode=mode,
            expected_public_manifest_sha256=PUBLIC_MANIFEST_SHA256,
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            requester=requester,
        )
        client.reset(0)
        return client, requester

    def bind_initial_context(self, client):
        normalized = client.normalize_initial_policy_context(
            [{"role": "user", "content": client.observe()}]
        )
        client.bind_policy_context(normalized, initial=True)
        return normalized

    def pressure(self):
        return PolicyContextPressure(
            action_prompt_tokens=70,
            candidate_prompt_tokens=80,
            max_prompt_tokens=100,
            max_model_tokens=200,
            max_response_tokens=20,
            max_observation_tokens=10,
        )

    def test_client_pins_all_server_metadata(self) -> None:
        requester = _FakeRequester()
        client, _ = self.make_client("native", requester)
        self.assertEqual(len(client), 22)
        requester.metadata = lambda: {
            **_FakeRequester().metadata(),
            "runner_sha256": "0" * 64,
        }
        with self.assertRaises(RuntimeError):
            self.make_client("native", requester)

    def test_client_pins_exact_triad_and_ordered_22_id_identity(self) -> None:
        self.assertEqual(
            CLIENT_MODES,
            ("native", "amg_compaction_only", "amg_memory"),
        )
        self.assertEqual(CLIENT_LITE_COMPETITION_IDS, LITE_COMPETITION_IDS)
        self.assertEqual(len(CLIENT_LITE_COMPETITION_IDS), 22)
        self.assertEqual(len(set(CLIENT_LITE_COMPETITION_IDS)), 22)
        for mode in CLIENT_MODES:
            client, requester = self.make_client(mode)
            metadata = requester.metadata()
            self.assertEqual(metadata["competition_ids"], list(LITE_COMPETITION_IDS))
            self.assertEqual(metadata["split_sha256"], SPLIT_SHA256)
            self.assertEqual(metadata["runtime_digest"], FAKE_RUNTIME_DIGEST)
            self.assertEqual(len(client), 22)
            for index, competition_id in enumerate(LITE_COMPETITION_IDS):
                client.reset(index)
                self.assertEqual(client._competition_id, competition_id)
                self.assertEqual(requester.competition_id, competition_id)

    def test_client_rejects_resource_contract_type_confusion(self) -> None:
        cases = (
            ("integer_as_boolean", "gpu_count", True),
            ("integer_as_float", "gpu_count", 1.0),
            ("boolean_as_integer", "network_disabled", 1),
        )
        for name, field, replacement in cases:
            metadata = _FakeRequester().metadata()
            contract = dict(metadata["resource_contract"])
            contract[field] = replacement
            metadata["resource_contract"] = contract
            requester = _FakeRequester()
            requester.metadata = lambda value=metadata: value
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.make_client("native", requester)
            with self.subTest(name=f"hash_{name}"), self.assertRaises(ValueError):
                _resource_contract_sha256(contract)

    def test_client_cannot_silently_expose_a_lite_subset(self) -> None:
        with self.assertRaises(ValueError):
            MLEBenchLiteEnvClient(
                env_server_base="http://mlebench-lite.invalid",
                mode="native",
                expected_public_manifest_sha256=PUBLIC_MANIFEST_SHA256,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
                data_len=21,
                requester=_FakeRequester(),
            )
        full = MLEBenchLiteEnvClient(
            env_server_base="http://mlebench-lite.invalid",
            mode="native",
            expected_public_manifest_sha256=PUBLIC_MANIFEST_SHA256,
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            data_len=22,
            requester=_FakeRequester(),
        )
        self.assertEqual(len(full), 22)

    def test_close_response_is_exactly_validated(self) -> None:
        client, requester = self.make_client("native")
        self.assertEqual(client.close(), {"closed": True})
        requester.close_response = {"closed": True, "unexpected": True}
        with self.assertRaises(RuntimeError):
            client.close()
        requester.close_response = {"closed": 1}
        with self.assertRaises(RuntimeError):
            client.close()

    def test_http_timeout_must_cover_the_shell_contract(self) -> None:
        with self.assertRaises(ValueError):
            MLEBenchLiteEnvClient(
                env_server_base="http://mlebench-lite.invalid",
                mode="native",
                expected_public_manifest_sha256=PUBLIC_MANIFEST_SHA256,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
                timeout=30.0,
                requester=_FakeRequester(),
            )
        contract = _resource_contract(
            max_actions=30,
            max_submission_bytes=100_000_000,
            max_shell_timeout_ms=3_600_000,
        )
        max_response_seconds = contract["max_step_response_ms"] / 1000.0
        with self.assertRaises(ValueError):
            MLEBenchLiteEnvClient(
                env_server_base="http://mlebench-lite.invalid",
                mode="native",
                expected_public_manifest_sha256=PUBLIC_MANIFEST_SHA256,
                expected_runner_sha256=FAKE_RUNNER_SHA256,
                expected_runtime_digest=FAKE_RUNTIME_DIGEST,
                timeout=max_response_seconds,
                requester=_FakeRequester(),
            )
        client = MLEBenchLiteEnvClient(
            env_server_base="http://mlebench-lite.invalid",
            mode="native",
            expected_public_manifest_sha256=PUBLIC_MANIFEST_SHA256,
            expected_runner_sha256=FAKE_RUNNER_SHA256,
            expected_runtime_digest=FAKE_RUNTIME_DIGEST,
            timeout=max_response_seconds + 0.001,
            requester=_FakeRequester(),
        )
        self.assertGreater(client.timeout * 1000.0, contract["max_step_response_ms"])

    def test_nonzero_nonfinite_reward_and_private_result_info_fail_closed(self) -> None:
        cases = (
            ("nonzero", 0.5, {}),
            ("nonfinite", float("nan"), {}),
            ("score", 0.0, {"score": 0.9}),
            ("private", 0.0, {"detail": "/private/answer.csv"}),
        )
        for name, reward, extra in cases:
            requester = _FakeRequester()
            requester.reward_override = reward
            requester.info_extra = extra
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.make_client("native", requester)

    def test_native_has_no_memory_prompt_or_compaction_candidate(self) -> None:
        client, _ = self.make_client("native")
        framing = "\n".join(message["content"] for message in client.policy_framing())
        lowered = framing.lower()
        self.assertNotIn("memory", lowered)
        self.assertNotIn("compaction", lowered)
        self.assertNotIn(".agent_memory", lowered)
        self.assertNotIn("/run/amg_memory", lowered)
        self.bind_initial_context(client)
        self.assertIsNone(client.policy_turn_candidate())
        self.assertIsNone(client.prepare_policy_turn(self.pressure()))

    def test_triad_pins_the_same_tools_resources_runtime_and_task_identity(
        self,
    ) -> None:
        clients = [self.make_client(mode)[0] for mode in CLIENT_MODES]
        expected = clients[0]._expected_metadata
        for client in clients[1:]:
            self.assertEqual(client._expected_metadata, expected)
        framings = [client.policy_framing()[0]["content"] for client in clients]
        self.assertEqual(framings[0], framings[1])
        for native_tool in ("inspect", "edit", "shell", "submit"):
            self.assertIn(native_tool, framings[0])
            self.assertIn(native_tool, framings[2])

    def test_compaction_only_has_shared_compaction_without_memory_capability(
        self,
    ) -> None:
        compact_only, requester = self.make_client("amg_compaction_only")
        framing = compact_only.policy_framing()[0]["content"].lower()
        self.assertNotIn("memory", framing)
        self.assertNotIn(".agent_memory", framing)
        self.assertNotIn("/run/amg_memory", framing)
        self.bind_initial_context(compact_only)
        candidate = compact_only.policy_turn_candidate()
        self.assertIsNotNone(candidate)
        self.assertEqual(compact_only.prepare_policy_turn(self.pressure()), candidate)

        compacted = compact_only.step("objective fixed; inspect train.py next")
        transition = compacted.info["context_transition"]
        self.assertEqual(transition["operation"], CONTEXT_OPERATION_REPLACE)
        self.assertIn("objective fixed", transition["messages"][-2]["content"])
        request = requester.calls[-1]["json"]
        self.assertEqual(request["control"], "compaction")
        self.assertEqual(request["expected_action_count"], 0)

        later = compact_only.step('inspect {"path":"/home/workspace/train.py"}')
        self.assertFalse(later.done)
        self.assertNotIn("control", requester.calls[-1]["json"])
        self.assertEqual(requester.calls[-1]["json"]["expected_action_count"], 1)

    def test_compacting_arms_share_trigger_request_transition_and_accounting(
        self,
    ) -> None:
        evidence = []
        summary = "objective fixed; inspect train.py next"
        for mode in ("amg_compaction_only", "amg_memory"):
            client, requester = self.make_client(mode)
            self.bind_initial_context(client)
            candidate = client.policy_turn_candidate()
            selected = client.prepare_policy_turn(self.pressure())
            compacted = client.step(summary)
            request = dict(requester.calls[-1]["json"])
            request.pop("action_id")
            request.pop("capability_token")
            transition = copy.deepcopy(compacted.info)
            transition["context_transition"]["messages"][0]["content"] = (
                "<mode-specific framing>"
            )
            evidence.append((candidate, selected, request, transition))
        self.assertEqual(evidence[0], evidence[1])

    def test_memory_compaction_is_server_counted_before_replace_then_read(self) -> None:
        client, requester = self.make_client("amg_memory")
        framing = "\n".join(message["content"] for message in client.policy_framing())
        self.assertIn("/run/amg_memory", framing)
        self.assertIn("inspect", framing)
        self.assertIn("edit", framing)
        self.assertNotIn("memory_write", framing)
        self.assertNotIn("memory_read", framing)
        self.assertNotIn(".agent_memory", framing)
        self.bind_initial_context(client)
        written = client.step(
            'edit {"path":"/run/amg_memory/notes.md","content":"durable clue"}'
        )
        self.assertEqual(
            written.info["env_info"]["external_memory_operation"], "write"
        )
        self.assertIsNotNone(client.policy_turn_candidate())
        self.assertIsNotNone(client.prepare_policy_turn(self.pressure()))

        compacted = client.step("durable note key notes.md; inspect next")
        transition = compacted.info["context_transition"]
        self.assertEqual(transition["operation"], CONTEXT_OPERATION_REPLACE)
        self.assertIn("notes.md", transition["messages"][-2]["content"])
        request = requester.calls[-1]["json"]
        self.assertEqual(request["control"], "compaction")
        self.assertEqual(request["expected_action_count"], 1)

        later = client.step(
            'inspect {"path":"/run/amg_memory/notes.md",'
            '"offset":0,"max_bytes":1024}'
        )
        self.assertFalse(later.done)
        self.assertEqual(later.info["env_info"]["external_memory_operation"], "read")
        self.assertNotIn("control", requester.calls[-1]["json"])
        self.assertEqual(requester.calls[-1]["json"]["expected_action_count"], 2)

    def test_memory_operation_receipt_fails_closed_outside_exact_marker(self) -> None:
        cases = (
            ("native", "read_write"),
            ("amg_compaction_only", "read_write"),
            ("amg_memory", "delete"),
        )
        for mode, marker in cases:
            with self.subTest(mode=mode, marker=marker):
                client, requester = self.make_client(mode)
                requester.info_extra = {"external_memory_operation": marker}
                with self.assertRaises(RuntimeError):
                    client.step('inspect {"path":"/home/workspace/train.py"}')

        client, requester = self.make_client("amg_memory")
        requester.info_extra = {"external_memory_operation": "read_write"}
        step = client.step('shell {"command":"fixture"}')
        self.assertEqual(
            step.info["env_info"]["external_memory_operation"], "read_write"
        )

    def test_compaction_counter_drift_fails_before_context_replacement(self) -> None:
        client, requester = self.make_client("amg_memory")
        self.bind_initial_context(client)
        client.prepare_policy_turn(self.pressure())
        requester.compaction_counter_drift = True
        with self.assertRaises(RuntimeError):
            client.step("handoff")

    def test_compaction_type_confusion_fails_before_context_replacement(self) -> None:
        cases = (
            ("accepted_integer", {"accepted": 1}, False),
            ("counter_action_count_float", {}, True),
            ("receipt_before_float", {"action_count_before": 0.0}, False),
            ("receipt_after_float", {"action_count_after": 1.0}, False),
        )
        for name, receipt_overrides, counter_as_float in cases:
            with self.subTest(name=name):
                client, requester = self.make_client("amg_memory")
                self.bind_initial_context(client)
                client.prepare_policy_turn(self.pressure())
                requester.compaction_receipt_overrides = receipt_overrides
                requester.compaction_action_count_as_float = counter_as_float
                with self.assertRaises(RuntimeError):
                    client.step("handoff")
                self.assertEqual(client._context_epoch, 0)

    def test_budget_terminal_compaction_does_not_replace_context(self) -> None:
        client, requester = self.make_client("amg_memory")
        self.bind_initial_context(client)
        client.prepare_policy_turn(self.pressure())
        requester.compaction_terminal = True
        terminal = client.step("handoff")
        self.assertTrue(terminal.done)
        self.assertNotEqual(
            terminal.info["context_transition"]["operation"],
            CONTEXT_OPERATION_REPLACE,
        )

    def test_lost_precontrol_infrastructure_terminal_replays_without_replace(
        self,
    ) -> None:
        client, requester = self.make_client("amg_memory")
        self.bind_initial_context(client)
        client.prepare_policy_turn(self.pressure())
        requester.precontrol_infrastructure_terminal = True
        requester.lose_next_step_response = True
        with self.assertRaises(requests.ConnectionError):
            client.step("handoff before deadline check")
        first_request = copy.deepcopy(requester.calls[-1]["json"])
        self.assertIsNotNone(client._pending_action_id)

        terminal = client.step("handoff before deadline check")
        second_request = requester.calls[-1]["json"]
        self.assertTrue(terminal.done)
        self.assertEqual(first_request, second_request)
        self.assertEqual(requester.counters["action_count"], 1)
        self.assertNotEqual(
            terminal.info["context_transition"]["operation"],
            CONTEXT_OPERATION_REPLACE,
        )
        self.assertIsNone(client._pending_action_id)

    def test_lost_submit_response_replays_the_exact_action_id(self) -> None:
        client, requester = self.make_client("native")
        requester.lose_next_step_response = True
        with self.assertRaises(requests.ConnectionError):
            client.step("submit")
        first_request = requester.calls[-1]["json"]
        terminal = client.step("submit")
        second_request = requester.calls[-1]["json"]
        self.assertTrue(terminal.done)
        self.assertEqual(first_request, second_request)
        self.assertEqual(requester.counters["action_count"], 1)
        self.assertIsNone(client._pending_action_id)

    def test_cumulative_resource_ledger_drift_fails_closed(self) -> None:
        client, requester = self.make_client("native")
        requester.cumulative_counter_drift = True
        with self.assertRaises(RuntimeError):
            client.step('inspect {"path":"/home/data/train.csv"}')
        self.assertIsNotNone(client._pending_action_id)

    def test_submission_handoff_without_exact_receipt_fails_closed(self) -> None:
        client, requester = self.make_client("native")
        requester.omit_terminal_receipt = True
        with self.assertRaises(RuntimeError):
            client.step("submit")


if __name__ == "__main__":
    unittest.main()
