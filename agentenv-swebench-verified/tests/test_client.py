from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_PATH = REPO_ROOT / "agentenv" / "agentenv" / "envs" / "swebench_verified.py"
TYPES_PATH = REPO_ROOT / "agentenv" / "agentenv" / "controller" / "types.py"
IMAGE_MANIFEST_SHA256 = "b" * 64


def load_client_module():
    saved = {
        name: sys.modules.get(name)
        for name in (
            "agentenv",
            "agentenv.controller",
            "agentenv.controller.types",
            "swebench_verified_client_under_test",
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
        "swebench_verified_client_under_test", CLIENT_PATH
    )
    assert client_spec is not None and client_spec.loader is not None
    client_module = importlib.util.module_from_spec(client_spec)
    sys.modules["swebench_verified_client_under_test"] = client_module
    client_spec.loader.exec_module(client_module)

    def restore() -> None:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    return client_module, types_module, restore


def metadata():
    return {
        "schema": "swebench_verified_external_patch_episode_v1",
        "task_count": 500,
        "full_benchmark_task_count": 500,
        "supported_arms": ["native", "amg_compaction_only", "amg_memory"],
        "model_labels": {
            "native": "qwen35-4b-native",
            "amg_compaction_only": "qwen35-4b-amg-compaction-only",
            "amg_memory": "qwen35-4b-amg-memory",
        },
        "policy_visible_fields": [
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
        ],
        "denied_grader_fields": [
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "eval_script",
            "eval_type",
            "grader_logs",
            "hints_text",
            "log_parser",
            "parser_state",
            "patch",
            "test_patch",
        ],
        "tool_contract": "codex_shell_command_apply_patch_v1",
        "tool_serialization": "qwen35_native_single_function_v1",
        "observation_contract": "bounded_policy_observation_v1",
        "max_observation_bytes": 6144,
        "max_observation_tokens": 8192,
        "evaluation_max_policy_turns": 250,
        "max_native_actions": 250,
        "compaction_consumes_policy_turn": True,
        "compaction_consumes_native_call": False,
        "run_capability_contract": "caller_supplied_run_bearer_first_claim_v1",
        "reward_contract": "external_official_grading_only",
        "patch_export_contract": "swebench_verified_exact_base_solution_diff_v1",
        "max_model_patch_bytes": 16 * 1024 * 1024,
        "testspec_contract": "swebench_v4_1_0_make_test_spec_binding_v1",
        "sandbox_contract": "swebench_verified_linux_namespace_oci_rootfs_v1",
        "official_grading_inside_adapter": False,
        "image_manifest": {
            "contract": "swebench_verified_linux_amd64_digest_tsv_v1",
            "tag_count": 500,
            "tag_ledger_sha256": (
                "b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a"
            ),
            "manifest_sha256": "b" * 64,
            "unique_digest_count": 497,
        },
        "prediction_contract": {
            "schema_fields": [
                "instance_id",
                "model_name_or_path",
                "model_patch",
            ],
            "task_count": 500,
            "instance_id_ledger_sha256": (
                "a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9"
            ),
            "model_labels": {
                "native": "qwen35-4b-native",
                "amg_compaction_only": "qwen35-4b-amg-compaction-only",
                "amg_memory": "qwen35-4b-amg-memory",
            },
        },
        "dataset": {
            "repository": "princeton-nlp/SWE-bench_Verified",
            "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
            "split": "test",
            "row_count": 500,
            "canonical_jsonl_sha256": (
                "392529c5e79ca273bf0b073be35169beb68c604a26d9aef5514912fc584fa6cb"
            ),
            "id_ledger_sha256": (
                "a6b0fd7c8c2969a0eef892e032250adcfa6d32362d395c246930e61b575ac9b9"
            ),
        },
    }


class Backend:
    def __init__(self, *, endpoint_metadata=None) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self.memory: dict[str, str] = {}
        self.native_steps = 0
        self.arm = "native"
        self.endpoint_metadata = endpoint_metadata or metadata()

    def request(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        if method == "GET" and path == "metadata":
            return self.endpoint_metadata
        if method == "POST" and path == "create":
            self.arm = str(kwargs["json"]["arm"])
            return {
                "id": 1,
                "capability": "test-client-slot-capability",
                "observation": "created",
                "reward": 0.0,
                "done": False,
                "info": {"schema": metadata()["schema"]},
            }
        if method == "POST" and path == "reset":
            self.memory.clear()
            self.native_steps = 0
            return {
                "observation": "Repair the fixture issue.",
                "reward": 0.0,
                "done": False,
                "info": {"schema": metadata()["schema"], "step": 0},
            }
        if method == "POST" and path == "step":
            self.native_steps += 1
            action = kwargs["json"]["action"]
            if action == (
                'shell_command {"command":"printf clue > '
                '/run/amg_memory/notes.md"}'
            ):
                self.memory["notes.md"] = "clue"
                observation = "memory written"
                action_kind = "shell_command"
                memory_operation = "write"
            elif action == 'shell_command {"command":"cat /run/amg_memory/notes.md"}':
                observation = self.memory.get("notes.md", "missing")
                action_kind = "shell_command"
                memory_operation = "read"
            else:
                observation = "dispatched"
                action_kind = "shell_command"
                memory_operation = None
            info = {
                "schema": metadata()["schema"],
                "step": self.native_steps,
                "action_kind": action_kind,
            }
            if self.arm == "amg_memory" and memory_operation is not None:
                info["external_memory_operation"] = memory_operation
            return {
                "observation": observation,
                "reward": 0.0,
                "done": False,
                "info": info,
            }
        if method == "POST" and path == "horizon":
            return {
                "observation": "exported",
                "reward": 0.0,
                "done": True,
                "info": {"schema": metadata()["schema"]},
            }
        if method == "GET" and path == "prediction":
            return {
                "instance_id": "task-0",
                "model_name_or_path": "qwen35-4b-native",
                "model_patch": "",
            }
        if method == "POST" and path == "predictions/assemble":
            return {"assembled": True, "row_count": 500}
        if method == "POST" and path == "close":
            return {"closed": True, "id": 1}
        raise AssertionError((method, path, kwargs))


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module, self.types, self.restore = load_client_module()
        module = self.module

        class TransportClient(module.SwebenchVerifiedEnvClient):
            def __init__(self, *, backend, **kwargs):
                self.backend = backend
                kwargs.setdefault("run_capability", "r" * 43)
                super().__init__(env_server_base="http://unused", **kwargs)

            def _request(self, method, path, **kwargs):
                return self.backend.request(method, path, **kwargs)

        self.TransportClient = TransportClient

    def tearDown(self) -> None:
        self.restore()

    def bind_initial_context(self, client):
        initial = [{"role": "user", "content": client.observe()}]
        normalized = client.normalize_initial_policy_context(initial)
        client.bind_policy_context(normalized, initial=True)
        return normalized

    def compaction_pressure(
        self,
        *,
        action_prompt_tokens=800,
        candidate_prompt_tokens=850,
    ):
        return self.types.PolicyContextPressure(
            action_prompt_tokens=action_prompt_tokens,
            candidate_prompt_tokens=candidate_prompt_tokens,
            max_prompt_tokens=1000,
            max_model_tokens=1200,
            max_response_tokens=100,
            max_observation_tokens=100,
        )

    def test_native_has_zero_memory_or_compaction_affordance(self) -> None:
        backend = Backend()
        client = self.TransportClient(
            backend=backend,
            arm="native",
            run_id="paired-native",
            image_manifest_sha256=IMAGE_MANIFEST_SHA256,
        )
        client.reset(0)
        prompt = client.conversation_start[0]["value"]
        self.assertNotIn(".agent_memory", prompt)
        self.assertNotIn("/run/amg_memory", prompt)
        self.assertNotIn("context compaction", prompt.lower())
        self.bind_initial_context(client)
        self.assertIsNone(client.policy_turn_candidate())
        self.assertIsNone(client.prepare_policy_turn(None))
        self.assertEqual(backend.memory, {})

    def test_client_rejects_malformed_step_fields(self) -> None:
        invalid_fields = (
            {"observation": {"not": "text"}},
            {"reward": "0.0"},
            {"reward": True},
            {"reward": float("inf")},
            {"done": "false"},
            {"info": []},
        )

        for drift in invalid_fields:
            with self.subTest(drift=drift):
                backend = Backend()
                request = backend.request

                def malformed(method, path, **kwargs):
                    response = request(method, path, **kwargs)
                    if method == "POST" and path == "step":
                        response.update(drift)
                    return response

                backend.request = malformed
                client = self.TransportClient(
                    backend=backend,
                    arm="native",
                    run_id="malformed-step",
                    image_manifest_sha256=IMAGE_MANIFEST_SHA256,
                )
                client.reset(0)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "step response types drifted",
                ):
                    client.step('shell_command {"command":"true"}')

    def test_memory_write_survives_policy_compaction_and_later_read(self) -> None:
        backend = Backend()
        client = self.TransportClient(
            backend=backend,
            arm="amg_memory",
            run_id="paired-memory",
            image_manifest_sha256=IMAGE_MANIFEST_SHA256,
        )
        client.reset(0)
        prompt = client.conversation_start[0]["value"]
        self.assertIn("/run/amg_memory", prompt)
        self.assertIn("shell_command", prompt)
        self.assertNotIn("memory_write", prompt)
        self.assertNotIn("memory_read", prompt)
        self.assertNotIn(".agent_memory", prompt)
        framing = self.bind_initial_context(client)
        write = (
            'shell_command {"command":"printf clue > '
            '/run/amg_memory/notes.md"}'
        )
        written = client.step(write)
        self.assertEqual(written.info["env_info"]["external_memory_operation"], "write")
        native_calls_before = len(
            [call for call in backend.calls if call[:2] == ("POST", "step")]
        )
        pressure = self.compaction_pressure()
        self.assertEqual(
            client.prepare_policy_turn(pressure),
            self.module.SBV_CONTEXT_COMPACTION_REQUEST,
        )
        compacted = client.step("objective fixed; durable note key notes.md")
        native_calls_after = len(
            [call for call in backend.calls if call[:2] == ("POST", "step")]
        )
        self.assertEqual(native_calls_before, native_calls_after)
        self.assertNotIn(
            "external_memory_operation",
            compacted.info["env_info"],
        )
        transition = compacted.info["context_transition"]
        self.assertEqual(transition["operation"], "replace_messages")
        self.assertEqual(transition["messages"][: len(framing)], framing)
        read = client.step(
            'shell_command {"command":"cat /run/amg_memory/notes.md"}'
        )
        self.assertEqual(read.state, "clue")
        self.assertEqual(
            read.info["env_info"]["external_memory_operation"],
            "read",
        )
        self.assertEqual(read.info["native_call_count_after"], 2)
        self.assertEqual(read.info["policy_step_after"], 3)

    def test_compaction_only_has_no_memory_affordance_or_store(self) -> None:
        backend = Backend()
        client = self.TransportClient(
            backend=backend,
            arm="amg_compaction_only",
            run_id="paired-compaction-only",
            image_manifest_sha256=IMAGE_MANIFEST_SHA256,
        )
        client.reset(0)
        prompt = client.conversation_start[0]["value"]
        self.assertNotIn(".agent_memory", prompt)
        self.assertNotIn("/run/amg_memory", prompt)
        self.assertNotIn("durable task memory", prompt.lower())
        framing = self.bind_initial_context(client)
        self.assertEqual(
            client.policy_turn_candidate(),
            self.module.SBV_CONTEXT_COMPACTION_REQUEST,
        )
        self.assertEqual(
            client.prepare_policy_turn(self.compaction_pressure()),
            self.module.SBV_CONTEXT_COMPACTION_REQUEST,
        )

        summary = "objective fixed; inspect the failing test next"
        compacted = client.step(summary)

        self.assertEqual(backend.memory, {})
        self.assertFalse(
            any(call[:2] == ("POST", "step") for call in backend.calls)
        )
        transition = compacted.info["context_transition"]
        self.assertEqual(transition["operation"], "replace_messages")
        self.assertEqual(transition["messages"][: len(framing)], framing)
        self.assertEqual(
            transition["messages"][-2:],
            [
                {"role": "assistant", "content": summary},
                {
                    "role": "user",
                    "content": self.types.POLICY_CONTINUATION_MARKER,
                },
            ],
        )
        self.assertEqual(compacted.info["native_call_count_after"], 0)
        self.assertEqual(compacted.info["policy_step_after"], 1)
        self.assertEqual(compacted.info["context_epoch_after"], 1)
        serialized = str(compacted.info).lower()
        for forbidden in (
            ".agent_memory",
            "memory_root",
            "memory_mount",
            "memory_endpoint",
            "memory_action",
            "memory_receipt",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_compacting_arms_share_trigger_transition_and_accounting(self) -> None:
        evidence = []
        summary = "objective fixed; inspect the failing test next"
        for arm in ("amg_compaction_only", "amg_memory"):
            backend = Backend()
            client = self.TransportClient(
                backend=backend,
                arm=arm,
                run_id=f"paired-{arm}",
                image_manifest_sha256=IMAGE_MANIFEST_SHA256,
            )
            expected_prompt = self.module.SBV_BASE_POLICY_SYSTEM_PROMPT
            if arm == "amg_memory":
                expected_prompt += self.module.SBV_MEMORY_ADDENDUM
            self.assertEqual(client.system_prompt, expected_prompt)
            self.assertEqual(
                client.policy_framing(),
                [{"role": "system", "content": expected_prompt}],
            )
            self.assertEqual(
                client.conversation_start[0]["value"],
                expected_prompt,
            )
            client.reset(0)
            self.bind_initial_context(client)
            candidate = client.policy_turn_candidate()
            below_trigger = client.prepare_policy_turn(
                self.compaction_pressure(
                    action_prompt_tokens=600,
                    candidate_prompt_tokens=650,
                )
            )
            selected = client.prepare_policy_turn(self.compaction_pressure())
            compacted = client.step(summary)
            normalized_info = deepcopy(compacted.info)
            normalized_info["context_transition"]["messages"][0]["content"] = (
                "<arm-specific framing>"
            )
            normalized_info["wrapper_evidence"]["arm"] = "<arm>"
            evidence.append(
                (
                    candidate,
                    below_trigger,
                    selected,
                    compacted.state,
                    compacted.reward,
                    compacted.done,
                    normalized_info,
                )
            )

        self.assertEqual(evidence[0], evidence[1])

    def test_non_memory_action_dispatch_is_identical_across_triad(self) -> None:
        clients = {}
        backends = {}
        for arm in ("native", "amg_compaction_only", "amg_memory"):
            backend = Backend()
            client = self.TransportClient(
                backend=backend,
                arm=arm,
                run_id=f"paired-{arm}",
                image_manifest_sha256=IMAGE_MANIFEST_SHA256,
            )
            client.reset(0)
            backend.calls.clear()
            clients[arm] = client
            backends[arm] = backend
        action = 'shell_command {"command":"printf same > source.txt"}'
        results = {arm: client.step(action) for arm, client in clients.items()}

        for arm in ("amg_compaction_only", "amg_memory"):
            self.assertEqual(backends["native"].calls, backends[arm].calls)
            self.assertEqual(results["native"].state, results[arm].state)
            self.assertEqual(
                results["native"].info["native_call_count_after"],
                results[arm].info["native_call_count_after"],
            )

    def test_unified_budget_counts_compaction_without_native_call_or_horizon_row(
        self,
    ) -> None:
        backend = Backend()
        client = self.TransportClient(
            backend=backend,
            arm="amg_memory",
            run_id="budget-run",
            image_manifest_sha256=IMAGE_MANIFEST_SHA256,
        )
        client.reset(0)
        self.bind_initial_context(client)
        pressure = self.compaction_pressure()
        client.prepare_policy_turn(pressure)
        compacted = client.step("short continuation")
        self.assertEqual(compacted.info["policy_step_after"], 1)
        self.assertEqual(compacted.info["native_call_count_after"], 0)
        finalized = client.finalize_policy_horizon()
        self.assertEqual(finalized.info["policy_step_after"], 1)
        self.assertEqual(finalized.info["native_call_count_after"], 0)
        self.assertEqual(
            len([call for call in backend.calls if call[:2] == ("POST", "horizon")]),
            1,
        )

    def test_slot_capability_is_kept_private_and_sent_to_every_slot_endpoint(
        self,
    ) -> None:
        backend = Backend()
        client = self.TransportClient(
            backend=backend,
            arm="native",
            run_id="capability-run",
            image_manifest_sha256=IMAGE_MANIFEST_SHA256,
        )
        self.assertNotIn("capability", client.info)
        create_call = next(
            call for call in backend.calls if call[:2] == ("POST", "create")
        )
        self.assertEqual(
            create_call[2]["headers"]["Authorization"],
            f"Bearer {'r' * 43}",
        )
        self.assertNotIn("run_capability", create_call[2]["json"])
        client.reset(0)
        client.step('shell_command {"command":"pwd"}')
        client.finalize_policy_horizon()
        client.prediction()
        client.assemble_predictions()
        client.close()

        protected = {
            "reset",
            "step",
            "horizon",
            "prediction",
            "predictions/assemble",
            "close",
        }
        calls = [call for call in backend.calls if call[1] in protected]
        self.assertEqual({call[1] for call in calls}, protected)
        for method, _path, kwargs in calls:
            transport = kwargs["params"] if method == "GET" else kwargs["json"]
            self.assertEqual(
                kwargs["headers"]["Authorization"],
                "Bearer test-client-slot-capability",
            )
            self.assertNotIn("capability", transport)
            self.assertEqual(transport["id"], 1)

    def test_rejects_a_smaller_panel(self) -> None:
        with self.assertRaisesRegex(ValueError, "full 500"):
            self.TransportClient(
                backend=Backend(),
                arm="native",
                run_id="bad-panel",
                data_len=499,
                image_manifest_sha256=IMAGE_MANIFEST_SHA256,
            )

    def test_rejects_unpinned_runtime_before_creating_a_slot(self) -> None:
        broken_endpoints = []
        wrong_testspec = metadata()
        wrong_testspec["testspec_contract"] = "moving-main"
        broken_endpoints.append(("TestSpec", wrong_testspec))
        wrong_images = metadata()
        wrong_images["image_manifest"]["tag_count"] = 499
        broken_endpoints.append(("image manifest", wrong_images))
        wrong_image_manifest_hash = metadata()
        wrong_image_manifest_hash["image_manifest"]["manifest_sha256"] = "c" * 64
        broken_endpoints.append(("image manifest hash", wrong_image_manifest_hash))
        wrong_predictions = metadata()
        wrong_predictions["prediction_contract"]["schema_fields"] = [
            "instance_id",
            "model_patch",
        ]
        broken_endpoints.append(("prediction", wrong_predictions))
        internal_grader = metadata()
        internal_grader["official_grading_inside_adapter"] = True
        broken_endpoints.append(("official_grading", internal_grader))

        for label, endpoint_metadata in broken_endpoints:
            with self.subTest(label=label):
                backend = Backend(endpoint_metadata=endpoint_metadata)
                with self.assertRaises(RuntimeError):
                    self.TransportClient(
                        backend=backend,
                        arm="native",
                        run_id="bad-runtime",
                        image_manifest_sha256=IMAGE_MANIFEST_SHA256,
                    )
                self.assertFalse(
                    any(call[:2] == ("POST", "create") for call in backend.calls)
                )


if __name__ == "__main__":
    unittest.main()
