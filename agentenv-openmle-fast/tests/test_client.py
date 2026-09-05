from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from agentenv_openmle_fast.environment import POLICY_PROMPT
from tests.support import RELEASE_REVISION


def _load_client_module():
    controller = types.ModuleType("agentenv.controller")

    class BaseEnvClient:
        def __init__(self, **_kwargs):
            pass

    class BaseTask:
        pass

    @dataclass
    class StepOutput:
        state: str
        reward: float | None
        done: bool
        info: dict

    def transition_info(**kwargs):
        return {
            "env_info": dict(kwargs.get("env_info") or {}),
            "action_submission": kwargs.get("action_submission"),
            "native_step_before": kwargs.get("native_step_before"),
            "native_step_after": kwargs.get("native_step_after"),
            "native_call_count_before": kwargs.get("native_call_count_before"),
            "native_call_count_after": kwargs.get("native_call_count_after"),
            "context_epoch_before": kwargs.get("context_epoch_before"),
            "context_epoch_after": kwargs.get("context_epoch_after"),
            "session_epoch_before": kwargs.get("session_epoch_before"),
            "session_epoch_after": kwargs.get("session_epoch_after"),
            "policy_step_before": kwargs.get("policy_step_before"),
            "policy_step_after": kwargs.get("policy_step_after"),
            "context_transition": (
                kwargs.get("context_transition")
                or {
                    "schema": "agentmemory_task_neutral_context_transition_v1",
                    "operation": "append",
                    "messages": [],
                }
            ),
            "wrapper_evidence": kwargs.get("wrapper_evidence"),
        }

    controller.BaseEnvClient = BaseEnvClient
    controller.BaseTask = BaseTask
    controller_types = types.ModuleType("agentenv.controller.types")
    @dataclass(frozen=True)
    class PolicyContextPressure:
        action_prompt_tokens: int
        candidate_prompt_tokens: int
        max_prompt_tokens: int
        max_model_tokens: int
        max_response_tokens: int
        max_observation_tokens: int
        action_observation_envelope_tokens: int = 0

        @property
        def projected_next_prompt_tokens_without_control(self):
            return (
                self.action_prompt_tokens
                + self.max_response_tokens
                + self.max_observation_tokens
                + self.action_observation_envelope_tokens
            )

        @property
        def effective_prompt_capacity(self):
            return min(
                self.max_prompt_tokens,
                self.max_model_tokens - self.max_response_tokens,
            )

    def context_transition(operation, *, messages=None):
        return {
            "schema": "agentmemory_task_neutral_context_transition_v1",
            "operation": operation,
            "messages": [dict(message) for message in messages or ()],
        }

    controller_types.CONTEXT_OPERATION_REPLACE = "replace_messages"
    controller_types.POLICY_CONTINUATION_MARKER = (
        "Continue the same task in the unchanged workspace."
    )
    controller_types.ConversationMessage = dict
    controller_types.PolicyContextPressure = PolicyContextPressure
    controller_types.StepOutput = StepOutput
    controller_types.build_task_neutral_context_transition = context_transition
    controller_types.build_task_neutral_transition_info = transition_info
    envs_dir = (
        Path(__file__).resolve().parents[2]
        / "agentenv"
        / "agentenv"
        / "envs"
    )
    agentenv = types.ModuleType("agentenv")
    agentenv.__path__ = [str(envs_dir.parent)]
    envs = types.ModuleType("agentenv.envs")
    envs.__path__ = [str(envs_dir)]
    sys.modules["agentenv"] = agentenv
    sys.modules["agentenv.envs"] = envs
    sys.modules["agentenv.controller"] = controller
    sys.modules["agentenv.controller.types"] = controller_types

    helper_name = "agentenv.envs.filesystem_checkpoint"
    helper_spec = importlib.util.spec_from_file_location(
        helper_name, envs_dir / "filesystem_checkpoint.py"
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    sys.modules[helper_name] = helper
    helper_spec.loader.exec_module(helper)

    module_name = "agentenv.envs.openmle_fast"
    spec = importlib.util.spec_from_file_location(
        module_name, envs_dir / "openmle_fast.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_CLIENT_MODULE = _load_client_module()
OPENMLE_FAST_POLICY_SYSTEM_PROMPT = _CLIENT_MODULE.OPENMLE_FAST_POLICY_SYSTEM_PROMPT
OpenMLEFastEnvClient = _CLIENT_MODULE.OpenMLEFastEnvClient
OPENMLE_CONTEXT_COMPACTION_REQUEST = (
    _CLIENT_MODULE.OPENMLE_CONTEXT_COMPACTION_REQUEST
)
OPENMLE_POLICY_CONTINUATION_MARKER = (
    _CLIENT_MODULE.OPENMLE_POLICY_CONTINUATION_MARKER
)


def qwen_call(name: str, **parameters: object) -> str:
    body = ["<tool_call>", f"<function={name}>"]
    for key, value in parameters.items():
        body.extend((f"<parameter={key}>", str(value), "</parameter>"))
    body.extend(("</function>", "</tool_call>"))
    return "\n".join(body)


def checkpoint_receipt(
    *,
    action_kind: str = "apply_patch",
    action_completed: bool = True,
    changed: bool = True,
    exists: bool = True,
    regular_file: bool = True,
    size_bytes: int = 31,
    sha256: str = "f" * 64,
) -> dict:
    return {
        "schema": "agentmemory_filesystem_checkpoint_receipt_v1",
        "path": ".agent_memory/CONTINUATION.md",
        "action_kind": action_kind,
        "action_completed": action_completed,
        "changed": changed,
        "exists": exists,
        "regular_file": regular_file,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


class _Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self.payload


class OpenMLEFastClientTest(unittest.TestCase):
    def test_client_and_server_prompt_contracts_are_byte_identical(self) -> None:
        self.assertEqual(OPENMLE_FAST_POLICY_SYSTEM_PROMPT, POLICY_PROMPT)

    def test_prompt_teaches_exact_action_grammar_without_task_content(self) -> None:
        prompt = OPENMLE_FAST_POLICY_SYSTEM_PROMPT
        self.assertIn("Use exactly one Qwen XML function call per response", prompt)
        self.assertIn("optional workdir must be exactly `.`", prompt)
        self.assertIn("from 1 through 20000", prompt)
        self.assertIn("Do not use `python -c`, `python3 -c`, a heredoc, or `bash -c`", prompt)
        self.assertIn("If an observation reports a parser error", prompt)
        self.assertIn("no reasoning, explanation, Markdown fence", prompt)
        self.assertIn("<function=shell_command>", prompt)
        self.assertNotIn("WRONG:", prompt)
        self.assertIn("Use the first turns efficiently", prompt)
        self.assertNotIn("Default first three actions", prompt)
        self.assertNotRegex(prompt, r"(?m)^Action [0-9]+:")
        self.assertIn("cat TASK.md; head -3 data/train.csv", prompt)
        self.assertIn("Every added file line starts with `+`", prompt)
        self.assertIn(
            "For creating or replacing `train.py`, prefer one shell_command with `printf`",
            prompt,
        )
        self.assertIn(
            "On a later turn, create workspace-relative `train.py` with one shell_command and `printf`",
            prompt,
        )
        self.assertNotIn("Do not use shell redirection to create Python code", prompt)
        self.assertIn("Do not use `python -c`, `python3 -c`, a heredoc, or `bash -c`", prompt)
        self.assertIn("Dependencies are already installed", prompt)
        self.assertIn("Never run `pip`, `pip3`, `conda`, `apt`, `ssh`, `curl`, `wget`, or `chmod`", prompt)
        self.assertIn("print one explicit measured metric line", prompt)
        self.assertIn("use only public labelled training data", prompt)
        self.assertIn("deterministic local validation split", prompt)
        self.assertIn(".agent_memory/CONTINUATION.md", prompt)
        self.assertIn("after a continuation marker, read it", prompt)
        self.assertIn("On the following turn, run `python train.py`", prompt)
        self.assertIn("Do not write the continuation note before the first measured validation", prompt)
        self.assertNotIn("After each meaningful experiment", prompt)
        self.assertIn("submit no later than action 27", prompt)
        self.assertIn("Training-set metrics do not count as local validation", prompt)
        self.assertIn("No network access is available", prompt)
        self.assertIn("n_jobs=1", prompt)
        self.assertIn("managed 15000 ms runtime", prompt)
        self.assertIn("Never use `/workspace/` in an apply_patch file path", prompt)
        self.assertIn("<function=submit>", prompt)
        self.assertIn("protected private data exactly once", prompt)
        self.assertIn("first submit is terminal", prompt)
        self.assertIn("there is no automatic submission", prompt)
        self.assertNotIn("evaluate_candidate", prompt)
        self.assertNotIn("best-so-far", prompt)
        self.assertNotIn("cement-sales-demand", prompt)

    def test_context_compaction_requires_a_relative_continuation_then_recovery(
        self,
    ) -> None:
        request = _CLIENT_MODULE.OPENMLE_CONTEXT_COMPACTION_REQUEST
        marker = getattr(
            _CLIENT_MODULE,
            "OPENMLE_POLICY_CONTINUATION_MARKER",
            None,
        )
        self.assertNotIn("but you may instead", request)
        self.assertIn("exactly one normal executable shell_command or apply_patch", request)
        self.assertIn("not measured yet", request)
        self.assertIn(
            "one concrete `next_action` that changes `train.py`", request
        )
        self.assertIsInstance(marker, str)
        self.assertIn("Earlier conversation was removed", marker)
        self.assertIn("read that file", marker)
        self.assertIn("immediately execute its `next_action`", marker)
        self.assertIn("modify `train.py` once before running it again", marker)
        self.assertIn("submit now", marker)
        self.assertIn("do not start a third iteration", marker)
        self.assertIn("Do not inspect the task or schema again", marker)
        self.assertIn("only one action remains", marker)
        self.assertIn("submit", marker)

    def metadata(self):
        prompt_sha = hashlib.sha256(
            OPENMLE_FAST_POLICY_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()
        return {
            "schema": "openmle_fast_public_metadata_v1",
            "domain_id": "openmle_fast",
            "contract_version": "openmle_fast_v1",
            "panel_id": "openmle-fast-unit-gate",
            "release_revision": RELEASE_REVISION,
            "openmle_tasks_revision": RELEASE_REVISION,
            "manifest_sha256": "a" * 64,
            "task_manifest_sha256": "a" * 64,
            "task_id_list_sha256": "d" * 64,
            "compact_panel_sha256": "e" * 64,
            "policy_prompt_sha256": prompt_sha,
            "task_count": 1,
            "role": "gate_only",
            "max_policy_actions": 30,
            "observation_max_bytes": 65536,
            "max_observation_tokens": 16_384,
            "recoverable_invalid_action_reward": -0.01,
            "boundary_contracts": dict(_CLIENT_MODULE._EXPECTED_BOUNDARIES),
            "contracts": {
                "action": _CLIENT_MODULE._EXPECTED_BOUNDARIES["actions"],
                "observation": _CLIENT_MODULE._EXPECTED_BOUNDARIES["observation"],
                "horizon": _CLIENT_MODULE._EXPECTED_BOUNDARIES["horizon"],
                "workspace": _CLIENT_MODULE._EXPECTED_BOUNDARIES["workspace"],
                "executor": _CLIENT_MODULE._EXPECTED_BOUNDARIES["executor"],
                "grader_boundary": _CLIENT_MODULE._EXPECTED_BOUNDARIES["grader"],
                "cleanup": _CLIENT_MODULE._EXPECTED_BOUNDARIES["cleanup"],
            },
            "runtime_source": {
                "outer_commit": "1" * 40,
                "inner_commit": "2" * 40,
            },
            "executor_runtime_digest": "sha256:" + "3" * 64,
            "implementation_digests": {
                "materializer_sha256": "4" * 64,
                "actions_sha256": "5" * 64,
            },
            "resource_limits": dict(_CLIENT_MODULE._FROZEN_RESOURCE_LIMITS),
            "limits": {
                "max_policy_actions": 30,
                "max_request_wall_seconds": 20.0,
            },
            "executor_coverage": {
                "formal_eligible": True,
                "backend_contract": "openmle_fast_linux_cgroup_namespace_runner_v1",
                "execution_counter_coverage": "complete",
                "fit_counter_coverage": "partial",
            },
            "audit_enabled": True,
            "active_slot_count": 0,
            "active_environment_count": 0,
            "active_workspace_count": 0,
        }

    def client_kwargs(self):
        return {
            "expected_manifest_sha256": "a" * 64,
            "expected_release_revision": RELEASE_REVISION,
            "expected_outer_commit": "1" * 40,
            "expected_inner_commit": "2" * 40,
            "expected_role": "gate_only",
            "expected_executor_runtime_digest": "sha256:" + "3" * 64,
            "expected_materializer_sha256": "4" * 64,
            "expected_actions_sha256": "5" * 64,
            "expected_max_observation_tokens": 16_384,
        }

    def step_response(
        self,
        *,
        observation="task",
        reward=0.0,
        done=False,
        action_count=0,
        action_kind="reset",
        terminal_reason=None,
        truncated=False,
        episode_id="episode-1",
        data_idx=0,
    ):
        counters = {
            "action_count": action_count,
            "execution_action_count": 0,
            "execution_attempt_count": 0,
            "execution_completed_count": 0,
            "nested_subprocess_count": 0,
            "fit_count": 0,
            "grading_count": 0,
            "managed_runtime_wall_seconds": 0.0,
            "raw_output_bytes": 0,
        }
        return {
            "observation": observation,
            "state": observation,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": {
                "schema": "openmle_fast_episode_v1",
                "episode_id": episode_id,
                "data_idx": data_idx,
                "task_id": "tiny-regression@1",
                "source_family": "TEST:tiny-regression",
                "public_tree_sha256": "6" * 64,
                "manifest_sha256": "a" * 64,
                "task_manifest_sha256": "a" * 64,
                "release_revision": RELEASE_REVISION,
                "manifest_role": "gate_only",
                "archive_sha256": "7" * 64,
                "package_identity_sha256": "8" * 64,
                "task_spec_sha256": "9" * 64,
                "grader_binding_sha256": "b" * 64,
                "runtime_source": {
                    "outer_commit": "1" * 40,
                    "inner_commit": "2" * 40,
                },
                "executor_runtime_digest": "sha256:" + "3" * 64,
                "implementation_digests": {
                    "materializer_sha256": "4" * 64,
                    "actions_sha256": "5" * 64,
                },
                "boundary_contracts": dict(_CLIENT_MODULE._EXPECTED_BOUNDARIES),
                "action_kind": action_kind,
                "action_status": "terminal" if done else "completed",
                "terminal": done,
                "truncated": truncated,
                "terminal_reason": terminal_reason,
                "runtime_success": False,
                "episode_success": False,
                "counters": counters,
                "counter_delta": {
                    **counters,
                    "action_count": (
                        0 if action_kind in {"reset", "policy_horizon"} else 1
                    ),
                },
                "fit_counter_coverage": "not_observed",
                "execution": None,
                "sandbox_freeze": None,
                "sandbox_teardown": None,
                "grade": None,
                "audit_digest": "c" * 64,
                "unaudited_evidence_sha256": None,
            },
        }

    def graded_terminal_response(self):
        terminal = self.step_response(
            observation="graded",
            reward=1.0,
            done=True,
            action_count=1,
            action_kind="submit",
            terminal_reason="graded_submission",
        )
        info = terminal["info"]
        info["action_status"] = "graded"
        info["runtime_success"] = True
        info["episode_success"] = True
        info["counters"]["grading_count"] = 1
        info["counter_delta"]["action_count"] = 1
        info["counter_delta"]["grading_count"] = 1
        info["grade"] = {
            "schema": "openmle_fast_grade_response_v1",
            "contract_version": "openmle_fast_v1",
            "request_id": "request-terminal",
            "episode_id": "episode-1",
            "task_id": "tiny-regression@1",
            "submission_sha256": "f" * 64,
            "submission_valid": True,
            "native_score": 0.0,
            "higher_is_better": False,
            "normalized_reward": 1.0,
            "improved_over_baseline": True,
            "runtime_success": True,
            "terminal_reason": "graded_submission",
            "classification": "graded",
            "audit_digest": "e" * 64,
        }
        return terminal

    def test_attests_before_create_and_translates_qwen_action(self) -> None:
        calls = []
        metadata = self.metadata()

        def request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            path = url.rsplit("/", 1)[-1]
            if path == "metadata":
                return _Response(metadata)
            if path == "create":
                return _Response({"id": 7, "observation": "unbound", "info": {}})
            if path == "reset":
                return _Response(self.step_response())
            if path == "step":
                return _Response(
                    self.step_response(
                        observation="ran",
                        action_count=1,
                        action_kind="shell_command",
                    )
                )
            if path == "horizon":
                return _Response(
                    self.step_response(
                        observation="done",
                        reward=-1.0,
                        done=True,
                        action_count=1,
                        action_kind="policy_horizon",
                        terminal_reason="action_budget_exhausted",
                    )
                )
            if path == "close":
                return _Response(
                    {
                        "schema": "openmle_fast_cleanup_receipt_v1",
                        "closed": True,
                        "already_closed": False,
                        "workspace_removed": True,
                        "retryable": False,
                        "failure_class": None,
                        "cleanup_contract": _CLIENT_MODULE._EXPECTED_BOUNDARIES[
                            "cleanup"
                        ],
                    }
                )
            raise AssertionError(path)

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000",
                **self.client_kwargs(),
            )
            self.assertTrue(calls[0][1].endswith("/metadata"))
            self.assertTrue(calls[1][1].endswith("/create"))
            client.reset(0)
            raw = qwen_call("shell_command", command="printf 'unchanged'")
            output = client.step(raw)
            submitted = 'shell_command {"command": "printf \'unchanged\'"}'
            self.assertEqual(calls[-1][2]["json"]["action"], submitted)
            self.assertEqual(output.info["action_submission"]["raw_policy_output"], raw)
            self.assertEqual(
                output.info["action_submission"]["submitted_action"], submitted
            )
            self.assertTrue(
                output.info["action_submission"]["tool_parser_normalized"]
            )
            self.assertEqual(output.info["policy_step_before"], 0)
            self.assertEqual(output.info["policy_step_after"], 1)
            terminal = client.finalize_policy_horizon()
            self.assertTrue(terminal.done)
            self.assertEqual(terminal.info["policy_step_after"], 1)
            client.close()

    def test_recoverable_parser_penalty_is_accepted_and_clears_on_valid_action(
        self,
    ) -> None:
        metadata = self.metadata()
        step_index = 0

        def request(method, url, **kwargs):
            nonlocal step_index
            path = url.rsplit("/", 1)[-1]
            if path == "metadata":
                return _Response(metadata)
            if path == "create":
                return _Response({"id": 7, "observation": "unbound", "info": {}})
            if path == "reset":
                return _Response(self.step_response())
            if path == "step":
                step_index += 1
                response = self.step_response(
                    observation="parser error" if step_index == 1 else "ran",
                    reward=-0.01 if step_index == 1 else 0.0,
                    action_count=step_index,
                    action_kind="parser_error" if step_index == 1 else "shell_command",
                )
                response["info"]["action_status"] = (
                    "parser_error" if step_index == 1 else "completed"
                )
                return _Response(response)
            raise AssertionError(path)

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000",
                **self.client_kwargs(),
            )
            client.reset(0)
            invalid = client.step("malformed")
            recovered = client.step(
                qwen_call("shell_command", command="pwd", workdir=".")
            )

        self.assertEqual(invalid.reward, -0.01)
        self.assertFalse(invalid.done)
        self.assertEqual(recovered.reward, 0.0)
        self.assertFalse(recovered.done)

    def test_step_normalizes_qwen_xml_with_upstream_parser_and_keeps_raw_audit(self) -> None:
        calls = []
        metadata = self.metadata()
        raw = """<tool_call>
<function=shell_command>
<parameter=command>
cat .agent_memory/CONTINUATION.md
</parameter>
<parameter=workdir>
.
</parameter>
</function>
</tool_call>"""

        def request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            path = url.rsplit("/", 1)[-1]
            if path == "metadata":
                return _Response(metadata)
            if path == "create":
                return _Response({"id": 7, "observation": "unbound", "info": {}})
            if path == "reset":
                return _Response(self.step_response())
            if path == "step":
                return _Response(
                    self.step_response(
                        observation="read",
                        action_count=1,
                        action_kind="shell_command",
                    )
                )
            raise AssertionError(path)

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000",
                **self.client_kwargs(),
            )
            client.reset(0)
            output = client.step(raw)

        posted = [call for call in calls if call[1].endswith("/step")][0][2]["json"]["action"]
        self.assertEqual(
            posted,
            'shell_command {"command": "cat .agent_memory/CONTINUATION.md", '
            '"workdir": "."}',
        )
        submission = output.info["action_submission"]
        self.assertEqual(submission["raw_policy_output"], raw)
        self.assertEqual(submission["submitted_action"], posted)
        self.assertEqual(submission["tool_parser"], "qwen3_coder")
        self.assertTrue(submission["tool_parser_normalized"])

    def test_context_pressure_executes_checkpoint_without_injecting_its_body(self) -> None:
        calls = []
        metadata = self.metadata()
        receipt = checkpoint_receipt(size_bytes=31, sha256="e" * 64)

        def request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            path = url.rsplit("/", 1)[-1]
            if path == "metadata":
                return _Response(metadata)
            if path == "create":
                return _Response({"id": 7, "observation": "unbound", "info": {}})
            if path == "reset":
                return _Response(self.step_response(observation="task framing"))
            if path == "step":
                response = self.step_response(
                    observation="action_status=completed",
                    action_count=1,
                    action_kind="apply_patch",
                )
                response["info"]["execution"] = {
                    "changed_paths": [".agent_memory/CONTINUATION.md"],
                    "filesystem_checkpoint": receipt,
                }
                return _Response(response)
            raise AssertionError(path)

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000",
                **self.client_kwargs(),
            )
            client.reset(0)
            initial = client.normalize_initial_policy_context(
                [
                    {"role": "system", "content": "legacy"},
                    {"role": "user", "content": client.observe()},
                ]
            )
            client.bind_policy_context(initial, initial=True)
            candidate = client.policy_turn_candidate()
            self.assertEqual(
                candidate, _CLIENT_MODULE.OPENMLE_CONTEXT_COMPACTION_REQUEST
            )
            self.assertIsNone(
                client.prepare_policy_turn(
                    _CLIENT_MODULE.PolicyContextPressure(
                        action_prompt_tokens=50,
                        candidate_prompt_tokens=80,
                        max_prompt_tokens=10_000,
                        max_model_tokens=10_032,
                        max_response_tokens=32,
                        max_observation_tokens=20,
                        action_observation_envelope_tokens=4,
                    )
                )
            )
            selected = client.prepare_policy_turn(
                _CLIENT_MODULE.PolicyContextPressure(
                    action_prompt_tokens=8_000,
                    candidate_prompt_tokens=8_030,
                    max_prompt_tokens=10_000,
                    max_model_tokens=10_032,
                    max_response_tokens=32,
                    max_observation_tokens=150,
                    action_observation_envelope_tokens=4,
                )
            )
            self.assertEqual(selected, candidate)

            secret_body = "next inspect train.csv"
            patch_body = f"""*** Begin Patch
*** Add File: .agent_memory/CONTINUATION.md
+{secret_body}
*** End Patch"""
            action = qwen_call("apply_patch", patch=patch_body)
            output = client.step(action)

        step_calls = [call for call in calls if call[1].endswith("/step")]
        self.assertEqual(len(step_calls), 1)
        self.assertEqual(
            step_calls[0][2]["json"]["action"], "apply_patch\n" + patch_body
        )
        self.assertEqual(metadata["max_policy_actions"], 30)
        self.assertEqual(
            (
                output.info["native_call_count_before"],
                output.info["native_call_count_after"],
                output.info["policy_step_before"],
                output.info["policy_step_after"],
                output.info["context_epoch_before"],
                output.info["context_epoch_after"],
            ),
            (0, 1, 0, 1, 0, 1),
        )
        replacement = output.info["context_transition"]["messages"]
        self.assertEqual(len(replacement), len(initial))
        self.assertEqual(replacement[0], initial[0])
        self.assertTrue(replacement[-1]["content"].startswith(initial[-1]["content"]))
        self.assertIn(".agent_memory/CONTINUATION.md", replacement[-1]["content"])
        self.assertIn(receipt["sha256"], replacement[-1]["content"])
        self.assertIn(
            _CLIENT_MODULE.OPENMLE_EXACT_CHECKPOINT_READ_ACTION,
            replacement[-1]["content"],
        )
        self.assertIn(
            "Do not overwrite `.agent_memory/CONTINUATION.md`",
            replacement[-1]["content"],
        )
        self.assertNotIn(secret_body, str(replacement))
        self.assertNotIn(action, str(replacement))
        self.assertNotIn("action_status=completed", str(replacement))
        self.assertFalse(any(message["role"] == "assistant" for message in replacement))
        evidence = output.info["wrapper_evidence"]
        self.assertEqual(evidence["event"], "context_compaction")
        self.assertTrue(evidence["continuation_persisted"])
        self.assertTrue(evidence["context_replaced"])
        self.assertFalse(evidence["retry_pending"])
        self.assertEqual(evidence["checkpoint_receipt"], receipt)
        self.assertFalse(evidence["checkpoint_action_in_successor_context"])
        self.assertFalse(evidence["checkpoint_content_in_successor_context"])

    def test_failed_checkpoint_write_rebuilds_context_and_forces_retry(self) -> None:
        metadata = self.metadata()

        def request(method, url, **kwargs):
            path = url.rsplit("/", 1)[-1]
            if path == "metadata":
                return _Response(metadata)
            if path == "create":
                return _Response({"id": 7, "observation": "unbound", "info": {}})
            if path == "reset":
                return _Response(self.step_response(observation="task framing"))
            if path == "step":
                response = self.step_response(
                    observation="parser_error: expected one exact action",
                    reward=-0.01,
                    action_count=1,
                    action_kind="parser_error",
                )
                response["info"]["action_status"] = "parser_error"
                response["info"]["execution"] = None
                return _Response(response)
            raise AssertionError(path)

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000",
                **self.client_kwargs(),
            )
            client.reset(0)
            initial = client.normalize_initial_policy_context(
                [{"role": "user", "content": client.observe()}]
            )
            client.bind_policy_context(initial, initial=True)
            selected = client.prepare_policy_turn(
                _CLIENT_MODULE.PolicyContextPressure(
                    action_prompt_tokens=120,
                    candidate_prompt_tokens=130,
                    max_prompt_tokens=300,
                    max_model_tokens=332,
                    max_response_tokens=32,
                    max_observation_tokens=150,
                    action_observation_envelope_tokens=4,
                )
            )
            self.assertEqual(selected, OPENMLE_CONTEXT_COMPACTION_REQUEST)
            malformed = 'apply_patch {"patch":"unfinished state"}'
            output = client.step(malformed)

        self.assertEqual(
            output.info["context_transition"]["operation"], "replace_messages"
        )
        replacement = output.info["context_transition"]["messages"]
        self.assertEqual(len(replacement), len(initial))
        self.assertEqual(replacement[:-1], initial[:-1])
        self.assertTrue(replacement[-1]["content"].startswith(initial[-1]["content"]))
        self.assertIn("Filesystem checkpoint was not accepted", str(replacement))
        self.assertNotIn(malformed, str(replacement))
        self.assertNotIn(
            "parser_error: expected one exact action", str(replacement)
        )
        self.assertEqual(
            (
                output.info["native_call_count_before"],
                output.info["native_call_count_after"],
                output.info["policy_step_before"],
                output.info["policy_step_after"],
                output.info["context_epoch_before"],
                output.info["context_epoch_after"],
            ),
            (0, 1, 0, 1, 0, 0),
        )
        evidence = output.info["wrapper_evidence"]
        self.assertFalse(evidence["continuation_persisted"])
        self.assertFalse(evidence["context_replaced"])
        self.assertTrue(evidence["retry_pending"])
        self.assertTrue(evidence["checkpoint_retry_context_rebuilt"])
        self.assertEqual(evidence["checkpoint_failure_reason"], "missing_receipt")
        self.assertFalse(evidence["checkpoint_action_in_successor_context"])
        self.assertFalse(evidence["checkpoint_observation_in_successor_context"])
        self.assertFalse(evidence["checkpoint_content_in_successor_context"])

        # Retry is selected even when ordinary pressure is low.
        retry = client.prepare_policy_turn(
            _CLIENT_MODULE.PolicyContextPressure(
                action_prompt_tokens=20,
                candidate_prompt_tokens=40,
                max_prompt_tokens=300,
                max_model_tokens=332,
                max_response_tokens=32,
                max_observation_tokens=20,
                action_observation_envelope_tokens=4,
            )
        )
        self.assertEqual(retry, OPENMLE_CONTEXT_COMPACTION_REQUEST)

    def test_continuation_receipt_requires_completed_write_to_exact_path(
        self,
    ) -> None:
        for action_kind in ("apply_patch", "shell_command"):
            receipt = checkpoint_receipt(action_kind=action_kind)
            self.assertTrue(
                _CLIENT_MODULE._continuation_write_succeeded(
                    {"execution": {"filesystem_checkpoint": receipt}}
                )
            )
        for receipt in (
            None,
            checkpoint_receipt(action_kind="parser_error"),
            checkpoint_receipt(changed=False),
            checkpoint_receipt(action_completed=False),
            checkpoint_receipt(size_bytes=0),
            checkpoint_receipt(size_bytes=8193),
        ):
            self.assertFalse(
                _CLIENT_MODULE._continuation_write_succeeded(
                    {"execution": {"filesystem_checkpoint": receipt}}
                )
            )

    def test_rejects_manifest_or_data_len_mismatch_before_create(self) -> None:
        metadata = self.metadata()
        with patch("requests.request", return_value=_Response(metadata)) as request:
            with self.assertRaises(RuntimeError):
                OpenMLEFastEnvClient(
                    "http://127.0.0.1:9000",
                    **{
                        **self.client_kwargs(),
                        "expected_manifest_sha256": "b" * 64,
                    },
                )
            self.assertEqual(request.call_count, 1)

    def test_shared_factory_requires_independent_attestation_pins(self) -> None:
        calls = []

        def request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/metadata"):
                return _Response(self.metadata())
            if url.endswith("/create"):
                return _Response({"id": 3, "observation": "unbound", "info": {}})
            raise AssertionError(url)

        with (
            patch("requests.request", side_effect=request),
            self.assertRaisesRegex(ValueError, "TASK_MANIFEST_SHA256"),
        ):
            OpenMLEFastEnvClient(
                "http://127.0.0.1:9000",
                data_len=None,
                timeout=2400,
            )
        self.assertEqual([call[0] for call in calls], ["GET"])

    def test_shared_factory_can_receive_independent_pins_from_environment(self) -> None:
        calls = []

        def request(method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/metadata"):
                return _Response(self.metadata())
            if url.endswith("/create"):
                return _Response({"id": 3, "observation": "unbound", "info": {}})
            raise AssertionError(url)

        environment = {
            "OPENMLE_FAST_TASK_MANIFEST_SHA256": "a" * 64,
            "OPENMLE_FAST_RELEASE_REVISION": RELEASE_REVISION,
            "OPENMLE_FAST_RUNTIME_OUTER_COMMIT": "1" * 40,
            "OPENMLE_FAST_RUNTIME_INNER_COMMIT": "2" * 40,
            "OPENMLE_FAST_MANIFEST_ROLE": "gate_only",
            "OPENMLE_FAST_EXECUTOR_RUNTIME_DIGEST": "sha256:" + "3" * 64,
            "OPENMLE_FAST_MATERIALIZER_SHA256": "4" * 64,
            "OPENMLE_FAST_ACTIONS_SHA256": "5" * 64,
            "OPENMLE_FAST_MAX_OBSERVATION_TOKENS": "16384",
        }
        with (
            patch.dict(_CLIENT_MODULE.os.environ, environment, clear=True),
            patch("requests.request", side_effect=request),
        ):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000",
                data_len=None,
                timeout=2400,
            )
        self.assertEqual(len(client), 1)
        self.assertEqual([call[0] for call in calls], ["GET", "POST"])

    def test_truncated_reset_is_marked_for_resampling(self) -> None:
        def request(_method, url, **_kwargs):
            if url.endswith("/metadata"):
                return _Response(self.metadata())
            if url.endswith("/create"):
                return _Response({"id": 3, "observation": "unbound", "info": {}})
            if url.endswith("/reset"):
                return _Response(
                    self.step_response(
                        reward=None,
                        done=True,
                        truncated=True,
                        action_count=4,
                        terminal_reason="reset_infrastructure_fault",
                    )
                )
            raise AssertionError(url)

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000", **self.client_kwargs()
            )
            with self.assertRaisesRegex(RuntimeError, "must be resampled"):
                client.reset(0)
        self.assertTrue(client.sample_excluded)

    def test_retryable_cleanup_failure_is_not_silently_accepted(self) -> None:
        def request(_method, url, **_kwargs):
            if url.endswith("/metadata"):
                return _Response(self.metadata())
            if url.endswith("/create"):
                return _Response({"id": 3, "observation": "unbound", "info": {}})
            if url.endswith("/close"):
                return _Response(
                    {
                        "schema": "openmle_fast_cleanup_receipt_v1",
                        "closed": False,
                        "already_closed": False,
                        "workspace_removed": False,
                        "retryable": True,
                        "failure_class": "cleanup_infrastructure_fault",
                        "cleanup_contract": _CLIENT_MODULE._EXPECTED_BOUNDARIES[
                            "cleanup"
                        ],
                    }
                )
            raise AssertionError(url)

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000", **self.client_kwargs()
            )
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                client.close()

    def test_rejects_reset_index_and_cross_step_episode_identity_drift(self) -> None:
        responses = [
            self.step_response(data_idx=1),
            self.step_response(),
            self.step_response(
                action_count=1,
                action_kind="parser_error",
                episode_id="episode-2",
            ),
        ]

        def request(_method, url, **_kwargs):
            if url.endswith("/metadata"):
                return _Response(self.metadata())
            if url.endswith("/create"):
                return _Response({"id": 3, "observation": "unbound", "info": {}})
            return _Response(responses.pop(0))

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000", **self.client_kwargs()
            )
            with self.assertRaisesRegex(RuntimeError, "wrong data_idx"):
                client.reset(0)
            client.reset(0)
            with self.assertRaisesRegex(RuntimeError, "identity drifted"):
                client.step("malformed")

    def test_rejects_ineligible_or_drifted_endpoint_contracts(self) -> None:
        cases = []
        ineligible = copy.deepcopy(self.metadata())
        ineligible["executor_coverage"]["formal_eligible"] = False
        cases.append(ineligible)
        missing_boundary = copy.deepcopy(self.metadata())
        missing_boundary["boundary_contracts"].pop("audit")
        cases.append(missing_boundary)
        limit_drift = copy.deepcopy(self.metadata())
        limit_drift["resource_limits"]["max_processes"] = 65
        cases.append(limit_drift)
        digest_drift = copy.deepcopy(self.metadata())
        digest_drift["implementation_digests"]["actions_sha256"] = "f" * 64
        cases.append(digest_drift)
        for metadata in cases:
            with (
                self.subTest(metadata=metadata),
                patch("requests.request", return_value=_Response(metadata)) as request,
            ):
                with self.assertRaises(RuntimeError):
                    OpenMLEFastEnvClient(
                        "http://127.0.0.1:9000",
                        **self.client_kwargs(),
                    )
                self.assertEqual(request.call_count, 1)

    def test_strict_response_validation_rejects_coercion_and_nonfinite_data(
        self,
    ) -> None:
        metadata = self.metadata()
        invalid = []
        nan_reward = self.step_response(action_count=1, action_kind="parser_error")
        nan_reward["reward"] = float("nan")
        invalid.append(nan_reward)
        string_done = self.step_response(action_count=1, action_kind="parser_error")
        string_done["done"] = "false"
        invalid.append(string_done)
        bad_counters = self.step_response(action_count=1, action_kind="parser_error")
        bad_counters["info"]["counters"]["execution_completed_count"] = 1
        invalid.append(bad_counters)
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                _CLIENT_MODULE._validate_step_response(
                    response,
                    metadata=metadata,
                    expected_action_count=1,
                    expected_action_delta=1,
                )
        with patch("requests.request", return_value=_Response(metadata)) as request:
            with self.assertRaises(ValueError):
                OpenMLEFastEnvClient(
                    "http://127.0.0.1:9000",
                    data_len=2,
                    **self.client_kwargs(),
                )
            self.assertEqual(request.call_count, 1)

    def test_terminal_step_binds_public_grade_identity_into_action_submission(
        self,
    ) -> None:
        terminal = self.graded_terminal_response()

        def request(_method, url, **_kwargs):
            if url.endswith("/metadata"):
                return _Response(self.metadata())
            if url.endswith("/create"):
                return _Response({"id": 3, "observation": "unbound", "info": {}})
            if url.endswith("/reset"):
                return _Response(self.step_response())
            if url.endswith("/step"):
                return _Response(terminal)
            raise AssertionError(url)

        with patch("requests.request", side_effect=request):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000", **self.client_kwargs()
            )
            client.reset(0)
            raw = qwen_call("submit")
            result = client.step(raw)
        self.assertEqual(
            result.info["action_submission"],
            {
                "raw_policy_output": raw,
                "request_id": "request-terminal",
                "episode_id": "episode-1",
                "submission_sha256": "f" * 64,
                "tool_contract": "qwen3_xml_single_call_v1",
                "tool_parser": "qwen3_coder",
                "tool_parser_normalized": True,
                "submitted_action": "submit",
            },
        )

    def test_rejects_inconsistent_public_grade_receipts(self) -> None:
        mutations = (
            ("reward", lambda value: value.__setitem__("reward", 0.5)),
            (
                "classification",
                lambda value: value["info"]["grade"].__setitem__(
                    "classification", "invalid_submission"
                ),
            ),
            (
                "native_score",
                lambda value: value["info"]["grade"].__setitem__(
                    "native_score", float("nan")
                ),
            ),
            (
                "episode_id",
                lambda value: value["info"]["grade"].__setitem__(
                    "episode_id", "other-episode"
                ),
            ),
            (
                "grading_count",
                lambda value: value["info"]["counter_delta"].__setitem__(
                    "grading_count", 0
                ),
            ),
        )
        for label, mutate in mutations:
            response = copy.deepcopy(self.graded_terminal_response())
            mutate(response)
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                _CLIENT_MODULE._validate_step_response(
                    response,
                    metadata=self.metadata(),
                    expected_action_count=1,
                    expected_action_delta=1,
                )

    def test_close_retries_a_retryable_receipt_within_client_deadline(self) -> None:
        calls = []
        close_receipts = [
            {
                "schema": "openmle_fast_cleanup_receipt_v1",
                "closed": False,
                "already_closed": False,
                "workspace_removed": False,
                "retryable": True,
                "failure_class": "cleanup_infrastructure_fault",
                "cleanup_contract": _CLIENT_MODULE._EXPECTED_BOUNDARIES["cleanup"],
            },
            {
                "schema": "openmle_fast_cleanup_receipt_v1",
                "closed": True,
                "already_closed": False,
                "workspace_removed": True,
                "retryable": False,
                "failure_class": None,
                "cleanup_contract": _CLIENT_MODULE._EXPECTED_BOUNDARIES["cleanup"],
            },
        ]

        def request(_method, url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith("/metadata"):
                return _Response(self.metadata())
            if url.endswith("/create"):
                return _Response({"id": 3, "observation": "unbound", "info": {}})
            if url.endswith("/close"):
                return _Response(close_receipts.pop(0))
            raise AssertionError(url)

        with (
            patch("requests.request", side_effect=request),
            patch.object(_CLIENT_MODULE.time, "sleep") as sleep,
        ):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000", **self.client_kwargs()
            )
            receipt = client.close()
        self.assertTrue(receipt["closed"])
        self.assertEqual(sum(url.endswith("/close") for url, _ in calls), 2)
        sleep.assert_called_once_with(_CLIENT_MODULE._CLOSE_RETRY_BACKOFF_SECONDS[0])

    def test_close_waits_through_bounded_runner_teardown_settle(self) -> None:
        calls = []
        retryable = {
            "schema": "openmle_fast_cleanup_receipt_v1",
            "closed": False,
            "already_closed": False,
            "workspace_removed": False,
            "retryable": True,
            "failure_class": "cleanup_infrastructure_fault",
            "cleanup_contract": _CLIENT_MODULE._EXPECTED_BOUNDARIES["cleanup"],
        }
        closed = {
            "schema": "openmle_fast_cleanup_receipt_v1",
            "closed": True,
            "already_closed": False,
            "workspace_removed": True,
            "retryable": False,
            "failure_class": None,
            "cleanup_contract": _CLIENT_MODULE._EXPECTED_BOUNDARIES["cleanup"],
        }
        close_receipts = [copy.deepcopy(retryable) for _ in range(5)] + [closed]

        def request(_method, url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith("/metadata"):
                return _Response(self.metadata())
            if url.endswith("/create"):
                return _Response({"id": 3, "observation": "unbound", "info": {}})
            if url.endswith("/close"):
                return _Response(close_receipts.pop(0))
            raise AssertionError(url)

        with (
            patch("requests.request", side_effect=request),
            patch.object(_CLIENT_MODULE.time, "sleep") as sleep,
        ):
            client = OpenMLEFastEnvClient(
                "http://127.0.0.1:9000", **self.client_kwargs()
            )
            receipt = client.close()
        self.assertTrue(receipt["closed"])
        self.assertEqual(sum(url.endswith("/close") for url, _ in calls), 6)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            list(_CLIENT_MODULE._CLOSE_RETRY_BACKOFF_SECONDS[:5]),
        )

    def test_rejects_nonfinite_timeouts_and_retired_manifest_roles(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(timeout=value), self.assertRaises(ValueError):
                OpenMLEFastEnvClient(
                    "http://127.0.0.1:9000",
                    timeout=value,
                    **self.client_kwargs(),
                )
        with patch("requests.request", return_value=_Response(self.metadata())):
            with self.assertRaisesRegex(ValueError, "role"):
                OpenMLEFastEnvClient(
                    "http://127.0.0.1:9000",
                    **{**self.client_kwargs(), "expected_role": "mechanism_gate"},
                )


if __name__ == "__main__":
    unittest.main()
