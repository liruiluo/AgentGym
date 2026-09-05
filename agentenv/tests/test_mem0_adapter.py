from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import (
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_REPLACE,
    PolicyActionBudget,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)
from agentenv.envs.mem0 import (
    MEM0_PROMPT_MARKER,
    MEM0_SOURCE_REVISION,
    Mem0AdapterConfig,
    Mem0EnvClientAdapter,
)


class FakeNativeClient(BaseEnvClient):
    def __init__(self) -> None:
        super().__init__("react")
        self.bound: list[list[dict[str, str]]] = []
        self.replace_next = False
        self.episode_source_identity: dict[str, object] | None = None

    def __len__(self) -> int:
        return 3

    def observe(self) -> str:
        return "initial task"

    def policy_framing(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": "native system"}]

    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        return [dict(message) for message in messages]

    def bind_policy_context(
        self, messages: Sequence[Mapping[str, str]], *, initial: bool = False
    ) -> None:
        del initial
        self.bound.append([dict(message) for message in messages])

    def policy_turn_candidate(self) -> str | None:
        return None

    def prepare_policy_turn(self, _pressure) -> str | None:
        return None

    def step(self, action: str) -> StepOutput:
        transition = (
            build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=[
                    {"role": "system", "content": "native replacement system"},
                    {"role": "user", "content": "What exact fact is needed next?"},
                ],
            )
            if self.replace_next
            else build_task_neutral_context_transition(CONTEXT_OPERATION_APPEND)
        )
        return StepOutput(
            state="native observation after action",
            reward=0.5,
            done=False,
            info=build_task_neutral_transition_info(
                env_info={"episode_source_identity": self.episode_source_identity},
                action_submission={"raw_policy_output": action},
                context_transition=transition,
                wrapper_evidence={"native": {"event": "step"}},
            ),
        )

    def reset(self, idx: int = 0) -> None:
        self.episode_source_identity = {
            "schema": "camg_native_episode_source_identity_v1",
            "route_id": "swesmith",
            "data_idx": idx,
            "instance_id": f"repo.issue-{idx}",
        }

    def finalize_policy_horizon(self) -> StepOutput | None:
        return None

    def close(self) -> None:
        return None


class FakeMemory:
    def __init__(self, config: dict, *, fail_add: bool = False) -> None:
        self.config = config
        self.fail_add = fail_add
        self.add_calls: list[tuple[list[dict[str, str]], str, bool]] = []
        self.search_calls: list[tuple[str, dict, int, float]] = []
        self.closed = False

    def _callback(self, prompt: int, completion: int) -> None:
        callback = self.config["llm"]["config"]["response_callback"]
        callback(
            object(),
            SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=prompt, completion_tokens=completion
                )
            ),
            {},
        )

    def add(self, messages, *, run_id: str, infer: bool):
        if self.fail_add:
            raise RuntimeError("official add failed")
        self.add_calls.append((messages, run_id, infer))
        self._callback(11, 7)
        self._callback(13, 5)
        return {"results": [{"id": "m1", "memory": "port 4317", "event": "ADD"}]}

    def search(self, query, *, filters, top_k, threshold):
        self.search_calls.append((query, filters, top_k, threshold))
        return {
            "results": [
                {"id": "m1", "memory": "Use callback port 4317", "score": 0.91}
            ]
        }

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self, *, fail_add: bool = False) -> None:
        self.fail_add = fail_add
        self.instances: list[FakeMemory] = []

    def __call__(self, config: dict) -> FakeMemory:
        memory = FakeMemory(config, fail_add=self.fail_add)
        self.instances.append(memory)
        return memory


class Mem0AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.native = FakeNativeClient()
        self.factory = FakeFactory()
        self.adapter = Mem0EnvClientAdapter(
            self.native,
            Mem0AdapterConfig(runtime_root=self.temp.name),
            memory_factory=self.factory,
        )
        self.adapter.reset(2)
        messages = self.adapter.policy_framing() + [
            {"role": "user", "content": self.adapter.observe()}
        ]
        self.adapter.bind_policy_context(messages, initial=True)

    def bind_budget(self, *, maximum: int = 30, consumed: int = 0) -> None:
        self.adapter.bind_policy_action_budget(
            PolicyActionBudget(maximum_steps=maximum, consumed_steps=consumed)
        )

    def tearDown(self) -> None:
        self.adapter.close()
        self.temp.cleanup()

    def test_config_is_loopback_only_and_validates_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback"):
            Mem0AdapterConfig(llm_base_url="http://10.0.0.2:8000/v1")
        with self.assertRaisesRegex(ValueError, "threshold"):
            Mem0AdapterConfig(threshold=float("nan"))

    def test_official_config_is_episode_private_and_pinned(self) -> None:
        config = self.factory.instances[-1].config
        self.assertEqual(config["version"], "v1.1")
        self.assertEqual(config["vector_store"]["provider"], "qdrant")
        self.assertEqual(config["vector_store"]["config"]["embedding_model_dims"], 1024)
        self.assertEqual(config["embedder"]["provider"], "openai")
        self.assertEqual(
            config["embedder"]["config"]["openai_base_url"],
            "http://127.0.0.1:65202/v1",
        )
        # BGE-M3 has a fixed 1024-dimensional output and rejects the OpenAI
        # ``dimensions`` request field.  The Qdrant schema still records 1024
        # above, while the embedder request must leave the native size intact.
        self.assertNotIn("embedding_dims", config["embedder"]["config"])
        self.assertEqual(config["llm"]["provider"], "openai")
        self.assertTrue(Path(config["history_db_path"]).is_relative_to(self.temp.name))

    def test_non_boundary_is_passthrough_with_zero_hidden_cost(self) -> None:
        self.bind_budget()
        output = self.adapter.step('shell_command {"command":"pwd"}')
        self.assertEqual(output.reward, 0.5)
        evidence = output.info["wrapper_evidence"]["mem0_adapter"]
        self.assertFalse(evidence["boundary_pipeline"])
        self.assertEqual(evidence["hidden_model_calls"], 0)
        self.assertEqual(evidence["operation_counts"], {})
        self.assertEqual(evidence["source_revision"], MEM0_SOURCE_REVISION)
        self.assertEqual(self.factory.instances[-1].add_calls, [])
        self.assertEqual(
            output.info["action_budget"],
            {
                "schema": "agentmemory_task_neutral_action_budget_v1",
                "maximum_steps": 30,
                "consumed_steps_before": 0,
                "policy_action_steps": 1,
                "auxiliary_steps": 0,
                "required_auxiliary_steps": 0,
                "consumed_steps_after": 1,
                "remaining_steps_after": 29,
                "atomic_operation_blocked": False,
                "terminate_after_action": False,
            },
        )

    def test_replace_boundary_runs_official_add_search_and_injects_results(self) -> None:
        self.native.replace_next = True
        self.bind_budget()
        output = self.adapter.step('shell_command {"command":"run tests"}')
        memory = self.factory.instances[-1]
        self.assertEqual(len(memory.add_calls), 1)
        messages, run_id, infer = memory.add_calls[0]
        self.assertTrue(infer)
        self.assertTrue(run_id.startswith("camg-"))
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(len(memory.search_calls), 1)
        replacement = output.info["context_transition"]["messages"]
        self.assertIn(MEM0_PROMPT_MARKER, replacement[0]["content"])
        self.assertIn("Use callback port 4317", replacement[0]["content"])
        evidence = output.info["wrapper_evidence"]["mem0_adapter"]
        self.assertTrue(evidence["boundary_pipeline"])
        self.assertEqual(evidence["hidden_model_calls"], 2)
        self.assertEqual(evidence["hidden_input_tokens"], 24)
        self.assertEqual(evidence["hidden_output_tokens"], 12)
        self.assertEqual(evidence["operation_counts"]["add"], 1)
        self.assertEqual(evidence["operation_counts"]["search"], 1)
        self.assertEqual(evidence["operation_counts"]["retrieved"], 1)
        self.assertEqual(output.info["action_budget"]["auxiliary_steps"], 2)
        self.assertEqual(output.info["action_budget"]["consumed_steps_after"], 3)

        self.adapter.bind_policy_context(replacement)
        self.assertNotIn(MEM0_PROMPT_MARKER, self.native.bound[-1][0]["content"])

    def test_replace_boundary_never_runs_a_partial_pipeline_at_budget_edge(self) -> None:
        self.native.replace_next = True
        self.bind_budget(maximum=30, consumed=28)

        output = self.adapter.step('shell_command {"command":"run tests"}')

        self.assertTrue(output.done)
        self.assertEqual(self.factory.instances[-1].add_calls, [])
        self.assertEqual(self.factory.instances[-1].search_calls, [])
        receipt = output.info["action_budget"]
        self.assertEqual(receipt["consumed_steps_before"], 28)
        self.assertEqual(receipt["consumed_steps_after"], 29)
        self.assertEqual(receipt["required_auxiliary_steps"], 2)
        self.assertEqual(receipt["auxiliary_steps"], 0)
        self.assertTrue(receipt["atomic_operation_blocked"])
        self.assertTrue(receipt["terminate_after_action"])
        evidence = output.info["wrapper_evidence"]
        self.assertEqual(evidence["outcome"], "terminal_failure")
        self.assertEqual(
            evidence["terminal_reason"], "combined_step_budget_exhausted"
        )
        self.assertTrue(evidence["mem0_adapter"]["boundary_requested"])
        self.assertFalse(evidence["mem0_adapter"]["boundary_pipeline"])

    def test_step_requires_fresh_budget_binding_before_native_side_effect(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "action budget"):
            self.adapter.step('shell_command {"command":"pwd"}')
        self.assertEqual(self.factory.instances[-1].add_calls, [])

    def test_reset_closes_and_removes_previous_episode_store(self) -> None:
        previous = self.factory.instances[-1]
        previous_dir = self.adapter._episode_dir
        self.assertTrue(previous_dir and previous_dir.is_dir())
        self.adapter.reset(1)
        self.assertTrue(previous.closed)
        self.assertFalse(previous_dir.exists())
        self.assertEqual(len(self.factory.instances), 2)

    def test_official_pipeline_failure_is_fail_closed(self) -> None:
        self.adapter.close()
        factory = FakeFactory(fail_add=True)
        self.adapter = Mem0EnvClientAdapter(
            self.native,
            Mem0AdapterConfig(runtime_root=self.temp.name),
            memory_factory=factory,
        )
        self.adapter.reset(0)
        self.adapter.bind_policy_context(
            [
                {"role": "system", "content": "native"},
                {"role": "user", "content": "task"},
            ],
            initial=True,
        )
        self.native.replace_next = True
        self.bind_budget()
        with self.assertRaisesRegex(RuntimeError, "official add failed"):
            self.adapter.step('shell_command {"command":"work"}')


if __name__ == "__main__":
    unittest.main()
