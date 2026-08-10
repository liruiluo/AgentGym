from __future__ import annotations

import json
from pathlib import Path
import unittest

from agentenv_agentmemory.literesearcher import (
    FrozenLiteResearchBackend,
    LiteResearcherWrapper,
    load_coverage_manifest,
)


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "literesearcher_stage1_coverage.json"
)


class FakeWorkspace:
    def __init__(self) -> None:
        self.reset_ids: list[str] = []
        self.actions: list[str] = []
        self.closed = False

    def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None:
        assert enabled
        self.reset_ids.append(episode_id)
        self.closed = False

    def apply(self, action: str, *, env_step: int, phase_index: int):
        self.actions.append(action)
        return type(
            "WorkspaceResult",
            (),
            {
                "message": f"workspace step={env_step} phase={phase_index}",
                "op": "SHELL_COMMAND",
            },
        )()

    def close(self) -> None:
        self.closed = True


class LiteResearcherIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coverage = load_coverage_manifest(FIXTURE)

    def test_manifest_is_exact_64_train_with_disjoint_heldout(self) -> None:
        self.assertEqual(self.coverage.task_count, 64)
        self.assertEqual(self.coverage.heldout_count, 8)
        train = {task.index for task in self.coverage.train}
        heldout = {task.index for task in self.coverage.heldout}
        self.assertEqual(train, set(range(64)))
        self.assertEqual(heldout, set(range(64, 72)))
        self.assertTrue(train.isdisjoint(heldout))

    def test_policy_facing_metadata_has_no_gold_or_mask_url(self) -> None:
        backend = FrozenLiteResearchBackend(self.coverage)
        wrapper = LiteResearcherWrapper(self.coverage, backend)
        metadata = wrapper.metadata()
        serialized = json.dumps(metadata, ensure_ascii=False)
        self.assertNotIn("targets", metadata)
        self.assertNotIn("mask_url", metadata)
        self.assertNotIn("targets", metadata["backend"])
        self.assertNotIn("mask_url", metadata["backend"])
        for task in self.coverage.train + self.coverage.heldout:
            self.assertNotIn(task.mask_url, serialized)
        self.assertFalse(metadata["backend"]["search_exposes_mask_url"])
        self.assertFalse(metadata["backend"]["search_exposes_targets"])

    def test_search_uses_opaque_urls_and_visit_is_the_only_evidence_surface(self) -> None:
        backend = FrozenLiteResearchBackend(self.coverage)
        task = self.coverage.train[0]
        results = backend.search(task.question)
        self.assertTrue(results)
        self.assertNotIn(task.mask_url, json.dumps(results))
        self.assertNotIn(task.targets[0], json.dumps(results))
        self.assertTrue(results[0]["url"].startswith("https://literesearcher.local/page/"))

        page = backend.visit(results[0]["url"], goal="answer the question")
        self.assertEqual(page["url"], results[0]["url"])
        self.assertIn(task.targets[0], page["content"])
        self.assertNotIn(task.mask_url, json.dumps(page))

    def test_malformed_tool_and_unknown_visit_do_not_fallback_to_live_web(self) -> None:
        backend = FrozenLiteResearchBackend(self.coverage)
        wrapper = LiteResearcherWrapper(self.coverage, backend)
        env_id = wrapper.create(data_idx=0)["id"]
        malformed = wrapper.step(
            env_id,
            '<tool_call>{"name":"search","arguments":{}}</tool_call>',
        )
        self.assertFalse(malformed["done"])
        self.assertEqual(malformed["info"]["status"], "invalid_action")
        self.assertEqual(malformed["info"]["native_environment_call_count"], 0)

        failed_visit = wrapper.step(
            env_id,
            '<tool_call>{"name":"visit","arguments":{"url":["https://example.invalid/unknown"],"goal":"x"}}</tool_call>',
        )
        self.assertFalse(failed_visit["done"])
        self.assertFalse(failed_visit["info"]["sample_excluded"])
        self.assertEqual(failed_visit["info"]["status"], "invalid_action")
        self.assertEqual(failed_visit["info"]["native_environment_call_count"], 0)
        wrapper.close(env_id)

    def test_gold_wrong_and_tampered_answers_have_terminal_binary_reward(self) -> None:
        backend = FrozenLiteResearchBackend(self.coverage)
        wrapper = LiteResearcherWrapper(self.coverage, backend)
        task = self.coverage.train[0]

        for answer, expected_reward in (
            (task.targets[0], 1.0),
            ("definitely-not-the-source-answer", 0.0),
            (task.targets[0][:-1] + "x", 0.0),
        ):
            env_id = wrapper.create(data_idx=0)["id"]
            result = wrapper.step(env_id, f"<answer>{answer}</answer>")
            self.assertTrue(result["done"])
            self.assertEqual(result["reward"], expected_reward)
            self.assertFalse(result["info"]["sample_excluded"])
            wrapper.close(env_id)

    def test_backend_failure_is_fail_closed_and_sample_excluded(self) -> None:
        task = self.coverage.train[0]
        backend = FrozenLiteResearchBackend(
            self.coverage,
            failing_search_queries={task.question},
        )
        wrapper = LiteResearcherWrapper(self.coverage, backend)
        env_id = wrapper.create(data_idx=0)["id"]
        result = wrapper.step(
            env_id,
            '<tool_call>{"name":"search","arguments":{"query":["'
            + task.question
            + '"]}}</tool_call>',
        )
        self.assertTrue(result["done"])
        self.assertEqual(result["reward"], 0.0)
        self.assertTrue(result["info"]["sample_excluded"])
        self.assertEqual(result["info"]["status"], "environment_error")
        self.assertNotIn(task.mask_url, json.dumps(result))
        wrapper.close(env_id)

    def test_workspace_factory_keeps_episode_workspaces_isolated(self) -> None:
        workspaces: dict[int, FakeWorkspace] = {}

        def factory(env_id: int) -> FakeWorkspace:
            workspace = FakeWorkspace()
            workspaces[env_id] = workspace
            return workspace

        wrapper = LiteResearcherWrapper(
            self.coverage,
            FrozenLiteResearchBackend(self.coverage),
            workspace_factory=factory,
        )
        first = wrapper.create(data_idx=0)
        second = wrapper.create(data_idx=1)
        wrapper.step(first["id"], "shell_command {\"command\":\"pwd\"}")
        wrapper.step(second["id"], "shell_command {\"command\":\"pwd\"}")
        self.assertEqual(len(workspaces[0].actions), 1)
        self.assertEqual(len(workspaces[1].actions), 1)
        self.assertNotEqual(workspaces[0].reset_ids, workspaces[1].reset_ids)
        wrapper.close(first["id"])
        self.assertTrue(workspaces[0].closed)
        self.assertFalse(workspaces[1].closed)
        wrapper.close(second["id"])

    def test_compaction_is_policy_authored_replacement_and_counts_as_step(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.coverage,
            FrozenLiteResearchBackend(self.coverage),
            max_policy_steps=4,
            model_context_tokens=1_024,
            max_response_tokens=128,
            compaction_margin_tokens=64,
        )
        env_id = wrapper.create(data_idx=0)["id"]
        blocked = wrapper.step(
            env_id,
            '<tool_call>{"name":"search","arguments":{"query":["history"]}}</tool_call>',
            context_token_count=900,
        )
        self.assertFalse(blocked["done"])
        self.assertEqual(blocked["info"]["status"], "compaction_required")
        self.assertEqual(blocked["info"]["native_environment_call_count"], 0)

        summary = "Keep the question and continue by searching the opaque local source URL."
        compacted = wrapper.step(
            env_id,
            f"<context_compaction>{summary}</context_compaction>",
            context_token_count=900,
        )
        self.assertFalse(compacted["done"])
        self.assertEqual(compacted["reward"], 0.0)
        self.assertEqual(compacted["info"]["status"], "context_compacted")
        transition = compacted["info"]["context_transition"]
        self.assertEqual(transition["operation"], "replace_messages")
        self.assertEqual(transition["continuity_id"], "stage1:00000")
        self.assertEqual(transition["workspace_path"], ".agent_memory")
        self.assertEqual(transition["messages"][-1]["content"], summary)
        self.assertTrue(transition["policy_authored"])
        self.assertEqual(compacted["info"]["compaction_count"], 1)
        self.assertEqual(compacted["info"]["native_environment_call_count"], 0)
        wrapper.close(env_id)

    def test_private_url_compaction_is_rejected_without_backend_call(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.coverage,
            FrozenLiteResearchBackend(self.coverage),
        )
        env_id = wrapper.create(data_idx=0)["id"]
        task = self.coverage.train[0]
        result = wrapper.step(
            env_id,
            f"<context_compaction>leak {task.mask_url}</context_compaction>",
            context_token_count=32_000,
        )
        self.assertFalse(result["done"])
        self.assertEqual(result["info"]["status"], "invalid_compaction")
        self.assertEqual(result["info"]["native_environment_call_count"], 0)
        wrapper.close(env_id)

    def test_nonterminal_action_at_turn_40_closes_without_a_hidden_41st_step(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.coverage,
            FrozenLiteResearchBackend(self.coverage),
            max_policy_steps=1,
        )
        env_id = wrapper.create(data_idx=0)["id"]
        result = wrapper.step(
            env_id,
            '<tool_call>{"name":"search","arguments":{"query":["history"]}}</tool_call>',
        )
        self.assertTrue(result["done"])
        self.assertEqual(result["info"]["status"], "max_policy_steps_exhausted")
        self.assertEqual(result["info"]["wrapper_evidence"]["max_policy_steps"], 1)
        wrapper.close(env_id)


if __name__ == "__main__":
    unittest.main()
