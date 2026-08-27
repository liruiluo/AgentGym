from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from agentenv_agentmemory.literesearcher import (
    FrozenLiteResearchBackend,
    LiteResearcherWrapper,
    load_coverage_manifest,
)
from agentenv_agentmemory.persistent_workspace import parse_workspace_action
from agentenv_agentmemory.workspace_patch import parse_workspace_patch


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "literesearcher_stage1_coverage.json"
)
SEMANTIC_AUDIT = FIXTURE.with_name("literesearcher_stage1_semantic_audit.json")


class FakeWorkspace:
    def __init__(self) -> None:
        self.reset_ids: list[str] = []
        self.actions: list[str] = []
        self.closed = False
        self.next_workspace_diff = {
            "added": [],
            "modified": [],
            "deleted": [],
            "directories_added": [],
            "directories_deleted": [],
        }
        self.next_tool_op = None

    def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None:
        assert enabled
        self.reset_ids.append(episode_id)
        self.closed = False

    def apply(self, action: str, *, env_step: int, phase_index: int):
        parsed = parse_workspace_action(action)
        if parsed is None:
            raise ValueError("expected a canonical workspace action")
        if parsed.tool_name == "apply_patch":
            parse_workspace_patch(parsed.tool_input)
        self.actions.append(action)
        tool_op = self.next_tool_op
        if tool_op is None:
            tool_op = (
                {"status": "executed", "exit_code": 0, "timed_out": False}
                if parsed.tool_name == "shell_command"
                else {"status": "executed", "transactional": True}
            )
        return type(
            "WorkspaceResult",
            (),
            {
                "message": f"workspace step={env_step} phase={phase_index}",
                "op": parsed.tool_name.upper(),
                "tool_op": tool_op,
                "workspace_diff": self.next_workspace_diff,
            },
        )()

    def close(self) -> None:
        self.closed = True


class LiteResearcherIntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.coverage = load_coverage_manifest(FIXTURE)

    def test_manifest_is_exact_64_train_with_disjoint_heldout(self) -> None:
        self.assertEqual(self.coverage.task_count, 64)
        self.assertEqual(self.coverage.heldout_count, 8)
        train = {task.index for task in self.coverage.train}
        heldout = {task.index for task in self.coverage.heldout}
        self.assertEqual(len(train), 64)
        self.assertEqual(len(heldout), 8)
        self.assertTrue(train.isdisjoint(heldout))
        self.assertEqual(sorted(train), [task.index for task in self.coverage.train])
        self.assertEqual(sorted(heldout), [task.index for task in self.coverage.heldout])
        expected_heldout = {5400, 5464, 5500, 5761, 5806, 5857, 6754, 6918}
        self.assertEqual(heldout, expected_heldout)
        self.assertFalse({5815, 5905, 5909} & heldout)
        with self.assertRaises(ValueError):
            self.coverage.task(-1)

    def test_wrapper_selects_train_or_heldout_without_exposing_gold(self) -> None:
        train_backend = FrozenLiteResearchBackend(self.coverage, split="train")
        heldout_backend = FrozenLiteResearchBackend(self.coverage, split="test")
        train = LiteResearcherWrapper(self.coverage, train_backend, split="train")
        heldout = LiteResearcherWrapper(
            self.coverage, heldout_backend, split="test"
        )
        self.assertEqual(train.metadata()["task_count"], 64)
        self.assertEqual(heldout.metadata()["task_count"], 8)
        self.assertEqual(heldout.metadata()["split"], "test")
        created = heldout.create(data_idx=0)
        self.assertEqual(created["observation"], self.coverage.heldout[0].question)
        self.assertEqual(created["info"]["data_idx"], 0)
        self.assertEqual(
            created["info"]["source_data_idx"], self.coverage.heldout[0].index
        )
        self.assertNotIn(
            self.coverage.heldout[0].targets[0],
            json.dumps(heldout.metadata(), ensure_ascii=False),
        )
        heldout.close(created["id"])

        with self.assertRaisesRegex(ValueError, "same LiteResearcher split"):
            LiteResearcherWrapper(
                self.coverage, train_backend, split="test"
            )

    def test_search_and_visit_are_strictly_split_local(self) -> None:
        train_backend = FrozenLiteResearchBackend(self.coverage, split="train")
        heldout_backend = FrozenLiteResearchBackend(self.coverage, split="test")
        train_urls = {task.public_url for task in self.coverage.train}
        heldout_urls = {task.public_url for task in self.coverage.heldout}

        train_results = train_backend.search(
            [task.question for task in self.coverage.heldout]
        )
        heldout_results = heldout_backend.search(
            [task.question for task in self.coverage.train]
        )
        self.assertTrue(
            {item["url"] for item in train_results}.issubset(train_urls)
        )
        self.assertTrue(
            {item["url"] for item in heldout_results}.issubset(heldout_urls)
        )
        self.assertTrue(
            {item["url"] for item in train_results}.isdisjoint(heldout_urls)
        )
        self.assertTrue(
            {item["url"] for item in heldout_results}.isdisjoint(train_urls)
        )
        with self.assertRaisesRegex(ValueError, "outside the frozen corpus"):
            train_backend.visit(self.coverage.heldout[0].public_url)
        with self.assertRaisesRegex(ValueError, "outside the frozen corpus"):
            heldout_backend.visit(self.coverage.train[0].public_url)

    def test_all_rows_self_search_top1_and_split_local_data_idx(self) -> None:
        for split, tasks in (
            ("train", self.coverage.train),
            ("test", self.coverage.heldout),
        ):
            backend = FrozenLiteResearchBackend(self.coverage, split=split)
            wrapper = LiteResearcherWrapper(self.coverage, backend, split=split)
            other_split = "test" if split == "train" else "train"
            other_urls = {
                task.public_url for task in self.coverage.tasks_for_split(other_split)
            }
            for data_idx, task in enumerate(tasks):
                with self.subTest(split=split, data_idx=data_idx, index=task.index):
                    created = wrapper.create(data_idx=data_idx)
                    self.assertEqual(created["info"]["data_idx"], data_idx)
                    self.assertEqual(created["info"]["source_data_idx"], task.index)
                    wrapper.close(created["id"])

                    hits = backend.search(task.question, top_k=5)
                    urls = [hit["url"] for hit in hits]
                    self.assertEqual(urls[0], task.public_url)
                    self.assertTrue(set(urls).isdisjoint(other_urls))

    def test_manifest_pages_are_source_backed_and_content_addressed(self) -> None:
        self.assertEqual(
            self.manifest["schema"], "agentmemory_literesearcher_coverage_v3"
        )
        self.assertEqual(
            self.manifest["page_fixture_contract"],
            "source_backed_semantically_reviewed_frozen_text_v1",
        )
        self.assertEqual(self.manifest["semantic_audit"]["approved_count"], 72)
        self.assertEqual(self.manifest["semantic_audit"]["rejected_count"], 21)
        self.assertEqual(self.manifest["semantic_audit"]["replacement_count"], 21)
        self.assertEqual(self.manifest["semantic_audit"]["source_backed_ratio"], 1.0)
        for task in self.coverage.train + self.coverage.heldout:
            self.assertEqual(
                task.content_sha256,
                hashlib.sha256(task.page_text.encode("utf-8")).hexdigest(),
            )
            self.assertTrue(task.resolved_url.startswith("https://en.wikipedia.org/wiki/"))
            self.assertEqual(task.extraction_method, "jina_reader_wikipedia_plaintext_v1")
            self.assertTrue(task.license_note)
            self.assertTrue(task.evidence_anchors)
            self.assertTrue(set(task.evidence_anchors).issubset(set(task.targets)))
            self.assertNotIn("Frozen source excerpt", task.page_text)
            self.assertNotIn(task.mask_url, task.page_text)
            self.assertNotIn(task.resolved_url, task.page_text)

    def test_semantic_audit_is_bound_and_rejects_tampering(self) -> None:
        audit = json.loads(SEMANTIC_AUDIT.read_text(encoding="utf-8"))
        self.assertEqual(
            audit["schema"], "agentmemory_literesearcher_semantic_audit_v1"
        )
        self.assertEqual(audit["summary"]["source_backed_count"], 72)
        self.assertEqual(audit["summary"]["source_backed_ratio"], 1.0)
        self.assertEqual(len(audit["approved"]), 72)
        self.assertEqual(len(audit["rejected"]), 21)
        self.assertEqual(
            {item["index"] for item in audit["rejected"]},
            {
                66,
                353,
                362,
                411,
                584,
                875,
                899,
                902,
                989,
                1489,
                1780,
                1878,
                2191,
                2315,
                2705,
                2911,
                3166,
                3838,
                3859,
                3874,
                4558,
            },
        )

        audit["approved"][0]["evidence_quote"] += " tampered"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / FIXTURE.name
            audit_path = root / SEMANTIC_AUDIT.name
            manifest_path.write_text(
                FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
            )
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantic audit SHA256"):
                load_coverage_manifest(manifest_path)

    def test_loader_rejects_placeholder_tamper_and_missing_provenance(self) -> None:
        cases = []

        placeholder = json.loads(json.dumps(self.manifest))
        placeholder_task = placeholder["train"][0]
        placeholder_task["page_text"] = (
            "Frozen source excerpt. The source evidence supports the requested answer: "
            + placeholder_task["targets"][0]
        )
        placeholder_task["content_sha256"] = hashlib.sha256(
            placeholder_task["page_text"].encode("utf-8")
        ).hexdigest()
        cases.append(placeholder)

        tampered = json.loads(json.dumps(self.manifest))
        tampered["train"][0]["page_text"] += " tampered"
        cases.append(tampered)

        missing = json.loads(json.dumps(self.manifest))
        del missing["train"][0]["resolved_url"]
        cases.append(missing)

        for case_index, payload in enumerate(cases):
            with self.subTest(case=case_index):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "coverage.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_coverage_manifest(path)

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

        page = backend.visit(results[0]["url"], goal=task.question)
        self.assertEqual(page["url"], results[0]["url"])
        pages = [page]
        for page_number in range(2, page["page_count"] + 1):
            pages.append(
                backend.visit(results[0]["url"], goal=task.question, page=page_number)
            )
        self.assertTrue(any(task.targets[0] in item["content"] for item in pages))
        self.assertNotIn(task.mask_url, json.dumps(pages))

    def test_visit_pages_bound_observations_and_keep_reviewed_evidence_reachable(self) -> None:
        audit = json.loads(SEMANTIC_AUDIT.read_text(encoding="utf-8"))
        evidence_by_index = {
            int(record["index"]): str(record["evidence_quote"])
            for record in audit["approved"]
        }
        for split, tasks in (
            ("train", self.coverage.train),
            ("test", self.coverage.heldout),
        ):
            backend = FrozenLiteResearchBackend(self.coverage, split=split)
            for task in tasks:
                with self.subTest(split=split, index=task.index):
                    first = backend.visit(task.public_url, goal=task.question, page=1)
                    self.assertGreaterEqual(first["page_count"], 1)
                    self.assertEqual(first["page"], 1)
                    self.assertLessEqual(len(first["content"]), 8192)
                    self.assertEqual(
                        first["next_page"],
                        2 if first["page_count"] > 1 else None,
                    )

                    pages = [first]
                    for page_number in range(2, first["page_count"] + 1):
                        pages.append(
                            backend.visit(
                                task.public_url,
                                goal=task.question,
                                page=page_number,
                            )
                        )
                    self.assertEqual(
                        [item["page"] for item in pages],
                        list(range(1, first["page_count"] + 1)),
                    )
                    self.assertTrue(
                        any(
                            evidence_by_index[task.index] in item["content"]
                            for item in pages
                        )
                    )
                    serialized = json.dumps(pages, ensure_ascii=False)
                    self.assertNotIn(task.mask_url, serialized)
                    self.assertNotIn(task.resolved_url, serialized)

                    with self.assertRaisesRegex(ValueError, "visit page"):
                        backend.visit(
                            task.public_url,
                            goal=task.question,
                            page=first["page_count"] + 1,
                        )

    def test_wrapper_visit_returns_one_bounded_page(self) -> None:
        backend = FrozenLiteResearchBackend(self.coverage)
        wrapper = LiteResearcherWrapper(self.coverage, backend)
        task = self.coverage.train[0]
        env_id = wrapper.create(data_idx=0)["id"]
        result = wrapper.step(
            env_id,
            '<tool_call>{"name":"visit","arguments":{"url":"'
            + task.public_url
            + '","goal":"'
            + task.question.replace('"', '\\"')
            + '","page":1}}</tool_call>',
        )
        observation = json.loads(result["observation"])
        self.assertFalse(result["done"])
        self.assertEqual(result["info"]["status"], "active")
        self.assertEqual(result["info"]["native_environment_call_count"], 1)
        self.assertEqual(observation["tool"], "visit")
        self.assertEqual(observation["page"]["page"], 1)
        self.assertLessEqual(len(observation["page"]["content"]), 8192)
        wrapper.close(env_id)

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
            '<tool_call>{"name":"visit","arguments":{"url":"https://example.invalid/unknown","goal":"x","page":1}}</tool_call>',
        )
        self.assertFalse(failed_visit["done"])
        self.assertFalse(failed_visit["info"]["sample_excluded"])
        self.assertEqual(failed_visit["info"]["status"], "invalid_action")
        self.assertEqual(failed_visit["info"]["native_environment_call_count"], 0)

        batched_visit = wrapper.step(
            env_id,
            '<tool_call>{"name":"visit","arguments":{"url":["https://literesearcher.local/page/one","https://literesearcher.local/page/two"],"goal":"x","page":1}}</tool_call>',
        )
        self.assertFalse(batched_visit["done"])
        self.assertEqual(batched_visit["info"]["status"], "invalid_action")
        self.assertEqual(batched_visit["info"]["native_environment_call_count"], 0)
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

    def test_all_rows_gold_wrong_and_tampered_rewards(self) -> None:
        for split, tasks in (
            ("train", self.coverage.train),
            ("test", self.coverage.heldout),
        ):
            backend = FrozenLiteResearchBackend(self.coverage, split=split)
            wrapper = LiteResearcherWrapper(self.coverage, backend, split=split)
            for data_idx, task in enumerate(tasks):
                wrong_answers = (
                    ("gold", task.targets[0], 1.0),
                    ("wrong", "definitely-not-the-source-answer", 0.0),
                    (
                        "tampered",
                        task.targets[0][:-1] + "x"
                        if len(task.targets[0]) > 1
                        else task.targets[0] + "x",
                        0.0,
                    ),
                )
                for kind, answer, expected_reward in wrong_answers:
                    with self.subTest(
                        split=split,
                        data_idx=data_idx,
                        index=task.index,
                        kind=kind,
                    ):
                        env_id = wrapper.create(data_idx=data_idx)["id"]
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
        self.assertEqual(wrapper.metadata()["active_environment_count"], 2)
        self.assertEqual(wrapper.metadata()["active_workspace_count"], 2)
        wrapper.step(first["id"], "shell_command {\"command\":\"pwd\"}")
        wrapper.step(second["id"], "shell_command {\"command\":\"pwd\"}")
        self.assertEqual(len(workspaces[0].actions), 1)
        self.assertEqual(len(workspaces[1].actions), 1)
        self.assertNotEqual(workspaces[0].reset_ids, workspaces[1].reset_ids)
        wrapper.close(first["id"])
        self.assertTrue(workspaces[0].closed)
        self.assertFalse(workspaces[1].closed)
        self.assertEqual(wrapper.metadata()["active_environment_count"], 1)
        self.assertEqual(wrapper.metadata()["active_workspace_count"], 1)
        wrapper.close(second["id"])
        self.assertEqual(wrapper.metadata()["active_environment_count"], 0)
        self.assertEqual(wrapper.metadata()["active_workspace_count"], 0)

    def test_workspace_accepts_bounded_visible_prefix_before_one_action(self) -> None:
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
        env_id = wrapper.create(data_idx=0)["id"]
        patch_prefix = "I found useful evidence and will preserve it.\n\n"
        executed_patch = (
            "apply_patch\n*** Begin Patch\n"
            "*** Add File: .agent_memory/research.md\n"
            "+evidence\n*** End Patch"
        )
        patch_action = patch_prefix + executed_patch
        patch_result = wrapper.step(env_id, patch_action)
        self.assertFalse(patch_result["done"])
        self.assertEqual(patch_result["info"]["status"], "active")
        self.assertEqual(workspaces[0].actions[-1], executed_patch)
        self.assertEqual(
            patch_result["info"]["action_submission"]["raw_policy_output"],
            patch_action,
        )
        self.assertEqual(
            patch_result["info"]["action_submission"]["reasoning_prefix_chars"],
            len(patch_prefix),
        )
        self.assertEqual(
            patch_result["info"]["action_submission"]
            ["executed_workspace_action_sha256"],
            hashlib.sha256(executed_patch.encode("utf-8")).hexdigest(),
        )

        shell_prefix = "I will read the saved evidence before answering.\n\n"
        executed_shell = (
            'shell_command {"command":"cat .agent_memory/research.md",'
            '"workdir":".","timeout_ms":10000}'
        )
        shell_result = wrapper.step(env_id, shell_prefix + executed_shell)
        self.assertFalse(shell_result["done"])
        self.assertEqual(shell_result["info"]["status"], "active")
        self.assertEqual(workspaces[0].actions[-1], executed_shell)
        self.assertEqual(
            shell_result["info"]["action_submission"]["reasoning_prefix_chars"],
            len(shell_prefix),
        )
        wrapper.close(env_id)

    def test_workspace_zero_prefix_actions_keep_legacy_execution_path(self) -> None:
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
        env_id = wrapper.create(data_idx=0)["id"]
        actions = (
            'shell_command {"command":"pwd"}',
            "apply_patch\n*** Begin Patch\n*** Add File: note.md\n+x\n*** End Patch",
            "apply_patch\n*** Begin Patch\n*** Update File: note.md\n@@\n"
            "-old\n+new\n apply_patch\n"
            ' shell_command {"command":"pwd"}\n*** End Patch',
        )
        for action in actions:
            result = wrapper.step(env_id, action)
            self.assertFalse(result["done"])
            self.assertEqual(result["info"]["status"], "active")
            self.assertEqual(workspaces[0].actions[-1], action)
            self.assertEqual(
                result["info"]["action_submission"]["reasoning_prefix_chars"],
                0,
            )
        wrapper.close(env_id)

    def test_workspace_prefix_boundary_is_measured_on_raw_prefix(self) -> None:
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
        accepted_prefix = "x" * 2047 + "\n"
        env_id = wrapper.create(data_idx=0)["id"]
        accepted = wrapper.step(
            env_id, accepted_prefix + 'shell_command {"command":"pwd"}'
        )
        self.assertEqual(accepted["info"]["status"], "active")
        self.assertEqual(
            accepted["info"]["action_submission"]["reasoning_prefix_chars"],
            2048,
        )
        wrapper.close(env_id)

        for rejected_prefix in ("x" * 2048 + "\n", " " * 2049 + "\n"):
            env_id = wrapper.create(data_idx=0)["id"]
            result = wrapper.step(
                env_id, rejected_prefix + 'shell_command {"command":"pwd"}'
            )
            self.assertEqual(result["info"]["status"], "invalid_action")
            self.assertEqual(workspaces[env_id].actions, [])
            wrapper.close(env_id)

    def test_workspace_prefix_parser_rejects_ambiguous_rows(self) -> None:
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
        rejected = (
            "reason\nshell_command {\"command\":\"pwd\"}\n"
            'shell_command {"command":"pwd"}',
            "reason\napply_patch\n*** Begin Patch\n*** Add File: x\n+x\n"
            "*** End Patch\napply_patch\n*** Begin Patch\n*** Add File: y\n"
            "+y\n*** End Patch",
            "<tool_call>{\"name\":\"search\",\"arguments\":"
            "{\"query\":[\"history\"]}}</tool_call>\n"
            "apply_patch\n*** Begin Patch\n*** Add File: x\n+x\n*** End Patch",
            "<answer>answer</answer>\n"
            "apply_patch\n*** Begin Patch\n*** Add File: x\n+x\n*** End Patch",
            "```text\napply_patch\n*** Begin Patch\n"
            "*** Add File: x\n+x\n*** End Patch",
            'reason\nshell_command {"command":"pwd"} trailing-junk',
            "reason\napply_patch\n*** Begin Patch\n*** Add File: x\n+x\n"
            "*** End Patch\ntrailing-junk",
            "shell_command not-json",
            "apply_patch\nnot-a-patch",
        )
        for action in rejected:
            with self.subTest(action=action[:80]):
                env_id = wrapper.create(data_idx=0)["id"]
                result = wrapper.step(env_id, action)
                self.assertFalse(result["done"])
                self.assertEqual(result["info"]["status"], "invalid_action")
                self.assertEqual(workspaces[env_id].actions, [])
                wrapper.close(env_id)

    def test_server_rejects_compaction_rows_because_client_owns_them(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.coverage,
            FrozenLiteResearchBackend(self.coverage),
        )
        env_id = wrapper.create(data_idx=0)["id"]
        result = wrapper.step(
            env_id,
            "<context_compaction>continue from notes.md</context_compaction>",
        )
        self.assertFalse(result["done"])
        self.assertEqual(result["info"]["status"], "invalid_action")
        self.assertEqual(result["info"]["native_environment_call_count"], 0)
        self.assertEqual(
            wrapper.metadata()["compaction_contract"],
            "task_neutral_filesystem_checkpoint_v2",
        )
        wrapper.close(env_id)

    def test_metadata_advertises_bounded_filesystem_checkpoint(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.coverage,
            FrozenLiteResearchBackend(self.coverage),
        )
        metadata = wrapper.metadata()
        self.assertEqual(
            metadata["compaction_contract"],
            "task_neutral_filesystem_checkpoint_v2",
        )
        self.assertEqual(
            metadata["continuation_checkpoint_path"],
            ".agent_memory/CONTINUATION.md",
        )
        self.assertEqual(metadata["continuation_checkpoint_max_bytes"], 8192)
        self.assertEqual(
            metadata["continuation_checkpoint_receipt_schema"],
            "agentmemory_continuation_checkpoint_v2",
        )
        self.assertEqual(metadata["workspace_memory_reward"], 0.0)
        self.assertTrue(metadata["compaction_calls_endpoint_step"])
        self.assertFalse(metadata["compaction_calls_research_backend"])

    def test_workspace_write_emits_valid_continuation_checkpoint_receipt(self) -> None:
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
        env_id = wrapper.create(data_idx=0)["id"]
        workspaces[env_id].next_workspace_diff = {
            "added": [
                {
                    "path": ".agent_memory/CONTINUATION.md",
                    "bytes": 127,
                    "sha256": "a" * 64,
                }
            ],
            "modified": [],
            "deleted": [],
            "directories_added": [".agent_memory"],
            "directories_deleted": [],
        }
        action = (
            "apply_patch\n*** Begin Patch\n*** Add File: "
            ".agent_memory/CONTINUATION.md\n+state\n*** End Patch"
        )
        result = wrapper.step(env_id, action)
        receipt = result["info"]["wrapper_evidence"][
            "continuation_checkpoint"
        ]
        self.assertEqual(
            receipt,
            {
                "schema": "agentmemory_continuation_checkpoint_v2",
                "path": ".agent_memory/CONTINUATION.md",
                "action_kind": "APPLY_PATCH",
                "action_execution_succeeded": True,
                "change_kind": "added",
                "before_sha256": None,
                "sha256": "a" * 64,
                "content_changed": True,
                "changed_in_action": True,
                "nonempty": True,
                "within_size_limit": True,
                "bytes": 127,
                "valid": True,
                "rejection_reason": None,
            },
        )
        wrapper.close(env_id)

    def test_failed_shell_with_partial_checkpoint_write_is_not_valid(self) -> None:
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
        env_id = wrapper.create(data_idx=0)["id"]
        workspaces[env_id].next_workspace_diff = {
            "added": [
                {
                    "path": ".agent_memory/CONTINUATION.md",
                    "bytes": 64,
                    "sha256": "e" * 64,
                }
            ],
            "modified": [],
            "deleted": [],
            "directories_added": [".agent_memory"],
            "directories_deleted": [],
        }
        workspaces[env_id].next_tool_op = {
            "status": "executed",
            "exit_code": 7,
            "timed_out": False,
        }
        result = wrapper.step(
            env_id,
            'shell_command {"command":"write-then-fail"}',
        )
        receipt = result["info"]["wrapper_evidence"][
            "continuation_checkpoint"
        ]
        self.assertFalse(receipt["valid"])
        self.assertFalse(receipt["action_execution_succeeded"])
        self.assertTrue(receipt["changed_in_action"])
        self.assertEqual(receipt["rejection_reason"], "action_execution_failed")
        wrapper.close(env_id)

    def test_continuation_checkpoint_receipt_rejects_missing_unchanged_or_oversized(self) -> None:
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
        cases = (
            (
                {
                    "added": [],
                    "modified": [],
                    "deleted": [],
                    "directories_added": [],
                    "directories_deleted": [],
                },
                "not_changed_in_action",
            ),
            (
                {
                    "added": [
                        {
                            "path": ".agent_memory/CONTINUATION.md",
                            "bytes": 0,
                            "sha256": "b" * 64,
                        }
                    ],
                    "modified": [],
                    "deleted": [],
                    "directories_added": [],
                    "directories_deleted": [],
                },
                "empty",
            ),
            (
                {
                    "added": [],
                    "modified": [
                        {
                            "before": {
                                "path": ".agent_memory/CONTINUATION.md",
                                "bytes": 100,
                                "sha256": "c" * 64,
                            },
                            "after": {
                                "path": ".agent_memory/CONTINUATION.md",
                                "bytes": 100,
                                "sha256": "c" * 64,
                            },
                        }
                    ],
                    "deleted": [],
                    "directories_added": [],
                    "directories_deleted": [],
                },
                "not_changed_in_action",
            ),
            (
                {
                    "added": [],
                    "modified": [
                        {
                            "before": {
                                "path": ".agent_memory/CONTINUATION.md",
                                "bytes": 100,
                                "sha256": "c" * 64,
                            },
                            "after": {
                                "path": ".agent_memory/CONTINUATION.md",
                                "bytes": 8193,
                                "sha256": "d" * 64,
                            },
                        }
                    ],
                    "deleted": [],
                    "directories_added": [],
                    "directories_deleted": [],
                },
                "too_large",
            ),
        )
        action = 'shell_command {"command":"true"}'
        for workspace_diff, reason in cases:
            with self.subTest(reason=reason):
                env_id = wrapper.create(data_idx=0)["id"]
                workspaces[env_id].next_workspace_diff = workspace_diff
                result = wrapper.step(env_id, action)
                receipt = result["info"]["wrapper_evidence"][
                    "continuation_checkpoint"
                ]
                self.assertFalse(receipt["valid"])
                self.assertEqual(receipt["rejection_reason"], reason)
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
