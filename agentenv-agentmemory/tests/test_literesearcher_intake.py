from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agentenv_agentmemory.literesearcher import (
    FrozenLiteResearchBackend,
    LiteResearcherWrapper,
    load_coverage_manifest,
)
from agentenv_agentmemory.literesearcher.wrapper import (
    _filesystem_checkpoint_receipt,
)


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
        self.files: dict[str, bytes] = {}
        self.closed = False

    def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None:
        assert enabled
        self.reset_ids.append(episode_id)
        self.files.clear()
        self.closed = False

    def apply(self, action: str, *, env_step: int, phase_index: int):
        self.actions.append(action)
        checkpoint_path = ".agent_memory/CONTINUATION.md"
        before = self.files.get(checkpoint_path)
        if "> .agent_memory/CONTINUATION.md" in action:
            self.files[checkpoint_path] = b"objective: answer\nnext_action: search source\n"
        after = self.files.get(checkpoint_path)
        added = []
        modified = []
        if before is None and after is not None:
            added.append(self._entry(checkpoint_path, after))
        elif before is not None and after is not None and before != after:
            modified.append(
                {
                    "before": self._entry(checkpoint_path, before),
                    "after": self._entry(checkpoint_path, after),
                }
            )
        stdout = (
            after.decode("utf-8")
            if "cat .agent_memory/CONTINUATION.md" in action and after is not None
            else ""
        )
        if "PARTIAL_CHECKPOINT_READ" in action:
            stdout = stdout[:5]
        exit_code = 7 if "FAIL_AFTER_WRITE" in action else 0
        timed_out = "TIMEOUT_AFTER_WRITE" in action
        message = stdout or f"workspace step={env_step} phase={phase_index}"
        return type(
            "WorkspaceResult",
            (),
            {
                "message": message,
                "op": "SHELL_COMMAND",
                "tool_op": {
                    "status": "executed",
                    "exit_code": exit_code,
                    "timed_out": timed_out,
                    "stdout": stdout,
                    "stdout_truncated": False,
                },
                "workspace_diff": {
                    "added": added,
                    "modified": modified,
                    "deleted": [],
                },
            },
        )()

    @staticmethod
    def _entry(path: str, content: bytes) -> dict:
        return {
            "path": path,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "kind": "file",
        }

    def snapshot(self) -> dict:
        return {
            "files": [
                self._entry(path, content)
                for path, content in sorted(self.files.items())
            ]
        }

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

    def test_wrapper_accepts_qwen35_native_search_and_visit(self) -> None:
        backend = FrozenLiteResearchBackend(self.coverage)
        wrapper = LiteResearcherWrapper(self.coverage, backend)
        task = self.coverage.train[0]
        env_id = wrapper.create(data_idx=0)["id"]
        query = json.dumps([task.question], ensure_ascii=False)
        searched = wrapper.step(
            env_id,
            "<tool_call>\n<function=search>\n<parameter=query>\n"
            + query
            + "\n</parameter>\n</function>\n</tool_call>",
        )
        self.assertFalse(searched["done"])
        self.assertEqual(searched["info"]["status"], "active")
        self.assertEqual(searched["info"]["native_environment_call_count"], 1)

        visited = wrapper.step(
            env_id,
            "<tool_call>\n<function=visit>\n<parameter=url>\n"
            + task.public_url
            + "\n</parameter>\n<parameter=goal>\n"
            + task.question
            + "\n</parameter>\n<parameter=page>\n1\n</parameter>\n"
            + "</function>\n</tool_call>",
        )
        self.assertFalse(visited["done"])
        self.assertEqual(visited["info"]["status"], "active")
        self.assertEqual(visited["info"]["native_environment_call_count"], 2)
        self.assertEqual(json.loads(visited["observation"])["tool"], "visit")
        wrapper.close(env_id)

    def test_native_search_keeps_query_array_strict(self) -> None:
        backend = FrozenLiteResearchBackend(self.coverage)
        wrapper = LiteResearcherWrapper(
            self.coverage,
            backend,
            invalid_action_penalty=-0.01,
        )
        env_id = wrapper.create(data_idx=0)["id"]
        result = wrapper.step(
            env_id,
            "<tool_call>\n<function=search>\n<parameter=query>\n"
            "not-a-json-array\n</parameter>\n</function>\n</tool_call>",
        )
        self.assertFalse(result["done"])
        self.assertEqual(result["reward"], -0.01)
        self.assertEqual(result["info"]["status"], "invalid_action")
        self.assertEqual(result["info"]["native_environment_call_count"], 0)
        recovered = wrapper.step(
            env_id,
            '<tool_call>{"name":"search","arguments":{"query":["history"]}}</tool_call>',
        )
        self.assertFalse(recovered["done"])
        self.assertEqual(recovered["reward"], 0.0)
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
            "policy_filesystem_checkpoint_then_client_replace_v2",
        )
        self.assertEqual(
            wrapper.metadata()["policy_workspace_tool_contract"],
            "qwen3_xml_function_call_v1",
        )
        wrapper.close(env_id)

    def test_qwen_xml_workspace_shell_executes_via_canonical_adapter(self) -> None:
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
        raw_action = """<tool_call>
<function=shell_command>
<parameter=command>
mkdir -p .agent_memory && printf checkpoint > .agent_memory/CONTINUATION.md
</parameter>
<parameter=workdir>
.
</parameter>
<parameter=timeout_ms>
10000
</parameter>
</function>
</tool_call>"""
        result = wrapper.step(env_id, raw_action)
        self.assertEqual(result["info"]["action_submission"]["raw_policy_output"], raw_action)
        self.assertEqual(
            workspaces[env_id].actions,
            [
                'shell_command {"command":"mkdir -p .agent_memory && printf '
                'checkpoint > .agent_memory/CONTINUATION.md",'
                '"timeout_ms":10000,"workdir":"."}'
            ],
        )
        evidence = result["info"]["wrapper_evidence"]
        self.assertEqual(evidence["workspace_policy_format"], "qwen3_xml")
        self.assertTrue(evidence["filesystem_checkpoint"]["action_completed"])
        self.assertTrue(evidence["filesystem_checkpoint"]["changed"])
        wrapper.close(env_id)

    def test_qwen_xml_workspace_call_must_be_the_complete_output(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.coverage,
            FrozenLiteResearchBackend(self.coverage),
            workspace_factory=lambda _env_id: FakeWorkspace(),
            invalid_action_penalty=-0.01,
        )
        env_id = wrapper.create(data_idx=0)["id"]
        result = wrapper.step(
            env_id,
            "prefix<tool_call><function=shell_command>"
            "<parameter=command>pwd</parameter>"
            "</function></tool_call>",
        )
        self.assertEqual(result["reward"], -0.01)
        self.assertEqual(result["info"]["status"], "invalid_action")
        wrapper.close(env_id)

    def test_workspace_checkpoint_receipt_attests_write_then_read(self) -> None:
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
        self.assertTrue(wrapper.metadata()["compaction_calls_backend"])
        env_id = wrapper.create(data_idx=0)["id"]
        action = (
            'shell_command {"command":"mkdir -p .agent_memory && printf checkpoint '
            '> .agent_memory/CONTINUATION.md","workdir":"."}'
        )
        written = wrapper.step(env_id, action)
        receipt = written["info"]["wrapper_evidence"]["filesystem_checkpoint"]
        self.assertEqual(receipt["schema"], "agentmemory_filesystem_checkpoint_receipt_v1")
        self.assertEqual(receipt["path"], ".agent_memory/CONTINUATION.md")
        self.assertTrue(receipt["action_completed"])
        self.assertTrue(receipt["changed"])
        self.assertGreater(receipt["size_bytes"], 0)

        read = wrapper.step(
            env_id,
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md",'
            '"workdir":"."}',
        )
        self.assertIn("objective: answer", read["observation"])
        read_receipt = read["info"]["wrapper_evidence"]["filesystem_checkpoint"]
        self.assertFalse(read_receipt["changed"])
        self.assertTrue(read_receipt["exists"])
        exact_read = read["info"]["wrapper_evidence"]["filesystem_checkpoint_read"]
        self.assertTrue(exact_read["observed"])
        self.assertEqual(exact_read["sha256"], receipt["sha256"])
        wrapper.close(env_id)

    def test_failed_or_timed_out_shell_write_does_not_complete_checkpoint(self) -> None:
        for marker in ("FAIL_AFTER_WRITE", "TIMEOUT_AFTER_WRITE"):
            with self.subTest(marker=marker):
                wrapper = LiteResearcherWrapper(
                    self.coverage,
                    FrozenLiteResearchBackend(self.coverage),
                    workspace_factory=lambda _env_id: FakeWorkspace(),
                )
                env_id = wrapper.create(data_idx=0)["id"]
                result = wrapper.step(
                    env_id,
                    'shell_command {"command":"mkdir -p .agent_memory && printf '
                    "checkpoint > .agent_memory/CONTINUATION.md; "
                    + marker
                    + '","workdir":"."}'
                )
                receipt = result["info"]["wrapper_evidence"][
                    "filesystem_checkpoint"
                ]
                self.assertTrue(receipt["changed"])
                self.assertFalse(receipt["action_completed"])
                wrapper.close(env_id)

    def test_partial_checkpoint_stdout_is_not_attested_as_a_read(self) -> None:
        wrapper = LiteResearcherWrapper(
            self.coverage,
            FrozenLiteResearchBackend(self.coverage),
            workspace_factory=lambda _env_id: FakeWorkspace(),
        )
        env_id = wrapper.create(data_idx=0)["id"]
        wrapper.step(
            env_id,
            'shell_command {"command":"mkdir -p .agent_memory && printf checkpoint '
            '> .agent_memory/CONTINUATION.md","workdir":"."}'
        )
        read = wrapper.step(
            env_id,
            'shell_command {"command":"cat .agent_memory/CONTINUATION.md; '
            'PARTIAL_CHECKPOINT_READ","workdir":"."}'
        )
        receipt = read["info"]["wrapper_evidence"]["filesystem_checkpoint_read"]
        self.assertFalse(receipt["observed"])
        wrapper.close(env_id)

    def test_checkpoint_receipt_rejects_metadata_only_or_stale_snapshot(self) -> None:
        path = ".agent_memory/CONTINUATION.md"
        before = {"path": path, "bytes": 4, "sha256": "a" * 64, "kind": "file"}
        after = {"path": path, "bytes": 4, "sha256": "b" * 64, "kind": "file"}
        cases = (
            (
                "metadata-only",
                {"added": [], "modified": [{"before": before, "after": before}], "deleted": []},
                {"files": [before]},
            ),
            (
                "stale-snapshot",
                {"added": [], "modified": [{"before": before, "after": after}], "deleted": []},
                {"files": [dict(after, sha256="c" * 64)]},
            ),
            (
                "missing-kind",
                {"added": [after], "modified": [], "deleted": []},
                {"files": [{key: value for key, value in after.items() if key != "kind"}]},
            ),
        )
        for label, diff, snapshot in cases:
            with self.subTest(case=label):
                result = SimpleNamespace(
                    op="SHELL_COMMAND",
                    tool_op={
                        "status": "executed",
                        "exit_code": 0,
                        "timed_out": False,
                    },
                    workspace_diff=diff,
                )
                receipt = _filesystem_checkpoint_receipt(
                    result=result,
                    workspace_snapshot=snapshot,
                )
                self.assertFalse(receipt["changed"])

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
