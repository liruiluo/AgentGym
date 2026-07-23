from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from agentenv_agentmemory.domains.formal_reasoning import (
    FORMAL_JUDGE_PROMPT_TEMPLATE_SHA256,
    FORMAL_REASONING_RUNTIME_IMPORT_RELATIVE_PATHS,
    FORMAL_REASONING_UPSTREAM_RELATIVE_PATHS,
    FormalReasoningFactory,
    _FORMAL_JUDGE_PROMPT_TEMPLATE,
    _judge_provenance,
    _normalize_upstream_judge_config,
    attest_formal_reasoning_upstream,
    load_formal_reasoning_tasks,
)
from agentenv_agentmemory.domains.memoryarena_dataset import (
    attest_injected_test_dataset,
)
from agentenv_agentmemory.runtime.wrapper import DomainEnvWrapper


class RecordingJudge:
    def __init__(self):
        self.calls = []

    def __call__(self, answer, ground_truth):
        self.calls.append((answer, ground_truth))
        passed = answer == ground_truth
        return passed, "yes" if passed else "no"


def write_tasks(path: Path) -> None:
    rows = [
        {
            "id": 10,
            "paper_name": "paper-a",
            "questions": ["Question one?", "Question two?"],
            "answers": ["answer-one", "answer-two"],
            "backgrounds": ["Shared definitions", ""],
        },
        {
            "id": 11,
            "paper_name": "paper-b",
            "questions": ["Only question?"],
            "answers": ["only-answer"],
            "backgrounds": ["Other definitions"],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class FormalReasoningDomainTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tasks_path = Path(self.tempdir.name) / "math.jsonl"
        write_tasks(self.tasks_path)
        self.judge = RecordingJudge()
        self.factory = FormalReasoningFactory(
            domain="math",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="formal_reasoning_math",
            ),
            judge=self.judge,
        )
        self.wrapper = DomainEnvWrapper(self.factory)
        created = self.wrapper.create()
        self.env_id = created["id"]
        self.created = created

    def tearDown(self):
        if self.env_id in self.wrapper.envs:
            self.wrapper.close(self.env_id)
        self.tempdir.cleanup()

    def test_reset_renders_original_math_agent_prompt_without_private_answer(self):
        self.assertIn("### BACKGROUND:\nShared definitions", self.created["observation"])
        self.assertIn("### PROBLEM:\nQuestion one?", self.created["observation"])
        self.assertNotIn("answer-one", self.created["observation"])
        self.assertEqual(self.created["info"]["phase_index"], 0)
        self.assertEqual(self.created["info"]["phase_count"], 2)

    def test_canonical_prompt_has_one_action_envelope_for_plain_answer(self):
        prompt = self.wrapper.metadata()["system_prompt"]
        self.assertIn("- <final answer text>", prompt)
        self.assertNotIn("- Action: <final answer text>", prompt)
        self.assertEqual(prompt.count("Action:\n"), 1)

    def test_plain_correct_answer_is_judged_rewarded_and_advances(self):
        correct = self.wrapper.step(
            self.env_id,
            "Reasoning that is not retained.\n\nAction: answer-one",
        )
        self.assertEqual(self.judge.calls, [("answer-one", "answer-one")])
        self.assertEqual(correct["reward"], 1.0)
        self.assertFalse(correct["done"])
        self.assertEqual(correct["info"]["phase_index"], 1)
        self.assertIn("Question two?", correct["observation"])
        self.assertNotIn("answer-two", correct["observation"])

    def test_incorrect_answer_is_zero_reward_and_terminates_without_advancing(self):
        wrong = self.wrapper.step(
            self.env_id,
            "Reasoning that is not retained.\n\nAction: wrong-answer",
        )
        self.assertEqual(self.judge.calls, [("wrong-answer", "answer-one")])
        self.assertEqual(wrong["reward"], 0.0)
        self.assertTrue(wrong["done"])
        self.assertEqual(wrong["info"]["phase_index"], 0)
        self.assertFalse(wrong["info"]["episode_success"])
        self.assertEqual(wrong["info"]["status"], "failed_on_incorrect_answer")
        self.assertNotIn("answer-one", wrong["observation"])

    def test_empty_answer_is_invalid_without_judging_or_terminating(self):
        for raw_output in ("   ", "Thought: no answer\n\nAction:   "):
            invalid = self.wrapper.step(self.env_id, raw_output)
            self.assertEqual(self.judge.calls, [])
            self.assertEqual(invalid["reward"], 0.0)
            self.assertFalse(invalid["done"])
            self.assertEqual(invalid["info"]["phase_index"], 0)
            self.assertEqual(
                invalid["info"]["action_execution"]["op"],
                "INVALID",
            )
            self.assertEqual(
                invalid["info"]["action_execution"]["status"],
                "invalid",
            )
            self.assertEqual(
                invalid["info"]["reward_components"][0]["name"],
                "invalid_action",
            )
            self.assertNotIn("committed", invalid["info"]["action_execution"])

        correct = self.wrapper.step(self.env_id, "Action: answer-one")
        self.assertEqual(self.judge.calls, [("answer-one", "answer-one")])
        self.assertFalse(correct["done"])
        self.assertEqual(correct["info"]["phase_index"], 1)

    def test_every_correct_subtask_gets_reward_and_full_chain_succeeds(self):
        first = self.wrapper.step(self.env_id, "Action: answer-one")
        final = self.wrapper.step(self.env_id, "Action: answer-two")
        self.assertEqual(first["reward"], 1.0)
        self.assertEqual(final["reward"], 1.0)
        self.assertTrue(final["done"])
        self.assertTrue(final["info"]["episode_success"])
        self.assertEqual(final["info"]["domain_evidence"]["correct_count"], 2)
        self.assertEqual(
            final["info"]["domain_evidence"]["phase_results"],
            [True, True],
        )

    def test_later_wrong_answer_terminates_after_preserving_earlier_positive_step(self):
        first = self.wrapper.step(self.env_id, "Action: answer-one")
        final = self.wrapper.step(self.env_id, "Action: wrong-answer")
        self.assertEqual(first["reward"], 1.0)
        self.assertEqual(final["reward"], 0.0)
        self.assertTrue(final["done"])
        self.assertFalse(final["info"]["episode_success"])
        self.assertEqual(final["info"]["phase_index"], 1)

    def test_phase_advance_clears_visible_trace_but_ltm_remains_retrievable(self):
        self.wrapper.step(
            self.env_id,
            'Action: ADD {"key": "definition", "value": "Shared definitions"}',
        )
        advanced = self.wrapper.step(self.env_id, "Action: answer-one")
        self.assertNotIn("[mem_0000] definition", advanced["observation"])
        retrieved = self.wrapper.step(
            self.env_id,
            'Action: RETRIEVE {"query": "definitions", "top_k": 3}',
        )
        self.assertIn("[mem_0000] definition: Shared definitions", retrieved["observation"])

    def test_judge_failure_is_excluded_without_advancing(self):
        def fail_judge(answer, ground_truth):
            del answer, ground_truth
            raise TimeoutError("judge unavailable")

        factory = FormalReasoningFactory(
            domain="math",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="formal_reasoning_math",
            ),
            judge=fail_judge,
        )
        wrapper = DomainEnvWrapper(factory)
        env_id = wrapper.create()["id"]
        failed = wrapper.step(env_id, "Action: any answer")
        self.assertTrue(failed["done"])
        self.assertTrue(failed["info"]["sample_excluded"])
        self.assertEqual(failed["info"]["phase_index"], 0)
        self.assertEqual(failed["info"]["status"], "infra_error")
        wrapper.close(env_id)

    def test_reset_uses_dataset_position_and_fails_out_of_range(self):
        reset = self.wrapper.reset(self.env_id, 1)
        self.assertIn("Only question?", reset["observation"])
        with self.assertRaisesRegex(IndexError, "outside"):
            self.wrapper.reset(self.env_id, 2)

    def test_math_and_physics_use_separate_surfaces(self):
        physics = FormalReasoningFactory(
            domain="phys",
            tasks_path=self.tasks_path,
            dataset_provenance=attest_injected_test_dataset(
                self.tasks_path,
                config="formal_reasoning_phys",
            ),
            judge=self.judge,
        )
        self.assertEqual(self.factory.domain_id, "formal_reasoning_math")
        self.assertEqual(physics.domain_id, "formal_reasoning_phys")
        self.assertNotEqual(self.factory.surface, physics.surface)

    def test_factory_rejects_dataset_provenance_for_other_formal_surface(self):
        physics_provenance = attest_injected_test_dataset(
            self.tasks_path,
            config="formal_reasoning_phys",
        )
        with self.assertRaisesRegex(RuntimeError, "dataset config mismatch"):
            FormalReasoningFactory(
                domain="math",
                tasks_path=self.tasks_path,
                dataset_provenance=physics_provenance,
                judge=self.judge,
            )

    def test_injected_judge_requires_injected_test_dataset_provenance(self):
        mislabeled = replace(
            attest_injected_test_dataset(
                self.tasks_path,
                config="formal_reasoning_math",
            ),
            mode="frozen_public_hf_dataset",
        )
        with self.assertRaisesRegex(RuntimeError, "injected-test dataset"):
            FormalReasoningFactory(
                domain="math",
                tasks_path=self.tasks_path,
                dataset_provenance=mislabeled,
                judge=self.judge,
            )

    def test_metadata_identifies_explicit_test_dataset_and_judge(self):
        metadata = self.factory.metadata()
        self.assertEqual(
            metadata["dataset_provenance"]["mode"],
            "injected_test_fixture",
        )
        judge = metadata["judge_provenance"]
        self.assertEqual(judge["mode"], "injected_test_double")
        self.assertEqual(
            judge["prompt_template_sha256"],
            FORMAL_JUDGE_PROMPT_TEMPLATE_SHA256,
        )
        self.assertRegex(judge["config_sha256"], r"^[0-9a-f]{64}$")


class FormalReasoningDatasetTest(unittest.TestCase):
    def test_judge_config_rejects_nonfinite_temperature_and_boolean_tokens(self):
        base = {
            "backend": "openai",
            "model_name": "judge-model",
            "base_url": "https://judge.example/v1",
            "temperature": 0.0,
            "max_tokens": 128,
        }
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(temperature=value), self.assertRaisesRegex(
                RuntimeError,
                "temperature must be finite",
            ):
                _normalize_upstream_judge_config({**base, "temperature": value})
        with self.assertRaisesRegex(RuntimeError, "max_tokens must be positive"):
            _normalize_upstream_judge_config({**base, "max_tokens": True})

    def test_factory_has_no_implicit_unattested_fixture_mode(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "fixture.jsonl"
            write_tasks(path)
            with self.assertRaisesRegex(TypeError, "dataset_provenance"):
                FormalReasoningFactory(  # type: ignore[call-arg]
                    domain="math",
                    tasks_path=path,
                    judge=RecordingJudge(),
                )

    def test_public_judge_provenance_hashes_endpoint_without_disclosing_it(self):
        base_url = "https://private-judge.example/v1"
        evidence: dict[str, Any] = _judge_provenance(
            {
                "backend": "openai",
                "model_name": "judge-model",
                "base_url": base_url,
                "temperature": 0.25,
                "max_tokens": 1234,
            },
            mode="upstream_memoryarena_judge",
        )
        self.assertEqual(evidence["backend"], "openai")
        self.assertEqual(evidence["model"], "judge-model")
        self.assertEqual(evidence["temperature"], 0.25)
        self.assertEqual(evidence["max_tokens"], 1234)
        self.assertRegex(evidence["endpoint_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(evidence["prompt_template_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(evidence["config_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn(base_url, json.dumps(evidence))

    def test_prompt_template_matches_pinned_upstream_judge_text(self):
        expected = (
            "\n"
            "            You are a math expert. \n"
            "            Determine if these two expressions are mathematically "
            "equivalent answer for the given question:\n"
            "            Question: None\n"
            "            Expression 1: candidate\n"
            "            Expression 2: reference\n"
            "\n"
            '            Respond only with "yes" or "no". '
        )
        rendered = _FORMAL_JUDGE_PROMPT_TEMPLATE["user"].format(
            query=None,
            action="candidate",
            ground_truth="reference",
        )
        self.assertEqual(rendered, expected)

    def test_rejects_misaligned_phase_arrays(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "bad.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": 0,
                        "paper_name": "bad",
                        "questions": ["q1", "q2"],
                        "answers": ["a1"],
                        "backgrounds": ["b1", "b2"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "misaligned"):
                load_formal_reasoning_tasks(path)

    def test_rejects_empty_questions_or_answers_but_allows_empty_background(self):
        for field in ("questions", "answers"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tempdir:
                path = Path(tempdir) / "empty.jsonl"
                row = {
                    "id": 0,
                    "paper_name": "empty",
                    "questions": ["question"],
                    "answers": ["answer"],
                    "backgrounds": [""],
                }
                row[field][0] = "   "
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, field):
                    load_formal_reasoning_tasks(path)


class FormalReasoningUpstreamAttestationTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for relative_path in FORMAL_REASONING_UPSTREAM_RELATIVE_PATHS:
            path = self.root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# pristine {relative_path}\n", encoding="utf-8")
        self._git("init")
        self._git("config", "user.email", "agentmemory-test@example.invalid")
        self._git("config", "user.name", "AgentMemory Test")
        self._git("add", ".")
        self._git("commit", "-m", "fixture")
        self.commit = self._git("rev-parse", "HEAD").strip()

    def tearDown(self):
        self.tempdir.cleanup()

    def _git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_attests_exact_commit_and_source_bundle(self):
        evidence = attest_formal_reasoning_upstream(
            self.root,
            expected_commit=self.commit,
        )
        self.assertEqual(evidence["memoryarena_commit"], self.commit)
        self.assertEqual(
            set(evidence["runtime_import_entry_files_sha256"]),
            set(FORMAL_REASONING_RUNTIME_IMPORT_RELATIVE_PATHS),
        )
        selected = {
            *evidence["runtime_import_entry_files_sha256"],
            *evidence["reference_entrypoint_files_sha256"],
        }
        self.assertEqual(selected, set(FORMAL_REASONING_UPSTREAM_RELATIVE_PATHS))
        self.assertEqual(
            evidence["pristine_git_scopes"],
            ["env", "agent/math.py", "run_math.py"],
        )
        self.assertRegex(evidence["env_git_tree_oid"], r"^[0-9a-f]{40}$")

    def test_rejects_modified_formal_reasoning_source(self):
        path = self.root / FORMAL_REASONING_UPSTREAM_RELATIVE_PATHS[0]
        path.write_text("# changed semantics\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            attest_formal_reasoning_upstream(
                self.root,
                expected_commit=self.commit,
            )

    def test_allows_unrelated_memoryarena_changes(self):
        unrelated = self.root / "docs/local_note.md"
        unrelated.parent.mkdir(parents=True, exist_ok=True)
        unrelated.write_text("# unrelated\n", encoding="utf-8")
        evidence = attest_formal_reasoning_upstream(
            self.root,
            expected_commit=self.commit,
        )
        self.assertEqual(evidence["memoryarena_commit"], self.commit)

    def test_allows_untracked_non_python_runtime_assets(self):
        asset = self.root / "env/env_systems/web_search_env/index/shard0.index"
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"frozen runtime asset")
        evidence = attest_formal_reasoning_upstream(
            self.root,
            expected_commit=self.commit,
        )
        self.assertEqual(evidence["memoryarena_commit"], self.commit)

    def test_rejects_untracked_transitive_env_source(self):
        transitive = self.root / "env/env_systems/web_shopping_env/local_patch.py"
        transitive.parent.mkdir(parents=True, exist_ok=True)
        transitive.write_text("# can affect package import\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "untracked or ignored Python"):
            attest_formal_reasoning_upstream(
                self.root,
                expected_commit=self.commit,
            )

    def test_rejects_ignored_python_that_can_change_package_imports(self):
        ignored = self.root / "env/env_systems/formal_reasoning_env/__init__.py"
        ignored.parent.mkdir(parents=True, exist_ok=True)
        ignored.write_text("# ignored import hook\n", encoding="utf-8")
        self._git("config", "core.excludesFile", "/dev/null")
        (self.root / ".gitignore").write_text(
            "env/env_systems/formal_reasoning_env/__init__.py\n",
            encoding="utf-8",
        )
        self._git("add", ".gitignore")
        self._git("commit", "-m", "ignore generated package hook")
        commit = self._git("rev-parse", "HEAD").strip()
        with self.assertRaisesRegex(RuntimeError, "ignored Python source"):
            attest_formal_reasoning_upstream(
                self.root,
                expected_commit=commit,
            )


if __name__ == "__main__":
    unittest.main()
