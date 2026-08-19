from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agentenv_agentmemory.domains import (
    BROWSECOMP_BM25_INTEGRATION_SURFACE,
    BROWSECOMP_SURFACES,
    FORMAL_REASONING_SURFACES_BY_MODE,
    TRAVEL_SURFACES,
)
from agentenv_agentmemory.env_wrapper import NATIVE_SURFACE
from agentenv_agentmemory.filesystem_webshop_env import PROCEDURAL_FILESYSTEM_SURFACE
from agentenv_agentmemory.compositional_recall_webshop_env import (
    COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
    COMPOSITIONAL_RECALL_SURFACE,
)
from agentenv_agentmemory.distractor_robustness_webshop_env import (
    DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
    DISTRACTOR_ROBUSTNESS_SURFACE,
)
from agentenv_agentmemory.intent_clarification_webshop_env import (
    INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
    INTENT_CLARIFICATION_SURFACE,
)
from agentenv_agentmemory.latent_preference_webshop_env import (
    LATENT_PREFERENCE_FILESYSTEM_SURFACE,
    LATENT_PREFERENCE_SURFACE,
)
from agentenv_agentmemory.literesearcher import (
    LITERESEARCHER_FULLPOOL_SURFACE,
    LITERESEARCHER_SURFACE,
)
from agentenv_agentmemory.negative_constraint_webshop_env import (
    NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
)
from agentenv_agentmemory.recency_override_webshop_env import (
    RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
)
from agentenv_agentmemory.procedural_webshop_env import PROCEDURAL_SURFACE
from agentenv_agentmemory.selective_memory_use_webshop_env import (
    SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
    SELECTIVE_MEMORY_USE_SURFACE,
)
from agentenv_agentmemory.launch import launch
from agentenv_agentmemory.reward_hierarchy import (
    FIRST_VALID_ADD_BONUS,
    FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS,
)


class DomainLaunchTest(unittest.TestCase):
    def _launch(self, arguments):
        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = Mock()
        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn}),
            patch.object(
                sys,
                "argv",
                ["agentmemory", *arguments],
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            launch()
            configured = dict(os.environ)
        return configured, uvicorn.run

    @staticmethod
    def _literesearcher_arguments(*, split="train"):
        return [
            "--surface",
            LITERESEARCHER_SURFACE,
            "--run-id",
            f"literesearcher-{split}-test",
            "--split",
            split,
            "--literesearcher-coverage-manifest",
            "/data/literesearcher-stage1.json",
            "--literesearcher-max-policy-steps",
            "40",
            "--literesearcher-top-k",
            "5",
            "--workspace-rg-binary",
            "/tools/rg",
            "--workspace-rg-sha256",
            "a" * 64,
            "--workspace-root-parent",
            "/runtime/literesearcher-workspaces",
        ]

    def test_literesearcher_launch_binds_only_native_research_inputs(self):
        configured, uvicorn = self._launch(
            self._literesearcher_arguments(split="test")
        )
        self.assertEqual(configured["AGENTMEMORY_SURFACE"], LITERESEARCHER_SURFACE)
        self.assertEqual(configured["AGENTMEMORY_LITERESEARCHER_SPLIT"], "test")
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_COVERAGE_MANIFEST"],
            "/data/literesearcher-stage1.json",
        )
        self.assertEqual(
            configured["AGENTMEMORY_WORKSPACE_ROOT_PARENT"],
            "/runtime/literesearcher-workspaces",
        )
        self.assertNotIn("MEMORYARENA_ROOT", configured)
        self.assertNotIn("MEMORYARENA_BASE_COMMIT", configured)
        uvicorn.assert_called_once()

    def test_literesearcher_refuses_memory_shaping(self):
        arguments = self._literesearcher_arguments()
        arguments.extend(["--memory-first-add-reward", "0.1"])
        with self.assertRaises(SystemExit):
            self._launch(arguments)

    @staticmethod
    def _literesearcher_fullpool_arguments(*, judge_key_file: str):
        return [
            "--surface",
            LITERESEARCHER_FULLPOOL_SURFACE,
            "--run-id",
            "literesearcher-fullpool-train-test",
            "--split",
            "train",
            "--literesearcher-full-pool-manifest",
            "/data/literesearcher-fullpool/manifest.json",
            "--literesearcher-full-pool-rows",
            "/data/literesearcher-fullpool/pool_rows.jsonl",
            "--literesearcher-source-root",
            "/data/literesearcher-fullpool/source",
            "--literesearcher-upstream-endpoint",
            "http://127.0.0.1:8018",
            "--literesearcher-backend-timeout-seconds",
            "33.5",
            "--literesearcher-judge-api-base",
            "http://127.0.0.1:18090/v1",
            "--literesearcher-judge-model",
            "qwen3-8b-judge",
            "--literesearcher-judge-api-key-file",
            judge_key_file,
            "--literesearcher-judge-timeout-seconds",
            "45.5",
            "--literesearcher-judge-max-retries",
            "4",
            "--literesearcher-max-policy-steps",
            "40",
            "--literesearcher-top-k",
            "5",
            "--workspace-rg-binary",
            "/tools/rg",
            "--workspace-rg-sha256",
            "a" * 64,
        ]

    def test_literesearcher_fullpool_launch_binds_upstream_judge_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "judge.key"
            key_file.write_text("private-key", encoding="utf-8")
            key_file.chmod(0o600)
            configured, uvicorn = self._launch(
                self._literesearcher_fullpool_arguments(
                    judge_key_file=str(key_file)
                )
            )
        self.assertEqual(
            configured["AGENTMEMORY_SURFACE"], LITERESEARCHER_FULLPOOL_SURFACE
        )
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_JUDGE_API_BASE"],
            "http://127.0.0.1:18090/v1",
        )
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_UPSTREAM_ENDPOINT"],
            "http://127.0.0.1:8018",
        )
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_BACKEND_TIMEOUT_SECONDS"],
            "33.5",
        )
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_JUDGE_MODEL"], "qwen3-8b-judge"
        )
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_JUDGE_API_KEY"], "private-key"
        )
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_JUDGE_TIMEOUT_SECONDS"],
            "45.5",
        )
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_JUDGE_MAX_RETRIES"], "4"
        )
        uvicorn.assert_called_once()

    def test_literesearcher_fullpool_launch_requires_judge_route(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "judge.key"
            key_file.write_text("private-key", encoding="utf-8")
            key_file.chmod(0o600)
            arguments = self._literesearcher_fullpool_arguments(
                judge_key_file=str(key_file)
            )
            base_index = arguments.index("--literesearcher-judge-api-base")
            del arguments[base_index : base_index + 2]
            with self.assertRaises(SystemExit):
                self._launch(arguments)

    def test_literesearcher_fullpool_launch_rejects_non_company_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "judge.key"
            key_file.write_text("private-key", encoding="utf-8")
            key_file.chmod(0o600)
            arguments = self._literesearcher_fullpool_arguments(
                judge_key_file=str(key_file)
            )
            model_index = arguments.index("--literesearcher-judge-model") + 1
            arguments[model_index] = "qwen-local"
            with self.assertRaises(SystemExit):
                self._launch(arguments)

    def test_literesearcher_fullpool_launch_accepts_original_company_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "judge.key"
            key_file.write_text("private-key", encoding="utf-8")
            key_file.chmod(0o600)
            arguments = self._literesearcher_fullpool_arguments(
                judge_key_file=str(key_file)
            )
            model_index = arguments.index("--literesearcher-judge-model") + 1
            arguments[model_index] = "kimi-k2.6"
            configured, uvicorn = self._launch(arguments)
        self.assertEqual(
            configured["AGENTMEMORY_LITERESEARCHER_JUDGE_MODEL"], "kimi-k2.6"
        )
        uvicorn.assert_called_once()

    @staticmethod
    def _procedural_arguments(*, split="train"):
        return [
            "--surface",
            PROCEDURAL_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            f"procedural-webshop-{split}-test",
            "--items-file",
            "/data/items.json",
            "--attributes-file",
            "/data/attrs.json",
            "--search-root",
            "/data/search",
            "--java-home",
            "/java",
            "--lucene-index-manifest",
            "/data/lucene.sha256",
            "--procedural-product-pool",
            "/data/certified-pool.json",
            "--procedural-product-pool-sha256",
            "b" * 64,
            "--procedural-task-count",
            "10000",
            "--procedural-generator-seed",
            "0",
            "--split",
            split,
        ]

    @classmethod
    def _filesystem_arguments(cls, *, split="train"):
        arguments = cls._procedural_arguments(split=split)
        surface_index = arguments.index("--surface") + 1
        arguments[surface_index] = PROCEDURAL_FILESYSTEM_SURFACE
        arguments.extend(
            [
                "--memory-prompt-mode",
                "natural_filesystem",
                "--workspace-rg-binary",
                "/opt/agentmemory/bin/rg",
                "--workspace-rg-sha256",
                "c" * 64,
            ]
        )
        return arguments

    @classmethod
    def _recency_filesystem_arguments(cls, *, split="train"):
        arguments = cls._programmatic_memory_arguments(
            surface=RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
            cli_prefix="recency_override",
            split=split,
        )
        arguments.extend(
            [
                "--workspace-rg-binary",
                "/opt/agentmemory/bin/rg",
                "--workspace-rg-sha256",
                "c" * 64,
            ]
        )
        prompt_index = arguments.index("--memory-prompt-mode") + 1
        arguments[prompt_index] = "natural_filesystem"
        return arguments

    @classmethod
    def _programmatic_filesystem_arguments(
        cls,
        *,
        surface: str,
        cli_prefix: str,
        task_count: int,
        split: str = "train",
    ):
        arguments = cls._programmatic_memory_arguments(
            surface=surface,
            cli_prefix=cli_prefix,
            split=split,
            task_count=task_count,
            memory_prompt_mode="natural_filesystem",
        )
        arguments.extend(
            [
                "--workspace-rg-binary",
                "/opt/agentmemory/bin/rg",
                "--workspace-rg-sha256",
                "c" * 64,
            ]
        )
        return arguments

    @staticmethod
    def _latent_preference_arguments(*, split="train"):
        return [
            "--surface",
            LATENT_PREFERENCE_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            f"latent-preference-{split}-test",
            "--items-file",
            "/data/items.json",
            "--attributes-file",
            "/data/attrs.json",
            "--search-root",
            "/data/search",
            "--java-home",
            "/java",
            "--lucene-index-manifest",
            "/data/lucene.sha256",
            "--latent-preference-product-pool",
            "/data/certified-latent-pool.json",
            "--latent-preference-product-pool-sha256",
            "c" * 64,
            "--latent-preference-task-count",
            "10000",
            "--latent-preference-generator-seed",
            "233",
            "--memory-prompt-mode",
            "latent_preference_sop",
            "--split",
            split,
        ]

    @staticmethod
    def _programmatic_memory_arguments(
        *,
        surface: str,
        cli_prefix: str,
        split: str = "train",
        task_count: int = 10000,
        memory_prompt_mode: str = "latent_preference_sop",
    ):
        rendered_prefix = cli_prefix.replace("_", "-")
        return [
            "--surface",
            surface,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            f"{rendered_prefix}-{split}-test",
            "--items-file",
            "/data/items.json",
            "--attributes-file",
            "/data/attrs.json",
            "--search-root",
            "/data/search",
            "--java-home",
            "/java",
            "--lucene-index-manifest",
            "/data/lucene.sha256",
            f"--{rendered_prefix}-product-pool",
            f"/data/{rendered_prefix}-pool.json",
            f"--{rendered_prefix}-product-pool-sha256",
            "d" * 64,
            f"--{rendered_prefix}-task-count",
            str(task_count),
            f"--{rendered_prefix}-generator-seed",
            "233",
            "--memory-prompt-mode",
            memory_prompt_mode,
            "--split",
            split,
        ]

    def test_travel_launches_bind_both_explicit_surfaces_reward_neutrally(self):
        for contract_mode, surface in TRAVEL_SURFACES.items():
            with self.subTest(contract_mode=contract_mode):
                configured, run = self._launch(
                    [
                        "--surface",
                        surface,
                        "--memoryarena-root",
                        "/memoryarena",
                        "--travel-tasks-path",
                        "/data/travel.jsonl",
                        "--travel-database-path",
                        "/data/travel-db",
                        "--memoryarena-base-commit",
                        "a" * 40,
                        "--run-id",
                        f"travel-{contract_mode}-test",
                    ]
                )
            self.assertEqual(configured["AGENTMEMORY_SURFACE"], surface)
            self.assertEqual(configured["AGENTMEMORY_FIRST_ADD_REWARD"], "0.0")
            self.assertEqual(
                configured["AGENTMEMORY_FIRST_LATER_RETRIEVE_REWARD"],
                "0.0",
            )
            self.assertEqual(configured["AGENTMEMORY_EXACT_REPEAT_REWARD"], "0.0")
            self.assertEqual(configured["AGENTMEMORY_INVALID_ACTION_REWARD"], "0.0")
            run.assert_called_once_with(
                "agentenv_agentmemory.server:app",
                host="0.0.0.0",
                port=8000,
                workers=1,
            )

    def test_travel_reward_overlay_requires_explicit_values(self):
        configured, _ = self._launch(
            [
                "--surface",
                TRAVEL_SURFACES["failfast"],
                "--memoryarena-root",
                "/memoryarena",
                "--travel-tasks-path",
                "/data/travel.jsonl",
                "--travel-database-path",
                "/data/travel-db",
                "--memoryarena-base-commit",
                "a" * 40,
                "--run-id",
                "travel-test",
                "--memory-first-add-reward",
                "0.1",
                "--memory-first-later-retrieve-reward",
                "0.1",
                "--memory-exact-repeat-reward",
                "-0.01",
                "--invalid-action-reward",
                "-0.01",
            ]
        )
        self.assertEqual(configured["AGENTMEMORY_FIRST_ADD_REWARD"], "0.1")
        self.assertEqual(configured["AGENTMEMORY_EXACT_REPEAT_REWARD"], "-0.01")

    def test_legacy_webshop_arguments_keep_their_environment_bindings(self):
        values = {
            "raw-data": "/data/raw.jsonl",
            "items-file": "/data/items.json",
            "attributes-file": "/data/attrs.json",
            "search-root": "/data/search",
            "java-home": "/java",
            "domain-data-path": "/data/domain.json",
            "lucene-index-manifest": "/data/lucene.sha256",
            "annotation-audit-summary": "/audit/summary.json",
            "annotation-audit-chains": "/audit/chains.jsonl",
            "annotation-manual-evidence": "/audit/manual.json",
            "annotation-gate-manifest": "/manifest/gate.json",
            "annotation-gate-manifest-sha256": "b" * 64,
        }
        arguments = [
            "--surface",
            NATIVE_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "webshop-test",
        ]
        for key, value in values.items():
            arguments.extend([f"--{key}", value])
        configured, _ = self._launch(arguments)
        self.assertEqual(
            configured["AGENTMEMORY_MEMORYARENA_RAW_PATH"], values["raw-data"]
        )
        self.assertEqual(
            configured["MEMORYARENA_WEBSHOP_ITEMS_FILE"],
            values["items-file"],
        )
        self.assertEqual(
            configured["AGENTMEMORY_ANNOTATION_GATE_MANIFEST"],
            values["annotation-gate-manifest"],
        )
        self.assertEqual(
            configured["AGENTMEMORY_FIRST_VALID_ADD_REWARD"],
            str(FIRST_VALID_ADD_BONUS),
        )
        self.assertEqual(
            configured["AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD"],
            str(FIRST_VALID_LATER_SESSION_RETRIEVE_BONUS),
        )
        self.assertEqual(configured["AGENTMEMORY_LTM_INVENTORY_MODE"], "hidden")
        self.assertEqual(configured["AGENTMEMORY_MEMORY_PROMPT_MODE"], "legacy")
        self.assertNotIn("AGENTMEMORY_FIRST_ADD_REWARD", configured)
        self.assertNotIn("AGENTMEMORY_TRAVEL_TASKS_PATH", configured)

    def test_smoke_service_requires_and_exports_runtime_source_identity(self):
        with self.assertRaises(SystemExit):
            self._launch(
                [
                    *self._latent_preference_arguments(),
                    "--service-role",
                    "smoke",
                ]
            )

        configured, _ = self._launch(
            [
                *self._latent_preference_arguments(),
                "--service-role",
                "smoke",
                "--runtime-source-id",
                "a" * 40,
            ]
        )
        self.assertEqual(configured["AGENTMEMORY_SERVICE_ROLE"], "smoke")
        self.assertEqual(configured["AGENTMEMORY_RUNTIME_SOURCE_ID"], "a" * 40)

    def test_procedural_webshop_binds_only_machine_verified_training_inputs(self):
        arguments = [
            "--surface",
            PROCEDURAL_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "procedural-webshop-test",
            "--items-file",
            "/data/items.json",
            "--attributes-file",
            "/data/attrs.json",
            "--search-root",
            "/data/search",
            "--java-home",
            "/java",
            "--lucene-index-manifest",
            "/data/lucene.sha256",
            "--procedural-product-pool",
            "/data/certified-pool.json",
            "--procedural-product-pool-sha256",
            "b" * 64,
            "--procedural-task-count",
            "10000",
            "--procedural-generator-seed",
            "0",
        ]

        configured, _ = self._launch(arguments)

        self.assertEqual(configured["AGENTMEMORY_SURFACE"], PROCEDURAL_SURFACE)
        self.assertEqual(configured["AGENTMEMORY_PROCEDURAL_TASK_COUNT"], "10000")
        self.assertEqual(configured["AGENTMEMORY_PROCEDURAL_GENERATOR_SEED"], "0")
        self.assertEqual(configured["AGENTMEMORY_SPLIT"], "train")
        self.assertNotIn("AGENTMEMORY_MEMORYARENA_RAW_PATH", configured)
        self.assertNotIn("AGENTMEMORY_ANNOTATION_GATE_MANIFEST", configured)
        self.assertNotIn("AGENTMEMORY_ANNOTATION_MANUAL_EVIDENCE", configured)

    def test_filesystem_webshop_binds_natural_prompt_and_zero_shaping(self):
        configured, _ = self._launch(self._filesystem_arguments())

        self.assertEqual(
            configured["AGENTMEMORY_SURFACE"], PROCEDURAL_FILESYSTEM_SURFACE
        )
        self.assertEqual(
            configured["AGENTMEMORY_MEMORY_PROMPT_MODE"], "natural_filesystem"
        )
        self.assertEqual(configured["AGENTMEMORY_LTM_INVENTORY_MODE"], "hidden")
        self.assertEqual(configured["AGENTMEMORY_LTM_TRANSITION_NOTICE_MODE"], "none")
        self.assertEqual(configured["AGENTMEMORY_ACTION_LISTING_MODE"], "separate")
        self.assertEqual(
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"],
            "/opt/agentmemory/bin/rg",
        )
        self.assertEqual(
            configured["AGENTMEMORY_WORKSPACE_RG_SHA256"],
            "c" * 64,
        )
        self.assertEqual(configured["AGENTMEMORY_FIRST_VALID_ADD_REWARD"], "0.0")
        self.assertEqual(
            configured["AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD"],
            "0.0",
        )

    def test_recency_filesystem_binds_recency_data_and_workspace_contract(self):
        configured, _ = self._launch(self._recency_filesystem_arguments())

        self.assertEqual(
            configured["AGENTMEMORY_SURFACE"],
            RECENCY_OVERRIDE_FILESYSTEM_SURFACE,
        )
        self.assertEqual(
            configured["AGENTMEMORY_MEMORY_PROMPT_MODE"],
            "natural_filesystem",
        )
        self.assertEqual(
            configured["AGENTMEMORY_RECENCY_OVERRIDE_TASK_COUNT"],
            "10000",
        )
        self.assertEqual(
            configured["AGENTMEMORY_WORKSPACE_RG_BINARY"],
            "/opt/agentmemory/bin/rg",
        )
        self.assertEqual(configured["AGENTMEMORY_FIRST_VALID_ADD_REWARD"], "0.0")
        self.assertEqual(
            configured["AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD"],
            "0.0",
        )

    def test_new_filesystem_surfaces_bind_their_own_data_and_zero_shaping(self):
        cases = (
            (
                LATENT_PREFERENCE_FILESYSTEM_SURFACE,
                "latent_preference",
                "AGENTMEMORY_LATENT_PREFERENCE",
                10_000,
            ),
            (
                DISTRACTOR_ROBUSTNESS_FILESYSTEM_SURFACE,
                "distractor_robustness",
                "AGENTMEMORY_DISTRACTOR_ROBUSTNESS",
                10_000,
            ),
            (
                COMPOSITIONAL_RECALL_FILESYSTEM_SURFACE,
                "compositional_recall",
                "AGENTMEMORY_COMPOSITIONAL_RECALL",
                10_000,
            ),
            (
                INTENT_CLARIFICATION_FILESYSTEM_SURFACE,
                "intent_clarification",
                "AGENTMEMORY_INTENT_CLARIFICATION",
                10_000,
            ),
            (
                SELECTIVE_MEMORY_USE_FILESYSTEM_SURFACE,
                "selective_memory_use",
                "AGENTMEMORY_SELECTIVE_MEMORY_USE",
                10_000,
            ),
            (
                NEGATIVE_CONSTRAINT_FILESYSTEM_SURFACE,
                "negative_constraint",
                "AGENTMEMORY_NEGATIVE_CONSTRAINT",
                9_999,
            ),
        )
        for surface, cli_prefix, env_prefix, task_count in cases:
            with self.subTest(surface=surface):
                configured, _ = self._launch(
                    self._programmatic_filesystem_arguments(
                        surface=surface,
                        cli_prefix=cli_prefix,
                        task_count=task_count,
                    )
                )
                self.assertEqual(configured["AGENTMEMORY_SURFACE"], surface)
                self.assertEqual(
                    configured["AGENTMEMORY_MEMORY_PROMPT_MODE"],
                    "natural_filesystem",
                )
                self.assertEqual(
                    configured[f"{env_prefix}_TASK_COUNT"],
                    str(task_count),
                )
                self.assertEqual(
                    configured["AGENTMEMORY_WORKSPACE_RG_BINARY"],
                    "/opt/agentmemory/bin/rg",
                )
                self.assertEqual(
                    configured["AGENTMEMORY_FIRST_VALID_ADD_REWARD"],
                    "0.0",
                )
                self.assertEqual(
                    configured[
                        "AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD"
                    ],
                    "0.0",
                )

    def test_filesystem_webshop_rejects_legacy_modes_and_reward_shaping(self):
        cases = (
            ("prompt", ["--memory-prompt-mode", "legacy"]),
            ("inventory", ["--ltm-inventory-mode", "keys"]),
            ("reward", ["--memory-first-add-reward", "0.1"]),
        )
        for label, replacement in cases:
            arguments = self._filesystem_arguments()
            flag = replacement[0]
            if flag in arguments:
                arguments[arguments.index(flag) + 1] = replacement[1]
            else:
                arguments.extend(replacement)
            with self.subTest(label=label), self.assertRaises(SystemExit):
                self._launch(arguments)

    def test_filesystem_webshop_requires_both_rg_pin_fields(self):
        for flag in ("--workspace-rg-binary", "--workspace-rg-sha256"):
            arguments = self._filesystem_arguments()
            index = arguments.index(flag)
            del arguments[index : index + 2]
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                self._launch(arguments)

    def test_filesystem_intervention_eval_requires_private_token_file(self):
        base = [
            *self._filesystem_arguments(),
            "--service-role",
            "intervention_eval",
            "--runtime-source-id",
            "d" * 40,
        ]
        with self.assertRaises(SystemExit):
            self._launch(base)

        with tempfile.TemporaryDirectory() as raw:
            token_path = Path(raw) / "token"
            token_path.write_text("t" * 48 + "\n", encoding="utf-8")
            token_path.chmod(0o600)
            configured, _ = self._launch(
                [
                    *base,
                    "--workspace-intervention-token-file",
                    str(token_path),
                ]
            )
        self.assertEqual(
            configured["AGENTMEMORY_SERVICE_ROLE"],
            "intervention_eval",
        )
        self.assertEqual(
            configured["AGENTMEMORY_WORKSPACE_INTERVENTION_TOKEN"],
            "t" * 48,
        )

        with self.assertRaises(SystemExit):
            self._launch(
                [
                    *self._filesystem_arguments(),
                    "--workspace-intervention-token-file",
                    "/tmp/not-allowed",
                ]
            )

    def test_procedural_webshop_rejects_odd_task_count(self):
        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = Mock()
        arguments = [
            "agentmemory",
            "--surface",
            PROCEDURAL_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "procedural-webshop-test",
            "--items-file",
            "/data/items.json",
            "--attributes-file",
            "/data/attrs.json",
            "--search-root",
            "/data/search",
            "--java-home",
            "/java",
            "--lucene-index-manifest",
            "/data/lucene.sha256",
            "--procedural-product-pool",
            "/data/certified-pool.json",
            "--procedural-product-pool-sha256",
            "b" * 64,
            "--procedural-task-count",
            "9999",
            "--procedural-generator-seed",
            "59",
        ]
        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn}),
            patch.object(sys, "argv", arguments),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(SystemExit),
        ):
            launch()
        uvicorn.run.assert_not_called()

    def test_latent_preference_binds_only_verified_programmatic_inputs(self):
        configured, _ = self._launch(self._latent_preference_arguments())

        self.assertEqual(
            configured["AGENTMEMORY_SURFACE"],
            LATENT_PREFERENCE_SURFACE,
        )
        self.assertEqual(
            configured["AGENTMEMORY_LATENT_PREFERENCE_TASK_COUNT"],
            "10000",
        )
        self.assertEqual(
            configured["AGENTMEMORY_LATENT_PREFERENCE_GENERATOR_SEED"],
            "233",
        )
        self.assertEqual(
            configured["AGENTMEMORY_MEMORY_PROMPT_MODE"],
            "latent_preference_sop",
        )
        self.assertEqual(
            configured["AGENTMEMORY_LATENT_PREFERENCE_PROVIDER_MODE"],
            "reseeded_stream",
        )
        self.assertNotIn("AGENTMEMORY_PROCEDURAL_PRODUCT_POOL", configured)
        self.assertNotIn("AGENTMEMORY_MEMORYARENA_RAW_PATH", configured)
        self.assertNotIn("AGENTMEMORY_ANNOTATION_GATE_MANIFEST", configured)

    def test_latent_preference_rejects_generic_memory_prompt_mode(self):
        arguments = self._latent_preference_arguments()
        mode_index = arguments.index("--memory-prompt-mode") + 1
        arguments[mode_index] = "legacy"
        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = Mock()
        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn}),
            patch.object(sys, "argv", ["agentmemory", *arguments]),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(SystemExit),
        ):
            launch()
        uvicorn.run.assert_not_called()

    def test_latent_preference_eval_defaults_to_fixed_window(self):
        configured, _ = self._launch(
            self._latent_preference_arguments(split="dev")
        )
        self.assertEqual(
            configured["AGENTMEMORY_LATENT_PREFERENCE_PROVIDER_MODE"],
            "fixed_window",
        )
        self.assertEqual(configured["AGENTMEMORY_SPLIT"], "dev")

    def test_latent_preference_rejects_odd_task_count(self):
        arguments = self._latent_preference_arguments()
        count_index = arguments.index("--latent-preference-task-count") + 1
        arguments[count_index] = "9999"
        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = Mock()
        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn}),
            patch.object(sys, "argv", ["agentmemory", *arguments]),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(SystemExit),
        ):
            launch()
        uvicorn.run.assert_not_called()

    def test_new_programmatic_memory_surfaces_bind_isolated_inputs(self):
        cases = (
            (
                DISTRACTOR_ROBUSTNESS_SURFACE,
                "distractor_robustness",
                "AGENTMEMORY_DISTRACTOR_ROBUSTNESS",
                10000,
            ),
            (
                COMPOSITIONAL_RECALL_SURFACE,
                "compositional_recall",
                "AGENTMEMORY_COMPOSITIONAL_RECALL",
                10000,
            ),
            (
                INTENT_CLARIFICATION_SURFACE,
                "intent_clarification",
                "AGENTMEMORY_INTENT_CLARIFICATION",
                10000,
                "latent_preference_sop",
            ),
            (
                SELECTIVE_MEMORY_USE_SURFACE,
                "selective_memory_use",
                "AGENTMEMORY_SELECTIVE_MEMORY_USE",
                10000,
                "selective_memory_sop",
            ),
        )
        normalized_cases = tuple(
            (*case, "latent_preference_sop") if len(case) == 4 else case
            for case in cases
        )
        for surface, cli_prefix, env_prefix, task_count, prompt_mode in normalized_cases:
            with self.subTest(surface=surface):
                configured, _ = self._launch(
                    self._programmatic_memory_arguments(
                        surface=surface,
                        cli_prefix=cli_prefix,
                        task_count=task_count,
                        memory_prompt_mode=prompt_mode,
                    )
                )
                self.assertEqual(configured["AGENTMEMORY_SURFACE"], surface)
                self.assertEqual(configured[f"{env_prefix}_TASK_COUNT"], "10000")
                self.assertEqual(configured[f"{env_prefix}_GENERATOR_SEED"], "233")
                self.assertEqual(
                    configured[f"{env_prefix}_PROVIDER_MODE"],
                    "reseeded_stream",
                )
                self.assertEqual(
                    configured["AGENTMEMORY_MEMORY_PROMPT_MODE"],
                    prompt_mode,
                )
                self.assertNotIn("AGENTMEMORY_PROCEDURAL_PRODUCT_POOL", configured)
                self.assertNotIn("AGENTMEMORY_MEMORYARENA_RAW_PATH", configured)
                if surface == SELECTIVE_MEMORY_USE_SURFACE:
                    self.assertEqual(
                        configured["AGENTMEMORY_FIRST_VALID_ADD_REWARD"],
                        "0.0",
                    )
                    self.assertEqual(
                        configured[
                            "AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD"
                        ],
                        "0.0",
                    )

    def test_new_programmatic_memory_eval_defaults_to_fixed_window(self):
        configured, _ = self._launch(
            self._programmatic_memory_arguments(
                surface=DISTRACTOR_ROBUSTNESS_SURFACE,
                cli_prefix="distractor_robustness",
                split="dev",
            )
        )
        self.assertEqual(
            configured["AGENTMEMORY_DISTRACTOR_ROBUSTNESS_PROVIDER_MODE"],
            "fixed_window",
        )

    def test_compositional_recall_requires_complete_four_task_orbits(self):
        with self.assertRaises(SystemExit):
            self._launch(
                self._programmatic_memory_arguments(
                    surface=COMPOSITIONAL_RECALL_SURFACE,
                    cli_prefix="compositional_recall",
                    task_count=10002,
                )
            )

    def test_selective_memory_requires_complete_four_task_orbits_and_zero_bonus(self):
        arguments = self._programmatic_memory_arguments(
            surface=SELECTIVE_MEMORY_USE_SURFACE,
            cli_prefix="selective_memory_use",
            task_count=10002,
            memory_prompt_mode="selective_memory_sop",
        )
        with self.assertRaises(SystemExit):
            self._launch(arguments)

        arguments = self._programmatic_memory_arguments(
            surface=SELECTIVE_MEMORY_USE_SURFACE,
            cli_prefix="selective_memory_use",
            memory_prompt_mode="selective_memory_sop",
        )
        arguments.extend(["--memory-first-add-reward", "0.1"])
        with self.assertRaises(SystemExit):
            self._launch(arguments)

    def test_procedural_provider_defaults_are_stream_for_train_and_fixed_for_eval(
        self,
    ):
        expected_modes = {
            "train": "reseeded_stream",
            "dev": "fixed_window",
            "test": "fixed_window",
        }
        for split, expected_mode in expected_modes.items():
            with self.subTest(split=split):
                configured, _ = self._launch(
                    self._procedural_arguments(split=split)
                )
                self.assertEqual(
                    configured["AGENTMEMORY_PROCEDURAL_PROVIDER_MODE"],
                    expected_mode,
                )
                self.assertEqual(configured["AGENTMEMORY_SPLIT"], split)

    def test_procedural_launch_rejects_invalid_stream_boundaries(self):
        invalid_suffixes = (
            (
                "dev_stream",
                ["--split", "dev", "--procedural-provider-mode", "reseeded_stream"],
            ),
            (
                "stream_start_orbit",
                ["--procedural-start-orbit", "1"],
            ),
            (
                "negative_start_orbit",
                ["--procedural-provider-mode", "fixed_window", "--procedural-start-orbit", "-1"],
            ),
        )
        base = self._procedural_arguments()
        # Each case owns its split arguments so argparse never receives two split flags.
        base_without_split = base[:-2]
        for name, suffix in invalid_suffixes:
            arguments = (
                base_without_split + suffix
                if name == "dev_stream"
                else base + suffix
            )
            with self.subTest(case=name), self.assertRaises(SystemExit):
                self._launch(arguments)

    def test_legacy_webshop_key_inventory_requires_explicit_flag(self):
        values = {
            "raw-data": "/data/raw.jsonl",
            "items-file": "/data/items.json",
            "attributes-file": "/data/attrs.json",
            "search-root": "/data/search",
            "java-home": "/java",
            "domain-data-path": "/data/domain.json",
            "lucene-index-manifest": "/data/lucene.sha256",
            "annotation-audit-summary": "/audit/summary.json",
            "annotation-audit-chains": "/audit/chains.jsonl",
            "annotation-manual-evidence": "/audit/manual.json",
            "annotation-gate-manifest": "/manifest/gate.json",
            "annotation-gate-manifest-sha256": "b" * 64,
        }
        arguments = [
            "--surface",
            NATIVE_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "webshop-inventory-test",
            "--ltm-inventory-mode",
            "keys",
        ]
        for key, value in values.items():
            arguments.extend([f"--{key}", value])

        configured, _ = self._launch(arguments)

        self.assertEqual(configured["AGENTMEMORY_LTM_INVENTORY_MODE"], "keys")

    def test_legacy_webshop_neutral_prompt_requires_explicit_flag(self):
        values = {
            "raw-data": "/data/raw.jsonl",
            "items-file": "/data/items.json",
            "attributes-file": "/data/attrs.json",
            "search-root": "/data/search",
            "java-home": "/java",
            "domain-data-path": "/data/domain.json",
            "lucene-index-manifest": "/data/lucene.sha256",
            "annotation-audit-summary": "/audit/summary.json",
            "annotation-audit-chains": "/audit/chains.jsonl",
            "annotation-manual-evidence": "/audit/manual.json",
            "annotation-gate-manifest": "/manifest/gate.json",
            "annotation-gate-manifest-sha256": "b" * 64,
        }
        arguments = [
            "--surface",
            NATIVE_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "webshop-neutral-prompt-test",
            "--memory-prompt-mode",
            "neutral",
        ]
        for key, value in values.items():
            arguments.extend([f"--{key}", value])

        configured, _ = self._launch(arguments)

        self.assertEqual(configured["AGENTMEMORY_MEMORY_PROMPT_MODE"], "neutral")

    def test_legacy_webshop_neutral_horizon_prompt_requires_explicit_flag(self):
        values = {
            "raw-data": "/data/raw.jsonl",
            "items-file": "/data/items.json",
            "attributes-file": "/data/attrs.json",
            "search-root": "/data/search",
            "java-home": "/java",
            "domain-data-path": "/data/domain.json",
            "lucene-index-manifest": "/data/lucene.sha256",
            "annotation-audit-summary": "/audit/summary.json",
            "annotation-audit-chains": "/audit/chains.jsonl",
            "annotation-manual-evidence": "/audit/manual.json",
            "annotation-gate-manifest": "/manifest/gate.json",
            "annotation-gate-manifest-sha256": "b" * 64,
        }
        arguments = [
            "--surface",
            NATIVE_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "webshop-neutral-horizon-prompt-test",
            "--memory-prompt-mode",
            "neutral_horizon",
        ]
        for key, value in values.items():
            arguments.extend([f"--{key}", value])

        configured, _ = self._launch(arguments)

        self.assertEqual(
            configured["AGENTMEMORY_MEMORY_PROMPT_MODE"],
            "neutral_horizon",
        )

    def test_webshop_neutral_horizon_responsibility_prompt_is_explicit(self):
        values = {
            "raw-data": "/data/raw.jsonl",
            "items-file": "/data/items.json",
            "attributes-file": "/data/attrs.json",
            "search-root": "/data/search",
            "java-home": "/java",
            "domain-data-path": "/data/domain.json",
            "lucene-index-manifest": "/data/lucene.sha256",
            "annotation-audit-summary": "/audit/summary.json",
            "annotation-audit-chains": "/audit/chains.jsonl",
            "annotation-manual-evidence": "/audit/manual.json",
            "annotation-gate-manifest": "/manifest/gate.json",
            "annotation-gate-manifest-sha256": "b" * 64,
        }
        arguments = [
            "--surface",
            NATIVE_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "webshop-neutral-horizon-responsibility-prompt-test",
            "--memory-prompt-mode",
            "neutral_horizon_responsibility",
        ]
        for key, value in values.items():
            arguments.extend([f"--{key}", value])

        configured, _ = self._launch(arguments)

        self.assertEqual(
            configured["AGENTMEMORY_MEMORY_PROMPT_MODE"],
            "neutral_horizon_responsibility",
        )

    def test_legacy_webshop_action_listing_defaults_to_separate_and_allows_unified(self):
        values = {
            "raw-data": "/data/raw.jsonl",
            "items-file": "/data/items.json",
            "attributes-file": "/data/attrs.json",
            "search-root": "/data/search",
            "java-home": "/java",
            "domain-data-path": "/data/domain.json",
            "lucene-index-manifest": "/data/lucene.sha256",
            "annotation-audit-summary": "/audit/summary.json",
            "annotation-audit-chains": "/audit/chains.jsonl",
            "annotation-manual-evidence": "/audit/manual.json",
            "annotation-gate-manifest": "/manifest/gate.json",
            "annotation-gate-manifest-sha256": "b" * 64,
        }
        arguments = [
            "--surface",
            NATIVE_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "webshop-unified-actions-test",
        ]
        for key, value in values.items():
            arguments.extend([f"--{key}", value])

        default_configured, _ = self._launch(arguments)
        unified_configured, _ = self._launch(
            [*arguments, "--action-listing-mode", "unified"]
        )

        self.assertEqual(
            default_configured["AGENTMEMORY_ACTION_LISTING_MODE"],
            "separate",
        )
        self.assertEqual(
            unified_configured["AGENTMEMORY_ACTION_LISTING_MODE"],
            "unified",
        )

    def test_legacy_webshop_reward_flags_bind_canonical_environment_names(self):
        values = {
            "raw-data": "/data/raw.jsonl",
            "items-file": "/data/items.json",
            "attributes-file": "/data/attrs.json",
            "search-root": "/data/search",
            "java-home": "/java",
            "domain-data-path": "/data/domain.json",
            "lucene-index-manifest": "/data/lucene.sha256",
            "annotation-audit-summary": "/audit/summary.json",
            "annotation-audit-chains": "/audit/chains.jsonl",
            "annotation-manual-evidence": "/audit/manual.json",
            "annotation-gate-manifest": "/manifest/gate.json",
            "annotation-gate-manifest-sha256": "b" * 64,
        }
        arguments = [
            "--surface",
            NATIVE_SURFACE,
            "--memoryarena-root",
            "/memoryarena",
            "--memoryarena-base-commit",
            "a" * 40,
            "--run-id",
            "webshop-reward-test",
            "--memory-first-add-reward",
            "0.2",
            "--memory-first-later-retrieve-reward",
            "0.1",
        ]
        for key, value in values.items():
            arguments.extend([f"--{key}", value])

        configured, _ = self._launch(arguments)

        self.assertEqual(configured["AGENTMEMORY_FIRST_VALID_ADD_REWARD"], "0.2")
        self.assertEqual(
            configured["AGENTMEMORY_FIRST_VALID_LATER_SESSION_RETRIEVE_REWARD"],
            "0.1",
        )

    def test_surface_specific_required_arguments_fail_closed(self):
        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = Mock()
        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn}),
            patch.object(
                sys,
                "argv",
                [
                    "agentmemory",
                    "--surface",
                    TRAVEL_SURFACES["failfast"],
                    "--memoryarena-root",
                    "/memoryarena",
                    "--memoryarena-base-commit",
                    "a" * 40,
                    "--run-id",
                    "missing-travel-paths",
                ],
            ),
        ):
            with self.assertRaises(SystemExit):
                launch()
        uvicorn.run.assert_not_called()

    def test_math_and_physics_launch_bind_judge_configuration(self):
        for contract_mode, surfaces in FORMAL_REASONING_SURFACES_BY_MODE.items():
            for domain, surface in surfaces.items():
                with self.subTest(domain=domain, contract_mode=contract_mode):
                    configured, _ = self._launch(
                        [
                            "--surface",
                            surface,
                            "--memoryarena-root",
                            "/memoryarena",
                            "--formal-reasoning-tasks-path",
                            "/data/formal.jsonl",
                            "--formal-reasoning-judge-model",
                            "judge-model",
                            "--formal-reasoning-judge-base-url",
                            "https://judge.example/v1",
                            "--formal-reasoning-judge-temperature",
                            "0.25",
                            "--formal-reasoning-judge-max-tokens",
                            "1234",
                            "--memoryarena-base-commit",
                            "a" * 40,
                            "--run-id",
                            f"formal-{domain}",
                        ]
                    )
                    self.assertEqual(configured["AGENTMEMORY_SURFACE"], surface)
                    self.assertEqual(
                        configured["AGENTMEMORY_FORMAL_REASONING_TASKS_PATH"],
                        "/data/formal.jsonl",
                    )
                    self.assertEqual(
                        configured["AGENTMEMORY_FORMAL_REASONING_JUDGE_MODEL"],
                        "judge-model",
                    )
                    self.assertEqual(
                        configured["AGENTMEMORY_FORMAL_REASONING_JUDGE_TEMPERATURE"],
                        "0.25",
                    )
                    self.assertEqual(
                        configured["AGENTMEMORY_FORMAL_REASONING_JUDGE_MAX_TOKENS"],
                        "1234",
                    )

    def test_browsecomp_launch_binds_every_production_input(self):
        for contract_mode, surface in BROWSECOMP_SURFACES.items():
            with self.subTest(contract_mode=contract_mode):
                configured, _ = self._launch(
                    [
                        "--surface",
                        surface,
                        "--memoryarena-root",
                        "/memoryarena",
                        "--browsecomp-tasks-path",
                        "/data/progressive-search.jsonl",
                        "--browsecomp-index-path",
                        "/data/indexes/shard*.index",
                        "--browsecomp-corpus-path",
                        "/data/corpus.jsonl",
                        "--browsecomp-corpus-manifest",
                        "/data/corpus.manifest.json",
                        "--browsecomp-embedding-provider",
                        "openai",
                        "--browsecomp-embedding-model",
                        "text-embedding-3-small",
                        "--browsecomp-judge-model",
                        "judge-model",
                        "--browsecomp-judge-max-tokens",
                        "1234",
                        "--browsecomp-api-base-url",
                        "https://api.example/v1",
                        "--memoryarena-base-commit",
                        "b" * 40,
                        "--run-id",
                        "browsecomp-test",
                    ]
                )
            self.assertEqual(configured["AGENTMEMORY_SURFACE"], surface)
            self.assertEqual(
                configured["AGENTMEMORY_BROWSECOMP_TASKS_PATH"],
                "/data/progressive-search.jsonl",
            )
            self.assertEqual(
                configured["MEMORYARENA_BROWSECOMP_INDEX_PATH"],
                "/data/indexes/shard*.index",
            )
            self.assertEqual(
                configured["MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST"],
                "/data/corpus.manifest.json",
            )
            self.assertEqual(
                configured["AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER"],
                "openai",
            )
            self.assertEqual(
                configured["AGENTMEMORY_BROWSECOMP_JUDGE_MAX_TOKENS"],
                "1234",
            )
            self.assertEqual(configured["OPENAI_BASE_URL"], "https://api.example/v1")

    def test_browsecomp_missing_required_input_fails_before_uvicorn(self):
        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = Mock()
        with (
            patch.dict(sys.modules, {"uvicorn": uvicorn}),
            patch.object(
                sys,
                "argv",
                [
                    "agentmemory",
                    "--surface",
                    BROWSECOMP_SURFACES["paper_eval"],
                    "--memoryarena-root",
                    "/memoryarena",
                    "--memoryarena-base-commit",
                    "b" * 40,
                    "--run-id",
                    "missing-browsecomp-inputs",
                ],
            ),
        ):
            with self.assertRaises(SystemExit):
                launch()
        uvicorn.run.assert_not_called()

    def test_browsecomp_bm25_launch_does_not_bind_dense_inputs(self):
        configured, _ = self._launch(
            [
                "--surface",
                BROWSECOMP_BM25_INTEGRATION_SURFACE,
                "--memoryarena-root",
                "/memoryarena",
                "--browsecomp-tasks-path",
                "/data/progressive-search.jsonl",
                "--browsecomp-bm25-index-path",
                "/data/lucene-index",
                "--browsecomp-judge-model",
                "judge-model",
                "--browsecomp-judge-max-tokens",
                "1234",
                "--browsecomp-api-base-url",
                "https://api.example/v1",
                "--memoryarena-base-commit",
                "b" * 40,
                "--run-id",
                "browsecomp-bm25-test",
            ]
        )
        self.assertEqual(
            configured["AGENTMEMORY_SURFACE"],
            BROWSECOMP_BM25_INTEGRATION_SURFACE,
        )
        self.assertEqual(
            configured["MEMORYARENA_BROWSECOMP_BM25_INDEX_PATH"],
            "/data/lucene-index",
        )
        for key in (
            "MEMORYARENA_BROWSECOMP_INDEX_PATH",
            "MEMORYARENA_BROWSECOMP_CORPUS_PATH",
            "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST",
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER",
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL",
        ):
            self.assertNotIn(key, configured)


if __name__ == "__main__":
    unittest.main()
