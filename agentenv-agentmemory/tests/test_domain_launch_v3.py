from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

from agentenv_agentmemory.domains import (
    BROWSECOMP_BM25_INTEGRATION_SURFACE,
    BROWSECOMP_SURFACES,
    FORMAL_REASONING_SURFACES_BY_MODE,
    SCIWORLD_SURFACES,
    TRAVEL_SURFACES,
)
from agentenv_agentmemory.env_wrapper import NATIVE_SURFACE
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

    def test_sciworld_launch_binds_backend_split_and_task_count(self):
        surface = SCIWORLD_SURFACES["sop_memory"]
        configured, run = self._launch(
            [
                "--surface",
                surface,
                "--memoryarena-root",
                "/memoryarena",
                "--memoryarena-base-commit",
                "a" * 40,
                "--run-id",
                "sciworld-sop-dev",
                "--sciworld-backend",
                "scienceworld",
                "--sciworld-task-count",
                "7",
                "--split",
                "dev",
            ]
        )
        self.assertEqual(configured["AGENTMEMORY_SURFACE"], surface)
        self.assertEqual(configured["AGENTMEMORY_SCIWORLD_BACKEND"], "scienceworld")
        self.assertEqual(configured["AGENTMEMORY_SCIWORLD_TASK_COUNT"], "7")
        self.assertEqual(configured["AGENTMEMORY_SPLIT"], "dev")
        run.assert_called_once()

    def test_sciworld_launch_rejects_all_split(self):
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
                    SCIWORLD_SURFACES["conductivity_memory"],
                    "--memoryarena-root",
                    "/memoryarena",
                    "--memoryarena-base-commit",
                    "a" * 40,
                    "--run-id",
                    "sciworld-all-invalid",
                    "--split",
                    "all",
                ],
            ),
            patch.dict(os.environ, {}, clear=True),
            self.assertRaises(SystemExit),
        ):
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
