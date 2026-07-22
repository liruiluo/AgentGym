from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentenv_agentmemory.domains import (
    BROWSECOMP_SURFACE,
    FORMAL_REASONING_SURFACES,
    TRAVEL_SURFACE,
)
from agentenv_agentmemory.env_wrapper import NATIVE_SURFACE
from agentenv_agentmemory.runtime import server_factory


class DomainServerFactoryTest(unittest.TestCase):
    @staticmethod
    def _factory(surface):
        return SimpleNamespace(surface=surface, task_count=1)

    def test_webshop_uses_unchanged_legacy_wrapper(self):
        sentinel = object()
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_SURFACE": NATIVE_SURFACE},
            clear=True,
        ), patch.object(
            server_factory,
            "AgentMemoryWrapper",
            return_value=sentinel,
        ) as wrapper:
            self.assertIs(server_factory.build_server(), sentinel)
        wrapper.assert_called_once_with()

    def test_travel_builds_explicit_reward_overlay(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks = root / "travel.jsonl"
            database = root / "database"
            memoryarena = root / "MemoryArena"
            tasks.write_text("{}\n", encoding="utf-8")
            database.mkdir()
            memoryarena.mkdir()
            environment = {
                "AGENTMEMORY_SURFACE": TRAVEL_SURFACE,
                "AGENTMEMORY_TRAVEL_TASKS_PATH": str(tasks),
                "MEMORYARENA_ROOT": str(memoryarena),
                "MEMORYARENA_TRAVEL_DATABASE_PATH": str(database),
                "MEMORYARENA_BASE_COMMIT": "a" * 40,
                "AGENTMEMORY_FIRST_ADD_REWARD": "0.1",
                "AGENTMEMORY_FIRST_LATER_RETRIEVE_REWARD": "0.1",
                "AGENTMEMORY_EXACT_REPEAT_REWARD": "-0.01",
                "AGENTMEMORY_INVALID_ACTION_REWARD": "-0.01",
            }
            factory = self._factory(TRAVEL_SURFACE)
            sentinel = object()
            with patch.dict(os.environ, environment, clear=True), patch.object(
                server_factory,
                "TravelPlannerFactory",
                return_value=factory,
            ) as factory_type, patch.object(
                server_factory,
                "DomainEnvWrapper",
                return_value=sentinel,
            ) as wrapper_type:
                self.assertIs(server_factory.build_server(), sentinel)

            factory_type.assert_called_once_with(
                tasks_path=tasks.resolve(),
                memoryarena_root=memoryarena.resolve(),
                database_path=database.resolve(),
                expected_memoryarena_commit="a" * 40,
            )
            args, kwargs = wrapper_type.call_args
            self.assertEqual(args, (factory,))
            policy = kwargs["reward_policy"]
            self.assertEqual(policy.first_add, 0.1)
            self.assertEqual(policy.first_later_phase_retrieve, 0.1)
            self.assertEqual(policy.exact_repeat, -0.01)
            self.assertEqual(kwargs["invalid_action_penalty"], -0.01)

    def test_unknown_surface_fails_closed(self):
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_SURFACE": "unknown"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "unknown AgentMemoryGym v3 surface"):
                server_factory.build_server()

    def test_math_and_physics_bind_explicit_failfast_factories(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks = root / "formal.jsonl"
            memoryarena = root / "MemoryArena"
            tasks.write_text("{}\n", encoding="utf-8")
            memoryarena.mkdir()
            for domain, surface in FORMAL_REASONING_SURFACES.items():
                environment = {
                    "AGENTMEMORY_SURFACE": surface,
                    "AGENTMEMORY_FORMAL_REASONING_TASKS_PATH": str(tasks),
                    "MEMORYARENA_ROOT": str(memoryarena),
                    "MEMORYARENA_BASE_COMMIT": "a" * 40,
                    "AGENTMEMORY_FORMAL_REASONING_JUDGE_MODEL": "judge-model",
                    "AGENTMEMORY_FORMAL_REASONING_JUDGE_BASE_URL": (
                        "https://judge.example/v1/"
                    ),
                    "AGENTMEMORY_FORMAL_REASONING_JUDGE_TEMPERATURE": "0.25",
                    "AGENTMEMORY_FORMAL_REASONING_JUDGE_MAX_TOKENS": "1234",
                }
                factory = self._factory(surface)
                sentinel = object()
                with self.subTest(domain=domain), patch.dict(
                    os.environ,
                    environment,
                    clear=True,
                ), patch.object(
                    server_factory,
                    "FormalReasoningFactory",
                    return_value=factory,
                ) as factory_type, patch.object(
                    server_factory,
                    "DomainEnvWrapper",
                    return_value=sentinel,
                ):
                    self.assertIs(server_factory.build_server(), sentinel)

                factory_type.assert_called_once_with(
                    domain=domain,
                    tasks_path=tasks.resolve(),
                    memoryarena_root=memoryarena.resolve(),
                    judge_config={
                        "backend": "openai",
                        "model_name": "judge-model",
                        "base_url": "https://judge.example/v1",
                        "temperature": 0.25,
                        "max_tokens": 1234,
                    },
                    expected_memoryarena_commit="a" * 40,
                )

    def test_browsecomp_binds_all_frozen_production_inputs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, paths = self._browsecomp_environment(root)
            factory = self._factory(BROWSECOMP_SURFACE)
            sentinel = object()
            provenance = {"mode": "frozen_public_assets"}
            with patch.dict(os.environ, environment, clear=True), patch.object(
                server_factory,
                "attest_browsecomp_search_assets",
                return_value=provenance,
            ) as attester, patch.object(
                server_factory,
                "BrowseCompPlusFactory",
                return_value=factory,
            ) as factory_type, patch.object(
                server_factory,
                "DomainEnvWrapper",
                return_value=sentinel,
            ):
                self.assertIs(server_factory.build_server(), sentinel)

            factory_type.assert_called_once_with(
                ground_truth_path=paths["ground_truth"].resolve(),
                decomposition_path=paths["decomposition"].resolve(),
                memoryarena_root=paths["memoryarena"].resolve(),
                index_path=str(paths["index_pattern"]),
                corpus_path=paths["corpus"].resolve(),
                embedding_model="text-embedding-3-small",
                provider="openai",
                judge_model="judge-model",
                search_asset_provenance=provenance,
                expected_memoryarena_commit="b" * 40,
            )
            attester.assert_called_once_with(
                index_pattern=str(paths["index_pattern"]),
                corpus_path=paths["corpus"].resolve(),
                corpus_manifest_path=paths["corpus_manifest"].resolve(),
                embedding_model="text-embedding-3-small",
            )

    def test_browsecomp_missing_configuration_fails_closed(self):
        required_keys = (
            "AGENTMEMORY_BROWSECOMP_GROUND_TRUTH_PATH",
            "AGENTMEMORY_BROWSECOMP_DECOMPOSITION_PATH",
            "MEMORYARENA_BROWSECOMP_INDEX_PATH",
            "MEMORYARENA_BROWSECOMP_CORPUS_PATH",
            "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST",
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER",
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL",
            "AGENTMEMORY_BROWSECOMP_JUDGE_MODEL",
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
        )
        with tempfile.TemporaryDirectory() as tempdir:
            environment, _ = self._browsecomp_environment(Path(tempdir))
            with patch.object(
                server_factory,
                "attest_browsecomp_search_assets",
                return_value={"mode": "test"},
            ):
                for key in required_keys:
                    broken = dict(environment)
                    del broken[key]
                    with self.subTest(missing=key), patch.dict(
                        os.environ,
                        broken,
                        clear=True,
                    ), self.assertRaises(RuntimeError):
                        server_factory.build_server()

    def test_browsecomp_missing_files_and_id_maps_fail_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, paths = self._browsecomp_environment(root)
            for key in (
                "AGENTMEMORY_BROWSECOMP_GROUND_TRUTH_PATH",
                "AGENTMEMORY_BROWSECOMP_DECOMPOSITION_PATH",
                "MEMORYARENA_BROWSECOMP_CORPUS_PATH",
                "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST",
            ):
                broken = dict(environment)
                broken[key] = str(root / f"missing-{key}")
                with self.subTest(missing_file=key), patch.dict(
                    os.environ,
                    broken,
                    clear=True,
                ), self.assertRaisesRegex(RuntimeError, "Required file"):
                    server_factory.build_server()

            paths["id_map"].unlink()
            with patch.dict(
                os.environ,
                environment,
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "lack id maps"):
                server_factory.build_server()

    def test_browsecomp_partial_shard_set_fails_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, paths = self._browsecomp_environment(root)
            paths["indexes"][3].unlink()
            paths["id_maps"][3].unlink()
            with patch.dict(
                os.environ,
                environment,
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "incomplete or unexpected"):
                server_factory.build_server()

    def test_browsecomp_invalid_api_route_and_empty_index_fail_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, _ = self._browsecomp_environment(root)
            invalid_url = dict(environment)
            invalid_url["OPENAI_BASE_URL"] = "not-a-url"
            with patch.dict(
                os.environ,
                invalid_url,
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "absolute HTTP"):
                server_factory.build_server()

            empty_index = dict(environment)
            empty_index["MEMORYARENA_BROWSECOMP_INDEX_PATH"] = str(
                root / "indexes" / "missing*.index"
            )
            with patch.dict(
                os.environ,
                empty_index,
                clear=True,
            ), self.assertRaisesRegex(RuntimeError, "no files"):
                server_factory.build_server()

    @staticmethod
    def _browsecomp_environment(root: Path):
        ground_truth = root / "ground-truth.jsonl"
        decomposition = root / "decomposition.jsonl"
        corpus = root / "corpus.jsonl"
        memoryarena = root / "MemoryArena"
        indexes = root / "indexes"
        memoryarena.mkdir()
        indexes.mkdir()
        ground_truth.write_text("{}\n", encoding="utf-8")
        decomposition.write_text("{}\n", encoding="utf-8")
        corpus.write_text(
            "".join(
                json.dumps({"docid": str(index), "text": f"doc {index}"}) + "\n"
                for index in range(4)
            ),
            encoding="utf-8",
        )
        corpus_manifest = root / "corpus.manifest.json"
        corpus_manifest.write_text("{}\n", encoding="utf-8")
        index_paths = []
        id_map_paths = []
        for shard in range(4):
            index = indexes / f"shard{shard}.index"
            id_map = indexes / f"shard{shard}_id_map.json"
            index.write_bytes(f"index-{shard}".encode())
            id_map.write_text(
                json.dumps({"ids": [str(shard)]}) + "\n",
                encoding="utf-8",
            )
            index_paths.append(index)
            id_map_paths.append(id_map)
        index_pattern = indexes / "shard*.index"
        environment = {
            "AGENTMEMORY_SURFACE": BROWSECOMP_SURFACE,
            "AGENTMEMORY_BROWSECOMP_GROUND_TRUTH_PATH": str(ground_truth),
            "AGENTMEMORY_BROWSECOMP_DECOMPOSITION_PATH": str(decomposition),
            "MEMORYARENA_ROOT": str(memoryarena),
            "MEMORYARENA_BROWSECOMP_INDEX_PATH": str(index_pattern),
            "MEMORYARENA_BROWSECOMP_CORPUS_PATH": str(corpus),
            "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST": str(corpus_manifest),
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER": "openai",
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL": "text-embedding-3-small",
            "AGENTMEMORY_BROWSECOMP_JUDGE_MODEL": "judge-model",
            "MEMORYARENA_BASE_COMMIT": "b" * 40,
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://api.example/v1",
        }
        return environment, {
            "ground_truth": ground_truth,
            "decomposition": decomposition,
            "corpus": corpus,
            "corpus_manifest": corpus_manifest,
            "memoryarena": memoryarena,
            "index_pattern": index_pattern,
            "indexes": index_paths,
            "id_maps": id_map_paths,
            "id_map": id_map_paths[0],
        }


if __name__ == "__main__":
    unittest.main()
