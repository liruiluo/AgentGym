from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentenv_agentmemory.domains import (
    BROWSECOMP_BM25_INTEGRATION_SURFACE,
    BROWSECOMP_SURFACES,
    FORMAL_REASONING_PAPER_EVAL_SURFACES,
    FORMAL_REASONING_SURFACES,
    FORMAL_REASONING_SURFACES_BY_MODE,
    TRAVEL_SURFACES,
)
from agentenv_agentmemory.env_wrapper import NATIVE_SURFACE
from agentenv_agentmemory.latent_preference_webshop_env import (
    LATENT_PREFERENCE_SURFACE,
)
from agentenv_agentmemory.procedural_webshop_env import PROCEDURAL_SURFACE
from agentenv_agentmemory.domains.formal_reasoning import FROZEN_MEMORYARENA_COMMIT
from agentenv_agentmemory.domains.browsecomp import (
    BROWSECOMP_BM25_INTEGRATION_BACKEND,
    BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
    BROWSECOMP_OPENROUTER_ENDPOINT,
)
from agentenv_agentmemory.domains.travel import TRAVEL_FROZEN_MEMORYARENA_COMMIT
from agentenv_agentmemory.runtime import server_factory


class DomainServerFactoryTest(unittest.TestCase):
    @staticmethod
    def _factory(surface):
        return SimpleNamespace(surface=surface, task_count=1)

    def test_webshop_uses_unchanged_legacy_wrapper(self):
        sentinel = object()
        with (
            patch.dict(
                os.environ,
                {"AGENTMEMORY_SURFACE": NATIVE_SURFACE},
                clear=True,
            ),
            patch.object(
                server_factory,
                "AgentMemoryWrapper",
                return_value=sentinel,
            ) as wrapper,
        ):
            self.assertIs(server_factory.build_server(), sentinel)
        wrapper.assert_called_once_with()

    def test_procedural_webshop_uses_separate_wrapper(self):
        sentinel = object()
        with (
            patch.dict(
                os.environ,
                {"AGENTMEMORY_SURFACE": PROCEDURAL_SURFACE},
                clear=True,
            ),
            patch.object(
                server_factory,
                "ProceduralAgentMemoryWrapper",
                return_value=sentinel,
            ) as wrapper,
        ):
            self.assertIs(server_factory.build_server(), sentinel)
        wrapper.assert_called_once_with()

    def test_latent_preference_uses_separate_wrapper(self):
        sentinel = object()
        with (
            patch.dict(
                os.environ,
                {"AGENTMEMORY_SURFACE": LATENT_PREFERENCE_SURFACE},
                clear=True,
            ),
            patch.object(
                server_factory,
                "LatentPreferenceAgentMemoryWrapper",
                return_value=sentinel,
            ) as wrapper,
        ):
            self.assertIs(server_factory.build_server(), sentinel)
        wrapper.assert_called_once_with()

    def test_travel_surfaces_bind_explicit_contract_modes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks = root / "travel.jsonl"
            database = root / "database"
            memoryarena = root / "MemoryArena"
            tasks.write_text("{}\n", encoding="utf-8")
            database.mkdir()
            memoryarena.mkdir()
            for contract_mode, surface in TRAVEL_SURFACES.items():
                environment = {
                    "AGENTMEMORY_SURFACE": surface,
                    "AGENTMEMORY_TRAVEL_TASKS_PATH": str(tasks),
                    "MEMORYARENA_ROOT": str(memoryarena),
                    "MEMORYARENA_TRAVEL_DATABASE_PATH": str(database),
                    "MEMORYARENA_BASE_COMMIT": TRAVEL_FROZEN_MEMORYARENA_COMMIT,
                }
                factory = self._factory(surface)
                dataset_provenance = object()
                sentinel = object()
                with (
                    self.subTest(contract_mode=contract_mode),
                    patch.dict(os.environ, environment, clear=True),
                    patch.object(
                        server_factory,
                        "TravelPlannerFactory",
                        return_value=factory,
                    ) as factory_type,
                    patch.object(
                        server_factory,
                        "attest_frozen_memoryarena_dataset",
                        return_value=dataset_provenance,
                    ) as dataset_attester,
                    patch.object(
                        server_factory,
                        "DomainEnvWrapper",
                        return_value=sentinel,
                    ) as wrapper_type,
                ):
                    self.assertIs(server_factory.build_server(), sentinel)

                factory_type.assert_called_once_with(
                    contract_mode=contract_mode,
                    tasks_path=tasks.resolve(),
                    dataset_provenance=dataset_provenance,
                    memoryarena_root=memoryarena.resolve(),
                    database_path=database.resolve(),
                    expected_memoryarena_commit=TRAVEL_FROZEN_MEMORYARENA_COMMIT,
                )
                dataset_attester.assert_called_once_with(
                    tasks.resolve(),
                    config="group_travel_planner",
                )
                args, kwargs = wrapper_type.call_args
                self.assertEqual(args, (factory,))
                policy = kwargs["reward_policy"]
                self.assertEqual(policy.first_add, 0.0)
                self.assertEqual(policy.first_later_phase_retrieve, 0.0)
                self.assertEqual(policy.exact_repeat, 0.0)
                self.assertEqual(kwargs["invalid_action_penalty"], 0.0)

    def test_travel_failfast_allows_explicit_reward_overlay(self):
        factory = self._factory(TRAVEL_SURFACES["failfast"])
        sentinel = object()
        environment = {
            "AGENTMEMORY_SURFACE": TRAVEL_SURFACES["failfast"],
            "AGENTMEMORY_FIRST_ADD_REWARD": "0.1",
            "AGENTMEMORY_FIRST_LATER_RETRIEVE_REWARD": "0.1",
            "AGENTMEMORY_EXACT_REPEAT_REWARD": "-0.01",
            "AGENTMEMORY_INVALID_ACTION_REWARD": "-0.01",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                server_factory,
                "build_domain_registry",
                return_value=SimpleNamespace(build=lambda surface: factory),
            ),
            patch.object(
                server_factory,
                "DomainEnvWrapper",
                return_value=sentinel,
            ) as wrapper_type,
        ):
            self.assertIs(server_factory.build_server(), sentinel)
        _, kwargs = wrapper_type.call_args
        self.assertEqual(kwargs["reward_policy"].first_add, 0.1)
        self.assertEqual(kwargs["reward_policy"].exact_repeat, -0.01)
        self.assertEqual(kwargs["invalid_action_penalty"], -0.01)

    def test_travel_paper_eval_rejects_nonzero_reward_overlay(self):
        environment = {
            "AGENTMEMORY_SURFACE": TRAVEL_SURFACES["paper_eval"],
            "AGENTMEMORY_FIRST_ADD_REWARD": "0.1",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                server_factory,
                "build_domain_registry",
                return_value=SimpleNamespace(build=lambda surface: object()),
            ),
            self.assertRaisesRegex(RuntimeError, "Travel paper_eval refuses"),
        ):
            server_factory.build_server()

    def test_travel_rejects_unfrozen_memoryarena_commit(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks = root / "travel.jsonl"
            database = root / "database"
            memoryarena = root / "MemoryArena"
            tasks.write_text("{}\n", encoding="utf-8")
            database.mkdir()
            memoryarena.mkdir()
            environment = {
                "AGENTMEMORY_SURFACE": TRAVEL_SURFACES["failfast"],
                "AGENTMEMORY_TRAVEL_TASKS_PATH": str(tasks),
                "MEMORYARENA_ROOT": str(memoryarena),
                "MEMORYARENA_TRAVEL_DATABASE_PATH": str(database),
                "MEMORYARENA_BASE_COMMIT": "a" * 40,
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(
                    RuntimeError,
                    "frozen Travel commit",
                ),
            ):
                server_factory.build_server()

    def test_unknown_surface_fails_closed(self):
        with patch.dict(
            os.environ,
            {"AGENTMEMORY_SURFACE": "unknown"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "unknown AgentMemoryGym v3 surface"
            ):
                server_factory.build_server()

    def test_math_and_physics_bind_explicit_contract_modes(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks = root / "formal.jsonl"
            memoryarena = root / "MemoryArena"
            tasks.write_text("{}\n", encoding="utf-8")
            memoryarena.mkdir()
            for contract_mode, surfaces in FORMAL_REASONING_SURFACES_BY_MODE.items():
                for domain, surface in surfaces.items():
                    environment = {
                        "AGENTMEMORY_SURFACE": surface,
                        "AGENTMEMORY_FORMAL_REASONING_TASKS_PATH": str(tasks),
                        "MEMORYARENA_ROOT": str(memoryarena),
                        "MEMORYARENA_BASE_COMMIT": FROZEN_MEMORYARENA_COMMIT,
                        "AGENTMEMORY_FORMAL_REASONING_JUDGE_MODEL": "judge-model",
                        "AGENTMEMORY_FORMAL_REASONING_JUDGE_BASE_URL": (
                            "https://judge.example/v1/"
                        ),
                        "AGENTMEMORY_FORMAL_REASONING_JUDGE_TEMPERATURE": "0.25",
                        "AGENTMEMORY_FORMAL_REASONING_JUDGE_MAX_TOKENS": "1234",
                    }
                    factory = self._factory(surface)
                    dataset_provenance = object()
                    sentinel = object()
                    with (
                        self.subTest(domain=domain, contract_mode=contract_mode),
                        patch.dict(
                            os.environ,
                            environment,
                            clear=True,
                        ),
                        patch.object(
                            server_factory,
                            "attest_frozen_memoryarena_dataset",
                            return_value=dataset_provenance,
                        ) as dataset_attester,
                        patch.object(
                            server_factory,
                            "FormalReasoningFactory",
                            return_value=factory,
                        ) as factory_type,
                        patch.object(
                            server_factory,
                            "DomainEnvWrapper",
                            return_value=sentinel,
                        ),
                    ):
                        self.assertIs(server_factory.build_server(), sentinel)

                    factory_type.assert_called_once_with(
                        domain=domain,
                        contract_mode=contract_mode,
                        tasks_path=tasks.resolve(),
                        dataset_provenance=dataset_provenance,
                        memoryarena_root=memoryarena.resolve(),
                        judge_config={
                            "backend": "openai",
                            "model_name": "judge-model",
                            "base_url": "https://judge.example/v1",
                            "temperature": 0.25,
                            "max_tokens": 1234,
                        },
                        expected_memoryarena_commit=FROZEN_MEMORYARENA_COMMIT,
                    )
                    dataset_attester.assert_called_once_with(
                        tasks.resolve(),
                        config=f"formal_reasoning_{domain}",
                    )

    def test_formal_paper_eval_rejects_nonzero_reward_overlay(self):
        environment = {
            "AGENTMEMORY_SURFACE": FORMAL_REASONING_PAPER_EVAL_SURFACES["math"],
            "AGENTMEMORY_FIRST_ADD_REWARD": "0.1",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(
                server_factory,
                "build_domain_registry",
                return_value=SimpleNamespace(build=lambda surface: object()),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "Formal Reasoning paper_eval refuses",
            ),
        ):
            server_factory.build_server()

    def test_formal_reasoning_rejects_unfrozen_memoryarena_commit(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tasks = root / "formal.jsonl"
            memoryarena = root / "MemoryArena"
            tasks.write_text("{}\n", encoding="utf-8")
            memoryarena.mkdir()
            environment = {
                "AGENTMEMORY_SURFACE": FORMAL_REASONING_SURFACES["math"],
                "AGENTMEMORY_FORMAL_REASONING_TASKS_PATH": str(tasks),
                "MEMORYARENA_ROOT": str(memoryarena),
                "MEMORYARENA_BASE_COMMIT": "a" * 40,
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_MODEL": "judge-model",
                "AGENTMEMORY_FORMAL_REASONING_JUDGE_BASE_URL": "https://judge.example/v1",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(
                    RuntimeError,
                    "frozen formal-reasoning commit",
                ),
            ):
                server_factory.build_server()

    def test_browsecomp_binds_all_frozen_production_inputs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, paths = self._browsecomp_environment(root)
            factory = self._factory(BROWSECOMP_SURFACES["paper_eval"])
            sentinel = object()
            dataset_provenance = object()
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(
                    server_factory,
                    "attest_frozen_memoryarena_dataset",
                    return_value=dataset_provenance,
                ) as dataset_attester,
                patch.object(
                    server_factory,
                    "BrowseCompPlusFactory",
                    return_value=factory,
                ) as factory_type,
                patch.object(
                    server_factory,
                    "DomainEnvWrapper",
                    return_value=sentinel,
                ),
            ):
                self.assertIs(server_factory.build_server(), sentinel)

            factory_type.assert_called_once_with(
                contract_mode="paper_eval",
                tasks_path=paths["tasks"].resolve(),
                dataset_provenance=dataset_provenance,
                memoryarena_root=paths["memoryarena"].resolve(),
                index_path=str(paths["index_pattern"]),
                corpus_path=paths["corpus"].resolve(),
                corpus_manifest_path=paths["corpus_manifest"].resolve(),
                embedding_model="text-embedding-3-small",
                provider="openai",
                embedding_endpoint="https://api.example/v1",
                judge_config={
                    "backend": "openai_responses",
                    "model_name": "judge-model",
                    "base_url": "https://api.example/v1",
                    "max_tokens": 8000,
                },
                expected_memoryarena_commit=BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
            )
            dataset_attester.assert_called_once_with(
                paths["tasks"].resolve(),
                config="progressive_search",
            )

    def test_browsecomp_bm25_surface_binds_only_lucene_backend_inputs(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, paths = self._browsecomp_environment(root)
            bm25_index = root / "lucene-index"
            bm25_index.mkdir()
            environment.update(
                {
                    "AGENTMEMORY_SURFACE": BROWSECOMP_BM25_INTEGRATION_SURFACE,
                    "MEMORYARENA_BROWSECOMP_BM25_INDEX_PATH": str(bm25_index),
                }
            )
            for key in (
                "MEMORYARENA_BROWSECOMP_INDEX_PATH",
                "MEMORYARENA_BROWSECOMP_CORPUS_PATH",
                "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST",
                "AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER",
                "AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL",
            ):
                environment.pop(key, None)
            factory = self._factory(BROWSECOMP_BM25_INTEGRATION_SURFACE)
            sentinel = object()
            dataset_provenance = object()
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(
                    server_factory,
                    "attest_frozen_memoryarena_dataset",
                    return_value=dataset_provenance,
                ),
                patch.object(
                    server_factory,
                    "BrowseCompPlusFactory",
                    return_value=factory,
                ) as factory_type,
                patch.object(
                    server_factory,
                    "DomainEnvWrapper",
                    return_value=sentinel,
                ),
            ):
                self.assertIs(server_factory.build_server(), sentinel)

            factory_type.assert_called_once_with(
                contract_mode="failfast",
                tasks_path=paths["tasks"].resolve(),
                dataset_provenance=dataset_provenance,
                memoryarena_root=paths["memoryarena"].resolve(),
                search_backend=BROWSECOMP_BM25_INTEGRATION_BACKEND,
                bm25_index_path=bm25_index.resolve(),
                judge_config={
                    "backend": "openai_responses",
                    "model_name": "judge-model",
                    "base_url": "https://api.example/v1",
                    "max_tokens": 8000,
                },
                expected_memoryarena_commit=BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
            )

    def test_browsecomp_paper_eval_rejects_openrouter_before_asset_attestation(self):
        with tempfile.TemporaryDirectory() as tempdir:
            environment, _ = self._browsecomp_environment(Path(tempdir))
            environment["AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER"] = "openrouter"
            environment["OPENROUTER_API_KEY"] = "test-openrouter-key"
            with (
                patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(RuntimeError, "requires the OpenAI"),
            ):
                server_factory.build_server()

    def test_browsecomp_failfast_binds_named_openrouter_embedding_route(self):
        with tempfile.TemporaryDirectory() as tempdir:
            environment, paths = self._browsecomp_environment(Path(tempdir))
            environment["AGENTMEMORY_SURFACE"] = BROWSECOMP_SURFACES["failfast"]
            environment["AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER"] = "openrouter"
            environment["OPENROUTER_API_KEY"] = "test-openrouter-key"
            factory = self._factory(BROWSECOMP_SURFACES["failfast"])
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(
                    server_factory,
                    "attest_frozen_memoryarena_dataset",
                    return_value=object(),
                ),
                patch.object(
                    server_factory,
                    "BrowseCompPlusFactory",
                    return_value=factory,
                ) as factory_type,
            ):
                server_factory.build_domain_registry().build(
                    BROWSECOMP_SURFACES["failfast"]
                )
            self.assertEqual(
                factory_type.call_args.kwargs["embedding_endpoint"],
                BROWSECOMP_OPENROUTER_ENDPOINT,
            )

    def test_browsecomp_missing_configuration_fails_closed(self):
        required_keys = (
            "AGENTMEMORY_BROWSECOMP_TASKS_PATH",
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
            for key in required_keys:
                broken = dict(environment)
                del broken[key]
                with (
                    self.subTest(missing=key),
                    patch.dict(
                        os.environ,
                        broken,
                        clear=True,
                    ),
                    self.assertRaises(RuntimeError),
                ):
                    server_factory.build_server()

    def test_browsecomp_missing_files_and_id_maps_fail_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, paths = self._browsecomp_environment(root)
            for key in (
                "AGENTMEMORY_BROWSECOMP_TASKS_PATH",
                "MEMORYARENA_BROWSECOMP_CORPUS_PATH",
                "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST",
            ):
                broken = dict(environment)
                broken[key] = str(root / f"missing-{key}")
                with (
                    self.subTest(missing_file=key),
                    patch.dict(
                        os.environ,
                        broken,
                        clear=True,
                    ),
                    self.assertRaisesRegex(RuntimeError, "Required file"),
                ):
                    server_factory.build_server()

            paths["id_map"].unlink()
            with (
                patch.dict(
                    os.environ,
                    environment,
                    clear=True,
                ),
                self.assertRaisesRegex(RuntimeError, "lack id maps"),
            ):
                server_factory.build_server()

    def test_browsecomp_partial_shard_set_fails_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, paths = self._browsecomp_environment(root)
            paths["tasks"].write_text(
                json.dumps(
                    {
                        "id": 0,
                        "questions": ["question"],
                        "answers": ["answer"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            paths["indexes"][3].unlink()
            paths["id_maps"][3].unlink()
            with (
                patch.dict(
                    os.environ,
                    environment,
                    clear=True,
                ),
                patch.object(
                    server_factory,
                    "attest_frozen_memoryarena_dataset",
                    return_value=SimpleNamespace(
                        mode="frozen_public_hf_dataset",
                        record_count=1,
                        phase_count=1,
                    ),
                ),
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "verify_memoryarena_dataset_provenance"
                ),
                patch(
                    "agentenv_agentmemory.domains.browsecomp."
                    "attest_browsecomp_upstream",
                    return_value={"mode": "pinned_pristine_upstream"},
                ),
                self.assertRaisesRegex(RuntimeError, "incomplete or unexpected"),
            ):
                server_factory.build_server()

    def test_browsecomp_invalid_api_route_and_empty_index_fail_closed(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            environment, _ = self._browsecomp_environment(root)
            invalid_url = dict(environment)
            invalid_url["OPENAI_BASE_URL"] = "not-a-url"
            with (
                patch.dict(
                    os.environ,
                    invalid_url,
                    clear=True,
                ),
                self.assertRaisesRegex(RuntimeError, "absolute HTTP"),
            ):
                server_factory.build_server()

            empty_index = dict(environment)
            empty_index["MEMORYARENA_BROWSECOMP_INDEX_PATH"] = str(
                root / "indexes" / "missing*.index"
            )
            with (
                patch.dict(
                    os.environ,
                    empty_index,
                    clear=True,
                ),
                self.assertRaisesRegex(RuntimeError, "no files"),
            ):
                server_factory.build_server()

    def test_browsecomp_rejects_unfrozen_commit_before_asset_use(self):
        with tempfile.TemporaryDirectory() as tempdir:
            environment, _ = self._browsecomp_environment(Path(tempdir))
            environment["MEMORYARENA_BASE_COMMIT"] = "a" * 40
            with (
                patch.dict(
                    os.environ,
                    environment,
                    clear=True,
                ),
                self.assertRaisesRegex(RuntimeError, "frozen Progressive Search"),
            ):
                server_factory.build_server()

    def test_paper_eval_refuses_nonzero_reward_overlay(self):
        with tempfile.TemporaryDirectory() as tempdir:
            environment, _ = self._browsecomp_environment(Path(tempdir))
            environment["AGENTMEMORY_FIRST_ADD_REWARD"] = "0.1"
            factory = self._factory(BROWSECOMP_SURFACES["paper_eval"])
            with (
                patch.dict(
                    os.environ,
                    environment,
                    clear=True,
                ),
                patch.object(
                    server_factory,
                    "attest_frozen_memoryarena_dataset",
                    return_value=object(),
                ),
                patch.object(
                    server_factory,
                    "BrowseCompPlusFactory",
                    return_value=factory,
                ),
                self.assertRaisesRegex(RuntimeError, "evaluation-only"),
            ):
                server_factory.build_server()

    @staticmethod
    def _browsecomp_environment(root: Path):
        tasks = root / "progressive-search.jsonl"
        corpus = root / "corpus.jsonl"
        memoryarena = root / "MemoryArena"
        indexes = root / "indexes"
        memoryarena.mkdir()
        indexes.mkdir()
        tasks.write_text("{}\n", encoding="utf-8")
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
            "AGENTMEMORY_SURFACE": BROWSECOMP_SURFACES["paper_eval"],
            "AGENTMEMORY_BROWSECOMP_TASKS_PATH": str(tasks),
            "MEMORYARENA_ROOT": str(memoryarena),
            "MEMORYARENA_BROWSECOMP_INDEX_PATH": str(index_pattern),
            "MEMORYARENA_BROWSECOMP_CORPUS_PATH": str(corpus),
            "MEMORYARENA_BROWSECOMP_CORPUS_MANIFEST": str(corpus_manifest),
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_PROVIDER": "openai",
            "AGENTMEMORY_BROWSECOMP_EMBEDDING_MODEL": "text-embedding-3-small",
            "AGENTMEMORY_BROWSECOMP_JUDGE_MODEL": "judge-model",
            "MEMORYARENA_BASE_COMMIT": BROWSECOMP_FROZEN_MEMORYARENA_COMMIT,
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://api.example/v1",
        }
        return environment, {
            "tasks": tasks,
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
