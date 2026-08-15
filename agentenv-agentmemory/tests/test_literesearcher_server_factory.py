from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agentenv_agentmemory.literesearcher import (
    LITERESEARCHER_FULLPOOL_SURFACE,
    UPSTREAM_LLM_JUDGE_CONTRACT,
)
from agentenv_agentmemory.runtime.server_factory import _build_literesearcher_wrapper


class _TaskSource:
    def public_metadata(self):
        return {"task_count": 1}

    def tasks_for_split(self, split):
        if split != "train":
            raise ValueError(split)
        return (object(),)


class _Judge:
    contract_id = UPSTREAM_LLM_JUDGE_CONTRACT

    def metadata(self):
        return {
            "contract": self.contract_id,
            "primary": "test",
            "fallback": "upstream_em_v1",
        }


class LiteResearcherServerFactoryTests(unittest.TestCase):
    def test_fullpool_factory_instantiates_upstream_judge_from_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rg = root / "rg"
            manifest = root / "manifest.json"
            rows = root / "rows.jsonl"
            source = root / "source"
            database = root / "corpus.sqlite"
            for path in (rg, manifest, rows, database):
                path.write_text("fixture", encoding="utf-8")
            source.mkdir()
            environment = {
                "AGENTMEMORY_WORKSPACE_RG_BINARY": str(rg),
                "AGENTMEMORY_WORKSPACE_RG_SHA256": "a" * 64,
                "AGENTMEMORY_LITERESEARCHER_SPLIT": "train",
                "AGENTMEMORY_LITERESEARCHER_FULL_POOL_MANIFEST": str(manifest),
                "AGENTMEMORY_LITERESEARCHER_FULL_POOL_ROWS": str(rows),
                "AGENTMEMORY_LITERESEARCHER_SOURCE_ROOT": str(source),
                "AGENTMEMORY_LITERESEARCHER_FTS_DATABASE": str(database),
                "AGENTMEMORY_LITERESEARCHER_JUDGE_API_BASE": "http://judge/v1",
                "AGENTMEMORY_LITERESEARCHER_JUDGE_MODEL": "qwen-judge",
                "AGENTMEMORY_LITERESEARCHER_JUDGE_API_KEY": "private-key",
                "AGENTMEMORY_LITERESEARCHER_JUDGE_TIMEOUT_SECONDS": "45.5",
                "AGENTMEMORY_LITERESEARCHER_JUDGE_MAX_RETRIES": "4",
            }
            task_source = _TaskSource()
            backend = Mock(tasks_source=task_source, split="train")
            judge = _Judge()
            sandbox = Mock(metadata={"kind": "test"})
            with (
                patch.dict(os.environ, environment, clear=True),
                patch(
                    "agentenv_agentmemory.runtime.server_factory."
                    "LinuxNamespaceShellSandbox.from_environment",
                    return_value=sandbox,
                ),
                patch(
                    "agentenv_agentmemory.runtime.server_factory.load_full_pool",
                    return_value=task_source,
                ),
                patch(
                    "agentenv_agentmemory.runtime.server_factory."
                    "SQLiteFTSLiteResearchBackend",
                    return_value=backend,
                ),
                patch(
                    "agentenv_agentmemory.runtime.server_factory."
                    "UpstreamCompatibleLLMJudge",
                    return_value=judge,
                ) as judge_type,
            ):
                wrapper = _build_literesearcher_wrapper(
                    LITERESEARCHER_FULLPOOL_SURFACE
                )
        self.assertIs(wrapper.judge, judge)
        judge_type.assert_called_once_with(
            api_base="http://judge/v1",
            model="qwen-judge",
            api_key="private-key",
            timeout_seconds=45.5,
            max_retries=4,
        )

    def test_fullpool_factory_selects_tantivy_backend_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rg = root / "rg"
            manifest = root / "manifest.json"
            rows = root / "rows.jsonl"
            source = root / "source"
            database = root / "corpus.sqlite"
            index = root / "tantivy"
            for path in (rg, manifest, rows, database):
                path.write_text("fixture", encoding="utf-8")
            source.mkdir()
            index.mkdir()
            environment = {
                "AGENTMEMORY_WORKSPACE_RG_BINARY": str(rg),
                "AGENTMEMORY_WORKSPACE_RG_SHA256": "a" * 64,
                "AGENTMEMORY_LITERESEARCHER_SPLIT": "train",
                "AGENTMEMORY_LITERESEARCHER_FULL_POOL_MANIFEST": str(manifest),
                "AGENTMEMORY_LITERESEARCHER_FULL_POOL_ROWS": str(rows),
                "AGENTMEMORY_LITERESEARCHER_SOURCE_ROOT": str(source),
                "AGENTMEMORY_LITERESEARCHER_FTS_DATABASE": str(database),
                "AGENTMEMORY_LITERESEARCHER_SEARCH_BACKEND": "tantivy",
                "AGENTMEMORY_LITERESEARCHER_TANTIVY_INDEX": str(index),
                "AGENTMEMORY_LITERESEARCHER_JUDGE_API_BASE": "http://judge/v1",
                "AGENTMEMORY_LITERESEARCHER_JUDGE_MODEL": "kimi-judge",
            }
            task_source = _TaskSource()
            backend = Mock(tasks_source=task_source, split="train")
            judge = _Judge()
            sandbox = Mock(metadata={"kind": "test"})
            with (
                patch.dict(os.environ, environment, clear=True),
                patch(
                    "agentenv_agentmemory.runtime.server_factory."
                    "LinuxNamespaceShellSandbox.from_environment",
                    return_value=sandbox,
                ),
                patch(
                    "agentenv_agentmemory.runtime.server_factory.load_full_pool",
                    return_value=task_source,
                ),
                patch(
                    "agentenv_agentmemory.runtime.server_factory."
                    "TantivyLiteResearchBackend",
                    return_value=backend,
                ) as backend_type,
                patch(
                    "agentenv_agentmemory.runtime.server_factory."
                    "UpstreamCompatibleLLMJudge",
                    return_value=judge,
                ),
            ):
                wrapper = _build_literesearcher_wrapper(
                    LITERESEARCHER_FULLPOOL_SURFACE
                )
        self.assertIs(wrapper.backend, backend)
        backend_type.assert_called_once_with(
            task_source,
            database.resolve(),
            index.resolve(),
            top_k=5,
        )


if __name__ == "__main__":
    unittest.main()
