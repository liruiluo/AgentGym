from __future__ import annotations

import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import agentenv_agentmemory.native_webshop_backend as native_webshop_backend
from agentenv_agentmemory.native_webshop_backend import (
    MemoryArenaNativeWebShopBackend,
    NATIVE_WEBSHOP_UPSTREAM_SCOPE,
    attest_native_webshop_upstream,
)


class _OriginalSignatureSimServer:
    """Test double matching MemoryArena@6cd9de1's public constructor."""

    constructor_called = False

    def __init__(
        self,
        base_url,
        file_path,
        filter_goals=None,
        limit_goals=-1,
        num_products=None,
        human_goals=0,
        show_attrs=False,
    ) -> None:
        self.__class__.constructor_called = True
        raise AssertionError("AMG must not invoke the upstream synthetic-goal constructor")


class NativeWebShopBackendCompatibilityTests(unittest.TestCase):
    def test_start_wraps_pristine_simserver_without_private_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            memoryarena = root / "MemoryArena"
            module_path = (
                memoryarena
                / "env/env_systems/web_shopping_env/runtime/service"
                / "web_agent_site/envs/web_agent_text_env.py"
            )
            module_path.parent.mkdir(parents=True)
            module_path.touch()
            items = root / "items.json"
            attributes = root / "attributes.json"
            items.touch()
            attributes.touch()
            search_root = root / "search_engine"
            (search_root / "indexes-full").mkdir(parents=True)
            java_home = root / "jre11"
            jvm = java_home / "lib/jvm/lib/server/libjvm.so"
            jvm.parent.mkdir(parents=True)
            jvm.touch()

            module = SimpleNamespace(
                __file__=str(module_path),
                SimServer=_OriginalSignatureSimServer,
                load_products=lambda **_kwargs: (
                    [
                        {
                            "asin": "B000000001",
                            "Title": "Fixture product",
                            "category": "fixture",
                            "query": "fixture query",
                            "product_category": "Fixture",
                        }
                    ],
                    {
                        "B000000001": {
                            "asin": "B000000001",
                            "Title": "Fixture product",
                        }
                    },
                    {"B000000001": 12.5},
                    None,
                ),
                init_search_engine=lambda **_kwargs: object(),
                map_action_to_html=lambda *_args, **_kwargs: "html",
                END_BUTTON="Buy Now",
            )
            engine_module = SimpleNamespace(
                SEARCH_ENGINE_ROOT=str(search_root),
                DEFAULT_ATTR_PATH=str(attributes),
            )
            backend = MemoryArenaNativeWebShopBackend(
                memoryarena_root=memoryarena,
                items_file=items,
                attributes_file=attributes,
                search_root=search_root,
                java_home=java_home,
            )

            imported = {
                backend.module_name: module,
                (
                    "env.env_systems.web_shopping_env.runtime.service."
                    "web_agent_site.engine.engine"
                ): engine_module,
            }
            random_state = random.getstate()
            _OriginalSignatureSimServer.constructor_called = False
            try:
                with (
                    mock.patch.object(
                        native_webshop_backend.importlib,
                        "import_module",
                        side_effect=lambda name: imported[name],
                    ),
                    mock.patch.object(
                        native_webshop_backend,
                        "attest_native_webshop_upstream",
                        return_value={
                            "mode": "pinned_pristine_upstream",
                            "memoryarena_commit": "f" * 40,
                        },
                    ),
                ):
                    backend.start()
            finally:
                random.setstate(random_state)

            self.assertFalse(_OriginalSignatureSimServer.constructor_called)
            self.assertIsInstance(backend._server, _OriginalSignatureSimServer)
            self.assertEqual(backend._server.goals[0]["instruction_text"], "")
            self.assertEqual(backend._server.cum_weights, [0.0, 1.0])
            self.assertEqual(
                backend._server.product_item_dict["B000000001"]["Price"],
                "$12.50",
            )
            self.assertEqual(backend.active_session_count(), 0)


class NativeWebShopUpstreamAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.runtime = self.root / NATIVE_WEBSHOP_UPSTREAM_SCOPE
        (self.runtime / "engine").mkdir(parents=True)
        (self.runtime / "envs").mkdir()
        (self.runtime / "engine/engine.py").write_text("VALUE = 1\n")
        (self.runtime / "envs/web_agent_text_env.py").write_text("VALUE = 2\n")
        (self.runtime / "templates").mkdir()
        (self.runtime / "templates/search_page.html").write_text("search\n")
        self._git("init")
        self._git("config", "user.email", "agentmemory-test@example.invalid")
        self._git("config", "user.name", "AgentMemory Test")
        self._git("add", ".")
        self._git("commit", "-m", "fixture")
        self.commit = self._git("rev-parse", "HEAD").strip()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_attests_exact_commit_and_runtime_bundle(self) -> None:
        evidence = attest_native_webshop_upstream(
            self.root,
            expected_commit=self.commit,
        )
        self.assertEqual(evidence["memoryarena_commit"], self.commit)
        self.assertEqual(evidence["source_scope"], NATIVE_WEBSHOP_UPSTREAM_SCOPE)
        self.assertEqual(evidence["source_file_count"], 3)
        self.assertRegex(evidence["source_bundle_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_modified_or_injected_runtime_source(self) -> None:
        tracked = self.runtime / "engine/engine.py"
        tracked.write_text("VALUE = 3\n")
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            attest_native_webshop_upstream(
                self.root,
                expected_commit=self.commit,
            )
        tracked.write_text("VALUE = 1\n")

        injected = self.runtime / "engine/injected.py"
        injected.write_text("VALUE = 4\n")
        with self.assertRaisesRegex(RuntimeError, "not pristine"):
            attest_native_webshop_upstream(
                self.root,
                expected_commit=self.commit,
            )

    def test_allows_unrelated_worktree_changes_but_rejects_wrong_commit(self) -> None:
        note = self.root / "notes/local.txt"
        note.parent.mkdir(parents=True)
        note.write_text("unrelated\n")
        evidence = attest_native_webshop_upstream(
            self.root,
            expected_commit=self.commit,
        )
        self.assertEqual(evidence["memoryarena_commit"], self.commit)
        with self.assertRaisesRegex(RuntimeError, "commit mismatch"):
            attest_native_webshop_upstream(
                self.root,
                expected_commit="0" * 40,
            )


if __name__ == "__main__":
    unittest.main()
