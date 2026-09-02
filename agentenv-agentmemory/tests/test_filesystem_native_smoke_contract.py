from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from agentenv_agentmemory.filesystem_webshop_env import (
    ProceduralFilesystemWebShopEnv,
)
from agentenv_agentmemory.native_webshop_backend import NativePage
from agentenv_agentmemory.procedural import (
    NaturalAttributeChainGenerator,
    VerifiedProceduralBundleProvider,
)
from tests.test_procedural_memory_data import FakeNativeBackend, make_fixture_pool
from tests.workspace_test_support import InProcessTestShellSandbox


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke"
    / "smoke_filesystem_memory_webshop_native.py"
)
SPEC = importlib.util.spec_from_file_location(
    "smoke_filesystem_memory_webshop_native",
    SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load smoke script: {SCRIPT}")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class VisibleAsinFakeNativeBackend(FakeNativeBackend):
    def step(self, session_token: str, action: str) -> NativePage:
        page = super().step(session_token, action)
        if not action.startswith("search["):
            return page
        return NativePage(
            observation="Search results: " + " ".join(page.clickables),
            url=page.url,
            has_search_bar=page.has_search_bar,
            clickables=page.clickables,
            purchase=page.purchase,
        )


class FilesystemNativeSmokeContractTests(unittest.TestCase):
    def test_four_intervention_arms_and_cleanup(self) -> None:
        pool, records, prices = make_fixture_pool(1)
        provider = VerifiedProceduralBundleProvider(
            generator=NaturalAttributeChainGenerator(pool=pool, seed=233),
            split="test",
            task_count=2,
        )
        backend = VisibleAsinFakeNativeBackend(records, prices)
        with tempfile.TemporaryDirectory() as temporary:
            env = ProceduralFilesystemWebShopEnv(
                provider=provider,
                backend=backend,
                env_uid="filesystem-native-smoke-contract",
                shell_sandbox=InProcessTestShellSandbox(),
                workspace_root_parent=Path(temporary),
            )
            roots = []
            try:
                _, reset_info = env.reset(data_idx=0)
                self.assertEqual(reset_info["orbit_index"], 0)
                task = SMOKE._task_for_index(provider, 0)
                results = []
                previous_root = None
                for arm in SMOKE.INTERVENTION_ARMS:
                    result, current_root = SMOKE._run_arm(
                        env,
                        backend=backend,
                        provider=provider,
                        task=task,
                        data_index=0,
                        arm=arm,
                        previous_root=previous_root,
                    )
                    results.append(result)
                    if current_root is not None:
                        roots.append(current_root)
                    previous_root = current_root

                self.assertEqual(
                    [item["observed_success"] for item in results],
                    [True, False, False, False],
                )
                self.assertEqual(
                    [item["dependent_done"] for item in results],
                    [False, True, True, True],
                )
                self.assertEqual(
                    results[0]["workspace_event_ops"],
                    ["APPLY_PATCH", "SHELL_COMMAND", "SHELL_COMMAND"],
                )
                self.assertEqual(results[-1]["workspace_event_ops"], [])
                self.assertTrue(
                    all(
                        item["evidence_scope"]
                        == "scripted_runtime_only_not_model_capability"
                        for item in results
                    )
                )
            finally:
                env.close()
                backend.close()

        self.assertTrue(all(not root.exists() for root in roots))
        self.assertEqual(backend.sessions, {})


if __name__ == "__main__":
    unittest.main()
