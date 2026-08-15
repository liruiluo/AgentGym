from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from agentenv_gaia_text.contracts import EvaluationArm, ProtocolContract
from agentenv_gaia_text.launch import build_manager_from_environment, launch
from support import protocol_kwargs, write_runtime_fixture


def _environment(runtime, arm: str) -> dict[str, str]:
    return {
        "GAIA_TEXT_ARM": arm,
        "GAIA_TEXT_BACKEND": "fixture",
        "GAIA_TEXT_MANIFEST": str(runtime.manifest),
        "GAIA_TEXT_QUESTIONS": str(runtime.questions),
        "GAIA_TEXT_QUESTIONS_SHA256": runtime.questions_sha256,
        "GAIA_TEXT_BACKEND_ASSET": str(runtime.backend),
        "GAIA_TEXT_BACKEND_SHA256": runtime.backend_sha256,
        "GAIA_TEXT_PREDICTIONS": str(runtime.predictions),
    }


def _contract(runtime) -> ProtocolContract:
    return ProtocolContract(**protocol_kwargs(runtime.rows))


@pytest.mark.parametrize(
    "arm",
    [EvaluationArm.NATIVE, EvaluationArm.AMG_COMPACTION_ONLY],
)
def test_memory_disabled_runtime_never_imports_or_constructs_memory(
    arm: EvaluationArm,
    tmp_path: Path,
) -> None:
    runtime = write_runtime_fixture(tmp_path)
    environment = _environment(runtime, arm.value)
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("agentenv_agentmemory"):
            raise AssertionError("memory-disabled arm imported the memory runtime")
        return real_import(name, *args, **kwargs)

    with (
        patch.dict(os.environ, environment, clear=True),
        patch("builtins.__import__", side_effect=guarded_import),
    ):
        manager = build_manager_from_environment(contract=_contract(runtime))
    assert manager.metadata()["arm"] == arm.value
    assert manager.metadata()["workspace_available"] is False
    assert manager.metadata()["active_workspace_count"] == 0
    assert manager.metadata()["compaction_available"] is (
        arm is EvaluationArm.AMG_COMPACTION_ONLY
    )
    assert str(tmp_path) not in str(manager.metadata())


@pytest.mark.parametrize(
    "arm",
    [EvaluationArm.NATIVE, EvaluationArm.AMG_COMPACTION_ONLY],
)
@pytest.mark.parametrize(
    "name",
    [
        "GAIA_TEXT_WORKSPACE_ROOT",
        "GAIA_TEXT_RG_BINARY",
        "GAIA_TEXT_RG_SHA256",
    ],
)
def test_memory_disabled_runtime_rejects_memory_environment_bundle(
    arm: EvaluationArm,
    name: str,
    tmp_path: Path,
) -> None:
    runtime = write_runtime_fixture(tmp_path)
    environment = {
        **_environment(runtime, arm.value),
        name: str(tmp_path / "hidden-memory-path"),
    }
    with pytest.raises(RuntimeError, match="external-memory environment") as error:
        build_manager_from_environment(
            contract=_contract(runtime),
            environment=environment,
        )
    assert str(tmp_path / "hidden-memory-path") not in str(error.value)


@pytest.mark.parametrize("question_sha256", [None, "not-a-digest"])
def test_runtime_requires_a_pinned_question_file_hash(
    question_sha256: str | None,
    tmp_path: Path,
) -> None:
    runtime = write_runtime_fixture(tmp_path)
    environment = _environment(runtime, "native")
    if question_sha256 is None:
        del environment["GAIA_TEXT_QUESTIONS_SHA256"]
    else:
        environment["GAIA_TEXT_QUESTIONS_SHA256"] = question_sha256
    with (
        patch.dict(os.environ, environment, clear=True),
        pytest.raises(RuntimeError, match="GAIA_TEXT_QUESTIONS_SHA256"),
    ):
        build_manager_from_environment(contract=_contract(runtime))


@pytest.mark.parametrize(
    "name",
    ["GAIA_GOLD_INPUT", "GAIA_SCORER", "GAIA_TEXT_GOLD_PATH", "GAIA_TEXT_SCORER_ONLY"],
)
def test_runtime_rejects_gold_and_scorer_environment(name: str, tmp_path: Path) -> None:
    runtime = write_runtime_fixture(tmp_path)
    environment = {**_environment(runtime, "native"), name: "/private/forbidden"}
    with (
        patch.dict(os.environ, environment, clear=True),
        pytest.raises(RuntimeError, match="gold/scorer"),
    ):
        build_manager_from_environment(contract=_contract(runtime))


def test_memory_runtime_lazily_builds_formal_workspace_factory(tmp_path: Path) -> None:
    runtime = write_runtime_fixture(tmp_path / "runtime")
    workspace_root = tmp_path / "workspaces"
    workspace_root.mkdir()
    rg = tmp_path / "rg"
    rg.write_bytes(b"pinned-rg")
    environment = {
        **_environment(runtime, "amg_memory"),
        "GAIA_TEXT_WORKSPACE_ROOT": str(workspace_root),
        "GAIA_TEXT_RG_BINARY": str(rg),
        "GAIA_TEXT_RG_SHA256": hashlib.sha256(rg.read_bytes()).hexdigest(),
    }

    class FakeLimits:
        def shell_limits(self):
            return "limits"

        def as_metadata(self):
            return {"test": True}

    class FakePersistentWorkspace:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def reset_episode(self, episode_id: str, *, enabled: bool = True):
            self.episode_id = episode_id

        def apply(self, action: str, *, env_step: int, phase_index: int):
            return SimpleNamespace(message="ok", op="SHELL_COMMAND")

        def close(self):
            pass

    fake_sandbox = SimpleNamespace(metadata={"contract": "formal-test"})
    persistent_module = SimpleNamespace(
        PersistentWorkspace=FakePersistentWorkspace,
        WorkspaceLimits=FakeLimits,
    )
    sandbox_module = SimpleNamespace(
        LinuxNamespaceShellSandbox=SimpleNamespace(
            from_environment=lambda **kwargs: fake_sandbox
        )
    )

    def import_module(name: str):
        if name == "agentenv_agentmemory.persistent_workspace":
            return persistent_module
        if name == "agentenv_agentmemory.workspace_sandbox":
            return sandbox_module
        raise AssertionError(name)

    with (
        patch.dict(os.environ, environment, clear=True),
        patch(
            "agentenv_gaia_text.launch.importlib.import_module",
            side_effect=import_module,
        ),
    ):
        manager = build_manager_from_environment(contract=_contract(runtime))
        env_id = manager.create()["id"]
        manager.reset(env_id, 0)
    metadata = manager.metadata()
    assert metadata["arm"] == "amg_memory"
    assert metadata["workspace_available"] is True
    assert metadata["workspace_runtime"] == {
        "sandbox": {"contract": "formal-test"},
        "limits": {"test": True},
        "host_paths_exposed_to_policy": False,
    }


def test_runtime_refuses_aliasing_input_output_paths(tmp_path: Path) -> None:
    runtime = write_runtime_fixture(tmp_path)
    environment = _environment(runtime, "native")
    environment["GAIA_TEXT_PREDICTIONS"] = str(runtime.questions)
    with (
        patch.dict(os.environ, environment, clear=True),
        pytest.raises(RuntimeError, match="distinct"),
    ):
        build_manager_from_environment(contract=_contract(runtime))


def test_launcher_always_binds_loopback_even_if_host_environment_is_public() -> None:
    fake_manager = object()
    with (
        patch.dict(
            os.environ,
            {"GAIA_TEXT_HOST": "0.0.0.0", "GAIA_TEXT_PORT": "8123"},
            clear=True,
        ),
        patch(
            "agentenv_gaia_text.launch.build_manager_from_environment",
            return_value=fake_manager,
        ),
        patch("agentenv_gaia_text.server.create_app", return_value="app"),
        patch("uvicorn.run") as run,
    ):
        launch()

    run.assert_called_once_with(
        "app",
        host="127.0.0.1",
        port=8123,
        log_level="info",
    )
