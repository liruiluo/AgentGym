from __future__ import annotations

import hashlib
import importlib
import os
import re
from collections.abc import Mapping
from pathlib import Path

from .backend import FixtureBackend, LiteResearcherBackend, SearchVisitBackend
from .contracts import PRODUCTION_PROTOCOL, EvaluationArm, ProtocolContract
from .dataset import GaiaTextDataset
from .submission import SubmissionStore
from .wrapper import GaiaTextEpisodeManager, WorkspaceFactory


def build_manager_from_environment(
    *,
    contract: ProtocolContract = PRODUCTION_PROTOCOL,
    environment: Mapping[str, str] | None = None,
) -> GaiaTextEpisodeManager:
    values = os.environ if environment is None else environment
    _reject_private_environment(values)
    arm = EvaluationArm(_required_text(values, "GAIA_TEXT_ARM"))
    manifest = _required_file(values, "GAIA_TEXT_MANIFEST")
    questions = _required_file(values, "GAIA_TEXT_QUESTIONS")
    backend_kind = _required_text(values, "GAIA_TEXT_BACKEND")
    backend_input = _backend_input(values, backend_kind)
    resolved_inputs = {manifest.resolve(), questions.resolve(), backend_input.resolve()}
    prediction_candidate = Path(
        _required_text(values, "GAIA_TEXT_PREDICTIONS")
    ).expanduser()
    if len(resolved_inputs) != 3 or prediction_candidate.resolve() in resolved_inputs:
        raise RuntimeError(
            "GAIA-Text manifest, questions, backend input, and predictions must be distinct"
        )
    predictions = _output_path(values, "GAIA_TEXT_PREDICTIONS")

    dataset = GaiaTextDataset.load(
        manifest,
        questions,
        expected_questions_sha256=_required_sha256(
            values, "GAIA_TEXT_QUESTIONS_SHA256"
        ),
        contract=contract,
    )
    backend = _build_backend(values, backend_kind, backend_input)
    submissions = SubmissionStore(dataset.task_ids, predictions)
    workspace_factory: WorkspaceFactory | None = None
    workspace_runtime = None
    if arm is EvaluationArm.AMG_MEMORY:
        workspace_factory, workspace_runtime = _memory_workspace_factory(values)

    return GaiaTextEpisodeManager(
        dataset,
        backend,
        submissions,
        arm=arm,
        workspace_factory=workspace_factory,
        workspace_runtime=workspace_runtime,
        max_policy_steps=_positive_integer(values, "GAIA_TEXT_MAX_POLICY_STEPS", 40),
    )


def _backend_input(
    environment: Mapping[str, str],
    backend_kind: str,
) -> Path:
    if backend_kind == "fixture":
        if any(
            value
            for name, value in environment.items()
            if name.startswith("GAIA_TEXT_LITERESEARCHER_")
            or name
            in {
                "GAIA_TEXT_SEARCH_RESULT_LIMIT",
                "GAIA_TEXT_VISIT_PAGE_LIMIT",
            }
        ):
            raise RuntimeError(
                "GAIA_TEXT_BACKEND fixture must not mix production backend inputs"
            )
        return _required_file(environment, "GAIA_TEXT_BACKEND_ASSET")
    if backend_kind == "production":
        if environment.get("GAIA_TEXT_BACKEND_ASSET") or environment.get(
            "GAIA_TEXT_BACKEND_SHA256"
        ):
            raise RuntimeError(
                "GAIA_TEXT_BACKEND production must not mix fixture backend inputs"
            )
        return _required_file(
            environment, "GAIA_TEXT_LITERESEARCHER_CERTIFICATE"
        )
    raise RuntimeError("GAIA_TEXT_BACKEND must be fixture or production")


def _build_backend(
    environment: Mapping[str, str],
    backend_kind: str,
    backend_input: Path,
) -> SearchVisitBackend:
    page_chars = _bounded_integer(
        environment,
        "GAIA_TEXT_VISIT_PAGE_CHARS",
        8192,
        minimum=1,
        maximum=1_000_000,
    )
    if backend_kind == "fixture":
        return FixtureBackend.load(
            backend_input,
            _required_sha256(environment, "GAIA_TEXT_BACKEND_SHA256"),
            page_chars=page_chars,
        )
    return LiteResearcherBackend.load(
        backend_input,
        _required_sha256(
            environment, "GAIA_TEXT_LITERESEARCHER_CERTIFICATE_SHA256"
        ),
        base_url=_required_text(
            environment, "GAIA_TEXT_LITERESEARCHER_BASE_URL"
        ),
        connect_timeout_ms=_bounded_integer(
            environment,
            "GAIA_TEXT_LITERESEARCHER_CONNECT_TIMEOUT_MS",
            2_000,
            minimum=1,
            maximum=300_000,
        ),
        read_timeout_ms=_bounded_integer(
            environment,
            "GAIA_TEXT_LITERESEARCHER_READ_TIMEOUT_MS",
            30_000,
            minimum=1,
            maximum=300_000,
        ),
        retry_count=_bounded_integer(
            environment,
            "GAIA_TEXT_LITERESEARCHER_RETRY_COUNT",
            2,
            minimum=0,
            maximum=10,
        ),
        retry_backoff_ms=_bounded_integer(
            environment,
            "GAIA_TEXT_LITERESEARCHER_RETRY_BACKOFF_MS",
            100,
            minimum=0,
            maximum=60_000,
        ),
        result_limit=_bounded_integer(
            environment,
            "GAIA_TEXT_SEARCH_RESULT_LIMIT",
            10,
            minimum=1,
            maximum=50,
        ),
        page_chars=page_chars,
        page_limit=_bounded_integer(
            environment,
            "GAIA_TEXT_VISIT_PAGE_LIMIT",
            256,
            minimum=1,
            maximum=10_000,
        ),
    )


def _memory_workspace_factory(
    environment: Mapping[str, str],
) -> tuple[WorkspaceFactory, dict[str, object]]:
    # Keep native construction independent of the optional memory package.
    persistent_module = importlib.import_module(
        "agentenv_agentmemory.persistent_workspace"
    )
    sandbox_module = importlib.import_module("agentenv_agentmemory.workspace_sandbox")
    PersistentWorkspace = persistent_module.PersistentWorkspace
    WorkspaceLimits = persistent_module.WorkspaceLimits
    LinuxNamespaceShellSandbox = sandbox_module.LinuxNamespaceShellSandbox

    workspace_root = _required_directory(environment, "GAIA_TEXT_WORKSPACE_ROOT")
    rg_binary = _required_file(environment, "GAIA_TEXT_RG_BINARY")
    expected_rg_sha256 = _required_sha256(environment, "GAIA_TEXT_RG_SHA256")
    observed_rg_sha256 = hashlib.sha256(rg_binary.read_bytes()).hexdigest()
    if observed_rg_sha256 != expected_rg_sha256:
        raise RuntimeError(
            "GAIA_TEXT_RG_SHA256 does not match GAIA_TEXT_RG_BINARY: "
            f"expected {expected_rg_sha256}, got {observed_rg_sha256}"
        )
    limits = WorkspaceLimits()
    sandbox = LinuxNamespaceShellSandbox.from_environment(
        limits=limits.shell_limits(),
        rg_binary=rg_binary,
        expected_rg_sha256=expected_rg_sha256,
    )

    def factory(env_id: int, task_id: str, episode_index: int):
        del task_id
        return PersistentWorkspace(
            workspace_id=f"gaia-text-env-{env_id}-episode-{episode_index}",
            shell_sandbox=sandbox,
            root_parent=workspace_root,
            limits=limits,
        )

    return factory, {
        "sandbox": dict(sandbox.metadata),
        "limits": limits.as_metadata(),
        "host_paths_exposed_to_policy": False,
    }


def launch() -> None:
    import uvicorn

    from .server import create_app

    manager = build_manager_from_environment()
    uvicorn.run(
        create_app(manager),
        host="127.0.0.1",
        port=_positive_integer(os.environ, "GAIA_TEXT_PORT", 8000),
        log_level=os.environ.get("GAIA_TEXT_LOG_LEVEL", "info"),
    )


def _reject_private_environment(environment: Mapping[str, str]) -> None:
    forbidden = sorted(
        name
        for name, value in environment.items()
        if value
        and "GAIA" in name.upper()
        and any(token in name.upper() for token in ("GOLD", "SCORER"))
    )
    if forbidden:
        raise RuntimeError(
            "GAIA-Text inference refuses gold/scorer environment variables: "
            + ", ".join(forbidden)
        )


def _required_text(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value.strip()


def _required_file(environment: Mapping[str, str], name: str) -> Path:
    path = Path(_required_text(environment, name)).expanduser()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must name a real file")
    return path


def _required_directory(environment: Mapping[str, str], name: str) -> Path:
    path = Path(_required_text(environment, name)).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{name} must name a real directory")
    return path


def _output_path(environment: Mapping[str, str], name: str) -> Path:
    path = Path(_required_text(environment, name)).expanduser()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"{name} must name a fresh output path")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(f"{name} parent must be a real existing directory")
    return path


def _required_sha256(environment: Mapping[str, str], name: str) -> str:
    value = _required_text(environment, name)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _bounded_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw = environment.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be in [{minimum}, {maximum}]")
    return value
