from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .workspace_patch import (
    WorkspacePatchError,
    apply_workspace_patch_transaction,
    parse_workspace_patch,
    replace_workspace_directory,
)
from .workspace_sandbox import (
    ShellSandbox,
    ShellSandboxError,
    ShellSandboxLimits,
)


WORKSPACE_TOOL_NAMES = ("shell_command", "apply_patch")
WORKSPACE_TOOL_OPS = ("SHELL_COMMAND", "APPLY_PATCH")
WORKSPACE_TOOL_CONTRACT = "codex_shell_command_apply_patch_v1"
WORKSPACE_STATE_SCHEMA_V1 = "agentmemory_workspace_transfer_state_v1"
WORKSPACE_STATE_SCHEMA = "agentmemory_workspace_transfer_state_v2"
WORKSPACE_SEED_MANIFEST_SCHEMA = "agentmemory_workspace_seed_manifest_v1"
WORKSPACE_CAUSAL_ARMS = (
    "correct",
    "blank",
    "swapped",
    "stale",
    "no_workspace",
)
_SHELL_ACTION_RE = re.compile(r"\Ashell_command\s+(\{.*\})\Z", re.DOTALL)
_APPLY_PATCH_PREFIX = "apply_patch\n"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class WorkspaceActionError(ValueError):
    pass


@dataclass(frozen=True)
class WorkspaceLimits:
    max_path_chars: int = 240
    max_files: int = 64
    max_directories: int = 64
    max_file_bytes: int = 64 * 1024
    max_total_bytes: int = 512 * 1024
    max_command_chars: int = 32 * 1024
    max_patch_bytes: int = 256 * 1024
    default_timeout_ms: int = 10_000
    max_timeout_ms: int = 30_000
    cpu_seconds: int = 10
    address_space_bytes: int = 1024 * 1024 * 1024
    max_processes: int = 32
    max_open_files: int = 64
    stdout_bytes: int = 16 * 1024
    stderr_bytes: int = 16 * 1024
    tmp_bytes: int = 64 * 1024 * 1024
    tmp_inodes: int = 512

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ValueError("workspace limits must be positive integers")
        if self.max_file_bytes > self.max_total_bytes:
            raise ValueError("max_file_bytes cannot exceed max_total_bytes")
        if self.default_timeout_ms > self.max_timeout_ms:
            raise ValueError("default_timeout_ms cannot exceed max_timeout_ms")

    def shell_limits(self) -> ShellSandboxLimits:
        return ShellSandboxLimits(
            workspace_bytes=self.max_total_bytes,
            workspace_inodes=self.max_files + self.max_directories + 1,
            max_files=self.max_files,
            max_directories=self.max_directories,
            max_file_bytes=self.max_file_bytes,
            max_path_chars=self.max_path_chars,
            default_timeout_ms=self.default_timeout_ms,
            max_timeout_ms=self.max_timeout_ms,
            cpu_seconds=self.cpu_seconds,
            address_space_bytes=self.address_space_bytes,
            max_processes=self.max_processes,
            max_open_files=self.max_open_files,
            stdout_bytes=self.stdout_bytes,
            stderr_bytes=self.stderr_bytes,
            tmp_bytes=self.tmp_bytes,
            tmp_inodes=self.tmp_inodes,
        )

    def as_metadata(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class WorkspaceAction:
    tool_name: str
    arguments: Mapping[str, Any] | None
    tool_input: str


@dataclass(frozen=True)
class WorkspaceActionResult:
    message: str
    op: str
    tool_op: dict[str, Any]
    workspace_diff: dict[str, Any]


@dataclass
class PersistentWorkspace:
    """Episode-scoped workspace behind the two canonical Codex tools."""

    workspace_id: str
    shell_sandbox: ShellSandbox
    root_parent: Path | None = None
    limits: WorkspaceLimits = field(default_factory=WorkspaceLimits)
    _root: Path | None = field(default=None, init=False, repr=False)
    _episode_id: str | None = field(default=None, init=False, repr=False)
    _enabled: bool = field(default=True, init=False, repr=False)
    _audit_events: list[dict[str, Any]] = field(default_factory=list, init=False)
    _event_counter: int = field(default=0, init=False)
    _causal_arm: str | None = field(default=None, init=False, repr=False)
    _control_event: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _seed_manifest: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, str) or not self.workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        if self.shell_sandbox is None:
            raise ValueError("the Codex workspace requires an isolated shell sandbox")
        if self.root_parent is not None:
            parent = Path(self.root_parent).expanduser().resolve()
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not parent.is_dir() or parent.is_symlink():
                raise ValueError("root_parent must be a real directory")
            self.root_parent = parent

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def host_root(self) -> Path:
        if not self._enabled:
            raise RuntimeError("this intervention has no workspace")
        if self._root is None:
            raise RuntimeError("workspace must be reset before use")
        return self._root

    @property
    def episode_id(self) -> str:
        if self._episode_id is None:
            raise RuntimeError("workspace must be reset before use")
        return self._episode_id

    @property
    def audit_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(_deepcopy_json(event) for event in self._audit_events)

    @property
    def causal_arm(self) -> str | None:
        return self._causal_arm

    @property
    def control_event(self) -> dict[str, Any] | None:
        if self._control_event is None:
            return None
        return _deepcopy_json(self._control_event)

    @property
    def seed_manifest(self) -> dict[str, Any] | None:
        if self._seed_manifest is None:
            return None
        return _deepcopy_json(self._seed_manifest)

    @property
    def provenance_summary(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        seeded = {
            item["path"]: item
            for item in (self._seed_manifest or {}).get("files", [])
        }
        current = {item["path"]: item for item in snapshot["files"]}
        unchanged_seed_paths = sorted(
            path
            for path in seeded.keys() & current.keys()
            if seeded[path]["sha256"] == current[path]["sha256"]
        )
        modified_seed_paths = sorted(
            path
            for path in seeded.keys() & current.keys()
            if seeded[path]["sha256"] != current[path]["sha256"]
        )
        deleted_seed_paths = sorted(seeded.keys() - current.keys())
        policy_created_paths = sorted(current.keys() - seeded.keys())
        policy_authored = bool(
            self._audit_events
            or modified_seed_paths
            or deleted_seed_paths
            or policy_created_paths
        )
        return {
            "schema": "agentmemory_workspace_provenance_summary_v1",
            "contains_harness_seed": self._seed_manifest is not None,
            "seed_manifest_sha256": (
                None
                if self._seed_manifest is None
                else self._seed_manifest["manifest_sha256"]
            ),
            "seed_file_count": len(seeded),
            "unchanged_seed_paths": unchanged_seed_paths,
            "modified_seed_paths": modified_seed_paths,
            "deleted_seed_paths": deleted_seed_paths,
            "policy_created_paths": policy_created_paths,
            "policy_action_count": len(self._audit_events),
            "policy_authored": policy_authored,
        }

    def reset_episode(self, episode_id: str, *, enabled: bool = True) -> None:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a non-empty string")
        if type(enabled) is not bool:
            raise ValueError("workspace enabled flag must be boolean")
        self.close()
        self._episode_id = episode_id.strip()
        self._enabled = enabled
        self._audit_events = []
        self._event_counter = 0
        self._causal_arm = None
        self._control_event = None
        self._seed_manifest = None
        if not enabled:
            return
        prefix = "agentmemory-" + _safe_component(self.workspace_id) + "-"
        root = Path(
            tempfile.mkdtemp(
                prefix=prefix,
                dir=None if self.root_parent is None else str(self.root_parent),
            )
        )
        os.chmod(root, 0o700)
        self._root = root.resolve()

    def close(self) -> None:
        root = self._root
        self._root = None
        self._episode_id = None
        self._enabled = True
        self._audit_events = []
        self._event_counter = 0
        self._causal_arm = None
        self._control_event = None
        self._seed_manifest = None
        if root is not None and root.exists():
            shutil.rmtree(root)

    def install_seed_files(
        self,
        files: Mapping[str, str | bytes],
        *,
        source_label: str,
    ) -> dict[str, Any]:
        """Install harness-authored ordinary files before the policy acts."""

        if not self.enabled:
            raise WorkspaceActionError("cannot seed an unavailable workspace")
        if self._causal_arm is not None:
            raise WorkspaceActionError("cannot seed a workspace after an intervention")
        if self._seed_manifest is not None:
            raise WorkspaceActionError("workspace seed files may be installed only once")
        if self._audit_events:
            raise WorkspaceActionError("workspace seed files must precede policy actions")
        if self.snapshot()["file_count"] or self.snapshot()["directory_count"]:
            raise WorkspaceActionError("workspace must be empty before seed files are installed")
        source_label = _require_nonempty_string(source_label, "source_label")
        if not isinstance(files, Mapping) or not files:
            raise WorkspaceActionError("workspace seed files must be a non-empty mapping")

        decoded: list[tuple[str, bytes]] = []
        for raw_path, raw_data in files.items():
            path = _normalize_relative_path(raw_path, limits=self.limits)
            if isinstance(raw_data, str):
                try:
                    data = raw_data.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise WorkspaceActionError(
                        "workspace seed text must be valid UTF-8"
                    ) from exc
            elif isinstance(raw_data, bytes):
                data = raw_data
            else:
                raise WorkspaceActionError(
                    "workspace seed file content must be text or bytes"
                )
            if len(data) > self.limits.max_file_bytes:
                raise WorkspaceActionError(
                    f"seed file cannot exceed {self.limits.max_file_bytes} bytes"
                )
            decoded.append((path, data))
        decoded.sort(key=lambda item: item[0])
        if len({path for path, _ in decoded}) != len(decoded):
            raise WorkspaceActionError("workspace seed paths must be unique")

        root = self.host_root
        staging = Path(
            tempfile.mkdtemp(prefix=".agentmemory-seed-", dir=root.parent)
        )
        candidate = staging / "workspace"
        try:
            candidate.mkdir(mode=0o700)
            for relative, data in decoded:
                path = candidate / relative
                path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                _write_private_regular_file(path, data)
            snapshot = self._snapshot_root(candidate)
            replace_workspace_directory(root, candidate)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        manifest_payload = {
            "schema": WORKSPACE_SEED_MANIFEST_SCHEMA,
            "source_label": source_label,
            "seed_tree_sha256": snapshot["tree_sha256"],
            "files": [
                {
                    "path": path,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for path, data in decoded
            ],
        }
        self._seed_manifest = {
            **manifest_payload,
            "manifest_sha256": _canonical_sha256(manifest_payload),
        }
        return self.seed_manifest or {}

    def export_state(self) -> dict[str, Any]:
        """Export one validated workspace tree for an evaluator intervention."""

        if not self.enabled:
            raise WorkspaceActionError("cannot export an unavailable workspace")
        snapshot = self.snapshot()
        root = self.host_root
        files = []
        for item in snapshot["files"]:
            data = _read_regular_file(
                root / item["path"],
                self.limits.max_file_bytes,
            )
            digest = hashlib.sha256(data).hexdigest()
            if digest != item["sha256"] or len(data) != item["bytes"]:
                raise WorkspaceActionError(
                    "workspace changed while its intervention state was exported"
                )
            files.append(
                {
                    "path": item["path"],
                    "bytes": item["bytes"],
                    "sha256": item["sha256"],
                    "content_base64": base64.b64encode(data).decode("ascii"),
                }
            )
        return {
            "schema": WORKSPACE_STATE_SCHEMA,
            "file_count": snapshot["file_count"],
            "directory_count": snapshot["directory_count"],
            "total_bytes": snapshot["total_bytes"],
            "directories": list(snapshot["directories"]),
            "files": files,
            "tree_sha256": snapshot["tree_sha256"],
            "seed_manifest": self.seed_manifest,
        }

    def install_causal_intervention(
        self,
        arm: str,
        *,
        state: Mapping[str, Any] | None = None,
        source_label: str | None = None,
    ) -> dict[str, Any]:
        """Replace or remove the workspace outside the policy action channel."""

        if arm not in WORKSPACE_CAUSAL_ARMS:
            raise WorkspaceActionError(
                "workspace causal arm must be one of: "
                + ", ".join(WORKSPACE_CAUSAL_ARMS)
            )
        if self._episode_id is None:
            raise WorkspaceActionError(
                "workspace must belong to an active episode before intervention"
            )
        if not self.enabled or self._causal_arm is not None:
            raise WorkspaceActionError(
                "workspace causal intervention may be installed only once per episode"
            )
        before = self.snapshot()
        before_seed_manifest = self.seed_manifest
        source_tree_sha256: str | None = None
        installed_seed_manifest: dict[str, Any] | None = None

        if arm == "no_workspace":
            if state is not None:
                raise WorkspaceActionError(
                    "no_workspace intervention must not carry workspace state"
                )
            root = self.host_root
            # Keep the live object usable if host deletion fails.  Flipping
            # these fields first would create a false no-workspace arm while
            # the policy-authored tree still existed on disk.
            shutil.rmtree(root)
            self._root = None
            self._enabled = False
            after = _empty_snapshot()
        else:
            if arm == "blank":
                if state is None:
                    state = _empty_workspace_state()
            elif state is None:
                raise WorkspaceActionError(
                    f"{arm} intervention requires an exported workspace state"
                )
            decoded = _decode_workspace_state(state, limits=self.limits)
            installed_seed_manifest = decoded["seed_manifest"]
            source_tree_sha256 = str(state["tree_sha256"])
            if arm == "blank" and source_tree_sha256 != _empty_snapshot()["tree_sha256"]:
                raise WorkspaceActionError(
                    "blank intervention requires an empty workspace state"
                )
            root = self.host_root
            staging = Path(
                tempfile.mkdtemp(
                    prefix=".agentmemory-intervention-",
                    dir=root.parent,
                )
            )
            candidate = staging / "workspace"
            try:
                candidate.mkdir(mode=0o700)
                for relative in decoded["directories"]:
                    (candidate / relative).mkdir(mode=0o700, parents=True)
                for relative, data in decoded["files"]:
                    path = candidate / relative
                    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    _write_private_regular_file(path, data)
                after = self._snapshot_root(candidate)
                if after["tree_sha256"] != source_tree_sha256:
                    raise WorkspaceActionError(
                        "workspace intervention state does not reproduce its tree hash"
                    )
                replace_workspace_directory(root, candidate)
            finally:
                shutil.rmtree(staging, ignore_errors=True)

        self._audit_events = []
        self._event_counter = 0
        self._causal_arm = arm
        self._seed_manifest = installed_seed_manifest
        self._control_event = {
            "schema": "agentmemory_workspace_causal_intervention_v1",
            "arm": arm,
            "source_label": source_label,
            "source_tree_sha256": source_tree_sha256,
            "workspace_tree_sha256_before": before["tree_sha256"],
            "workspace_tree_sha256_after": after["tree_sha256"],
            "seed_manifest_sha256_before": (
                None
                if before_seed_manifest is None
                else before_seed_manifest["manifest_sha256"]
            ),
            "seed_manifest_sha256_after": (
                None
                if installed_seed_manifest is None
                else installed_seed_manifest["manifest_sha256"]
            ),
            "workspace_enabled_after": self.enabled,
            "policy_action": False,
            "task_reward": 0.0,
        }
        return self.control_event or {}

    def apply(
        self,
        action: str,
        *,
        env_step: int,
        phase_index: int,
    ) -> WorkspaceActionResult | None:
        parsed = parse_workspace_action(action)
        if parsed is None:
            return None
        if not self.enabled:
            raise WorkspaceActionError(
                "this intervention does not provide a persistent workspace"
            )
        before_snapshot = self.snapshot()
        try:
            if parsed.tool_name == "shell_command":
                message, detail = self._run_shell(parsed)
            elif parsed.tool_name == "apply_patch":
                message, detail = self._run_apply_patch(parsed)
            else:  # pragma: no cover - parser owns the closed tool set.
                raise WorkspaceActionError(
                    f"unsupported workspace tool: {parsed.tool_name}"
                )
        except (ShellSandboxError, WorkspacePatchError) as exc:
            raise WorkspaceActionError(str(exc)) from exc
        after_snapshot = self.snapshot()
        workspace_diff = _snapshot_diff(before_snapshot, after_snapshot)
        request_sha256 = hashlib.sha256(parsed.tool_input.encode("utf-8")).hexdigest()
        event = {
            **detail,
            "event_id": self._event_counter,
            "op": parsed.tool_name.upper(),
            "tool_name": parsed.tool_name,
            "step": _require_nonnegative_int(env_step, "env_step"),
            "phase_index": _require_nonnegative_int(phase_index, "phase_index"),
            "episode_id": self.episode_id,
            "request_sha256": request_sha256,
            "workspace_tree_sha256_before": before_snapshot["tree_sha256"],
            "workspace_tree_sha256_after": after_snapshot["tree_sha256"],
            "workspace_diff": workspace_diff,
            "status": "executed",
        }
        self._event_counter += 1
        self._audit_events.append(_deepcopy_json(event))
        return WorkspaceActionResult(
            message=message,
            op=parsed.tool_name.upper(),
            tool_op=_deepcopy_json(event),
            workspace_diff=_deepcopy_json(workspace_diff),
        )

    def snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return _empty_snapshot()
        return self._snapshot_root(self.host_root)

    def render_contract(self) -> str:
        if not self.enabled:
            return "Persistent workspace: unavailable in this intervention."
        return "\n".join(
            [
                "Persistent workspace tools:",
                "The private workspace persists across shopping sessions in this episode.",
                'Canonical shell form: shell_command {"command":"rg -n pattern .","workdir":".","timeout_ms":10000}',
                "The literal shell_command prefix and one separating space are required; a bare JSON object, markdown code fence, or explanation is invalid.",
                "apply_patch followed on the next line by one *** Begin Patch ... *** End Patch patch.",
                "shell_command runs in a networkless, resource-bounded workspace sandbox.",
                "apply_patch supports Add File, Update File, Delete File, and Move to.",
                "Both tools have zero task reward. Paths and workdir are workspace-relative.",
            ]
        )

    def _run_shell(
        self,
        parsed: WorkspaceAction,
    ) -> tuple[str, dict[str, Any]]:
        payload = dict(parsed.arguments or {})
        _require_fields(
            payload,
            required={"command"},
            optional={"workdir", "timeout_ms"},
        )
        command = _require_nonempty_string(payload["command"], "command")
        if len(command) > self.limits.max_command_chars:
            raise WorkspaceActionError(
                f"command cannot exceed {self.limits.max_command_chars} characters"
            )
        workdir = _normalize_relative_path(
            payload.get("workdir", "."),
            limits=self.limits,
            allow_root=True,
        )
        timeout_ms = payload.get("timeout_ms", self.limits.default_timeout_ms)
        timeout_ms = _require_positive_int(timeout_ms, "timeout_ms")
        if timeout_ms > self.limits.max_timeout_ms:
            raise WorkspaceActionError(
                f"timeout_ms cannot exceed {self.limits.max_timeout_ms}"
            )
        result = self.shell_sandbox.run(
            self.host_root,
            command=command,
            workdir=workdir,
            timeout_ms=timeout_ms,
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        output_sections = []
        if stdout:
            output_sections.append(stdout)
        if stderr:
            output_sections.append("[stderr]\n" + stderr)
        output = "\n".join(output_sections) or "<no output>"
        message = (
            f"Exit code: {result.exit_code}\n"
            f"Wall time: {result.elapsed_ms / 1000.0:.3f} seconds\n"
            f"Output:\n{output}"
        )
        return message, {
            "command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
            "command_bytes": len(command.encode("utf-8")),
            "workdir": workdir,
            "timeout_ms": timeout_ms,
            "exit_code": result.exit_code,
            "elapsed_ms": result.elapsed_ms,
            "timed_out": result.timed_out,
            "termination_reason": result.termination_reason,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_bytes": len(result.stdout),
            "stderr_bytes": len(result.stderr),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "stdout_truncated": result.stdout_truncated,
            "stderr_truncated": result.stderr_truncated,
            "sandbox_contract": result.sandbox_contract,
            "model_uid": result.model_uid,
        }

    def _run_apply_patch(
        self,
        parsed: WorkspaceAction,
    ) -> tuple[str, dict[str, Any]]:
        patch_text = parsed.tool_input
        patch_bytes = patch_text.encode("utf-8")
        if len(patch_bytes) > self.limits.max_patch_bytes:
            raise WorkspaceActionError(
                f"apply_patch input cannot exceed {self.limits.max_patch_bytes} bytes"
            )
        operations = parse_workspace_patch(patch_text)
        result = apply_workspace_patch_transaction(
            self.host_root,
            operations,
            normalize_path=lambda value: _normalize_relative_path(
                value,
                limits=self.limits,
            ),
            validate_tree=lambda root: self._snapshot_root(root),
        )
        return "Done!", {
            "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
            "patch_bytes": len(patch_bytes),
            "operation_count": len(operations),
            "changed_paths": list(result.changed_paths),
            "added_paths": list(result.added_paths),
            "updated_paths": list(result.updated_paths),
            "deleted_paths": list(result.deleted_paths),
            "transactional": True,
        }

    def _snapshot_root(self, root: Path) -> dict[str, Any]:
        root = Path(root).resolve()
        files: list[dict[str, Any]] = []
        directories: list[str] = []
        total_bytes = 0
        for current, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
        ):
            current_path = Path(current)
            clean_directories = []
            for name in sorted(directory_names):
                path = current_path / name
                info = os.lstat(path)
                relative = path.relative_to(root).as_posix()
                _validate_snapshot_path(relative, self.limits)
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise WorkspaceActionError(
                        "workspace may contain only real directories and regular files"
                    )
                directories.append(relative)
                clean_directories.append(name)
            directory_names[:] = clean_directories
            for name in sorted(file_names):
                path = current_path / name
                info = os.lstat(path)
                relative = path.relative_to(root).as_posix()
                _validate_snapshot_path(relative, self.limits)
                if (
                    stat.S_ISLNK(info.st_mode)
                    or not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                ):
                    raise WorkspaceActionError(
                        "workspace may not contain symlinks, hard links, or special files"
                    )
                if info.st_size > self.limits.max_file_bytes:
                    raise WorkspaceActionError(
                        f"file cannot exceed {self.limits.max_file_bytes} bytes"
                    )
                data = _read_regular_file(path, self.limits.max_file_bytes)
                total_bytes += len(data)
                files.append(
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data),
                        "kind": "file",
                    }
                )
        if len(files) > self.limits.max_files:
            raise WorkspaceActionError(
                f"workspace cannot exceed {self.limits.max_files} files"
            )
        if len(directories) > self.limits.max_directories:
            raise WorkspaceActionError(
                f"workspace cannot exceed {self.limits.max_directories} directories"
            )
        if total_bytes > self.limits.max_total_bytes:
            raise WorkspaceActionError(
                f"workspace cannot exceed {self.limits.max_total_bytes} bytes"
            )
        files.sort(key=lambda item: item["path"])
        directories.sort()
        manifest = json.dumps(
            {"directories": directories, "files": files},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "schema": "agentmemory_workspace_snapshot_v2",
            "file_count": len(files),
            "directory_count": len(directories),
            "total_bytes": total_bytes,
            "directories": directories,
            "files": files,
            "tree_sha256": hashlib.sha256(manifest).hexdigest(),
        }


def parse_workspace_action(action: str) -> WorkspaceAction | None:
    if not isinstance(action, str):
        raise WorkspaceActionError(
            f"action must be a string, got {type(action).__name__}"
        )
    text = action.strip()
    shell_match = _SHELL_ACTION_RE.fullmatch(text)
    if shell_match is not None:
        try:
            payload = json.loads(shell_match.group(1))
        except json.JSONDecodeError as exc:
            raise WorkspaceActionError(
                f"shell_command payload must be valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise WorkspaceActionError("shell_command payload must be a JSON object")
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return WorkspaceAction(
            tool_name="shell_command",
            arguments=payload,
            tool_input=canonical,
        )
    if text.startswith(_APPLY_PATCH_PREFIX):
        patch_text = text[len(_APPLY_PATCH_PREFIX) :]
        return WorkspaceAction(
            tool_name="apply_patch",
            arguments=None,
            tool_input=patch_text,
        )
    prefix = text.split(None, 1)[0] if text else ""
    if prefix.lower() in WORKSPACE_TOOL_NAMES:
        raise WorkspaceActionError(
            "workspace action must use canonical shell_command JSON or apply_patch newline syntax"
        )
    return None


def _normalize_relative_path(
    value: Any,
    *,
    limits: WorkspaceLimits,
    allow_root: bool = False,
) -> str:
    raw = _require_nonempty_string(value, "path")
    if "\x00" in raw or "\\" in raw:
        raise WorkspaceActionError("path contains a forbidden character")
    if len(raw) > limits.max_path_chars:
        raise WorkspaceActionError(
            f"path cannot exceed {limits.max_path_chars} characters"
        )
    if raw == ".":
        if allow_root:
            return raw
        raise WorkspaceActionError("a file path cannot name the workspace root")
    if raw.startswith(("/", "~")):
        raise WorkspaceActionError("path must be relative to the workspace")
    path = PurePosixPath(raw)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise WorkspaceActionError(
            "path contains an empty, dot, or parent component"
        )
    return path.as_posix()


def _snapshot_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_files = {item["path"]: item for item in before["files"]}
    after_files = {item["path"]: item for item in after["files"]}
    added = [after_files[path] for path in sorted(after_files.keys() - before_files.keys())]
    deleted = [before_files[path] for path in sorted(before_files.keys() - after_files.keys())]
    modified = [
        {"before": before_files[path], "after": after_files[path]}
        for path in sorted(before_files.keys() & after_files.keys())
        if before_files[path]["sha256"] != after_files[path]["sha256"]
    ]
    before_directories = set(before.get("directories", []))
    after_directories = set(after.get("directories", []))
    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "directories_added": sorted(after_directories - before_directories),
        "directories_deleted": sorted(before_directories - after_directories),
    }


def _empty_snapshot() -> dict[str, Any]:
    manifest = json.dumps(
        {"directories": [], "files": []},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "agentmemory_workspace_snapshot_v2",
        "file_count": 0,
        "directory_count": 0,
        "total_bytes": 0,
        "directories": [],
        "files": [],
        "tree_sha256": hashlib.sha256(manifest).hexdigest(),
    }


def _empty_workspace_state() -> dict[str, Any]:
    snapshot = _empty_snapshot()
    return {
        "schema": WORKSPACE_STATE_SCHEMA,
        "file_count": 0,
        "directory_count": 0,
        "total_bytes": 0,
        "directories": [],
        "files": [],
        "tree_sha256": snapshot["tree_sha256"],
        "seed_manifest": None,
    }


def _decode_workspace_state(
    state: Mapping[str, Any],
    *,
    limits: WorkspaceLimits,
) -> dict[str, Any]:
    if not isinstance(state, Mapping) or state.get("schema") not in {
        WORKSPACE_STATE_SCHEMA_V1,
        WORKSPACE_STATE_SCHEMA,
    }:
        raise WorkspaceActionError("workspace intervention state has an invalid schema")
    state_schema = state.get("schema")
    raw_directories = state.get("directories")
    raw_files = state.get("files")
    if not isinstance(raw_directories, list) or not isinstance(raw_files, list):
        raise WorkspaceActionError(
            "workspace intervention state requires directory and file lists"
        )
    directories: list[str] = []
    for raw in raw_directories:
        normalized = _normalize_relative_path(raw, limits=limits)
        if normalized != raw:
            raise WorkspaceActionError(
                "workspace intervention directory paths must be canonical"
            )
        directories.append(normalized)
    if directories != sorted(set(directories)):
        raise WorkspaceActionError(
            "workspace intervention directories must be sorted and unique"
        )

    files: list[tuple[str, bytes]] = []
    file_paths: list[str] = []
    total_bytes = 0
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise WorkspaceActionError(
                "workspace intervention file records must be objects"
            )
        raw_path = item.get("path")
        path = _normalize_relative_path(raw_path, limits=limits)
        if path != raw_path:
            raise WorkspaceActionError(
                "workspace intervention file paths must be canonical"
            )
        encoded = item.get("content_base64")
        if not isinstance(encoded, str):
            raise WorkspaceActionError(
                "workspace intervention file content must be base64 text"
            )
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
            raise WorkspaceActionError(
                "workspace intervention file content is not canonical base64"
            ) from exc
        expected_size = item.get("bytes")
        expected_sha256 = item.get("sha256")
        if (
            type(expected_size) is not int
            or expected_size != len(data)
            or expected_size > limits.max_file_bytes
            or not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
            or hashlib.sha256(data).hexdigest() != expected_sha256
        ):
            raise WorkspaceActionError(
                "workspace intervention file size or digest is inconsistent"
            )
        files.append((path, data))
        file_paths.append(path)
        total_bytes += len(data)
    if file_paths != sorted(set(file_paths)):
        raise WorkspaceActionError(
            "workspace intervention files must be sorted and unique"
        )
    if set(file_paths).intersection(directories):
        raise WorkspaceActionError(
            "workspace intervention path cannot be both a file and directory"
        )
    expected_tree_sha256 = state.get("tree_sha256")
    if (
        not isinstance(expected_tree_sha256, str)
        or _SHA256_RE.fullmatch(expected_tree_sha256) is None
        or state.get("file_count") != len(files)
        or state.get("directory_count") != len(directories)
        or state.get("total_bytes") != total_bytes
        or len(files) > limits.max_files
        or len(directories) > limits.max_directories
        or total_bytes > limits.max_total_bytes
    ):
        raise WorkspaceActionError(
            "workspace intervention manifest counts, bytes, or tree digest are invalid"
        )
    seed_manifest = None
    if state_schema == WORKSPACE_STATE_SCHEMA:
        seed_manifest = _decode_seed_manifest(
            state.get("seed_manifest"),
            limits=limits,
        )
    return {
        "directories": directories,
        "files": files,
        "seed_manifest": seed_manifest,
    }


def _decode_seed_manifest(
    raw_manifest: Any,
    *,
    limits: WorkspaceLimits,
) -> dict[str, Any] | None:
    if raw_manifest is None:
        return None
    if not isinstance(raw_manifest, Mapping):
        raise WorkspaceActionError("workspace seed manifest must be an object")
    required = {
        "schema",
        "source_label",
        "seed_tree_sha256",
        "files",
        "manifest_sha256",
    }
    if set(raw_manifest) != required:
        raise WorkspaceActionError("workspace seed manifest has invalid fields")
    if raw_manifest.get("schema") != WORKSPACE_SEED_MANIFEST_SCHEMA:
        raise WorkspaceActionError("workspace seed manifest has an invalid schema")
    source_label = _require_nonempty_string(
        raw_manifest.get("source_label"),
        "seed source_label",
    )
    seed_tree_sha256 = raw_manifest.get("seed_tree_sha256")
    if (
        not isinstance(seed_tree_sha256, str)
        or _SHA256_RE.fullmatch(seed_tree_sha256) is None
    ):
        raise WorkspaceActionError("workspace seed tree digest is invalid")
    raw_files = raw_manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise WorkspaceActionError("workspace seed manifest requires files")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"path", "bytes", "sha256"}:
            raise WorkspaceActionError("workspace seed file record is invalid")
        raw_path = item.get("path")
        path = _normalize_relative_path(raw_path, limits=limits)
        size = item.get("bytes")
        digest = item.get("sha256")
        if (
            path != raw_path
            or type(size) is not int
            or size < 0
            or size > limits.max_file_bytes
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise WorkspaceActionError("workspace seed file metadata is invalid")
        files.append({"path": path, "bytes": size, "sha256": digest})
        total_bytes += size
    paths = [item["path"] for item in files]
    if paths != sorted(set(paths)):
        raise WorkspaceActionError("workspace seed paths must be sorted and unique")
    if len(files) > limits.max_files or total_bytes > limits.max_total_bytes:
        raise WorkspaceActionError("workspace seed manifest exceeds workspace limits")
    payload = {
        "schema": WORKSPACE_SEED_MANIFEST_SCHEMA,
        "source_label": source_label,
        "seed_tree_sha256": seed_tree_sha256,
        "files": files,
    }
    manifest_sha256 = raw_manifest.get("manifest_sha256")
    if (
        not isinstance(manifest_sha256, str)
        or _SHA256_RE.fullmatch(manifest_sha256) is None
        or manifest_sha256 != _canonical_sha256(payload)
    ):
        raise WorkspaceActionError("workspace seed manifest digest is invalid")
    return {**payload, "manifest_sha256": manifest_sha256}


def _write_private_regular_file(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()


def _read_regular_file(path: Path, limit: int) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise WorkspaceActionError("workspace path is not a private regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise WorkspaceActionError("workspace file exceeds the configured limit")
        return data
    finally:
        os.close(descriptor)


def _validate_snapshot_path(relative: str, limits: WorkspaceLimits) -> None:
    if len(relative) > limits.max_path_chars:
        raise WorkspaceActionError(
            f"workspace path cannot exceed {limits.max_path_chars} characters"
        )


def _require_fields(
    payload: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - set(payload)
    extra = set(payload) - required - optional
    if missing:
        raise WorkspaceActionError(
            f"missing field(s): {', '.join(sorted(missing))}"
        )
    if extra:
        raise WorkspaceActionError(
            f"unexpected field(s): {', '.join(sorted(extra))}"
        )


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise WorkspaceActionError(f"{name} must be a string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise WorkspaceActionError(f"{name} must be valid UTF-8 text") from exc
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    parsed = _require_string(value, name)
    if not parsed:
        raise WorkspaceActionError(f"{name} must be a non-empty string")
    return parsed


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise WorkspaceActionError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise WorkspaceActionError(f"{name} must be a non-negative integer")
    return value


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return normalized[:48] or "workspace"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
