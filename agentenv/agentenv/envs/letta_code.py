"""Episode-private Letta Code v0.31.12 MemFS adapter for CAMG.

This is a narrow Python port of Letta Code's public ``memory`` and
``memory_apply_patch`` tool contract.  It keeps the native CAMG wrapper in
charge of task state/reward while preserving the baseline's defining
mechanism: a git-backed MemFS, reason-bearing writes, root core memory in the
system context, and child memory read on demand.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from agentenv.controller.env import BaseEnvClient
from agentenv.controller.types import (
    CONTEXT_OPERATION_APPEND,
    CONTEXT_OPERATION_REPLACE,
    TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA,
    PolicyContextPressure,
    StepOutput,
    build_task_neutral_context_transition,
    build_task_neutral_transition_info,
)

from .agemem import _copy_messages
from .agentmemory import parse_filesystem_env_action


LETTA_CODE_SOURCE_REVISION = "787b856f9db9f5030dc2976618e1d1f909f61612"
LETTA_ADAPTER_SCHEMA = "camg_letta_code_adapter_v1"
LETTA_ACTION_SCHEMA = "camg_letta_code_action_v1"
LETTA_PROMPT_MARKER = "[CAMG_LETTA_CODE_MEMFS_V0.31.12]"
_OPEN = "<letta_memory_call>"
_CLOSE = "</letta_memory_call>"
_CALL = re.compile(rf"\A{re.escape(_OPEN)}\s*(.*?)\s*{re.escape(_CLOSE)}\Z", re.S)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class LettaInvocation(NamedTuple):
    name: str
    arguments: dict[str, Any]


class LettaActionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LettaCodeAdapterConfig:
    runtime_root: str = "/tmp/camg-letta-code-memfs"
    max_file_bytes: int = 32768
    max_total_bytes: int = 131072
    max_observation_bytes: int = 24576
    max_files: int = 64
    invalid_action_reward: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.invalid_action_reward, bool):
            raise TypeError("Letta Code invalid_action_reward must be numeric")
        try:
            reward = float(self.invalid_action_reward)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Letta Code invalid_action_reward must be finite and non-positive"
            ) from exc
        if not math.isfinite(reward) or reward > 0.0:
            raise ValueError(
                "Letta Code invalid_action_reward must be finite and non-positive"
            )
        object.__setattr__(self, "invalid_action_reward", reward)

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "LettaCodeAdapterConfig":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("Letta Code adapter config must be a mapping")
        unknown = sorted(set(map(str, value)) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("unknown Letta Code adapter config fields: " + ", ".join(unknown))
        result = cls(**{str(key): item for key, item in value.items()})
        if any(
            isinstance(getattr(result, field), bool)
            or not isinstance(getattr(result, field), int)
            or getattr(result, field) <= 0
            for field in (
                "max_file_bytes",
                "max_total_bytes",
                "max_observation_bytes",
                "max_files",
            )
        ):
            raise ValueError("Letta Code byte limits must be positive integers")
        return result


def parse_letta_action(action: str) -> LettaInvocation | None:
    if not isinstance(action, str):
        raise TypeError("policy action must be text")
    if _OPEN not in action and _CLOSE not in action:
        return None
    match = _CALL.fullmatch(action.strip())
    if match is None:
        raise ValueError("Letta Code action must contain one exact letta_memory_call block")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Letta Code payload is invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"name", "arguments"}:
        raise ValueError("Letta Code invocation must contain only name and arguments")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if name not in {"memory", "memory_apply_patch"} or not isinstance(arguments, Mapping):
        raise ValueError("unsupported Letta Code tool invocation")
    return LettaInvocation(str(name), {str(k): deepcopy(v) for k, v in arguments.items()})


def parse_memory_filesystem_read(action: str) -> str | None:
    """Return the requested MemFS path for one strict ordinary ``cat`` action.

    Letta Code does not expose a ``read`` memory-tool command.  Deferred memory
    is read through the normal shell tool and ``$MEMORY_DIR``.  The CAMG task
    sandbox is distinct from the episode-private MemFS, so the adapter only
    intercepts a side-effect-free single-file ``cat`` and rejects any broader
    shell expression that mentions the memory directory.
    """

    if not isinstance(action, str):
        raise TypeError("policy action must be text")
    mentions_memory = any(
        marker in action
        for marker in ("$MEMORY_DIR", "${MEMORY_DIR}", ".letta_memory/")
    )
    if not mentions_memory:
        return None
    try:
        action_name, arguments = parse_filesystem_env_action(action)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LettaActionError(
            "invalid_memory_read",
            "MemFS reads must be one ordinary shell_command cat action",
        ) from exc
    if action_name != "shell_command" or not isinstance(arguments, Mapping):
        raise LettaActionError(
            "invalid_memory_read",
            "MemFS reads must use shell_command",
        )
    if not set(arguments) <= {"command", "workdir", "timeout_ms"}:
        raise LettaActionError(
            "invalid_memory_read",
            "MemFS read shell_command contains unsupported arguments",
        )
    if arguments.get("workdir", ".") != ".":
        raise LettaActionError(
            "invalid_memory_read",
            "MemFS reads must use the default task workdir",
        )
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        raise LettaActionError("invalid_memory_read", "MemFS read command is empty")
    try:
        words = shlex.split(command, posix=True)
    except ValueError as exc:
        raise LettaActionError("invalid_memory_read", "MemFS read shell syntax is invalid") from exc
    if len(words) == 3 and words[:2] == ["cat", "--"]:
        target = words[2]
    elif len(words) == 2 and words[0] == "cat":
        target = words[1]
    else:
        raise LettaActionError(
            "invalid_memory_read",
            "MemFS reads permit only: cat $MEMORY_DIR/<path>.md",
        )
    if not target.startswith(("$MEMORY_DIR/", "${MEMORY_DIR}/", ".letta_memory/")):
        raise LettaActionError(
            "invalid_memory_read",
            "MemFS cat target must be rooted at $MEMORY_DIR",
        )
    return target


class LettaCodeEnvClientAdapter(BaseEnvClient):
    def __init__(
        self,
        native_client: BaseEnvClient,
        config: LettaCodeAdapterConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(native_client, BaseEnvClient):
            raise TypeError("Letta Code adapter requires a BaseEnvClient")
        super().__init__(native_client.action_format)
        self.native_client = native_client
        self.config = (
            config
            if isinstance(config, LettaCodeAdapterConfig)
            else LettaCodeAdapterConfig.from_mapping(config)
        )
        self._repo: Path | None = None
        self._current_policy_context: list[dict[str, str]] | None = None
        self._native_control_selected = False
        self._episode_source_identity: dict[str, Any] | None = None
        self._memory_action_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_client, name)

    def __len__(self) -> int:
        return len(self.native_client)

    @property
    def sample_excluded(self) -> bool:
        return bool(getattr(self.native_client, "sample_excluded", False))

    def observe(self) -> str:
        return str(self.native_client.observe())

    def policy_framing(self) -> list[dict[str, str]]:
        method = getattr(self.native_client, "policy_framing", None)
        return self._inject_prompt(_copy_messages(method() if callable(method) else ()))

    def normalize_initial_policy_context(
        self, messages: Sequence[Mapping[str, str]]
    ) -> list[dict[str, str]]:
        normalized = self.native_client.normalize_initial_policy_context(
            self._strip_prompt(_copy_messages(messages))
        )
        return self._inject_prompt(_copy_messages(normalized))

    def bind_policy_context(
        self, messages: Sequence[Mapping[str, str]], *, initial: bool = False
    ) -> None:
        adapted = self._inject_prompt(_copy_messages(messages))
        self._current_policy_context = deepcopy(adapted)
        self.native_client.bind_policy_context(
            self._strip_prompt(adapted), initial=initial
        )

    def policy_turn_candidate(self) -> str | None:
        self._native_control_selected = False
        return self.native_client.policy_turn_candidate()

    def prepare_policy_turn(self, pressure: PolicyContextPressure | None) -> str | None:
        selected = self.native_client.prepare_policy_turn(pressure)
        self._native_control_selected = selected is not None
        return selected

    def step(self, action: str) -> StepOutput:
        if not isinstance(action, str):
            raise TypeError("Letta Code-adapted policy action must be text")
        if self._native_control_selected:
            self._native_control_selected = False
            return self._wrap_native(self.native_client.step(action))
        try:
            invocation = parse_letta_action(action)
        except ValueError as exc:
            return self._error(action, "memory_action_parse_error", str(exc))
        if invocation is None:
            try:
                read_target = parse_memory_filesystem_read(action)
            except LettaActionError as exc:
                return self._error(action, exc.code, str(exc))
            if read_target is None:
                return self._wrap_native(self.native_client.step(action))
            self._memory_action_count += 1
            try:
                path = self._path(read_target)
                if not path.is_file():
                    raise LettaActionError(
                        "not_found", "memory file does not exist"
                    )
                content = path.read_text(encoding="utf-8")
            except (UnicodeError, OSError) as exc:
                return self._error(
                    action, "memory_read_failed", str(exc), increment=False
                )
            except LettaActionError as exc:
                return self._error(action, exc.code, str(exc), increment=False)
            return self._memory_read_step(action, path, content)

        self._memory_action_count += 1
        try:
            payload = self._execute(invocation)
        except LettaActionError as exc:
            return self._error(action, exc.code, str(exc), increment=False)
        return self._memory_step(action, payload, accepted=True)

    def reset(self, idx: int = 0) -> Any:
        self._cleanup_repo()
        self._current_policy_context = None
        self._native_control_selected = False
        self._episode_source_identity = None
        self._memory_action_count = 0
        response = self.native_client.reset(idx)
        identity = getattr(self.native_client, "episode_source_identity", None)
        if not isinstance(identity, Mapping) or not identity:
            raise RuntimeError("Letta Code native reset lacks episode source identity")
        self._episode_source_identity = deepcopy(dict(identity))
        root = Path(self.config.runtime_root)
        root.mkdir(parents=True, exist_ok=True)
        owner = os.environ.get("AGENTMEMORY_PROCESS_OWNER", "local")
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        self._repo = Path(tempfile.mkdtemp(prefix=f"{owner}.{digest}.", dir=root))
        self._git("init", "-q")
        self._git("config", "user.name", "CAMG Letta Code")
        self._git("config", "user.email", "camg-letta-code@local")
        (self._repo / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
        self._git("add", "MEMORY.md")
        self._git("commit", "-q", "-m", "Initialize episode-private MemFS")
        return response

    def finalize_policy_horizon(self) -> StepOutput | None:
        output = self.native_client.finalize_policy_horizon()
        return None if output is None else self._wrap_native(output)

    def close(self) -> Any:
        try:
            return self.native_client.close()
        finally:
            self._cleanup_repo()

    def _execute(self, invocation: LettaInvocation) -> dict[str, Any]:
        args = invocation.arguments
        reason = args.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise LettaActionError("missing_reason", "reason must be non-empty")
        if "\x00" in reason:
            raise LettaActionError("invalid_reason", "reason contains a NUL byte")
        try:
            if invocation.name == "memory_apply_patch":
                patch = args.get("input")
                if (
                    set(args) != {"reason", "input"}
                    or not isinstance(patch, str)
                    or not patch.strip()
                ):
                    raise LettaActionError(
                        "invalid_patch",
                        "memory_apply_patch requires exactly reason and input",
                    )
                changed = self._apply_patch(patch)
                operation = "apply_patch"
            else:
                command = str(args.get("command") or "")
                changed = self._memory_command(command, args)
                operation = command
            self._validate_tree()
            self._enforce_limits()
            self._git("add", "-A")
            if not self._git_output("status", "--porcelain"):
                raise LettaActionError(
                    "no_effective_change", "memory write made no effective change"
                )
            self._git("commit", "-q", "-m", reason.strip())
            commit = self._git_output("rev-parse", "HEAD").strip()
            if not _COMMIT.fullmatch(commit):
                raise RuntimeError("Letta Code git commit identity is invalid")
            return {
                "operation": operation,
                "reason": reason.strip(),
                "changed_paths": sorted(set(changed)),
                "commit_sha": commit,
            }
        except Exception:
            self._rollback_repo()
            raise

    def _memory_command(self, command: str, args: Mapping[str, Any]) -> list[str]:
        argument_contracts = {
            "str_replace": {"command", "reason", "file_path", "old_string", "new_string"},
            "insert": {"command", "reason", "file_path", "insert_line", "insert_text"},
            "delete": {"command", "reason", "file_path"},
            "rename": {"command", "reason", "old_path", "new_path"},
            "update_description": {"command", "reason", "file_path", "description"},
            "create": {"command", "reason", "file_path", "description", "file_text"},
        }
        if command not in argument_contracts:
            raise LettaActionError(
                "unsupported_command", f"unsupported memory command {command!r}"
            )
        allowed = argument_contracts[command]
        required = allowed - ({"description", "file_text"} if command == "create" else set())
        missing = sorted(key for key in required if key not in args)
        unknown = sorted(set(args) - allowed)
        if missing or unknown:
            raise LettaActionError(
                "invalid_arguments",
                f"{command} arguments drift: missing={missing!r}, unknown={unknown!r}",
            )
        if command == "rename":
            old = self._path(args.get("old_path"))
            new = self._path(args.get("new_path"))
            if self._is_index(old) or self._is_index(new):
                raise LettaActionError(
                    "invalid_rename", "MEMORY.md indexes cannot be renamed"
                )
            if not old.is_file() or old.is_symlink() or new.exists():
                raise LettaActionError("invalid_rename", "rename source/destination is invalid")
            self._assert_indexed(new)
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
            return [self._relative(old), self._relative(new)]
        path = self._path(
            args.get("file_path"), allow_existing_directory=command == "delete"
        )
        if command == "create":
            if path.exists():
                raise LettaActionError("already_exists", "memory file already exists")
            description = args.get("description")
            if not self._is_index(path) and (
                not isinstance(description, str) or not description.strip()
            ):
                raise LettaActionError("missing_description", "regular memory files require description")
            body = args.get("file_text", "")
            if not isinstance(body, str):
                raise LettaActionError("invalid_file_text", "file_text must be text")
            self._assert_indexed(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._render(path, body, description), encoding="utf-8")
        elif command == "delete":
            if self._is_index(path) or not path.exists() or path.is_symlink():
                raise LettaActionError("invalid_delete", "MEMORY.md indexes cannot be deleted or path is missing")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            if not path.is_file():
                raise LettaActionError("not_found", "memory file does not exist")
            original = path.read_text(encoding="utf-8")
            front, body = self._split_frontmatter(path, original)
            if command == "str_replace":
                old = args.get("old_string")
                new = args.get("new_string")
                if (
                    not isinstance(old, str)
                    or not old
                    or not isinstance(new, str)
                    or old not in body
                ):
                    raise LettaActionError("old_string_not_found", "old_string was not found")
                body = body.replace(old, new, 1)
            elif command == "insert":
                text = args.get("insert_text")
                line = args.get("insert_line")
                if (
                    not isinstance(text, str)
                    or not text
                    or isinstance(line, bool)
                    or not isinstance(line, (int, float))
                ):
                    raise LettaActionError("invalid_insert", "insert requires line and text")
                if not float(line).is_integer():
                    raise LettaActionError("invalid_insert", "insert_line must be an integer")
                lines = body.split("\n") if body else []
                position = min(max(int(line), 1) - 1, len(lines))
                lines[position:position] = text.split("\n")
                body = "\n".join(lines)
            elif command == "update_description":
                description = args.get("description")
                if self._is_index(path) or not isinstance(description, str) or not description.strip():
                    raise LettaActionError("invalid_description", "description must be non-empty")
                front["description"] = description.strip()
            path.write_text(
                self._render(
                    path,
                    body,
                    front.get("description"),
                    name=front.get("name"),
                ),
                encoding="utf-8",
            )
        return [self._relative(path)]

    def _apply_patch(self, text: str) -> list[str]:
        # Letta Code uses the Codex-style Begin/End Patch grammar.  Reuse the
        # repository's tested patch engine when available; this narrow port
        # supports Add/Delete and anchored Update hunks used by the baseline.
        lines = text.strip().splitlines()
        if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
            raise LettaActionError("invalid_patch", "patch must use Begin/End Patch markers")
        changed: list[str] = []
        i = 1
        while i < len(lines) - 1:
            header = lines[i]
            if header.startswith("*** Add File: "):
                path = self._path(header.removeprefix("*** Add File: "))
                i += 1
                body: list[str] = []
                while i < len(lines) - 1 and not lines[i].startswith("*** "):
                    if not lines[i].startswith("+"):
                        raise LettaActionError("invalid_patch", "Add File lines require +")
                    body.append(lines[i][1:])
                    i += 1
                if path.exists():
                    raise LettaActionError("already_exists", "patch add target exists")
                self._assert_indexed(path)
                content = "\n".join(body) + "\n"
                if self._is_index(path):
                    if content.startswith("---\n"):
                        raise LettaActionError(
                            "invalid_frontmatter",
                            "MEMORY.md indexes must not have frontmatter",
                        )
                elif content.startswith("---\n"):
                    front, plain_body = self._split_frontmatter(path, content)
                    content = self._render(
                        path,
                        plain_body,
                        front["description"],
                        name=front["name"],
                    )
                else:
                    content = self._render(
                        path,
                        content,
                        f"Memory block {self._label_without_suffix(path)}",
                    )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                changed.append(self._relative(path))
                continue
            if header.startswith("*** Delete File: "):
                path = self._path(header.removeprefix("*** Delete File: "))
                if path.name == "MEMORY.md" or not path.is_file():
                    raise LettaActionError("invalid_delete", "patch delete target is invalid")
                path.unlink()
                changed.append(self._relative(path))
                i += 1
                continue
            if header.startswith("*** Update File: "):
                path = self._path(header.removeprefix("*** Update File: "))
                if not path.is_file():
                    raise LettaActionError("not_found", "patch update target is missing")
                i += 1
                old: list[str] = []
                new: list[str] = []
                while i < len(lines) - 1 and not lines[i].startswith("*** "):
                    line = lines[i]
                    if line.startswith("@@"):
                        i += 1
                        continue
                    if line.startswith((" ", "-")):
                        old.append(line[1:])
                    if line.startswith((" ", "+")):
                        new.append(line[1:])
                    i += 1
                content = path.read_text(encoding="utf-8")
                before = "\n".join(old)
                after = "\n".join(new)
                if not before or before not in content:
                    raise LettaActionError("patch_context_not_found", "patch context was not found")
                path.write_text(content.replace(before, after, 1), encoding="utf-8")
                changed.append(self._relative(path))
                continue
            raise LettaActionError("invalid_patch", f"unsupported patch directive {header!r}")
        if not changed:
            raise LettaActionError("invalid_patch", "patch has no file operations")
        return changed

    def _memory_step(
        self,
        raw_action: str,
        payload: Mapping[str, Any],
        *,
        accepted: bool,
    ) -> StepOutput:
        observation = self._bounded_observation(
            "[Letta Code memory result]\n"
            + json.dumps(
                dict(payload),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if accepted:
            replacement = self._bound_context() + [
                {"role": "assistant", "content": raw_action},
                {"role": "user", "content": observation},
            ]
            transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_REPLACE,
                messages=self._inject_prompt(replacement),
            )
        else:
            transition = build_task_neutral_context_transition(
                CONTEXT_OPERATION_APPEND
            )
        evidence = {
            "schema": LETTA_ADAPTER_SCHEMA,
            "event": "memory_tool_action",
            "operation": payload.get("operation"),
            "accepted": accepted,
            "reason": payload.get("reason"),
            "commit_sha": payload.get("commit_sha"),
            "changed_paths": payload.get("changed_paths", []),
            "memory_action_index": self._memory_action_count,
            "episode_private": True,
            "git_backed": True,
            "hidden_model_calls": 0,
            "hidden_input_tokens": 0,
            "hidden_output_tokens": 0,
            "hidden_latency_ms": 0,
            "source_revision": LETTA_CODE_SOURCE_REVISION,
            "context_operation": transition["operation"],
        }
        if not accepted:
            evidence["error_code"] = payload.get("error_code")
        return StepOutput(
            state=observation,
            reward=0.0 if accepted else float(self.config.invalid_action_reward),
            done=False,
            info=build_task_neutral_transition_info(
                env_info={"episode_source_identity": self._identity()},
                action_submission={
                    "schema": LETTA_ACTION_SCHEMA,
                    "raw_policy_output": raw_action,
                    "submitted_action": raw_action,
                    "parser_status": "letta_code_adapter",
                    "accepted": accepted,
                    "operation": payload.get("operation"),
                    "error_code": payload.get("error_code"),
                },
                context_transition=transition,
                wrapper_evidence={"letta_code_adapter": evidence},
            ),
        )

    def _memory_read_step(
        self, raw_action: str, path: Path, content: str
    ) -> StepOutput:
        relative = self._relative(path)
        observation = self._bounded_observation(
            f"[Letta Code MemFS read: {relative}]\n{content}"
        )
        evidence = {
            "schema": LETTA_ADAPTER_SCHEMA,
            "event": "memory_filesystem_read",
            "operation": "read",
            "accepted": True,
            "reason": None,
            "commit_sha": None,
            "changed_paths": [],
            "read_path": relative,
            "read_bytes": len(content.encode("utf-8")),
            "memory_action_index": self._memory_action_count,
            "episode_private": True,
            "git_backed": True,
            "hidden_model_calls": 0,
            "hidden_input_tokens": 0,
            "hidden_output_tokens": 0,
            "hidden_latency_ms": 0,
            "source_revision": LETTA_CODE_SOURCE_REVISION,
            "context_operation": CONTEXT_OPERATION_APPEND,
        }
        return StepOutput(
            state=observation,
            reward=0.0,
            done=False,
            info=build_task_neutral_transition_info(
                env_info={"episode_source_identity": self._identity()},
                action_submission={
                    "schema": LETTA_ACTION_SCHEMA,
                    "raw_policy_output": raw_action,
                    "submitted_action": raw_action,
                    "parser_status": "letta_code_adapter",
                    "accepted": True,
                    "operation": "read",
                    "error_code": None,
                },
                context_transition=build_task_neutral_context_transition(
                    CONTEXT_OPERATION_APPEND
                ),
                wrapper_evidence={"letta_code_adapter": evidence},
            ),
        )

    def _error(self, action: str, code: str, message: str, *, increment: bool = True) -> StepOutput:
        if increment:
            self._memory_action_count += 1
        return self._memory_step(
            action,
            {"operation": None, "reason": None, "commit_sha": None, "error_code": code, "error": message},
            accepted=False,
        )

    def _wrap_native(self, output: StepOutput) -> StepOutput:
        if not isinstance(output, StepOutput):
            raise TypeError("native client step must return StepOutput")
        info = deepcopy(dict(output.info)) if isinstance(output.info, Mapping) else {}
        transition = info.get("context_transition")
        if isinstance(transition, Mapping):
            transition = deepcopy(dict(transition))
            if transition.get("schema") == TASK_NEUTRAL_CONTEXT_TRANSITION_SCHEMA and transition.get("operation") == CONTEXT_OPERATION_REPLACE:
                transition["messages"] = self._inject_prompt(_copy_messages(transition.get("messages", ())))
            info["context_transition"] = transition
        evidence = deepcopy(dict(info.get("wrapper_evidence") or {}))
        evidence["letta_code_adapter"] = {
            "schema": LETTA_ADAPTER_SCHEMA,
            "event": "native_action_passthrough",
            "memory_action_count": self._memory_action_count,
            "episode_private": True,
            "git_backed": True,
            "hidden_model_calls": 0,
            "hidden_input_tokens": 0,
            "hidden_output_tokens": 0,
            "hidden_latency_ms": 0,
            "source_revision": LETTA_CODE_SOURCE_REVISION,
        }
        info["wrapper_evidence"] = evidence
        env_info = deepcopy(dict(info.get("env_info") or {}))
        env_info["episode_source_identity"] = self._identity()
        info["env_info"] = env_info
        return StepOutput(state=str(output.state), reward=output.reward, done=bool(output.done), info=info)

    def _inject_prompt(self, messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        normalized = self._strip_prompt(_copy_messages(messages))
        core = self._core_memory()
        prompt = _prompt(core)
        for message in normalized:
            if message["role"] == "system":
                message["content"] += "\n\n" + prompt
                return normalized
        return [{"role": "system", "content": prompt}, *normalized]

    @staticmethod
    def _strip_prompt(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
        normalized = _copy_messages(messages)
        for message in normalized:
            if message["role"] != "system":
                continue
            index = message["content"].find(LETTA_PROMPT_MARKER)
            if index >= 0:
                message["content"] = message["content"][:index].rstrip("\n")
        return [
            message
            for message in normalized
            if message["role"] != "system" or message["content"]
        ]

    def _core_memory(self) -> str:
        if self._repo is None:
            return ""
        parts: list[str] = []
        for path in sorted(self._repo.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            parts.append(f"## {path.name}\n{text}")
        return "\n\n".join(parts)

    def _bound_context(self) -> list[dict[str, str]]:
        if not self._current_policy_context:
            raise LettaActionError(
                "policy_context_unbound",
                "Letta Code memory write requires a bound policy context",
            )
        return self._strip_prompt(deepcopy(self._current_policy_context))

    def _bounded_observation(self, text: str) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= self.config.max_observation_bytes:
            return text
        suffix = b"\n[truncated]"
        keep = max(self.config.max_observation_bytes - len(suffix), 0)
        return (encoded[:keep] + suffix).decode("utf-8", "ignore")

    def _path(
        self, value: Any, *, allow_existing_directory: bool = False
    ) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise LettaActionError("invalid_path", "memory path must be non-empty")
        raw = value.strip().replace("\\", "/")
        for prefix in (
            "${MEMORY_DIR}/",
            "$MEMORY_DIR/",
            ".letta_memory/",
            "memory/",
            "./",
        ):
            if raw.startswith(prefix):
                raw = raw[len(prefix) :]
                break
        raw_path = Path(raw)
        repo = self._require_repo().resolve()
        if raw_path.is_absolute():
            absolute = raw_path.resolve()
            try:
                raw = absolute.relative_to(repo).as_posix()
            except ValueError as exc:
                raise LettaActionError(
                    "invalid_path", "absolute memory path is outside $MEMORY_DIR"
                ) from exc
            raw_path = Path(raw)
        if not raw or ".." in raw_path.parts or any(
            not part or part.startswith(".") for part in raw_path.parts
        ):
            raise LettaActionError(
                "invalid_path", "memory path is not a safe MemFS-relative path"
            )
        candidate = (repo / raw_path).resolve()
        if allow_existing_directory and candidate.is_dir():
            path = candidate
        else:
            if raw_path.name.lower() != "memory.md" and raw_path.suffix == "":
                raw_path = raw_path.with_suffix(".md")
            if raw_path.suffix.lower() != ".md":
                raise LettaActionError(
                    "invalid_path", "memory files must use the Markdown extension"
                )
            path = (repo / raw_path).resolve()
        if path != repo and repo not in path.parents:
            raise LettaActionError("invalid_path", "memory path escapes MemFS")
        return path

    def _split_frontmatter(self, path: Path, text: str) -> tuple[dict[str, str], str]:
        if self._is_index(path):
            if text.startswith("---\n") or text.startswith("---\r\n"):
                raise LettaActionError(
                    "invalid_frontmatter", "MEMORY.md indexes must not have frontmatter"
                )
            return {}, text
        match = re.match(r"\A---\r?\n([\s\S]*?)\r?\n---\r?\n?", text)
        if match is None:
            raise LettaActionError(
                "invalid_frontmatter", "memory file is missing required frontmatter"
            )
        fields: dict[str, str] = {}
        keys: list[str] = []
        for line in match.group(1).splitlines():
            if ":" not in line:
                raise LettaActionError(
                    "invalid_frontmatter", "frontmatter contains an invalid line"
                )
            key, raw_value = line.split(":", 1)
            key = key.strip()
            keys.append(key)
            value = raw_value.strip()
            if value.startswith('"') and value.endswith('"'):
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise LettaActionError(
                        "invalid_frontmatter", "frontmatter JSON string is invalid"
                    ) from exc
                value = decoded if isinstance(decoded, str) else ""
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1].replace("''", "'")
            fields[key] = value
        if keys != ["name", "description"] and set(keys) != {
            "name",
            "description",
        }:
            raise LettaActionError(
                "invalid_frontmatter",
                "memory frontmatter must contain exactly name and description",
            )
        if len(keys) != 2 or not all(fields.get(key, "").strip() for key in keys):
            raise LettaActionError(
                "invalid_frontmatter", "name and description must be non-empty"
            )
        return {
            "name": fields["name"].strip(),
            "description": fields["description"].strip(),
        }, text[match.end() :]

    def _render(self, path: Path, body: str, description: Any, *, name: Any = None) -> str:
        if self._is_index(path):
            if body.startswith("---\n") or body.startswith("---\r\n"):
                raise LettaActionError(
                    "invalid_frontmatter", "MEMORY.md indexes must not have frontmatter"
                )
            return body if not body or body.endswith("\n") else body + "\n"
        safe_name = str(name or self._default_memory_name(path)).strip()
        safe_description = str(description or "").strip()
        if not safe_name or not safe_description or "\n" in safe_name or "\n" in safe_description:
            raise LettaActionError("invalid_frontmatter", "name/description must be one non-empty line")
        header = (
            "---\n"
            f"name: {json.dumps(safe_name, ensure_ascii=False)}\n"
            f"description: {json.dumps(safe_description, ensure_ascii=False)}\n"
            "---\n"
        )
        return header + body

    def _label_without_suffix(self, path: Path) -> str:
        relative = self._relative(path)
        return relative[:-3] if relative.lower().endswith(".md") else relative

    @staticmethod
    def _default_memory_name(path: Path) -> str:
        stem = path.name[:-3] if path.name.lower().endswith(".md") else path.name
        words = [word for word in re.split(r"[-_]", stem) if word]
        return " ".join(word[:1].upper() + word[1:] for word in words) or "Memory"

    @staticmethod
    def _is_index(path: Path) -> bool:
        return path.name == "MEMORY.md"

    def _assert_indexed(self, path: Path) -> None:
        repo = self._require_repo().resolve()
        relative = path.relative_to(repo)
        if relative == Path("MEMORY.md"):
            return
        if not (repo / "MEMORY.md").is_file():
            raise LettaActionError(
                "missing_memory_index", "Memory requires a root MEMORY.md index"
            )
        current = repo
        directories = relative.parts[:-1]
        for part in directories:
            current = current / part
            marker = current / "MEMORY.md"
            if marker == path:
                break
            if not marker.is_file():
                raise LettaActionError(
                    "missing_memory_index",
                    f"Memory requires {marker.relative_to(repo)} before writing {relative}",
                )

    def _validate_tree(self) -> None:
        repo = self._require_repo().resolve()
        root_index = repo / "MEMORY.md"
        if not root_index.is_file() or root_index.is_symlink():
            raise LettaActionError(
                "invalid_memory_tree", "MemFS requires a regular root MEMORY.md"
            )
        for path in repo.rglob("*"):
            if ".git" in path.relative_to(repo).parts:
                continue
            if path.is_symlink():
                raise LettaActionError(
                    "invalid_memory_tree", "MemFS does not permit symbolic links"
                )
            if path.is_dir():
                continue
            if not path.is_file() or path.suffix.lower() != ".md":
                raise LettaActionError(
                    "invalid_memory_tree", "MemFS contains a non-Markdown file"
                )
            self._assert_indexed(path)
            self._split_frontmatter(path, path.read_text(encoding="utf-8"))

    def _enforce_limits(self) -> None:
        repo = self._require_repo()
        files = [
            path
            for path in repo.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(repo).parts
        ]
        sizes = [path.stat().st_size for path in files]
        if (
            len(files) > self.config.max_files
            or any(size > self.config.max_file_bytes for size in sizes)
            or sum(sizes) > self.config.max_total_bytes
        ):
            raise LettaActionError("memory_capacity_exceeded", "MemFS byte limit exceeded")

    def _relative(self, path: Path) -> str:
        return path.relative_to(self._require_repo()).as_posix()

    def _require_repo(self) -> Path:
        if self._repo is None:
            raise LettaActionError("memfs_unavailable", "episode MemFS is unavailable")
        return self._repo

    def _git(self, *args: str) -> None:
        completed = subprocess.run(["git", "-C", str(self._require_repo()), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise LettaActionError("git_failure", completed.stderr.strip() or "git command failed")

    def _git_output(self, *args: str) -> str:
        completed = subprocess.run(["git", "-C", str(self._require_repo()), *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            raise LettaActionError("git_failure", completed.stderr.strip() or "git command failed")
        return completed.stdout

    def _rollback_repo(self) -> None:
        repo = self._require_repo()
        for command in (
            ["git", "-C", str(repo), "reset", "--hard", "HEAD"],
            ["git", "-C", str(repo), "clean", "-fd"],
        ):
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode:
                raise RuntimeError(
                    "Letta Code failed to roll back a rejected MemFS mutation: "
                    + (completed.stderr.strip() or "git cleanup failed")
                )

    def _identity(self) -> dict[str, Any]:
        if not isinstance(self._episode_source_identity, Mapping):
            raise RuntimeError("Letta Code episode source identity is unavailable")
        return deepcopy(dict(self._episode_source_identity))

    def _cleanup_repo(self) -> None:
        if self._repo is not None:
            shutil.rmtree(self._repo, ignore_errors=True)
            self._repo = None


def _prompt(core_memory: str) -> str:
    rendered_core = core_memory or "(empty)"
    return f"""{LETTA_PROMPT_MARKER}
This baseline exposes an episode-private, git-backed Letta Code MemFS pinned to
v0.31.12 ({LETTA_CODE_SOURCE_REVISION}). Root memory is compiled below into the
system context; child files are read on demand. Every write requires a non-empty
reason and creates a real git commit. A memory tool consumes one ordinary policy
response. Native task actions remain unchanged and cannot be combined with a
memory action.

For a memory write, call exactly one tool and no prose:
<letta_memory_call>{{"name":"memory","arguments":{{"command":"create|str_replace|insert|delete|rename|update_description","reason":"...",...}}}}</letta_memory_call>
or
<letta_memory_call>{{"name":"memory_apply_patch","arguments":{{"reason":"...","input":"*** Begin Patch\\n...\\n*** End Patch"}}}}</letta_memory_call>

The memory tool has no read command. Read a deferred file through the ordinary
CAMG shell tool, for example exactly:
shell_command {{"command":"cat $MEMORY_DIR/reference/fact.md"}}
Do not use shell commands to modify $MEMORY_DIR; all writes go through one of
the two memory tools above. Root and child MEMORY.md files are frontmatter-free
indexes. Every other memory Markdown file has exactly name and description
frontmatter, and each child directory needs its own MEMORY.md index first.

Current root core memory:
{rendered_core}"""


__all__ = [
    "LETTA_CODE_SOURCE_REVISION",
    "LETTA_ADAPTER_SCHEMA",
    "LETTA_PROMPT_MARKER",
    "LettaCodeAdapterConfig",
    "LettaCodeEnvClientAdapter",
    "parse_memory_filesystem_read",
    "parse_letta_action",
]
