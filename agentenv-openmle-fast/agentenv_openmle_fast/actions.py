from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

ActionKind = Literal["shell_command", "apply_patch", "submit", "parser_error"]
MAX_COMMAND_BYTES = 32 * 1024
MAX_PATCH_BYTES = 256 * 1024
MAX_SHELL_TIMEOUT_MS = 20_000
QWEN_TOOL_CALL_RE = re.compile(
    r"\A<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>\Z",
    flags=re.DOTALL,
)
QWEN_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>(.*?)</parameter>",
    flags=re.DOTALL,
)


class OpenMLEFastActionError(RuntimeError):
    pass


class OpenMLEFastProtectedPathError(OpenMLEFastActionError):
    pass


@dataclass(frozen=True)
class ParsedPolicyAction:
    kind: ActionKind
    raw_output: str
    arguments: Mapping[str, Any] | None = None
    patch: str | None = None
    error: str | None = None

    def as_evidence(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "arguments": None if self.arguments is None else dict(self.arguments),
            "patch_sha256": None,
            "error": self.error,
        }


@dataclass(frozen=True)
class PatchResult:
    changed_paths: tuple[str, ...]


class _DuplicateKey(ValueError):
    pass


def parse_policy_action(raw_output: str) -> ParsedPolicyAction:
    if not isinstance(raw_output, str):
        raise TypeError("OpenMLE-fast policy output must be text")
    text = raw_output.strip()
    qwen_action = _parse_qwen_action(raw_output, text)
    if qwen_action is not None:
        return qwen_action
    if text in {"submit", "submit {}"}:
        return ParsedPolicyAction(kind="submit", raw_output=raw_output)
    if text.startswith("shell_command "):
        return _parse_shell(raw_output, text[len("shell_command ") :])
    if text.startswith("apply_patch\n"):
        return _parse_patch(raw_output, text[len("apply_patch\n") :])
    return _parser_error(
        raw_output, "expected exactly shell_command, apply_patch, or submit"
    )


def _parse_qwen_action(
    raw_output: str, text: str
) -> ParsedPolicyAction | None:
    match = QWEN_TOOL_CALL_RE.fullmatch(text)
    if match is None:
        return None
    action_name = match.group(1)
    if action_name not in {"shell_command", "apply_patch", "submit"}:
        return _parser_error(raw_output, f"unsupported Qwen tool {action_name!r}")

    body = match.group(2)
    matches = list(QWEN_PARAMETER_RE.finditer(body))
    if QWEN_PARAMETER_RE.sub("", body).strip():
        return _parser_error(raw_output, "Qwen tool body contains non-parameter text")

    arguments: dict[str, str] = {}
    for parameter in matches:
        key = parameter.group(1)
        if not key or key in arguments:
            return _parser_error(raw_output, f"invalid or duplicate Qwen parameter {key!r}")
        value = parameter.group(2)
        # Match veRL's Qwen3XMLToolParser: remove at most the formatting newline
        # immediately inside each parameter tag, while preserving intentional
        # spaces and additional newlines in command or patch payloads.
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        arguments[key] = value

    if action_name == "submit":
        if arguments or body.strip():
            return _parser_error(raw_output, "submit accepts no parameters")
        return ParsedPolicyAction(kind="submit", raw_output=raw_output)
    if action_name == "apply_patch":
        if set(arguments) != {"patch"}:
            return _parser_error(raw_output, "apply_patch requires exactly one patch parameter")
        return _parse_patch(raw_output, arguments["patch"])
    if "command" not in arguments or set(arguments) - {
        "command",
        "timeout_ms",
        "workdir",
    }:
        return _parser_error(
            raw_output,
            "shell_command requires command, with optional timeout_ms and workdir",
        )
    payload: dict[str, Any] = {"command": arguments["command"]}
    if "workdir" in arguments:
        payload["workdir"] = arguments["workdir"]
    if "timeout_ms" in arguments:
        try:
            payload["timeout_ms"] = int(arguments["timeout_ms"])
        except ValueError:
            return _parser_error(raw_output, "shell_command timeout_ms must be an integer")
    return _validate_shell_payload(raw_output, payload)


def _parse_shell(raw_output: str, payload_text: str) -> ParsedPolicyAction:
    try:
        payload = json.loads(payload_text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, _DuplicateKey) as exc:
        return _parser_error(raw_output, f"shell_command payload is invalid: {exc}")
    return _validate_shell_payload(raw_output, payload)


def _validate_shell_payload(
    raw_output: str, payload: Any
) -> ParsedPolicyAction:
    if not isinstance(payload, dict) or set(payload) - {
        "command",
        "timeout_ms",
        "workdir",
    }:
        return _parser_error(
            raw_output,
            "shell_command accepts command, timeout_ms, and fixed workdir only",
        )
    if "workdir" in payload and payload["workdir"] != ".":
        return _parser_error(raw_output, "shell_command workdir must be exactly '.'")
    command = payload.get("command")
    if not isinstance(command, str) or not command or "\x00" in command:
        return _parser_error(
            raw_output, "shell_command command must be non-empty UTF-8 text"
        )
    try:
        command_bytes = command.encode("utf-8")
    except UnicodeEncodeError:
        return _parser_error(raw_output, "shell_command command is not valid UTF-8")
    if len(command_bytes) > MAX_COMMAND_BYTES:
        return _parser_error(raw_output, "shell_command command exceeds 32 KiB")
    timeout = payload.get("timeout_ms", MAX_SHELL_TIMEOUT_MS)
    if type(timeout) is not int or not 0 < timeout <= MAX_SHELL_TIMEOUT_MS:
        return _parser_error(
            raw_output, "shell_command timeout exceeds the frozen limit"
        )
    return ParsedPolicyAction(
        kind="shell_command",
        raw_output=raw_output,
        arguments={"command": command, "timeout_ms": timeout},
    )


def _parse_patch(raw_output: str, patch: str) -> ParsedPolicyAction:
    try:
        patch_bytes = patch.encode("utf-8")
    except UnicodeEncodeError:
        return _parser_error(raw_output, "apply_patch payload is not valid UTF-8")
    if len(patch_bytes) > MAX_PATCH_BYTES:
        return _parser_error(raw_output, "apply_patch payload exceeds 256 KiB")
    lines = patch.splitlines()
    if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        return _parser_error(raw_output, "apply_patch payload is incomplete")
    if lines.count("*** Begin Patch") != 1 or lines.count("*** End Patch") != 1:
        return _parser_error(raw_output, "apply_patch payload has nested delimiters")
    return ParsedPolicyAction(kind="apply_patch", raw_output=raw_output, patch=patch)


def _parser_error(raw_output: str, message: str) -> ParsedPolicyAction:
    return ParsedPolicyAction(kind="parser_error", raw_output=raw_output, error=message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate key {key!r}")
        result[key] = value
    return result


def apply_workspace_patch(workspace: Path | str, patch: str) -> PatchResult:
    root = Path(workspace).absolute()
    if root.is_symlink() or not root.is_dir():
        raise OpenMLEFastActionError("workspace must be a real directory")
    operations = _parse_patch_operations(patch)
    staged: dict[str, bytes | None] = {}
    for operation, relative, body in operations:
        path = _workspace_path(root, relative)
        if _protected(relative):
            raise OpenMLEFastProtectedPathError("TASK.md and data are immutable")
        if operation == "add":
            if path.exists() or path.is_symlink():
                raise OpenMLEFastActionError(f"patch add target exists: {relative}")
            staged[relative] = _added_bytes(body)
        elif operation == "delete":
            staged[relative] = None
            _read_workspace_regular(path, relative)
        elif operation == "update":
            original = _read_workspace_regular(path, relative).decode("utf-8")
            staged[relative] = _updated_text(original, body).encode("utf-8")
        else:  # pragma: no cover
            raise OpenMLEFastActionError("unsupported patch operation")

    changed: list[str] = []
    for relative, payload in staged.items():
        path = _workspace_path(root, relative)
        if payload is None:
            path.unlink()
            changed.append(relative)
            continue
        _prepare_patch_parent(root, path.parent)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            # The exact sandbox runs as a fresh unprivileged uid.  Files
            # authored by the root-owned patch controller must remain readable
            # and writable across later policy actions by that sandbox identity.
            os.chmod(temporary, 0o666)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        changed.append(relative)
    return PatchResult(changed_paths=tuple(changed))


def _parse_patch_operations(patch: str) -> list[tuple[str, str, list[str]]]:
    lines = patch.splitlines()
    if len(lines) < 3 or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise OpenMLEFastActionError("invalid apply_patch envelope")
    headers = {
        "*** Add File: ": "add",
        "*** Update File: ": "update",
        "*** Delete File: ": "delete",
    }
    operations: list[tuple[str, str, list[str]]] = []
    index = 1
    while index < len(lines) - 1:
        matched = next(
            (
                (prefix, kind)
                for prefix, kind in headers.items()
                if lines[index].startswith(prefix)
            ),
            None,
        )
        if matched is None:
            raise OpenMLEFastActionError("apply_patch expected a file operation")
        prefix, kind = matched
        relative = _normalize_patch_path(lines[index][len(prefix) :])
        index += 1
        body: list[str] = []
        while index < len(lines) - 1 and not any(
            lines[index].startswith(prefix_value) for prefix_value in headers
        ):
            body.append(lines[index])
            index += 1
        if kind == "delete" and body:
            raise OpenMLEFastActionError("delete operation must not have a body")
        operations.append((kind, relative, body))
    if not operations or len({relative for _, relative, _ in operations}) != len(
        operations
    ):
        raise OpenMLEFastActionError("patch paths must be non-empty and unique")
    return operations


def _normalize_patch_path(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw:
        raise OpenMLEFastActionError("patch path is invalid")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise OpenMLEFastActionError("patch path must stay within the workspace")
    return path.as_posix()


def _protected(relative: str) -> bool:
    path = PurePosixPath(relative)
    first = unicodedata.normalize("NFC", path.parts[0]).casefold()
    return first in {"task.md", "data"}


def _workspace_path(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise OpenMLEFastActionError("patch path escapes the workspace") from exc
    _require_real_parent_chain(root, candidate.parent)
    return candidate


def _prepare_patch_parent(root: Path, parent: Path) -> None:
    relative = parent.absolute().relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            os.mkdir(current, 0o777)
            os.chmod(current, 0o777)
            continue
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OpenMLEFastActionError("patch path traverses a non-directory")
    _require_real_parent_chain(root, parent)


def _require_real_parent_chain(root: Path, parent: Path) -> None:
    relative = parent.absolute().relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if not current.exists():
            continue
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise OpenMLEFastActionError("patch path traverses a non-directory")


def _read_workspace_regular(path: Path, relative: str) -> bytes:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise OpenMLEFastActionError(f"patch target is missing: {relative}") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise OpenMLEFastActionError("patch target must be an independent regular file")
    return path.read_bytes()


def _added_bytes(body: list[str]) -> bytes:
    if not body or any(not line.startswith("+") for line in body):
        raise OpenMLEFastActionError("added-file lines must start with +")
    return ("\n".join(line[1:] for line in body) + "\n").encode("utf-8")


def _updated_text(original: str, body: list[str]) -> str:
    chunks: list[list[str]] = []
    current: list[str] | None = None
    for line in body:
        if line.startswith("@@"):
            current = []
            chunks.append(current)
        elif current is None:
            raise OpenMLEFastActionError("update requires an @@ chunk")
        elif line.startswith((" ", "+", "-")):
            current.append(line)
        else:
            raise OpenMLEFastActionError("update line lacks a patch prefix")
    if not chunks:
        raise OpenMLEFastActionError("update contains no chunks")
    lines = original.splitlines()
    trailing_newline = original.endswith("\n")
    cursor = 0
    for chunk in chunks:
        old = [line[1:] for line in chunk if not line.startswith("+")]
        new = [line[1:] for line in chunk if not line.startswith("-")]
        position = _find_subsequence(lines, old, cursor)
        if position is None:
            raise OpenMLEFastActionError("update context does not match the file")
        lines[position : position + len(old)] = new
        cursor = position + len(new)
    result = "\n".join(lines)
    if trailing_newline:
        result += "\n"
    return result


def _find_subsequence(values: list[str], needle: list[str], start: int) -> int | None:
    if not needle:
        return start
    for index in range(start, len(values) - len(needle) + 1):
        if values[index : index + len(needle)] == needle:
            return index
    return None
