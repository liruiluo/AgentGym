from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


PATCH_BEGIN = "*** Begin Patch"
PATCH_END = "*** End Patch"


class WorkspacePatchError(ValueError):
    pass


@dataclass(frozen=True)
class PatchHunk:
    section_hint: str | None
    lines: tuple[str, ...]
    end_of_file: bool


@dataclass(frozen=True)
class PatchOperation:
    kind: str
    path: str
    destination: str | None = None
    content_lines: tuple[str, ...] = ()
    hunks: tuple[PatchHunk, ...] = ()


@dataclass(frozen=True)
class WorkspacePatchResult:
    changed_paths: tuple[str, ...]
    added_paths: tuple[str, ...]
    updated_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]


def parse_workspace_patch(text: str) -> tuple[PatchOperation, ...]:
    """Parse the native Codex ``*** Begin Patch`` envelope.

    This parser deliberately accepts only the Add/Update/Delete grammar used by
    ``apply_patch``. It never delegates policy-authored text to a shell or the
    host ``patch`` executable.
    """

    if not isinstance(text, str):
        raise WorkspacePatchError("apply_patch input must be text")
    if "\x00" in text:
        raise WorkspacePatchError("apply_patch input contains a NUL byte")
    lines = text.splitlines()
    if not lines or lines[0] != PATCH_BEGIN or lines[-1] != PATCH_END:
        raise WorkspacePatchError(
            "apply_patch input must start with '*** Begin Patch' and end with "
            "'*** End Patch'"
        )
    if len(lines) == 2:
        raise WorkspacePatchError("apply_patch contains no file operations")

    operations: list[PatchOperation] = []
    cursor = 1
    while cursor < len(lines) - 1:
        header = lines[cursor]
        if header.startswith("*** Add File: "):
            path = _header_path(header, "*** Add File: ")
            cursor += 1
            content: list[str] = []
            while cursor < len(lines) - 1 and not lines[cursor].startswith("*** "):
                line = lines[cursor]
                if not line.startswith("+"):
                    raise WorkspacePatchError(
                        f"Add File content for {path!r} must use '+' lines"
                    )
                content.append(line[1:])
                cursor += 1
            operations.append(
                PatchOperation(kind="add", path=path, content_lines=tuple(content))
            )
            continue

        if header.startswith("*** Delete File: "):
            path = _header_path(header, "*** Delete File: ")
            cursor += 1
            if cursor < len(lines) - 1 and not lines[cursor].startswith("*** "):
                raise WorkspacePatchError(
                    f"Delete File for {path!r} must not contain a body"
                )
            operations.append(PatchOperation(kind="delete", path=path))
            continue

        if header.startswith("*** Update File: "):
            path = _header_path(header, "*** Update File: ")
            cursor += 1
            destination: str | None = None
            if cursor < len(lines) - 1 and lines[cursor].startswith("*** Move to: "):
                destination = _header_path(lines[cursor], "*** Move to: ")
                cursor += 1
            hunks: list[PatchHunk] = []
            while cursor < len(lines) - 1 and not lines[cursor].startswith("*** "):
                hunk_header = lines[cursor]
                if not hunk_header.startswith("@@"):
                    raise WorkspacePatchError(
                        f"Update File for {path!r} expected a '@@' hunk header"
                    )
                section_hint = hunk_header[2:].strip() or None
                cursor += 1
                hunk_lines: list[str] = []
                end_of_file = False
                while cursor < len(lines) - 1:
                    line = lines[cursor]
                    if line == "*** End of File":
                        end_of_file = True
                        cursor += 1
                        break
                    if line.startswith("@@") or line.startswith("*** "):
                        break
                    if not line or line[0] not in " +-":
                        raise WorkspacePatchError(
                            f"Update File hunk for {path!r} has an invalid line prefix"
                        )
                    hunk_lines.append(line)
                    cursor += 1
                if not hunk_lines:
                    raise WorkspacePatchError(
                        f"Update File hunk for {path!r} is empty"
                    )
                hunks.append(
                    PatchHunk(
                        section_hint=section_hint,
                        lines=tuple(hunk_lines),
                        end_of_file=end_of_file,
                    )
                )
            if not hunks:
                raise WorkspacePatchError(
                    f"Update File for {path!r} contains no hunks"
                )
            operations.append(
                PatchOperation(
                    kind="update",
                    path=path,
                    destination=destination,
                    hunks=tuple(hunks),
                )
            )
            continue

        raise WorkspacePatchError(f"unsupported apply_patch header: {header!r}")

    _reject_duplicate_patch_paths(operations)
    return tuple(operations)


def apply_workspace_patch_transaction(
    root: Path,
    operations: Sequence[PatchOperation],
    *,
    normalize_path: Callable[[str], str],
    validate_tree: Callable[[Path], None],
) -> WorkspacePatchResult:
    """Apply all operations to a staging tree and install it atomically."""

    root = Path(root).resolve()
    if not root.is_dir() or root.is_symlink():
        raise WorkspacePatchError("workspace root must be a real directory")
    parent = root.parent
    staging = Path(tempfile.mkdtemp(prefix=".agentmemory-patch-", dir=parent))
    candidate = staging / "workspace"
    try:
        shutil.copytree(root, candidate, symlinks=True)
        added: list[str] = []
        updated: list[str] = []
        deleted: list[str] = []
        for operation in operations:
            source = normalize_path(operation.path)
            if operation.kind == "add":
                _apply_add(candidate, source, operation.content_lines)
                added.append(source)
            elif operation.kind == "delete":
                _apply_delete(candidate, source)
                deleted.append(source)
            elif operation.kind == "update":
                destination = (
                    None
                    if operation.destination is None
                    else normalize_path(operation.destination)
                )
                _apply_update(candidate, source, destination, operation.hunks)
                if destination is None or destination == source:
                    updated.append(source)
                else:
                    deleted.append(source)
                    added.append(destination)
            else:  # pragma: no cover - parser owns the closed operation set.
                raise WorkspacePatchError(
                    f"unsupported parsed patch operation: {operation.kind}"
                )
        validate_tree(candidate)
        replace_workspace_directory(root, candidate)
        changed = sorted(set(added) | set(updated) | set(deleted))
        return WorkspacePatchResult(
            changed_paths=tuple(changed),
            added_paths=tuple(sorted(set(added))),
            updated_paths=tuple(sorted(set(updated))),
            deleted_paths=tuple(sorted(set(deleted))),
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _apply_add(root: Path, relative: str, content_lines: Sequence[str]) -> None:
    path = root / relative
    _require_contained(root, path)
    if path.exists() or path.is_symlink():
        raise WorkspacePatchError(f"Add File target already exists: {relative}")
    _create_real_parents(root, path.parent)
    content = "\n".join(content_lines)
    if content_lines:
        content += "\n"
    _write_utf8(path, content)


def _apply_delete(root: Path, relative: str) -> None:
    path = root / relative
    _require_regular_file(root, path, relative)
    path.unlink()
    _remove_empty_parents(root, path.parent)


def _apply_update(
    root: Path,
    relative: str,
    destination: str | None,
    hunks: Sequence[PatchHunk],
) -> None:
    source = root / relative
    _require_regular_file(root, source, relative)
    try:
        original = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise WorkspacePatchError(
            f"Update File target is not UTF-8 text: {relative}"
        ) from exc
    updated = _apply_hunks(original, hunks, relative)
    target_relative = relative if destination is None else destination
    target = root / target_relative
    if target != source:
        _require_contained(root, target)
        if target.exists() or target.is_symlink():
            raise WorkspacePatchError(
                f"Move destination already exists: {target_relative}"
            )
        _create_real_parents(root, target.parent)
    _write_utf8(target, updated)
    if target != source:
        source.unlink()
        _remove_empty_parents(root, source.parent)


def _apply_hunks(text: str, hunks: Sequence[PatchHunk], relative: str) -> str:
    had_final_newline = text.endswith("\n")
    lines = text.splitlines()
    cursor = 0
    for hunk in hunks:
        if hunk.section_hint is not None:
            cursor = _find_section(lines, hunk.section_hint, cursor, relative)
        old_lines = [line[1:] for line in hunk.lines if line[0] in " -"]
        new_lines = [line[1:] for line in hunk.lines if line[0] in " +"]
        if old_lines:
            index = _find_sequence(
                lines,
                old_lines,
                cursor,
                require_eof=hunk.end_of_file,
            )
            if index is None:
                raise WorkspacePatchError(
                    f"Update File context was not found in {relative}: "
                    f"{old_lines[0]!r}"
                )
        else:
            index = len(lines) if hunk.end_of_file else cursor
        lines[index : index + len(old_lines)] = new_lines
        cursor = index + len(new_lines)
    rendered = "\n".join(lines)
    if lines and had_final_newline:
        rendered += "\n"
    return rendered


def _find_section(
    lines: Sequence[str],
    hint: str,
    start: int,
    relative: str,
) -> int:
    for candidate in (hint, hint.strip()):
        for index in range(start, len(lines)):
            if lines[index] == candidate or lines[index].strip() == candidate:
                return index + 1
    raise WorkspacePatchError(
        f"Update File section was not found in {relative}: {hint!r}"
    )


def _find_sequence(
    lines: Sequence[str],
    pattern: Sequence[str],
    start: int,
    *,
    require_eof: bool,
) -> int | None:
    last = len(lines) - len(pattern)
    if last < start:
        return None
    for normalize in (
        lambda value: value,
        lambda value: value.rstrip(),
        lambda value: value.strip(),
    ):
        normalized_pattern = [normalize(value) for value in pattern]
        indexes = [last] if require_eof else range(start, last + 1)
        for index in indexes:
            if [normalize(value) for value in lines[index : index + len(pattern)]] == normalized_pattern:
                return index
    return None


def _header_path(header: str, prefix: str) -> str:
    path = header[len(prefix) :]
    if not path or path != path.strip():
        raise WorkspacePatchError(f"invalid apply_patch path header: {header!r}")
    return path


def _reject_duplicate_patch_paths(operations: Sequence[PatchOperation]) -> None:
    seen: set[str] = set()
    for operation in operations:
        touched = [operation.path]
        if operation.destination is not None:
            touched.append(operation.destination)
        for path in touched:
            if path in seen:
                raise WorkspacePatchError(
                    f"apply_patch touches the same path more than once: {path}"
                )
            seen.add(path)


def _require_contained(root: Path, path: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise WorkspacePatchError("apply_patch path escapes the workspace") from exc


def _require_regular_file(root: Path, path: Path, relative: str) -> None:
    _require_contained(root, path)
    if not path.exists() or path.is_symlink() or not path.is_file():
        raise WorkspacePatchError(f"patch target is not a regular file: {relative}")
    if os.stat(path).st_nlink != 1:
        raise WorkspacePatchError(f"patch target is hard-linked: {relative}")


def _create_real_parents(root: Path, parent: Path) -> None:
    _require_contained(root, parent)
    relative_parts = parent.relative_to(root).parts
    cursor = root
    for part in relative_parts:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            if cursor.is_symlink() or not cursor.is_dir():
                raise WorkspacePatchError(
                    "apply_patch parent is not a real directory"
                )
        else:
            cursor.mkdir(mode=0o700)


def _write_utf8(path: Path, content: str) -> None:
    try:
        data = content.encode("utf-8")
    except UnicodeEncodeError as exc:  # pragma: no cover - Python str invariant.
        raise WorkspacePatchError("apply_patch content is not valid UTF-8") from exc
    descriptor, temporary = tempfile.mkstemp(prefix=".apply-patch-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _remove_empty_parents(root: Path, parent: Path) -> None:
    while parent != root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def replace_workspace_directory(destination: Path, source: Path) -> None:
    previous = destination.with_name(
        destination.name + f".previous-{os.getpid()}"
    )
    if previous.exists():
        shutil.rmtree(previous)
    os.replace(destination, previous)
    try:
        os.replace(source, destination)
    except Exception:
        os.replace(previous, destination)
        raise
    else:
        shutil.rmtree(previous)
