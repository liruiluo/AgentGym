from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import MODEL_LABELS, require_arm
from .workspace import VerifiedWorkspace, git_environment, git_object_directory


PATCH_EXPORT_CONTRACT = "swebench_verified_exact_base_solution_diff_v1"
MAX_MODEL_PATCH_BYTES = 16 * 1024 * 1024
MAX_GIT_METADATA_BYTES = 64 * 1024 * 1024
MAX_GIT_STDERR_BYTES = 64 * 1024
GIT_COMMAND_TIMEOUT_SECONDS = 300
PREDICTION_SCHEMA_FIELDS = (
    "instance_id",
    "model_name_or_path",
    "model_patch",
)
ARTIFACT_ROOTS = (
    ".agent_memory",
    ".agent_logs",
    ".agent_receipts",
    ".agent_telemetry",
)
_RUN_ID_RE = re.compile(r"\A[A-Za-z0-9_.-]+\Z")
_RUN_CAPABILITY_CLAIM_FILE = ".run-capability.sha256"
_RUN_CAPABILITY_CLAIM_RE = re.compile(rb"\A[0-9a-f]{64}\n\Z")


class PatchExportError(RuntimeError):
    pass


class UnsupportedSolutionState(PatchExportError):
    pass


class GitOutputLimitError(PatchExportError):
    pass


class PredictionStoreError(RuntimeError):
    pass


class RunCapabilityMismatch(PredictionStoreError):
    pass


class SolutionPatchExporter:
    def export(self, workspace: VerifiedWorkspace) -> str:
        nested_git = find_nested_git_metadata(workspace.policy_root)
        if nested_git:
            raise UnsupportedSolutionState(
                "solution workspace contains unsupported nested Git metadata"
            )
        index_path = workspace.private_root / "export.index"
        lock_path = workspace.private_root / "export.index.lock"
        private_git_dir = workspace.private_root / "export.git"
        for path in (index_path, lock_path):
            if path.exists() or path.is_symlink():
                path.unlink()
        if private_git_dir.exists() or private_git_dir.is_symlink():
            raise PatchExportError("private export repository already exists")
        initialize_private_export_repository(workspace, private_git_dir)
        environment = git_environment()
        environment.update(
            {
                "GIT_INDEX_FILE": str(index_path),
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            }
        )
        command = self.git_command(workspace, git_dir=private_git_dir)
        try:
            run_git(
                [*command, "read-tree", workspace.base_commit],
                environment,
                "initialize private export index",
            )
            gitlinks = base_gitlinks(
                workspace,
                environment,
                git_dir=private_git_dir,
            )
            validate_base_gitlink_placeholders(workspace.policy_root, gitlinks)
            excluded_paths = [*ARTIFACT_ROOTS, *gitlinks]
            pathspecs = ["."] + [
                f":(top,literal,exclude){path}" for path in excluded_paths
            ]
            try:
                run_git(
                    [*command, "add", "-f", "-A", "--", *pathspecs],
                    environment,
                    "stage solution workspace",
                )
            except PatchExportError as exc:
                raise UnsupportedSolutionState(
                    "solution workspace could not be staged"
                ) from exc
            try:
                completed = run_git(
                    [
                        *command,
                        "diff",
                        "--cached",
                        "--binary",
                        "--full-index",
                        "--no-ext-diff",
                        "--no-color",
                        "--src-prefix=a/",
                        "--dst-prefix=b/",
                        workspace.base_commit,
                        "--",
                    ],
                    environment,
                    "export solution diff",
                    stdout_limit=MAX_MODEL_PATCH_BYTES,
                )
            except PatchExportError as exc:
                raise UnsupportedSolutionState(
                    "solution diff could not be exported within the bounds"
                ) from exc
            try:
                return completed.stdout.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UnsupportedSolutionState(
                    "solution diff is not valid UTF-8"
                ) from exc
        finally:
            for path in (index_path, lock_path):
                if path.exists() or path.is_symlink():
                    path.unlink()
            shutil.rmtree(private_git_dir, ignore_errors=True)

    def prediction_row(
        self, workspace: VerifiedWorkspace, *, arm: str
    ) -> dict[str, str]:
        normalized_arm = require_arm(arm)
        try:
            patch = self.export(workspace)
        except (GitOutputLimitError, UnsupportedSolutionState):
            patch = ""
        return {
            "instance_id": workspace.instance_id,
            "model_name_or_path": MODEL_LABELS[normalized_arm],
            "model_patch": patch,
        }

    @staticmethod
    def git_command(
        workspace: VerifiedWorkspace,
        *,
        git_dir: Path | None = None,
    ) -> list[str]:
        return [
            "git",
            "-c",
            "core.fileMode=true",
            f"--git-dir={git_dir or workspace.git_dir}",
            f"--work-tree={workspace.policy_root}",
        ]


class PredictionStore:
    """Store one row per arm/run/task and assemble canonical-order JSONL."""

    def __init__(self, root: Path | str, *, instance_ids: Sequence[str]) -> None:
        ids = tuple(instance_ids)
        if not ids or len(set(ids)) != len(ids):
            raise PredictionStoreError("instance_ids must be non-empty and unique")
        if any(not isinstance(value, str) or not value for value in ids):
            raise PredictionStoreError("instance_ids must contain non-empty text")
        self.instance_ids = ids
        raw_root = Path(root).expanduser()
        created = False
        try:
            info = raw_root.lstat()
        except FileNotFoundError:
            raw_root.mkdir(parents=True, mode=0o700)
            created = True
            info = raw_root.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise PredictionStoreError("prediction root must be a real directory")
        if created:
            os.chmod(raw_root, 0o700, follow_symlinks=False)
        elif stat.S_IMODE(info.st_mode) & 0o077:
            raise PredictionStoreError("prediction root must be a private directory")
        self.root = raw_root.resolve(strict=True)

    def claim_run(
        self,
        *,
        arm: str,
        run_id: str,
        capability_digest: bytes,
    ) -> Path:
        """Atomically bind an arm/run namespace to one bearer digest."""

        normalized_arm = require_arm(arm)
        normalized_run = validate_run_id(run_id)
        if (
            not isinstance(capability_digest, bytes)
            or len(capability_digest) != hashlib.sha256().digest_size
        ):
            raise PredictionStoreError(
                "run capability digest must be one SHA-256 digest"
            )
        run_root = ensure_private_directory(
            self.root,
            (normalized_arm, normalized_run),
            create=True,
        )
        destination = run_root / _RUN_CAPABILITY_CLAIM_FILE
        payload = capability_digest.hex().encode("ascii") + b"\n"
        temporary = create_temporary_path(run_root, ".run-capability-")
        try:
            os.chmod(temporary, 0o600)
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()

        claimed_digest = read_run_capability_digest(destination)
        if not hmac.compare_digest(claimed_digest, capability_digest):
            raise RunCapabilityMismatch("run capability does not own arm/run_id")
        return destination

    def write(
        self,
        *,
        arm: str,
        run_id: str,
        data_idx: int,
        row: Mapping[str, Any],
    ) -> Path:
        normalized_arm = require_arm(arm)
        normalized_run = validate_run_id(run_id)
        normalized_row = self.validate_row(
            normalized_arm, data_idx=data_idx, row=row
        )
        rows_root = ensure_private_directory(
            self.root,
            (normalized_arm, normalized_run, "rows"),
            create=True,
        )
        destination = rows_root / f"{data_idx:04d}.json"
        payload = canonical_row_bytes(normalized_row)
        temporary = create_temporary_path(rows_root, ".prediction-")
        try:
            os.chmod(temporary, 0o600)
            temporary.write_bytes(payload)
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise PredictionStoreError(
                    "duplicate prediction for arm/run/data_idx"
                ) from exc
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        return destination

    def read(self, *, arm: str, run_id: str, data_idx: int) -> dict[str, str]:
        normalized_arm = require_arm(arm)
        normalized_run = validate_run_id(run_id)
        rows_root = ensure_private_directory(
            self.root,
            (normalized_arm, normalized_run, "rows"),
            create=False,
        )
        path = rows_root / (
            f"{validate_data_idx(data_idx, len(self.instance_ids)):04d}.json"
        )
        if not path.is_file() or path.is_symlink():
            raise PredictionStoreError("prediction is unavailable")
        value = json.loads(path.read_text(encoding="utf-8"))
        return self.validate_row(normalized_arm, data_idx=data_idx, row=value)

    def assemble(self, *, arm: str, run_id: str) -> Path:
        normalized_arm = require_arm(arm)
        normalized_run = validate_run_id(run_id)
        output_root = ensure_private_directory(
            self.root,
            (normalized_arm, normalized_run),
            create=True,
        )
        rows_root = ensure_private_directory(
            self.root,
            (normalized_arm, normalized_run, "rows"),
            create=False,
        )
        validate_prediction_inventory(rows_root, len(self.instance_ids))
        destination = output_root / f"{MODEL_LABELS[normalized_arm]}.jsonl"
        temporary = create_temporary_path(output_root, ".predictions-")
        try:
            os.chmod(temporary, 0o600)
            with temporary.open("wb") as handle:
                for data_idx in range(len(self.instance_ids)):
                    try:
                        row = self.read(
                            arm=normalized_arm,
                            run_id=normalized_run,
                            data_idx=data_idx,
                        )
                    except PredictionStoreError as exc:
                        raise PredictionStoreError(
                            "prediction ledger is incomplete"
                        ) from exc
                    handle.write(canonical_row_bytes(row))
            os.replace(temporary, destination)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
        return destination

    def validate_row(
        self, arm: str, *, data_idx: int, row: Mapping[str, Any]
    ) -> dict[str, str]:
        index = validate_data_idx(data_idx, len(self.instance_ids))
        if not isinstance(row, Mapping) or tuple(row) != PREDICTION_SCHEMA_FIELDS:
            raise PredictionStoreError(
                "prediction row must have the exact ordered schema"
            )
        normalized: dict[str, str] = {}
        for key in PREDICTION_SCHEMA_FIELDS:
            value = row.get(key)
            if not isinstance(value, str):
                raise PredictionStoreError(f"prediction {key} must be text")
            normalized[key] = value
        if normalized["instance_id"] != self.instance_ids[index]:
            raise PredictionStoreError("prediction instance_id disagrees with data_idx")
        if normalized["model_name_or_path"] != MODEL_LABELS[arm]:
            raise PredictionStoreError(
                "prediction model_name_or_path is not arm-pinned"
            )
        if len(normalized["model_patch"].encode("utf-8")) > MAX_MODEL_PATCH_BYTES:
            raise PredictionStoreError("prediction model_patch exceeds the byte cap")
        return normalized

    def rows_root(self, arm: str, run_id: str) -> Path:
        return self.root / arm / run_id / "rows"

    def public_metadata(self) -> dict[str, object]:
        ledger = "".join(
            f"{instance_id}\n" for instance_id in self.instance_ids
        ).encode()
        return {
            "schema_fields": list(PREDICTION_SCHEMA_FIELDS),
            "task_count": len(self.instance_ids),
            "instance_id_ledger_sha256": hashlib.sha256(ledger).hexdigest(),
            "model_labels": dict(MODEL_LABELS),
        }


def ensure_private_directory(
    root: Path,
    components: Sequence[str],
    *,
    create: bool,
) -> Path:
    current = root
    validate_private_directory(current)
    for component in components:
        current = current / component
        try:
            current.lstat()
        except FileNotFoundError:
            if not create:
                raise PredictionStoreError("prediction directory is unavailable")
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
        validate_private_directory(current)
    return current


def validate_private_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise PredictionStoreError("prediction directory is unavailable") from exc
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        raise PredictionStoreError(
            "prediction path component must be a real private directory"
        )


def read_run_capability_digest(path: Path) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PredictionStoreError("run capability claim is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_nlink != 1
        ):
            raise PredictionStoreError(
                "run capability claim must be a regular private file"
            )
        payload = os.read(descriptor, 66)
        if os.read(descriptor, 1):
            raise PredictionStoreError(
                "run capability claim has non-canonical content"
            )
    finally:
        os.close(descriptor)
    if _RUN_CAPABILITY_CLAIM_RE.fullmatch(payload) is None:
        raise PredictionStoreError(
            "run capability claim has non-canonical content"
        )
    return bytes.fromhex(payload[:-1].decode("ascii"))


def validate_prediction_inventory(rows_root: Path, task_count: int) -> None:
    expected = {f"{data_idx:04d}.json" for data_idx in range(task_count)}
    entries = tuple(rows_root.iterdir())
    actual = {entry.name for entry in entries}
    if actual - expected:
        raise PredictionStoreError("prediction ledger contains unexpected entries")
    if expected - actual:
        raise PredictionStoreError("prediction ledger is incomplete")
    for entry in entries:
        try:
            info = entry.lstat()
        except FileNotFoundError as exc:
            raise PredictionStoreError("prediction ledger is incomplete") from exc
        if not stat.S_ISREG(info.st_mode):
            raise PredictionStoreError(
                "prediction ledger contains an unexpected non-regular entry"
            )


def run_git(
    command: list[str],
    environment: Mapping[str, str],
    label: str,
    *,
    stdout_limit: int = MAX_GIT_METADATA_BYTES,
    stderr_limit: int = MAX_GIT_STDERR_BYTES,
) -> subprocess.CompletedProcess[bytes]:
    with (
        tempfile.TemporaryFile() as stdout_file,
        tempfile.TemporaryFile() as stderr_file,
    ):
        process = subprocess.Popen(
            command,
            cwd=command_worktree(command),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        deadline = time.monotonic() + GIT_COMMAND_TIMEOUT_SECONDS
        failure: PatchExportError | None = None
        while process.poll() is None:
            if os.fstat(stdout_file.fileno()).st_size > stdout_limit:
                failure = GitOutputLimitError(
                    f"git output exceeded {stdout_limit} bytes while trying to {label}"
                )
                break
            if os.fstat(stderr_file.fileno()).st_size > stderr_limit:
                failure = GitOutputLimitError(
                    f"git stderr exceeded {stderr_limit} bytes while trying to {label}"
                )
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = PatchExportError(f"git timed out while trying to {label}")
                break
            try:
                process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                continue
        if failure is not None:
            terminate_process_group(process)
            raise failure
        stdout_size = os.fstat(stdout_file.fileno()).st_size
        if stdout_size > stdout_limit:
            raise GitOutputLimitError(
                f"git output exceeded {stdout_limit} bytes while trying to {label}"
            )
        stderr_size = os.fstat(stderr_file.fileno()).st_size
        if stderr_size > stderr_limit:
            raise GitOutputLimitError(
                f"git stderr exceeded {stderr_limit} bytes while trying to {label}"
            )
        stdout_file.seek(0)
        stdout = stdout_file.read(stdout_limit + 1)
        stderr_file.seek(max(0, stderr_size - 4096))
        stderr = stderr_file.read(4096)
        completed = subprocess.CompletedProcess(
            command,
            process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise PatchExportError(f"git failed to {label}: {detail}")
    return completed


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def command_worktree(command: Sequence[str]) -> Path:
    prefix = "--work-tree="
    for argument in command:
        if argument.startswith(prefix):
            return Path(argument[len(prefix) :])
    raise PatchExportError("git export command has no work tree")


def base_gitlinks(
    workspace: VerifiedWorkspace,
    environment: Mapping[str, str],
    *,
    git_dir: Path,
) -> tuple[str, ...]:
    completed = run_git(
        [
            *SolutionPatchExporter.git_command(workspace, git_dir=git_dir),
            "ls-tree",
            "-r",
            "-z",
            workspace.base_commit,
        ],
        environment,
        "inspect base gitlinks",
    )
    paths: list[str] = []
    for entry in completed.stdout.split(b"\0"):
        if not entry:
            continue
        metadata, separator, raw_path = entry.partition(b"\t")
        if not separator:
            raise PatchExportError("base tree entry has an invalid format")
        mode = metadata.split(b" ", 1)[0]
        if mode == b"160000":
            try:
                paths.append(raw_path.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise PatchExportError("base gitlink path is not UTF-8") from exc
    return tuple(paths)


def initialize_private_export_repository(
    workspace: VerifiedWorkspace,
    destination: Path,
) -> None:
    completed = subprocess.run(
        ["git", "init", "--quiet", "--bare", str(destination)],
        capture_output=True,
        text=True,
        env=git_environment(),
    )
    if completed.returncode != 0:
        raise PatchExportError("cannot initialize the private export repository")
    try:
        objects = git_object_directory(workspace.mirror_root, workspace.git_dir)
        if "\n" in str(objects):
            raise PatchExportError("mirror object path contains a newline")
        (destination / "objects" / "info" / "alternates").write_text(
            f"{objects}\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def find_nested_git_metadata(root: Path) -> tuple[str, ...]:
    found: list[str] = []
    for current_root, directories, files in os.walk(root, followlinks=False):
        current = Path(current_root)
        if ".git" in directories:
            directories.remove(".git")
            found.append(str((current / ".git").relative_to(root)))
        if ".git" in files:
            found.append(str((current / ".git").relative_to(root)))
    return tuple(sorted(found))


def validate_base_gitlink_placeholders(
    policy_root: Path,
    gitlinks: Sequence[str],
) -> None:
    for relative in gitlinks:
        path = policy_root / relative
        if not path.exists() and not path.is_symlink():
            raise UnsupportedSolutionState(
                "solution workspace deleted an unsupported base gitlink"
            )
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            continue
        raise UnsupportedSolutionState(
            "solution workspace changed an unsupported base gitlink"
        )


def validate_run_id(run_id: str) -> str:
    if (
        not isinstance(run_id, str)
        or run_id in {".", ".."}
        or _RUN_ID_RE.fullmatch(run_id) is None
    ):
        raise PredictionStoreError("run_id contains unsupported characters")
    return run_id


def validate_data_idx(data_idx: int, task_count: int) -> int:
    if isinstance(data_idx, bool) or not isinstance(data_idx, int):
        raise PredictionStoreError("data_idx must be an integer")
    if not 0 <= data_idx < task_count:
        raise PredictionStoreError("data_idx is outside the dataset")
    return data_idx


def canonical_row_bytes(row: Mapping[str, str]) -> bytes:
    return (
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def create_temporary_path(root: Path, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=root)
    os.close(descriptor)
    return Path(raw_path)
