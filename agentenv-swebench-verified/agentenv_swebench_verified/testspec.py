from __future__ import annotations

import io
import importlib
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .protocol import (
    HARNESS_REVISION,
    HARNESS_TAG,
    policy_projection,
    require_nonempty_text,
)


TESTSPEC_BINDING_CONTRACT = "swebench_v4_1_0_make_test_spec_binding_v1"
MAX_HARNESS_SOURCE_ARCHIVE_BYTES = 128 * 1024 * 1024
_TESTSPEC_IMPORT_LOCK = threading.Lock()
_TESTSPEC_MODULE_NAME = "swebench.harness.test_spec.test_spec"


class TestSpecBindingError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedTestSpecBinding:
    instance_id: str
    repo: str
    base_commit: str
    instance_image_key: str
    platform: str
    namespace: str
    source_revision: str
    source_tag: str
    contract: str = TESTSPEC_BINDING_CONTRACT
    _private_test_spec: Any = field(default=None, repr=False, compare=False)

    def runtime_metadata(self) -> dict[str, str]:
        return {
            "contract": self.contract,
            "instance_image_key": self.instance_image_key,
            "platform": self.platform,
            "namespace": self.namespace,
            "source_revision": self.source_revision,
            "source_tag": self.source_tag,
        }


@dataclass
class _ProcessTestSpecImport:
    source_revision: str
    source_tag: str
    make_test_spec: Callable[..., Any]
    materialized_source: tempfile.TemporaryDirectory[str]
    import_root: Path
    module: Any


_PROCESS_TESTSPEC_IMPORT: _ProcessTestSpecImport | None = None


class OfficialTestSpecResolver:
    """Bind a private dataset row through the exact pinned harness checkout."""

    def __init__(
        self,
        *,
        source_root: Path | str,
        expected_revision: str = HARNESS_REVISION,
        expected_tag: str = HARNESS_TAG,
    ) -> None:
        self.source_root = require_real_directory(
            Path(source_root).expanduser(), "SWE-bench harness source"
        )
        self.expected_revision = expected_revision
        self.expected_tag = expected_tag
        self._make_test_spec: Callable[..., Any] | None = None

    def close(self) -> None:
        self._make_test_spec = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def resolve(self, instance: Mapping[str, Any]) -> VerifiedTestSpecBinding:
        make_test_spec = self.load_make_test_spec()
        try:
            projected = policy_projection(instance)
            require_nonempty_text(instance, "version")
            if not isinstance(instance.get("test_patch"), str):
                raise ValueError("test_patch must be text")
            for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                if not isinstance(instance.get(key), (str, list)):
                    raise ValueError(f"{key} has invalid type")
            test_spec = make_test_spec(dict(instance), namespace="swebench")
        except Exception as exc:
            raise TestSpecBindingError(
                "pinned v4.1.0 make_test_spec could not bind the private row"
            ) from exc

        instance_id = required_attribute(test_spec, "instance_id")
        repo = required_attribute(test_spec, "repo")
        image_key = required_attribute(test_spec, "instance_image_key")
        platform = required_attribute(test_spec, "platform")
        if instance_id != projected["instance_id"] or repo != projected["repo"]:
            raise TestSpecBindingError(
                "TestSpec identity disagrees with the dataset row"
            )
        if platform != "linux/x86_64":
            raise TestSpecBindingError("Verified policy runtime requires linux/x86_64")
        if not image_key.startswith("swebench/") or not image_key.endswith(":latest"):
            raise TestSpecBindingError(
                "TestSpec instance image key is outside the swebench namespace"
            )
        return VerifiedTestSpecBinding(
            instance_id=instance_id,
            repo=repo,
            base_commit=projected["base_commit"],
            instance_image_key=image_key,
            platform=platform,
            namespace="swebench",
            source_revision=self.expected_revision,
            source_tag=self.expected_tag,
            _private_test_spec=test_spec,
        )

    def load_make_test_spec(self) -> Callable[..., Any]:
        if self._make_test_spec is not None:
            return self._make_test_spec
        with _TESTSPEC_IMPORT_LOCK:
            if self._make_test_spec is not None:
                return self._make_test_spec
            return self._load_make_test_spec()

    def _load_make_test_spec(self) -> Callable[..., Any]:
        global _PROCESS_TESTSPEC_IMPORT

        self._validate_source_checkout()
        process_import = active_process_testspec_import()
        if process_import is not None:
            if (
                process_import.source_revision != self.expected_revision
                or process_import.source_tag != self.expected_tag
            ):
                raise TestSpecBindingError(
                    "process is already bound to a different pinned harness"
                )
            self._make_test_spec = process_import.make_test_spec
            return process_import.make_test_spec

        materialized, import_root = materialize_pinned_source(
            self.source_root,
            self.expected_revision,
        )
        try:
            assert_loaded_modules_belong_to(import_root)
            sys.path.insert(0, str(import_root))
            try:
                module = importlib.import_module(_TESTSPEC_MODULE_NAME)
            except Exception as exc:
                raise TestSpecBindingError(
                    "cannot import make_test_spec from the materialized pinned source"
                ) from exc
            finally:
                try:
                    sys.path.remove(str(import_root))
                except ValueError:
                    pass
            module_path = Path(module.__file__).resolve()
            try:
                module_path.relative_to(import_root)
            except ValueError as exc:
                raise TestSpecBindingError(
                    "make_test_spec was imported from outside the materialized source"
                ) from exc
            make_test_spec = getattr(module, "make_test_spec", None)
            if not callable(make_test_spec):
                raise TestSpecBindingError(
                    "pinned harness has no callable make_test_spec"
                )
        except Exception:
            remove_materialized_modules(import_root)
            materialized.cleanup()
            raise

        process_import = _ProcessTestSpecImport(
            source_revision=self.expected_revision,
            source_tag=self.expected_tag,
            make_test_spec=make_test_spec,
            materialized_source=materialized,
            import_root=import_root,
            module=module,
        )
        _PROCESS_TESTSPEC_IMPORT = process_import
        self._make_test_spec = make_test_spec
        return make_test_spec

    def _validate_source_checkout(self) -> None:
        replacement_refs = git_text(
            self.source_root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/replace",
            label="harness replacement refs",
        )
        if replacement_refs:
            raise TestSpecBindingError(
                "harness checkout contains forbidden replacement refs"
            )
        index_state = git_text(
            self.source_root,
            "ls-files",
            "-v",
            label="harness index flags",
        )
        unsafe_index_entries = [
            line
            for line in index_state.splitlines()
            if line and (line[0].islower() or line[0] == "S")
        ]
        if unsafe_index_entries:
            raise TestSpecBindingError(
                "harness checkout contains unsafe assume-unchanged or "
                "skip-worktree index flags"
            )
        revision = git_text(
            self.source_root,
            "rev-parse",
            "--verify",
            "HEAD^{commit}",
            label="harness revision",
        )
        if revision != self.expected_revision:
            raise TestSpecBindingError(
                "harness revision mismatch: expected "
                f"{self.expected_revision}, got {revision}"
            )
        status = git_text(
            self.source_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            label="harness checkout state",
        )
        if status:
            raise TestSpecBindingError(
                "harness checkout is not clean at the pinned revision"
            )
        tags = set(
            git_text(
                self.source_root,
                "tag",
                "--points-at",
                "HEAD",
                label="harness tag",
            ).splitlines()
        )
        if self.expected_tag not in tags:
            raise TestSpecBindingError(
                f"harness tag mismatch: {self.expected_tag} does not point at HEAD"
            )


def active_process_testspec_import() -> _ProcessTestSpecImport | None:
    global _PROCESS_TESTSPEC_IMPORT

    process_import = _PROCESS_TESTSPEC_IMPORT
    if process_import is None:
        return None
    module_is_live = (
        sys.modules.get(_TESTSPEC_MODULE_NAME) is process_import.module
        and process_import.import_root.is_dir()
        and Path(process_import.module.__file__).is_file()
    )
    if not module_is_live:
        remove_materialized_modules(process_import.import_root)
        process_import.materialized_source.cleanup()
        _PROCESS_TESTSPEC_IMPORT = None
        return None
    assert_loaded_modules_belong_to(process_import.import_root)
    return process_import


def remove_materialized_modules(import_root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "swebench" and not name.startswith("swebench."):
            continue
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        try:
            Path(raw_path).resolve().relative_to(import_root)
        except ValueError:
            continue
        sys.modules.pop(name, None)


def materialize_pinned_source(
    source_root: Path,
    revision: str,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    temporary = tempfile.TemporaryDirectory(
        prefix="swebench-verified-harness-source-"
    )
    root = Path(temporary.name).resolve(strict=True)
    git_dir = root / "source.git"
    import_root = root / "checkout"
    try:
        initialized = subprocess.run(
            ["git", "init", "--quiet", "--bare", str(git_dir)],
            capture_output=True,
            env=git_environment(),
        )
        if initialized.returncode != 0:
            raise TestSpecBindingError(
                "cannot initialize the materialized harness source view"
            )
        objects = git_object_directory(source_root)
        if "\n" in str(objects):
            raise TestSpecBindingError(
                "harness object directory contains a newline"
            )
        (git_dir / "objects" / "info" / "alternates").write_text(
            f"{objects}\n",
            encoding="utf-8",
        )
        (git_dir / "info" / "attributes").write_text(
            "* -export-ignore -export-subst\n"
            "** -export-ignore -export-subst\n",
            encoding="utf-8",
        )
        archived = subprocess.run(
            [
                "git",
                f"--git-dir={git_dir}",
                "archive",
                "--format=tar",
                revision,
                "--",
                "swebench",
            ],
            capture_output=True,
            env=git_environment(),
        )
        if archived.returncode != 0:
            raise TestSpecBindingError(
                "cannot archive the pinned harness package source"
            )
        if (
            not archived.stdout
            or len(archived.stdout) > MAX_HARNESS_SOURCE_ARCHIVE_BYTES
        ):
            raise TestSpecBindingError(
                "materialized harness source archive has an invalid size"
            )
        import_root.mkdir(mode=0o700)
        extract_source_archive(archived.stdout, import_root)
        expected_module = (
            import_root / "swebench" / "harness" / "test_spec" / "test_spec.py"
        )
        if not expected_module.is_file() or expected_module.is_symlink():
            raise TestSpecBindingError(
                "materialized harness source lacks make_test_spec"
            )
        return temporary, import_root
    except Exception:
        temporary.cleanup()
        raise


def git_object_directory(source_root: Path) -> Path:
    raw = git_text(
        source_root,
        "rev-parse",
        "--git-path",
        "objects",
        label="harness object directory",
    )
    if not raw:
        raise TestSpecBindingError("harness object directory is empty")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = source_root / candidate
    return require_real_directory(candidate, "harness object directory")


def extract_source_archive(payload: bytes, destination: Path) -> None:
    seen: set[PurePosixPath] = set()
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:")
    except tarfile.TarError as exc:
        raise TestSpecBindingError(
            "materialized harness source archive is invalid"
        ) from exc
    with archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or str(relative) != member.name.rstrip("/")
                or relative in seen
            ):
                raise TestSpecBindingError(
                    "materialized harness source has an unsafe archive path"
                )
            seen.add(relative)
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                target.chmod(0o700)
                continue
            if not member.isfile():
                raise TestSpecBindingError(
                    "materialized harness source has an unsupported entry"
                )
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source = archive.extractfile(member)
            if source is None:
                raise TestSpecBindingError(
                    "materialized harness source entry is unreadable"
                )
            with source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle)
            target.chmod(0o700 if member.mode & 0o111 else 0o600)


def assert_loaded_modules_belong_to(source_root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "swebench" and not name.startswith("swebench."):
            continue
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        try:
            Path(raw_path).resolve().relative_to(source_root)
        except ValueError as exc:
            raise TestSpecBindingError(
                "a swebench module is already loaded from another checkout"
            ) from exc


def required_attribute(value: Any, name: str) -> str:
    item = getattr(value, name, None)
    if not isinstance(item, str) or not item:
        raise TestSpecBindingError(f"TestSpec {name} must be non-empty text")
    return item


def git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
    )
    return environment


def git_text(root: Path, *arguments: str, label: str) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve(strict=True)}",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-C",
            str(root),
            *arguments,
        ],
        capture_output=True,
        text=True,
        env=git_environment(),
    )
    if completed.returncode != 0:
        raise TestSpecBindingError(f"cannot verify {label}")
    return completed.stdout.strip()


def require_real_directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TestSpecBindingError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise TestSpecBindingError(f"{label} must be a real directory")
    return path.resolve(strict=True)
