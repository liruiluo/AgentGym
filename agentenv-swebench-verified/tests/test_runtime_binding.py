from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import os
import py_compile
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from agentenv_swebench_verified.images import (
    PRODUCTION_IMAGE_PINS,
    FrozenImagePins,
    VerifiedImageManifest,
    VerifiedImageManifestError,
)
from agentenv_swebench_verified.launch import (
    ENV_PREFIX,
    limits_from_environment,
    required_path,
)
from agentenv_swebench_verified.protocol import HARNESS_REVISION, HARNESS_TAG
from agentenv_swebench_verified.sandbox import (
    VerifiedSandboxError,
    resolve_cached_profile_image,
)
from agentenv_swebench_verified.testspec import (
    OfficialTestSpecResolver,
    TestSpecBindingError,
)
import agentenv_swebench_verified.testspec as testspec_module


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "harness"
        module = self.source / "swebench" / "harness" / "test_spec"
        module.mkdir(parents=True)
        for package in (
            self.source / "swebench",
            self.source / "swebench" / "harness",
            module,
        ):
            (package / "__init__.py").write_text("", encoding="utf-8")
        (module / "test_spec.py").write_text(
            "from types import SimpleNamespace\n"
            "CALLS = []\n"
            "def make_test_spec(instance, namespace=None):\n"
            "    CALLS.append({'instance': dict(instance), 'namespace': namespace})\n"
            "    return SimpleNamespace(\n"
            "        instance_id=instance['instance_id'],\n"
            "        repo=instance['repo'],\n"
            "        platform='linux/x86_64',\n"
            "        instance_image_key=(\n"
            "            namespace + '/sweb.eval.x86_64.' +\n"
            "            instance['instance_id'].lower().replace('__', '_1776_') +\n"
            "            ':latest'\n"
            "        ),\n"
            "        eval_script='SECRET_EVAL_SCRIPT',\n"
            "        FAIL_TO_PASS=['SECRET_F2P'],\n"
            "        PASS_TO_PASS=['SECRET_P2P'],\n"
            "    )\n",
            encoding="utf-8",
        )
        self.git("init", "-q", "-b", "main")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")
        self.git("add", ".")
        self.git("commit", "-q", "-m", "fake pinned harness")
        self.git("tag", HARNESS_TAG)
        self.revision = self.git("rev-parse", "HEAD").strip()
        self.row = {
            "instance_id": "Owner__Repo-1",
            "repo": "owner/repo",
            "base_commit": "a" * 40,
            "problem_statement": "Repair it",
            "version": "1.0",
            "test_patch": "SECRET_TEST_PATCH",
            "FAIL_TO_PASS": "[]",
            "PASS_TO_PASS": "[]",
        }

    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name == "swebench" or name.startswith("swebench."):
                sys.modules.pop(name, None)
        self.temporary.cleanup()

    def git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_production_harness_and_image_pins_match_the_audit(self) -> None:
        self.assertEqual(
            HARNESS_REVISION,
            "726c5461e2ef52d83cf1ea2107870a8bb3328d57",
        )
        self.assertEqual(HARNESS_TAG, "v4.1.0")
        self.assertEqual(PRODUCTION_IMAGE_PINS.tag_count, 500)
        self.assertEqual(
            PRODUCTION_IMAGE_PINS.tag_ledger_sha256,
            "b69e618cfcfd2a59c3897e3f4856dbd88c4eeb921a5b24467a90bff6fa48581a",
        )

    def test_policy_shell_limits_are_frozen_to_current_training_contract(self) -> None:
        names = {
            f"{ENV_PREFIX}STDOUT_BYTES",
            f"{ENV_PREFIX}STDERR_BYTES",
            f"{ENV_PREFIX}DEFAULT_TIMEOUT_MS",
            f"{ENV_PREFIX}MAX_TIMEOUT_MS",
        }
        clean_environment = {key: value for key, value in os.environ.items() if key not in names}
        with patch.dict(os.environ, clean_environment, clear=True):
            limits = limits_from_environment(6144)
        self.assertEqual(limits.stdout_bytes, 3072)
        self.assertEqual(limits.stderr_bytes, 3072)
        self.assertEqual(limits.default_timeout_ms, 120_000)
        self.assertEqual(limits.max_timeout_ms, 120_000)

        for name, value in (
            ("STDOUT_BYTES", "3073"),
            ("STDERR_BYTES", "3071"),
            ("DEFAULT_TIMEOUT_MS", "119999"),
            ("MAX_TIMEOUT_MS", "120001"),
        ):
            with self.subTest(name=name):
                with patch.dict(
                    os.environ,
                    clean_environment | {f"{ENV_PREFIX}{name}": value},
                    clear=True,
                ):
                    with self.assertRaisesRegex(ValueError, "frozen"):
                        limits_from_environment(6144)

    def test_resolver_uses_only_the_expected_checkout(self) -> None:
        resolver = OfficialTestSpecResolver(
            source_root=self.source,
            expected_revision=self.revision,
            expected_tag=HARNESS_TAG,
        )

        binding = resolver.resolve(self.row)

        self.assertEqual(binding.source_revision, self.revision)
        self.assertEqual(binding.source_tag, HARNESS_TAG)
        self.assertEqual(binding.namespace, "swebench")
        self.assertEqual(binding.platform, "linux/x86_64")
        self.assertEqual(
            binding.instance_image_key,
            "swebench/sweb.eval.x86_64.owner_1776_repo-1:latest",
        )
        self.assertNotIn("SECRET", repr(binding))
        module = sys.modules["swebench.harness.test_spec.test_spec"]
        self.assertEqual(module.CALLS[-1]["namespace"], "swebench")
        self.assertEqual(
            module.CALLS[-1]["instance"]["test_patch"],
            "SECRET_TEST_PATCH",
        )

    def test_concurrent_first_resolves_share_one_pinned_import(self) -> None:
        resolver = OfficialTestSpecResolver(
            source_root=self.source,
            expected_revision=self.revision,
            expected_tag=HARNESS_TAG,
        )
        materialize = testspec_module.materialize_pinned_source

        def slow_materialize(*args, **kwargs):
            time.sleep(0.05)
            return materialize(*args, **kwargs)

        with patch.object(
            testspec_module,
            "materialize_pinned_source",
            side_effect=slow_materialize,
        ):
            with ThreadPoolExecutor(max_workers=16) as executor:
                bindings = list(executor.map(resolver.resolve, [self.row] * 16))

        self.assertEqual(len(bindings), 16)
        self.assertTrue(
            all(binding.instance_id == self.row["instance_id"] for binding in bindings)
        )

    def test_independent_resolvers_share_one_process_wide_pinned_import(self) -> None:
        resolvers = [
            OfficialTestSpecResolver(
                source_root=self.source,
                expected_revision=self.revision,
                expected_tag=HARNESS_TAG,
            )
            for _ in range(16)
        ]

        with ThreadPoolExecutor(max_workers=16) as executor:
            bindings = list(
                executor.map(
                    lambda resolver: resolver.resolve(self.row),
                    resolvers,
                )
            )

        self.assertEqual(len(bindings), 16)
        self.assertTrue(
            all(binding.instance_id == self.row["instance_id"] for binding in bindings)
        )
        module_path = Path(
            sys.modules["swebench.harness.test_spec.test_spec"].__file__
        )
        for resolver in resolvers:
            resolver.close()
        self.assertTrue(module_path.is_file())

        fresh = OfficialTestSpecResolver(
            source_root=self.source,
            expected_revision=self.revision,
            expected_tag=HARNESS_TAG,
        )
        self.assertEqual(fresh.resolve(self.row).instance_id, self.row["instance_id"])

    def test_stale_process_binding_evicts_all_materialized_modules(self) -> None:
        resolver = OfficialTestSpecResolver(
            source_root=self.source,
            expected_revision=self.revision,
            expected_tag=HARNESS_TAG,
        )
        resolver.resolve(self.row)
        sys.modules.pop("swebench.harness.test_spec.test_spec")

        fresh = OfficialTestSpecResolver(
            source_root=self.source,
            expected_revision=self.revision,
            expected_tag=HARNESS_TAG,
        )
        self.assertEqual(fresh.resolve(self.row).instance_id, self.row["instance_id"])

    def test_resolver_rejects_a_revision_or_tag_mismatch_before_import(self) -> None:
        with self.assertRaisesRegex(TestSpecBindingError, "revision"):
            OfficialTestSpecResolver(
                source_root=self.source,
                expected_revision="f" * 40,
                expected_tag=HARNESS_TAG,
            ).resolve(self.row)
        self.assertNotIn("swebench.harness.test_spec.test_spec", sys.modules)

        self.git("tag", "-d", HARNESS_TAG)
        self.git("tag", "wrong-tag")
        with self.assertRaisesRegex(TestSpecBindingError, "tag"):
            OfficialTestSpecResolver(
                source_root=self.source,
                expected_revision=self.revision,
                expected_tag=HARNESS_TAG,
            ).resolve(self.row)

    def test_resolver_rejects_a_dirty_checkout_before_import(self) -> None:
        target = self.source / "swebench" / "harness" / "test_spec" / "test_spec.py"
        target.write_text(
            target.read_text(encoding="utf-8") + "\nDIRTY = True\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(TestSpecBindingError, "not clean"):
            OfficialTestSpecResolver(
                source_root=self.source,
                expected_revision=self.revision,
                expected_tag=HARNESS_TAG,
            ).resolve(self.row)
        self.assertNotIn("swebench.harness.test_spec.test_spec", sys.modules)

    def test_resolver_rejects_assume_unchanged_source_edits(self) -> None:
        relative = "swebench/harness/test_spec/test_spec.py"
        target = self.source / relative
        self.git("update-index", "--assume-unchanged", relative)
        target.write_text(
            target.read_text(encoding="utf-8") + "\nBYPASS = True\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(TestSpecBindingError, "index flags"):
            OfficialTestSpecResolver(
                source_root=self.source,
                expected_revision=self.revision,
                expected_tag=HARNESS_TAG,
            ).resolve(self.row)
        self.assertNotIn("swebench.harness.test_spec.test_spec", sys.modules)

    def test_resolver_rejects_replacement_refs(self) -> None:
        target = self.source / "swebench" / "harness" / "test_spec" / "test_spec.py"
        target.write_text(
            target.read_text(encoding="utf-8") + "\nREPLACEMENT = True\n",
            encoding="utf-8",
        )
        self.git("add", ".")
        self.git("commit", "-q", "-m", "replacement source")
        replacement = self.git("rev-parse", "HEAD").strip()
        self.git("update-ref", "HEAD", self.revision)
        self.git("replace", self.revision, replacement)

        with self.assertRaisesRegex(TestSpecBindingError, "replacement refs"):
            OfficialTestSpecResolver(
                source_root=self.source,
                expected_revision=self.revision,
                expected_tag=HARNESS_TAG,
            ).resolve(self.row)
        self.assertNotIn("swebench.harness.test_spec.test_spec", sys.modules)

    def test_resolver_imports_a_materialized_tree_without_ignored_residue(self) -> None:
        target = self.source / "swebench" / "harness" / "test_spec" / "test_spec.py"
        excluded = self.source / ".git" / "info" / "exclude"
        excluded.write_text(
            excluded.read_text(encoding="utf-8") + "\n*.pyc\n",
            encoding="utf-8",
        )
        malicious_source = self.root / "ignored_residue.py"
        malicious_source.write_text(
            target.read_text(encoding="utf-8")
            + "\nIGNORED_PYC_EXECUTED = True\n",
            encoding="utf-8",
        )
        cache_path = Path(importlib.util.cache_from_source(str(target)))
        cache_path.parent.mkdir(parents=True)
        py_compile.compile(
            str(malicious_source),
            cfile=str(cache_path),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
        self.assertEqual(self.git("status", "--porcelain", "--untracked-files=all"), "")

        resolver = OfficialTestSpecResolver(
            source_root=self.source,
            expected_revision=self.revision,
            expected_tag=HARNESS_TAG,
        )
        resolver.resolve(self.row)

        module = sys.modules["swebench.harness.test_spec.test_spec"]
        self.assertFalse(hasattr(module, "IGNORED_PYC_EXECUTED"))
        self.assertFalse(Path(module.__file__).resolve().is_relative_to(self.source))

    def test_image_manifest_allows_shared_digests_and_resolves_exact_tag(self) -> None:
        tags = (
            "swebench/sweb.eval.x86_64.owner_1776_repo-1:latest",
            "swebench/sweb.eval.x86_64.owner_1776_repo-2:latest",
        )
        digest = "sha256:" + "a" * 64
        path = self.root / "digests.tsv"
        path.write_text(
            "".join(f"{tag}\t{digest}\n" for tag in tags),
            encoding="utf-8",
        )
        ledger = "".join(f"{tag}\n" for tag in tags).encode()
        pins = FrozenImagePins(
            tag_count=2,
            tag_ledger_sha256=hashlib.sha256(ledger).hexdigest(),
        )
        manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = VerifiedImageManifest(
            path,
            pins=pins,
            expected_manifest_sha256=manifest_sha256,
        )
        resolver = OfficialTestSpecResolver(
            source_root=self.source,
            expected_revision=self.revision,
            expected_tag=HARNESS_TAG,
        )
        binding = resolver.resolve(self.row)

        self.assertEqual(manifest.resolve(binding), digest)
        self.assertEqual(manifest.aliases_for_digest(digest), tags)
        self.assertEqual(manifest.unique_digest_count, 1)
        self.assertEqual(manifest.manifest_sha256, manifest_sha256)

        with self.assertRaisesRegex(
            VerifiedImageManifestError,
            "manifest SHA-256",
        ):
            VerifiedImageManifest(
                path,
                pins=pins,
                expected_manifest_sha256="f" * 64,
            )

    def test_image_manifest_rejects_duplicate_or_foreign_tags(self) -> None:
        tag = "swebench/sweb.eval.x86_64.owner_1776_repo-1:latest"
        digest = "sha256:" + "b" * 64
        path = self.root / "bad.tsv"
        path.write_text(f"{tag}\t{digest}\n{tag}\t{digest}\n", encoding="utf-8")
        ledger = f"{tag}\n{tag}\n".encode()
        pins = FrozenImagePins(2, hashlib.sha256(ledger).hexdigest())
        with self.assertRaisesRegex(VerifiedImageManifestError, "unique"):
            VerifiedImageManifest(
                path,
                pins=pins,
                expected_manifest_sha256=hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            )

        path.write_text(f"foreign/image:latest\t{digest}\n", encoding="utf-8")
        pins = FrozenImagePins(
            1,
            hashlib.sha256(b"foreign/image:latest\n").hexdigest(),
        )
        with self.assertRaisesRegex(VerifiedImageManifestError, "tag"):
            VerifiedImageManifest(
                path,
                pins=pins,
                expected_manifest_sha256=hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
            )

    def test_digest_cache_accepts_an_allowed_stored_tag_alias(self) -> None:
        digest = "sha256:" + "c" * 64
        requested = "swebench/sweb.eval.x86_64.owner_1776_repo-1:latest"
        stored = "swebench/sweb.eval.x86_64.owner_1776_repo-2:latest"
        cache_root = self.root / "oci"
        cache_dir = cache_root / f"sha256-{digest.removeprefix('sha256:')}"
        cache_dir.mkdir(parents=True)
        metadata = cache_dir / "metadata.json"
        metadata.write_text(
            '{"repo_profile_image":"' + stored + '"}',
            encoding="utf-8",
        )

        self.assertEqual(
            resolve_cached_profile_image(
                cache_root,
                digest=digest,
                allowed_images=(requested, stored),
            ),
            stored,
        )

        metadata.write_text(
            '{"repo_profile_image":"foreign/image:latest"}',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(VerifiedSandboxError, "alias"):
            resolve_cached_profile_image(
                cache_root,
                digest=digest,
                allowed_images=(requested, stored),
            )

    def test_runtime_paths_fail_closed(self) -> None:
        shared = self.root / "shared"
        shared.mkdir(mode=0o755)
        linked = self.root / "linked"
        linked.symlink_to(shared, target_is_directory=True)
        variable = f"{ENV_PREFIX}PREDICTIONS_ROOT"

        with patch.dict(os.environ, {variable: str(linked)}):
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                required_path("PREDICTIONS_ROOT", file=False, create=True)
        self.assertEqual(stat.S_IMODE(shared.stat().st_mode), 0o755)

        with patch.dict(os.environ, {variable: str(shared)}):
            with self.assertRaisesRegex(RuntimeError, "private"):
                required_path("PREDICTIONS_ROOT", file=False, create=True)
        self.assertEqual(stat.S_IMODE(shared.stat().st_mode), 0o755)

        with patch.dict(os.environ, {variable: "/"}):
            with self.assertRaisesRegex(RuntimeError, "dedicated leaf"):
                required_path("PREDICTIONS_ROOT", file=False, create=True)


if __name__ == "__main__":
    unittest.main()
