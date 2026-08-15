from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from agentenv_mlebench_lite.config import (
    RUNTIME_CONFIG_SCHEMA,
    MLEBenchLiteConfigError,
    load_runtime_config,
)

from tests.support import FAKE_RUNTIME_DIGEST, sha256_bytes, write_fixture


class MLEBenchLiteConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="mlebench-lite-config-")
        self.root = Path(self.temporary.name).resolve()
        self.fixture = write_fixture(self.root)
        self.runner = self.root / "sandbox-runner"
        self.runner.write_bytes(b"#!/bin/sh\nexit 99\n")
        self.runner.chmod(0o500)
        self.forbidden = self.root / "host-secrets"
        self.forbidden.mkdir()
        self.value = {
            "schema": RUNTIME_CONFIG_SCHEMA,
            "upstream_root": str(self.fixture["upstream_root"].resolve()),
            "data_root": str(self.fixture["data_root"].resolve()),
            "public_manifest_path": str(self.fixture["manifest_path"].resolve()),
            "public_manifest_sha256": self.fixture["manifest_sha256"],
            "episodes_root": str((self.root / "episodes-runtime").resolve()),
            "handoff_root": str((self.root / "handoffs-runtime").resolve()),
            "sandbox_runner_path": str(self.runner.resolve()),
            "sandbox_runner_sha256": sha256_bytes(self.runner.read_bytes()),
            "sandbox_runtime_digest": FAKE_RUNTIME_DIGEST,
            "sandbox_runner_uid": os.geteuid(),
            "max_actions": 30,
            "max_submission_bytes": 100_000_000,
            "max_shell_timeout_ms": 120_000,
            "episode_timeout_ms": 600_000,
            "max_total_execution_ms": 300_000,
            "cpu_limit_cores": 4,
            "memory_limit_bytes": 8_000_000_000,
            "pids_limit": 128,
            "writable_bytes_limit": 200_000_000,
            "writable_inodes_limit": 10_000,
            "gpu_count": 1,
            "forbidden_roots": [str(self.forbidden.resolve())],
        }
        self.config_path = self.root / "runtime.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, value=None) -> Path:
        self.config_path.write_text(
            json.dumps(self.value if value is None else value), encoding="utf-8"
        )
        return self.config_path

    def test_strict_external_config_loads_all_pins_and_budgets(self) -> None:
        config = load_runtime_config(self.write())
        self.assertEqual(config.public_manifest_sha256, self.fixture["manifest_sha256"])
        self.assertEqual(config.max_actions, 30)
        self.assertEqual(config.max_submission_bytes, 100_000_000)
        self.assertEqual(config.max_shell_timeout_ms, 120_000)
        self.assertEqual(config.episode_timeout_ms, 600_000)
        self.assertEqual(config.max_total_execution_ms, 300_000)
        self.assertEqual(config.cpu_limit_cores, 4)
        self.assertEqual(config.memory_limit_bytes, 8_000_000_000)
        self.assertEqual(config.pids_limit, 128)
        self.assertEqual(config.writable_bytes_limit, 200_000_000)
        self.assertEqual(config.writable_inodes_limit, 10_000)
        self.assertEqual(config.gpu_count, 1)
        self.assertEqual(config.sandbox_runner_path, self.runner.resolve())
        self.assertEqual(config.sandbox_runner_uid, os.geteuid())

    def test_sandbox_runner_uid_is_strict_nonnegative_and_zero_is_valid(self) -> None:
        for value in (-1, True, 1.0, "0"):
            with self.subTest(value=value), self.assertRaises(MLEBenchLiteConfigError):
                load_runtime_config(
                    self.write({**self.value, "sandbox_runner_uid": value})
                )
        config = load_runtime_config(
            self.write({**self.value, "sandbox_runner_uid": 0})
        )
        self.assertEqual(config.sandbox_runner_uid, 0)

    def test_missing_extra_relative_symlink_and_overlap_fail_closed(self) -> None:
        variants = []
        missing = dict(self.value)
        missing.pop("max_actions")
        variants.append(("missing", missing))
        extra = {**self.value, "unexpected": True}
        variants.append(("extra", extra))
        relative = {**self.value, "data_root": "relative/data"}
        variants.append(("relative", relative))
        overlap = {
            **self.value,
            "episodes_root": str(
                (self.fixture["data_root"] / "runtime-episodes").resolve()
            ),
        }
        variants.append(("overlap", overlap))
        runner_link = self.root / "runner-link"
        runner_link.symlink_to(self.runner)
        symlink = {**self.value, "sandbox_runner_path": str(runner_link)}
        variants.append(("symlink", symlink))
        for name, value in variants:
            with self.subTest(name=name), self.assertRaises(MLEBenchLiteConfigError):
                load_runtime_config(self.write(value))


if __name__ == "__main__":
    unittest.main()
