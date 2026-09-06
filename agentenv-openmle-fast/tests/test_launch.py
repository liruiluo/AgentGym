from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agentenv_openmle_fast.executor import OpenMLEFastResourceLimits
from agentenv_openmle_fast.launch import (
    _required_float,
    _validate_timeout_margins,
)


class OpenMLEFastLaunchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = OpenMLEFastResourceLimits.frozen_v1()

    def test_grader_client_timeout_covers_total_not_only_worker_wall(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "total grader wall"):
            _validate_timeout_margins(
                limits=self.limits,
                grader_timeout=9.5,
                grader_margin=2.0,
                client_timeout=200.0,
                client_margin=5.0,
            )

    def test_frozen_grader_deadlines_preserve_nested_timeout_order(self) -> None:
        self.assertEqual(self.limits.grader_worker_wall_ms, 7_000)
        self.assertEqual(self.limits.grader_total_wall_ms, 7_500)
        self.assertLess(
            self.limits.grader_worker_wall_ms,
            self.limits.grader_total_wall_ms,
        )
        self.assertLess(
            self.limits.grader_total_wall_ms / 1000.0 + 2.0,
            10.0,
        )

    def test_ppo_client_timeout_covers_episode_capped_step(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "episode-capped step"):
            _validate_timeout_margins(
                limits=self.limits,
                grader_timeout=10.0,
                grader_margin=2.0,
                client_timeout=30.0,
                client_margin=5.0,
            )

    def test_timeout_margins_accept_frozen_safe_values(self) -> None:
        _validate_timeout_margins(
            limits=self.limits,
            grader_timeout=10.0,
            grader_margin=2.0,
            client_timeout=200.0,
            client_margin=5.0,
        )

    def test_nonfinite_timeouts_fail_closed(self) -> None:
        for raw in ("nan", "inf", "-inf"):
            with (
                self.subTest(raw=raw),
                patch.dict(os.environ, {"OPENMLE_TEST_TIMEOUT": raw}),
                self.assertRaisesRegex(RuntimeError, "finite"),
            ):
                _required_float("OPENMLE_TEST_TIMEOUT")
        for value in (float("nan"), float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(RuntimeError, "finite"),
            ):
                _validate_timeout_margins(
                    limits=self.limits,
                    grader_timeout=value,
                    grader_margin=1.0,
                    client_timeout=200.0,
                    client_margin=5.0,
                )


if __name__ == "__main__":
    unittest.main()
