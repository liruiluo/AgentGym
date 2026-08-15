from __future__ import annotations

import unittest

from agentenv_openmle_fast.executor import OpenMLEFastResourceLimits
from agentenv_openmle_fast.launch import _validate_timeout_margins


class OpenMLEFastLaunchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = OpenMLEFastResourceLimits.frozen_v1()

    def test_grader_client_timeout_covers_total_not_only_worker_wall(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "total grader wall"):
            _validate_timeout_margins(
                limits=self.limits,
                grader_timeout=5.5,
                grader_margin=1.0,
                client_timeout=200.0,
                client_margin=5.0,
            )

    def test_ppo_client_timeout_covers_episode_capped_step(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "episode-capped step"):
            _validate_timeout_margins(
                limits=self.limits,
                grader_timeout=7.0,
                grader_margin=1.0,
                client_timeout=30.0,
                client_margin=5.0,
            )

    def test_timeout_margins_accept_frozen_safe_values(self) -> None:
        _validate_timeout_margins(
            limits=self.limits,
            grader_timeout=7.0,
            grader_margin=1.0,
            client_timeout=200.0,
            client_margin=5.0,
        )


if __name__ == "__main__":
    unittest.main()
