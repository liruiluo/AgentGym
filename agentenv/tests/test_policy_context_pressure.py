from __future__ import annotations

import unittest

from agentenv.controller.types import PolicyContextPressure


class PolicyContextPressureTests(unittest.TestCase):
    def test_candidate_can_be_shorter_after_history_normalization(self) -> None:
        pressure = PolicyContextPressure(
            action_prompt_tokens=140,
            candidate_prompt_tokens=130,
            max_prompt_tokens=300,
            max_model_tokens=332,
            max_response_tokens=32,
            max_observation_tokens=100,
            action_observation_envelope_tokens=4,
        )

        self.assertEqual(pressure.action_prompt_tokens, 140)
        self.assertEqual(pressure.candidate_prompt_tokens, 130)
        self.assertEqual(pressure.projected_next_prompt_tokens_without_control, 276)
        self.assertEqual(pressure.effective_prompt_capacity, 300)


if __name__ == "__main__":
    unittest.main()
