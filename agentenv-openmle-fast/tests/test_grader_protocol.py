from __future__ import annotations

import unittest

from agentenv_openmle_fast.grader_protocol import (
    GradeRequest,
    GradeResult,
    GraderProtocolError,
)


class GraderProtocolTaskIdentityTest(unittest.TestCase):
    def _request(self, **overrides) -> GradeRequest:
        values = {
            "request_id": "request-1",
            "episode_id": "episode-1",
            "task_id": "-lionel-messi-all-club-goals@1",
            "grader_binding_sha256": "a" * 64,
            "package_identity_sha256": "b" * 64,
            "baseline_score": 0.21667120456429992,
            "ideal_score": 1.0,
            "higher_is_better": True,
            "submission": b"row_id,Type\nT000001,Tap-in\n",
        }
        values.update(overrides)
        return GradeRequest.build(**values)

    def test_leading_hyphen_task_id_roundtrips(self) -> None:
        request = self._request()
        decoded = GradeRequest.from_payload(
            request.payload(),
            max_submission_bytes=64 * 1024 * 1024,
        )
        self.assertEqual(decoded, request)

    def test_leading_hyphen_task_id_roundtrips_in_grade_response(self) -> None:
        request = self._request()
        result = GradeResult(
            request_id=request.request_id,
            episode_id=request.episode_id,
            task_id=request.task_id,
            grader_binding_sha256=request.grader_binding_sha256,
            package_identity_sha256=request.package_identity_sha256,
            baseline_score=request.baseline_score,
            ideal_score=request.ideal_score,
            submission_sha256=request.submission_sha256,
            submission_valid=True,
            native_score=0.5,
            higher_is_better=True,
            normalized_reward=0.1,
            improved_over_baseline=True,
            runtime_success=True,
            terminal_reason="graded_submission",
            classification="graded",
            audit_digest="c" * 64,
        )
        self.assertEqual(GradeResult.from_payload(result.payload()), result)

    def test_request_and_episode_ids_remain_strict(self) -> None:
        for field in ("request_id", "episode_id"):
            with self.subTest(field=field), self.assertRaises(GraderProtocolError):
                self._request(**{field: "-not-allowed"})


if __name__ == "__main__":
    unittest.main()
