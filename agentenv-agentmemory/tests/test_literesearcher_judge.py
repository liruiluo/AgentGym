from __future__ import annotations

import json
import unittest
from unittest.mock import patch
from urllib import error

from agentenv_agentmemory.literesearcher.judge import (
    NormalizedExactLiteResearchJudge,
    UpstreamCompatibleLLMJudge,
    upstream_em,
)


class _Response:
    def __init__(self, content: str) -> None:
        self.payload = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.payload


class LiteResearcherJudgeTests(unittest.TestCase):
    def test_normalized_exact_does_not_accept_semantic_alias(self) -> None:
        result = NormalizedExactLiteResearchJudge().judge(
            "What was the recorder called?",
            ("PSX (digital video recorder)",),
            "PSX",
        )
        self.assertFalse(result.correct)

    def test_upstream_judge_accepts_semantic_equivalence(self) -> None:
        judge = UpstreamCompatibleLLMJudge(
            api_base="http://judge.example/v1",
            model="qwen",
            api_key="secret",
        )
        response = _Response(
            '{"reasoning":"The short name is equivalent.","judgment":"Correct"}'
        )
        with patch(
            "agentenv_agentmemory.literesearcher.judge.request.urlopen",
            return_value=response,
        ) as urlopen:
            result = judge.judge(
                "What was the recorder called?",
                ("PSX (digital video recorder)",),
                "PSX",
            )
        self.assertTrue(result.correct)
        self.assertEqual(result.method, "llm_judge")
        self.assertEqual(result.attempts, 1)
        self.assertGreaterEqual(result.latency_seconds, 0.0)
        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent["model"], "qwen")
        self.assertIn("PSX (digital video recorder)", sent["messages"][0]["content"])

    def test_upstream_judge_parses_incorrect_without_correct_substring_bug(self) -> None:
        judge = UpstreamCompatibleLLMJudge(
            api_base="http://judge.example/v1",
            model="qwen",
        )
        response = _Response(
            '{"reasoning":"The answer is too broad.","judgment":"Incorrect"}'
        )
        with patch(
            "agentenv_agentmemory.literesearcher.judge.request.urlopen",
            return_value=response,
        ):
            result = judge.judge("Which cup?", ("Styrofoam cup",), "Styrofoam")
        self.assertFalse(result.correct)

    def test_failed_judge_retries_then_uses_upstream_em(self) -> None:
        judge = UpstreamCompatibleLLMJudge(
            api_base="http://judge.example/v1",
            model="qwen",
            max_retries=3,
        )
        with (
            patch(
                "agentenv_agentmemory.literesearcher.judge.request.urlopen",
                side_effect=error.URLError("offline"),
            ) as urlopen,
            patch("agentenv_agentmemory.literesearcher.judge.time.sleep"),
        ):
            result = judge.judge("Which laws?", ("penal laws",), "The Penal Laws")
        self.assertTrue(result.correct)
        self.assertEqual(result.method, "upstream_em_fallback")
        self.assertEqual(result.attempts, 3)
        self.assertGreaterEqual(result.latency_seconds, 0.0)
        self.assertEqual(urlopen.call_count, 3)

    def test_metadata_redacts_endpoint_and_key(self) -> None:
        judge = UpstreamCompatibleLLMJudge(
            api_base="https://judge.example/private/v1",
            model="qwen",
            api_key="secret",
        )
        metadata = judge.metadata()
        serialized = json.dumps(metadata)
        self.assertNotIn("judge.example", serialized)
        self.assertNotIn("secret", serialized)
        self.assertEqual(metadata["fallback"], "upstream_em_v1")

    def test_upstream_em_matches_released_normalization_order(self) -> None:
        self.assertTrue(upstream_em("The Penal Laws", ("penal laws",)))
        self.assertTrue(
            upstream_em(
                "PSX digital video recorder",
                ("PSX (digital video recorder)",),
            )
        )


if __name__ == "__main__":
    unittest.main()
