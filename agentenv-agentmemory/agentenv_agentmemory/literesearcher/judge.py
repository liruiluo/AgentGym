from __future__ import annotations

import hashlib
import json
import re
import string
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, parse, request

UPSTREAM_LLM_JUDGE_CONTRACT = "upstream_llm_with_em_fallback_v1"
NORMALIZED_EXACT_JUDGE_CONTRACT = "normalized_exact_v1"

_EVALUATION_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is semantically equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Please evaluate the answer and return a JSON object with the following format:
{{
  "reasoning": "A concise explanation of why the predicted answer is equivalent or not equivalent to the labeled answer.",
  "judgment": "Correct"
}}

If the answers are not equivalent, the "judgment" field should be "Incorrect".
Output ONLY the JSON object, without any markdown formatting or additional text.
"""


@dataclass(frozen=True)
class LiteResearchJudgeResult:
    correct: bool
    method: str
    attempts: int
    latency_seconds: float = 0.0


class LiteResearchJudge(Protocol):
    contract_id: str

    def judge(
        self,
        question: str,
        targets: Sequence[str],
        answer: str,
    ) -> LiteResearchJudgeResult:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


def normalized_exact(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def upstream_em_normalized(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def upstream_em(answer: str, targets: Sequence[str]) -> bool:
    normalized_answer = upstream_em_normalized(answer)
    return any(
        upstream_em_normalized(target) == normalized_answer for target in targets
    )


class NormalizedExactLiteResearchJudge:
    contract_id = NORMALIZED_EXACT_JUDGE_CONTRACT

    def judge(
        self,
        question: str,
        targets: Sequence[str],
        answer: str,
    ) -> LiteResearchJudgeResult:
        del question
        normalized = normalized_exact(answer)
        correct = bool(normalized) and normalized in {
            normalized_exact(target) for target in targets
        }
        return LiteResearchJudgeResult(
            correct=correct,
            method="normalized_exact",
            attempts=0,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "contract": self.contract_id,
            "primary": "normalized_exact",
            "fallback": "none",
            "semantic_equivalence": False,
        }


class UpstreamCompatibleLLMJudge:
    contract_id = UPSTREAM_LLM_JUDGE_CONTRACT

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        parsed = parse.urlparse(api_base)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("judge api_base must be a plain HTTP(S) service URL")
        if not model.strip():
            raise ValueError("judge model must be nonempty")
        if timeout_seconds <= 0:
            raise ValueError("judge timeout_seconds must be positive")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("judge max_retries must be an integer")
        if max_retries < 1:
            raise ValueError("judge max_retries must be a positive integer")
        self.api_base = api_base.rstrip("/")
        self.model = model.strip()
        self.api_key = api_key.strip() or "EMPTY"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max_retries

    @staticmethod
    def _parse_judgment(value: str) -> bool | None:
        text = re.sub(r"```(?:json)?\s*", "", value.strip(), flags=re.IGNORECASE)
        text = text.replace("```", "").strip()
        candidates = [text]
        match = re.search(r'\{[^{}]*"judgment"[^{}]*\}', text, flags=re.DOTALL)
        if match is not None:
            candidates.insert(0, match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            judgment = payload.get("judgment") if isinstance(payload, dict) else None
            if isinstance(judgment, str):
                normalized = judgment.strip().casefold()
                if normalized == "correct":
                    return True
                if normalized == "incorrect":
                    return False
        return None

    def _request(self, question: str, targets: Sequence[str], answer: str) -> bool:
        prompt = _EVALUATION_PROMPT.format(
            question=question,
            correct_answer=", ".join(targets),
            response=answer,
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 512,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = request.Request(
            f"{self.api_base}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("judge response content must be text")
        judgment = self._parse_judgment(content)
        if judgment is None:
            raise ValueError("judge response does not contain Correct or Incorrect")
        return judgment

    def judge(
        self,
        question: str,
        targets: Sequence[str],
        answer: str,
    ) -> LiteResearchJudgeResult:
        if (
            not question.strip()
            or not targets
            or any(not target.strip() for target in targets)
        ):
            raise ValueError("judge requires a question and nonempty targets")
        started_at = time.monotonic()
        for attempt in range(1, self.max_retries + 1):
            try:
                return LiteResearchJudgeResult(
                    correct=self._request(question, targets, answer),
                    method="llm_judge",
                    attempts=attempt,
                    latency_seconds=time.monotonic() - started_at,
                )
            except (
                error.URLError,
                TimeoutError,
                OSError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ):
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)
        return LiteResearchJudgeResult(
            correct=upstream_em(answer, targets),
            method="upstream_em_fallback",
            attempts=self.max_retries,
            latency_seconds=time.monotonic() - started_at,
        )

    def metadata(self) -> dict[str, Any]:
        endpoint_digest = hashlib.sha256(self.api_base.encode("utf-8")).hexdigest()
        return {
            "contract": self.contract_id,
            "primary": "openai_compatible_semantic_equivalence",
            "fallback": "upstream_em_v1",
            "model": self.model,
            "endpoint_sha256": endpoint_digest,
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "semantic_equivalence": True,
        }
