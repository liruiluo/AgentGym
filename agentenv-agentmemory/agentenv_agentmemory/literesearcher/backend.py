from __future__ import annotations

from collections import Counter
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import LiteResearcherCoverage, LiteResearcherTask


VISIT_PAGE_CHARS = 8192
VISIT_PAGE_OVERLAP_CHARS = 1024
_VISIT_PAGE_STRIDE_CHARS = VISIT_PAGE_CHARS - VISIT_PAGE_OVERLAP_CHARS


class LiteResearchBackendError(RuntimeError):
    """Fail-closed backend error; no fallback to live web or exact match."""


class LiteResearchRequestError(ValueError):
    """Policy request rejected without treating it as infrastructure failure."""


@dataclass(frozen=True)
class SearchHit:
    url: str
    title: str
    snippet: str
    rank: int
    score: float | None = None

    def public_record(self) -> dict[str, Any]:
        record = {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "rank": self.rank,
        }
        if self.score is not None:
            record["score"] = self.score
        return record


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
        if len(token) > 1
    )


def _overlapping_windows(text: str) -> tuple[str, ...]:
    return tuple(
        text[start : start + VISIT_PAGE_CHARS]
        for start in range(0, len(text), _VISIT_PAGE_STRIDE_CHARS)
    )


def _rank_windows_by_goal(text: str, goal: str) -> tuple[str, ...]:
    windows = _overlapping_windows(text)
    query_terms = set(_tokens(goal))
    if len(windows) == 1 or not query_terms:
        return windows

    documents = [Counter(_tokens(window)) for window in windows]
    average_length = sum(sum(document.values()) for document in documents) / len(documents)
    document_frequency = {
        term: sum(term in document for document in documents) for term in query_terms
    }
    k1 = 1.5
    b = 0.75

    def score(document: Counter[str]) -> float:
        length = sum(document.values())
        length_ratio = length / average_length if average_length else 0.0
        total = 0.0
        for term in query_terms:
            frequency = document.get(term, 0)
            if not frequency:
                continue
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(
                1.0 + (len(documents) - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            denominator = frequency + k1 * (1.0 - b + b * length_ratio)
            total += inverse_frequency * frequency * (k1 + 1.0) / denominator
        return total

    ranked_indices = sorted(
        range(len(windows)),
        key=lambda index: (-score(documents[index]), index),
    )
    return tuple(windows[index] for index in ranked_indices)


class FrozenLiteResearchBackend:
    """Small deterministic search/page backend for the Stage-1 intake gate.

    Search results expose opaque local URLs.  The upstream ``mask_url`` and
    answer aliases remain server-private in the coverage object and never enter
    ``search`` results or service metadata.
    """

    contract_id = "literesearcher_frozen_search_page_backend_v2"

    def __init__(
        self,
        coverage: LiteResearcherCoverage,
        *,
        split: str = "train",
        top_k: int = 5,
        failing_search_queries: Iterable[str] = (),
        failing_visit_urls: Iterable[str] = (),
    ) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.coverage = coverage
        self.tasks_source = coverage
        self.split = split
        self.tasks = coverage.tasks_for_split(split)
        self.top_k = top_k
        self._fail_search = {str(item) for item in failing_search_queries}
        self._fail_visit = {str(item) for item in failing_visit_urls}
        self._by_url = {task.public_url: task for task in self.tasks}
        self._search_tokens = {
            task.public_url: set(_tokens(task.question + " " + task.page_title))
            for task in self.tasks
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "backend_contract": self.contract_id,
            "coverage_manifest_sha256": self.coverage.manifest_sha256,
            "split": self.split,
            "active_count": len(self.tasks),
            "train_count": self.coverage.task_count,
            "heldout_count": self.coverage.heldout_count,
            "search_result_url_namespace": "opaque_local_fixture_url_v1",
            "search_exposes_mask_url": False,
            "search_exposes_targets": False,
            "visit_exposes_mask_url": False,
            "visit_pagination_contract": "goal_bm25_overlapping_chars_v1",
            "visit_page_chars": VISIT_PAGE_CHARS,
            "visit_page_overlap_chars": VISIT_PAGE_OVERLAP_CHARS,
            "visit_page_order_public_inputs": ["goal", "page_text"],
            "live_network": False,
            "failure_mode": "fail_closed",
        }

    def search(
        self,
        query: str | list[str],
        *,
        top_k: int | None = None,
        mask_url: str = "",
    ) -> list[dict[str, Any]]:
        del mask_url
        queries = [query] if isinstance(query, str) else query
        if not isinstance(queries, list) or not queries or any(
            not isinstance(item, str) for item in queries
        ):
            raise LiteResearchRequestError(
                "search query must be a non-empty string or list of strings"
            )
        query_text = " ".join(item.strip() for item in queries).strip()
        if not query_text:
            raise LiteResearchRequestError("search query must not be empty")
        if query_text in self._fail_search:
            raise LiteResearchBackendError("frozen search backend rejected the query")
        query_tokens = set(_tokens(query_text))
        scored: list[tuple[int, int, LiteResearcherTask]] = []
        for task in self.tasks:
            overlap = len(query_tokens & self._search_tokens[task.public_url])
            if overlap:
                scored.append((-overlap, task.index, task))
        scored.sort(key=lambda item: (item[0], item[1]))
        limit = self.top_k if top_k is None else top_k
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise LiteResearchRequestError("search top_k must be a positive integer")
        hits = []
        for rank, (_, _, task) in enumerate(scored[:limit], start=1):
            # The snippet intentionally contains no page body or target alias.
            hits.append(
                SearchHit(
                    url=task.public_url,
                    title=task.page_title,
                    snippet="Indexed source page matching the query terms.",
                    rank=rank,
                ).public_record()
            )
        return hits

    def visit(self, url: str, *, goal: str = "", page: int = 1) -> dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise LiteResearchRequestError("visit URL must be a non-empty string")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise LiteResearchRequestError("visit page must be a positive integer")
        url = url.strip()
        if url in self._fail_visit:
            raise LiteResearchBackendError("frozen page backend rejected the URL")
        try:
            task = self._by_url[url]
        except KeyError as exc:
            raise LiteResearchRequestError(
                "visit URL is outside the frozen corpus"
            ) from exc
        goal_text = str(goal)
        pages = _rank_windows_by_goal(task.page_text, goal_text)
        if page > len(pages):
            raise LiteResearchRequestError(
                f"visit page {page} exceeds page_count {len(pages)}"
            )
        return {
            "url": task.public_url,
            "title": task.page_title,
            "content": pages[page - 1],
            "goal": goal_text,
            "page": page,
            "page_count": len(pages),
            "next_page": page + 1 if page < len(pages) else None,
        }

    def private_task(self, url: str) -> LiteResearcherTask:
        """Verifier-only lookup; wrappers must not put this result in observations."""

        try:
            return self._by_url[url]
        except KeyError as exc:
            raise LiteResearchRequestError("unknown frozen page URL") from exc
