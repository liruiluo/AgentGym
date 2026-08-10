from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import LiteResearcherCoverage, LiteResearcherTask


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

    def public_record(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "rank": self.rank,
        }


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)
        if len(token) > 1
    )


class FrozenLiteResearchBackend:
    """Small deterministic search/page backend for the Stage-1 intake gate.

    Search results expose opaque local URLs.  The upstream ``mask_url`` and
    answer aliases remain server-private in the coverage object and never enter
    ``search`` results or service metadata.
    """

    contract_id = "literesearcher_frozen_search_page_backend_v1"

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
            "live_network": False,
            "failure_mode": "fail_closed",
        }

    def search(self, query: str | list[str], *, top_k: int | None = None) -> list[dict[str, Any]]:
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

    def visit(self, url: str, *, goal: str = "") -> dict[str, Any]:
        if not isinstance(url, str) or not url.strip():
            raise LiteResearchRequestError("visit URL must be a non-empty string")
        url = url.strip()
        if url in self._fail_visit:
            raise LiteResearchBackendError("frozen page backend rejected the URL")
        try:
            task = self._by_url[url]
        except KeyError as exc:
            raise LiteResearchRequestError(
                "visit URL is outside the frozen corpus"
            ) from exc
        return {
            "url": task.public_url,
            "title": task.page_title,
            "content": task.page_text,
            "goal": str(goal),
        }

    def private_task(self, url: str) -> LiteResearcherTask:
        """Verifier-only lookup; wrappers must not put this result in observations."""

        try:
            return self._by_url[url]
        except KeyError as exc:
            raise LiteResearchRequestError("unknown frozen page URL") from exc
