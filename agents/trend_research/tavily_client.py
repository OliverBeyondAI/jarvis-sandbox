"""
Tavily API Client for the Trend Research system.

Wraps the Tavily SDK with domain-specific search methods, retry logic,
and structured result parsing into Pydantic Source models.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from .config import Config
from .models import Source


class TavilyResearchClient:
    """
    High-level Tavily search client tailored for trend research.

    Provides specialized search methods (trend discovery, deep dive,
    news scan) that map to different Tavily search configurations.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self._client: Any = None

    # ------------------------------------------------------------------
    # Lazy client initialization
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Lazy-initialize the TavilyClient."""
        if self._client is None:
            from tavily import TavilyClient

            api_key = self.config.tavily_api_key
            if not api_key:
                raise RuntimeError("TAVILY_API_KEY is required for web search")
            self._client = TavilyClient(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # Search methods (all async)
    # ------------------------------------------------------------------

    async def search_trends(
        self,
        domain: str,
        *,
        max_results: int | None = None,
    ) -> list[Source]:
        """
        Discover emerging trends in a domain.

        Args:
            domain: Topic area to search (e.g. "artificial intelligence", "biotech").
            max_results: Override default max results.
        """
        query = f"emerging trends in {domain} 2026"
        return await self._search(
            query=query,
            max_results=max_results or self.config.tavily_max_results,
            search_depth=self.config.tavily_search_depth,
        )

    async def deep_dive(
        self,
        topic: str,
        *,
        max_results: int | None = None,
    ) -> list[Source]:
        """
        Perform a deep research dive on a specific topic or trend.

        Args:
            topic: Specific topic to research in depth.
            max_results: Override default max results.
        """
        return await self._search(
            query=topic,
            max_results=max_results or self.config.tavily_max_results,
            search_depth="advanced",
        )

    async def search_news(
        self,
        topic: str,
        *,
        max_results: int | None = None,
    ) -> list[Source]:
        """
        Search for recent news articles about a topic.

        Args:
            topic: Topic to find news about.
            max_results: Override default max results.
        """
        query = f"{topic} latest news developments"
        return await self._search(
            query=query,
            max_results=max_results or min(self.config.tavily_max_results, 5),
            search_depth="basic",
        )

    async def search_raw(
        self,
        query: str,
        *,
        max_results: int | None = None,
        search_depth: str | None = None,
    ) -> list[Source]:
        """
        Run an arbitrary search query and return structured Sources.

        Args:
            query: Free-form search query.
            max_results: Override default max results.
            search_depth: "basic" or "advanced".
        """
        return await self._search(
            query=query,
            max_results=max_results or self.config.tavily_max_results,
            search_depth=search_depth or self.config.tavily_search_depth,
        )

    # ------------------------------------------------------------------
    # Batch search (parallel fan-out)
    # ------------------------------------------------------------------

    async def batch_search(
        self,
        queries: list[str],
        *,
        max_results_per_query: int | None = None,
    ) -> dict[str, list[Source]]:
        """
        Execute multiple searches in parallel and return results keyed by query.

        Args:
            queries: List of search queries.
            max_results_per_query: Max results per individual query.
        """
        tasks = [
            self.search_raw(q, max_results=max_results_per_query)
            for q in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, list[Source]] = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                print(f"[tavily] Search failed for '{query}': {result}", file=sys.stderr)
                output[query] = []
            else:
                output[query] = result
        return output

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _search(
        self,
        query: str,
        max_results: int,
        search_depth: str,
    ) -> list[Source]:
        """Execute a Tavily search and convert results to Source models."""

        def _sync_search() -> list[dict[str, Any]]:
            client = self._get_client()
            response = client.search(
                query=query,
                max_results=min(max_results, 10),
                search_depth=search_depth,
            )
            return response.get("results", [])

        print(f"[tavily] Searching: {query!r}", file=sys.stderr)
        raw_results = await asyncio.to_thread(_sync_search)

        sources: list[Source] = []
        for r in raw_results:
            sources.append(
                Source(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    relevance_score=r.get("score", 0.0),
                )
            )
        return sources
