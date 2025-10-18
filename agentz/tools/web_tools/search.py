"""Web search tool implementation using DuckDuckGo's HTML endpoint.

The implementation mirrors the Baidu scraper found in tmp/baidu_search.py but
targets DuckDuckGo so it works globally without an API key. Results are parsed
from the returned HTML and exposed as a simple tool function that can be called
by the agent runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

from agents import function_tool

DEFAULT_BASE_URL = "https://duckduckgo.com/html/"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://duckduckgo.com",
}
DEFAULT_TIMEOUT = 15


class WebSearchError(RuntimeError):
    """Raised when the web search fails."""


@dataclass
class WebSearchConfig:
    """Configuration for the web search client."""

    base_url: str = DEFAULT_BASE_URL
    headers: dict[str, str] = None  # type: ignore[assignment]
    region: Optional[str] = None
    timeout: int = DEFAULT_TIMEOUT
    banned_sites: Optional[Iterable[str]] = None
    content_filter: Any = None  # Placeholder for future ContentFilter integration

    def __post_init__(self) -> None:
        merged_headers = dict(DEFAULT_HEADERS)
        if self.headers:
            merged_headers.update(self.headers)
        self.headers = merged_headers


class DuckDuckGoSearchClient:
    """DuckDuckGo search client that scrapes HTML results."""

    def __init__(self, config: Optional[WebSearchConfig] = None) -> None:
        self.config = config or WebSearchConfig()

        # Prepare banned site lookup for fast filtering
        self._banned_sites = {
            site.lower()
            for site in self.config.banned_sites or []
        }

        # Placeholder for future content filtering integration
        self._content_filter = self.config.content_filter

    async def search(self, query: str, num_results: int = 5) -> dict[str, Any]:
        """Execute a search query and return structured results."""
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        raw_results = await self._fetch_results(query)
        parsed_results = self._parse_results(raw_results)

        if not parsed_results:
            logger.warning(f"No results found for query: {query}")

        filtered_results = self._apply_filters(parsed_results, num_results)

        return {
            "query": query,
            "results": filtered_results,
            "total_results": len(parsed_results),
            "provider": "duckduckgo",
        }

    async def _fetch_results(self, query: str) -> str:
        params = {
            "q": query,
            "kl": self.config.region or "us-en",
        }

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(
                    self.config.base_url,
                    params=params,
                    headers=self.config.headers,
                ) as response:
                    response.raise_for_status()
                    return await response.text(encoding="utf-8", errors="ignore")
            except aiohttp.ClientError as exc:
                logger.error(f"Web search request failed: {exc}")
                raise WebSearchError("Failed to complete web search request.") from exc

    def _parse_results(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, Any]] = []

        for idx, item in enumerate(soup.select("div.result"), start=1):
            title_anchor = item.select_one("a.result__a")
            if not title_anchor:
                continue

            title = title_anchor.get_text(strip=True)
            url = self._normalize_url(title_anchor.get("href", ""))
            snippet_node = item.select_one("a.result__snippet")
            if snippet_node is None:
                snippet_node = item.select_one("div.result__snippet")

            snippet = (
                " ".join(snippet_node.stripped_strings) if snippet_node else ""
            )

            results.append(
                {
                    "rank": idx,
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )

        return results

    def _apply_filters(
        self,
        results: list[dict[str, Any]],
        num_results: int,
    ) -> list[dict[str, Any]]:
        filtered = [
            result
            for result in results
            if not self._is_banned(result.get("url", ""))
        ]

        if self._content_filter and hasattr(self._content_filter, "filter_results"):
            try:
                filtered = self._content_filter.filter_results(
                    filtered,
                    num_results,
                    key="url",
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(f"Content filter failed, falling back to defaults: {exc}")

        if num_results > 0:
            filtered = filtered[:num_results]

        return filtered

    def _is_banned(self, url: str) -> bool:
        if not url or not self._banned_sites:
            return False
        hostname = urlparse(url).hostname or ""
        return any(hostname.endswith(site) for site in self._banned_sites)

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Decode DuckDuckGo redirect URLs to their destination."""
        if not url:
            return url

        if url.startswith("/"):
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            uddg = params.get("uddg")
            if uddg:
                return unquote(uddg[0])

        return url


@function_tool
async def web_search(
    query: str,
    num_results: int = 5,
    region: Optional[str] = None,
    banned_sites: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Search the web for a query using DuckDuckGo.

    Args:
        query: User provided search query string.
        num_results: Maximum number of results to return.
        region: Optional region code (DuckDuckGo `kl` parameter, e.g., ``us-en``).
        banned_sites: Iterable of banned hostnames that should be filtered out.

    Returns:
        Dictionary containing the original query, provider metadata, and a list
        of search results with rank, title, url, and snippet fields.
    """
    config = WebSearchConfig(region=region, banned_sites=banned_sites)
    client = DuckDuckGoSearchClient(config)
    return await client.search(query, num_results=num_results)


__all__ = [
    "web_search",
    "DuckDuckGoSearchClient",
    "WebSearchConfig",
    "WebSearchError",
]
