"""Web search tool using SerpAPI for Google search."""

from typing import Union, Dict, Any, List, Optional
import os
import requests
from agents import function_tool
from agents.run_context import RunContextWrapper
from agentz.context.data_store import DataStore
from loguru import logger


@function_tool
async def search_google(
    ctx: RunContextWrapper[DataStore],
    query: str,
    num_results: int = 10
) -> Union[List[Dict[str, Any]], str]:
    """Use Google search engine to search information for the given query.

    This tool uses SerpAPI to perform Google searches. It requires a SERPAPI_API_KEY
    environment variable to be set.

    Args:
        ctx: Pipeline context wrapper for accessing the data store
        query: The query to be searched
        num_results: The number of search results to retrieve (default: 10)

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary represents
        a search result. Each dictionary contains:
            - 'result_id': A number in order (1-indexed)
            - 'title': The title of the webpage
            - 'snippet': A brief description/snippet from the webpage
            - 'link': The URL of the webpage
            - 'position': The position in search results

        Example:
        [
            {
                'result_id': 1,
                'title': 'OpenAI',
                'snippet': 'An organization focused on ensuring that artificial...',
                'link': 'https://www.openai.com',
                'position': 1
            },
            ...
        ]

        Or error message string if the search fails.
    """
    try:
        # Get API key from environment
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SERPAPI_API_KEY environment variable not set. Please set it to use search_google."

        logger.info(f"Performing Google search for query: {query}")

        # Prepare SerpAPI request
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": num_results,
            "hl": "en"
        }

        # Make the API request
        response = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=30
        )
        response.raise_for_status()

        # Parse the response
        data = response.json()

        # Check for API errors
        if "error" in data:
            error_msg = f"SerpAPI error: {data['error']}"
            logger.error(error_msg)
            return error_msg

        # Extract organic search results
        organic_results = data.get("organic_results", [])

        if not organic_results:
            logger.warning(f"No search results found for query: {query}")
            return []

        # Format the results to match the expected structure
        formatted_results = []
        for idx, result in enumerate(organic_results[:num_results], 1):
            formatted_result = {
                "result_id": idx,
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "link": result.get("link", ""),
                "position": result.get("position", idx)
            }
            formatted_results.append(formatted_result)

        logger.info(f"Successfully retrieved {len(formatted_results)} search results")

        # Store results in context if available
        data_store = ctx.context
        if data_store is not None:
            cache_key = f"search_google:{query}:{num_results}"
            data_store.set(
                cache_key,
                formatted_results,
                data_type="search_results",
                metadata={
                    "query": query,
                    "num_results": num_results,
                    "result_count": len(formatted_results)
                }
            )
            logger.debug(f"Cached search results with key: {cache_key}")

        return formatted_results

    except requests.exceptions.RequestException as e:
        error_msg = f"Error making search request: {str(e)}"
        logger.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Error performing Google search: {str(e)}"
        logger.error(error_msg)
        return error_msg
