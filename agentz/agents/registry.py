from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
from loguru import logger
import asyncio
import inspect
from pydantic import BaseModel, Field

# Canonical name -> factory(config) -> Agent (sync or async)
ALL_AGENT_FACTORIES: Dict[str, Callable[..., Any]] = {}
# Alias -> canonical name
_AGENT_ALIASES: Dict[str, str] = {}


class ToolAgentOutput(BaseModel):
    """Standard output for all tool agents"""
    output: str
    sources: list[str] = Field(default_factory=list)


def _canon(name: str) -> str:
    return name.strip().lower()


def register_agent(name: str, *, aliases: Optional[List[str]] = None):
    """
    Decorator to register an agent factory under a canonical name.

    Usage:
        @register_agent("observe_agent", aliases=["observe"])
        def create_observe_agent(config): ...
    """
    def deco(fn: Callable[..., Any]):
        key = _canon(name)
        if key in ALL_AGENT_FACTORIES:
            raise ValueError(f"Agent '{name}' already registered.")
        ALL_AGENT_FACTORIES[key] = fn
        for a in (aliases or []):
            _AGENT_ALIASES[_canon(a)] = key
        return fn
    return deco


def _resolve_name(name: str) -> str:
    key = _canon(name)
    return _AGENT_ALIASES.get(key, key)


def list_agents() -> List[str]:
    return sorted(ALL_AGENT_FACTORIES.keys())


async def _build_one_async(agent_name: str, config: Any) -> Any:
    """Internal async builder that supports sync or async factories."""
    canonical = _resolve_name(agent_name)
    if canonical not in ALL_AGENT_FACTORIES:
        available = sorted(ALL_AGENT_FACTORIES.keys())
        raise ValueError(
            f"Unknown agent name: {agent_name}. Available agents: {available}"
        )
    factory = ALL_AGENT_FACTORIES[canonical]
    # Support async factory function or async return
    if inspect.iscoroutinefunction(factory):
        return await factory(config)
    result = factory(config)
    if inspect.isawaitable(result):
        return await result
    return result


def _sync_run(coro):
    """Run an async coroutine safely from sync context if needed."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # If we're already in a running loop, block until done
    # (Typical for CLIs and most services; notebooks may already have a loop)
    return loop.run_until_complete(coro)


def create_agents(
    agent_names: Union[str, List[str]],
    config: Any,
) -> Union[Any, Dict[str, Any]]:
    """Create one or more agents from agent name(s).

    Supports both manager agents (observe, evaluate, routing, writer) and
    worker agents (data_loader, data_analysis, preprocessing, etc.).

    Args:
        agent_names: Single agent name (str) or list of agent names
        config: LLM configuration with full_config containing agent prompts

    Returns:
        - If agent_names is a string: returns a single Agent instance
        - If agent_names is a list: returns Dict mapping agent names to Agent instances

    Raises:
        ValueError: If an unknown agent name is provided

    Examples:
        # Create single agent
        observe_agent = create_agents("observe_agent", config)

        # Create multiple agents
        agents = create_agents(["observe_agent", "data_loader_agent"], config)
        observe_agent = agents["observe_agent"]
    """
    # Single name
    if isinstance(agent_names, str):
        agent_name = agent_names
        agent = _sync_run(_build_one_async(agent_name, config))
        logger.info(f"Created agent: {agent_name}")
        return agent

    # Multiple names
    if not isinstance(agent_names, list):
        raise ValueError("agent_names must be a string or a list of strings.")

    async def _build_many():
        out: Dict[str, Any] = {}
        for name in agent_names:
            out[name] = await _build_one_async(name, config)
            logger.info(f"Created agent: {name}")
        return out

    return _sync_run(_build_many())
