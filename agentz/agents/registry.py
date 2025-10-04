from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from loguru import logger
import asyncio
import inspect
import importlib
import pkgutil
from pydantic import BaseModel, Field

try:
    from agents import Agent
except Exception as e:
    raise ImportError("Expected `from agents import Agent` to be importable") from e

# Canonical name -> factory(config) -> Agent (sync or async)
ALL_AGENT_FACTORIES: Dict[str, Callable[..., Any]] = {}
# Alias -> canonical name
_AGENT_ALIASES: Dict[str, str] = {}

# Auto-discovery guard
_DISCOVERY_DONE = False


def _try_import_module(mod: str) -> None:
    """Best-effort module import (silently ignore failures)."""
    try:
        importlib.import_module(mod)
    except Exception:
        pass


def _import_all_under_agents_root() -> None:
    """Recursively import all submodules under agentz.agents."""
    try:
        pkg = importlib.import_module("agentz.agents")
    except Exception:
        return

    pkg_path = getattr(pkg, "__path__", None)
    if not pkg_path:
        return

    for _, name, _ in pkgutil.walk_packages(pkg_path, prefix=pkg.__name__ + "."):
        _try_import_module(name)


def _ensure_registry_populated(agent_name_hint: Optional[str] = None) -> None:
    """Ensure agent registry is populated via lazy auto-discovery.

    Args:
        agent_name_hint: Optional agent name to try convention-based imports first
    """
    global _DISCOVERY_DONE

    if _DISCOVERY_DONE and ALL_AGENT_FACTORIES:
        return

    # 1) Try convention-based imports first (cheap, specific)
    if agent_name_hint:
        leaf = agent_name_hint.strip().lower()
        for mod in (
            f"agentz.agents.{leaf}",
            f"agentz.agents.{leaf}_agent",
            f"agentz.agents.manager_agents.{leaf}",
            f"agentz.agents.worker_agents.{leaf}",
        ):
            _try_import_module(mod)

        # Check if we found it
        if ALL_AGENT_FACTORIES.get(leaf):
            logger.debug(f"Found agent '{leaf}' via convention-based import")
            return

    # 2) Full recursive import under agentz.agents (one-time)
    if not _DISCOVERY_DONE:
        _import_all_under_agents_root()
        _DISCOVERY_DONE = True
        logger.debug(f"Auto-discovered {len(ALL_AGENT_FACTORIES)} agent factories")


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

    # Auto-discover agents if not found
    if canonical not in ALL_AGENT_FACTORIES:
        _ensure_registry_populated(agent_name_hint=canonical)

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


class AgentRegistry:
    """Simple registry for building agents from names and specs."""

    @staticmethod
    def register(name: str, builder: Callable[..., Any], *, aliases: Optional[List[str]] = None) -> None:
        """Register an agent builder function."""
        key = _canon(name)
        if key in ALL_AGENT_FACTORIES:
            raise ValueError(f"Agent '{name}' already registered.")
        ALL_AGENT_FACTORIES[key] = builder
        for a in (aliases or []):
            _AGENT_ALIASES[_canon(a)] = key

    @staticmethod
    def build(name: str, spec: Optional[Dict[str, Any]] = None, config: Optional[Any] = None) -> Agent:
        """Build an agent from a name and optional spec dict.

        Args:
            name: Agent name to look up in registry
            spec: Optional dict with additional agent parameters (if None, will try to get from config)
            config: Optional config object for factory function

        Returns:
            Agent instance
        """
        canonical = _resolve_name(name)

        # Auto-discover agents if not found
        if canonical not in ALL_AGENT_FACTORIES:
            _ensure_registry_populated(agent_name_hint=canonical)

        if canonical not in ALL_AGENT_FACTORIES:
            available = sorted(ALL_AGENT_FACTORIES.keys())
            raise ValueError(f"Unknown agent name: {name}. Available agents: {available}")

        factory = ALL_AGENT_FACTORIES[canonical]

        # If spec not provided, try to get from config
        if spec is None and config is not None:
            try:
                from agentz.configuration.base import get_agent_spec
                spec = get_agent_spec(config, canonical)
            except (ValueError, AttributeError):
                # Config doesn't have this agent spec, that's ok
                pass

        # Call factory with config and spec
        if config:
            try:
                sig = inspect.signature(factory)
                if len(sig.parameters) >= 2:
                    # Factory accepts (config, spec)
                    result = factory(config, spec)
                else:
                    # Factory only accepts config
                    result = factory(config)
            except TypeError:
                # Fallback: just pass config
                result = factory(config)
        elif spec and "instructions" in spec:
            # No config but have spec - build directly
            params = spec.get("params", {})
            return Agent(
                name=spec.get("name", name),
                instructions=spec["instructions"],
                **params
            )
        else:
            raise ValueError(f"Cannot build agent '{name}' without config or valid spec")

        # Handle async results
        if inspect.isawaitable(result):
            return _sync_run(result)
        return result


def coerce_to_agent(path: str, value: Any, config: Optional[Any] = None) -> Agent:
    """Coerce various input types to an Agent instance.

    Args:
        path: Dot-path for this agent (used to infer type from leaf name)
        value: Can be:
            - Agent instance (returned as-is)
            - dict with "type" key -> use as registry name
            - dict with "name" and "instructions" -> build directly
            - str -> treat as registry key
        config: Optional config for registry builders

    Returns:
        Agent instance
    """
    # Already an Agent
    if isinstance(value, Agent):
        return value

    # Dict with type key
    if isinstance(value, dict):
        if "type" in value:
            agent_type = value["type"]
            spec = {k: v for k, v in value.items() if k != "type"}
            return AgentRegistry.build(agent_type, spec=spec, config=config)

        # Dict with name/instructions - build directly
        if "name" in value and "instructions" in value:
            return Agent(name=value["name"], instructions=value["instructions"],
                       **{k: v for k, v in value.items() if k not in {"name", "instructions"}})

        # Try to infer type from path
        leaf_name = path.split(".")[-1]
        return AgentRegistry.build(leaf_name, spec=value, config=config)

    # String - treat as registry key
    if isinstance(value, str):
        return AgentRegistry.build(value, config=config)

    raise TypeError(f"Cannot coerce {type(value)} to Agent at path '{path}'")


class AgentStore:
    """Tree structure for managing agents with dot-path addressing.

    Supports:
    - Dot-path addressing: "manager.observe", "workers.training.trainer"
    - Agent storage from YAML, dict, or Agent objects
    - Group management
    - Config merging with precedence
    """

    def __init__(self, config: Optional[Any] = None):
        """Initialize the agent store.

        Args:
            config: Optional config object passed to agent builders
        """
        self._tree: Dict[str, Any] = {}
        self._config = config

    def add(self, path: str, value: Union[Agent, dict, str], replace: bool = True) -> Agent:
        """Add an agent at a dot-path location.

        Args:
            path: Dot-separated path (e.g., "manager.observe")
            value: Agent instance, dict spec, or registry name
            replace: If False, raise error on duplicate

        Returns:
            The added Agent instance
        """
        parts = path.split(".")
        node = self._tree

        # Navigate/create intermediate groups
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            elif not isinstance(node[part], dict):
                raise ValueError(f"Path '{path}' conflicts with existing non-group value")
            node = node[part]

        leaf = parts[-1]

        # Check for existing value
        if leaf in node and not replace:
            raise ValueError(f"Agent already exists at path '{path}' and replace=False")

        # Coerce to Agent
        agent = coerce_to_agent(path, value, self._config)

        # Log replacement
        if leaf in node:
            logger.debug(f"Replacing agent at '{path}'")

        node[leaf] = agent
        return agent

    def get(self, path: str) -> Agent:
        """Get an agent at a dot-path location.

        Args:
            path: Dot-separated path

        Returns:
            Agent instance

        Raises:
            KeyError: If path not found
        """
        parts = path.split(".")
        node = self._tree

        for part in parts:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"No agent found at path '{path}'")
            node = node[part]

        if not isinstance(node, Agent):
            raise KeyError(f"Path '{path}' is a group, not an agent")

        return node

    def list(self, path: str = "") -> Dict[str, Any]:
        """List agents/groups at a path (returns subtree).

        Args:
            path: Dot-separated path (empty for root)

        Returns:
            Dict of agents/groups at that path
        """
        if not path:
            return dict(self._tree)

        parts = path.split(".")
        node = self._tree

        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return {}
            node = node[part]

        if isinstance(node, dict):
            return dict(node)

        # Single agent
        return {parts[-1]: node}

    def ensure_group(self, path: str) -> None:
        """Ensure a group exists at the given path.

        Args:
            path: Dot-separated path for the group
        """
        parts = path.split(".")
        node = self._tree

        for part in parts:
            if part not in node:
                node[part] = {}
            elif not isinstance(node[part], dict):
                raise ValueError(f"Cannot create group at '{path}': conflicts with existing agent")
            node = node[part]

    def merge_config(self, mapping: Dict[str, Any], root: str = "") -> None:
        """Merge a config mapping into the store.

        Walks the mapping; leaves (Agent instances or specs) become agents.
        Precedence: last writer wins.

        Args:
            mapping: Dict of agents/groups to merge
            root: Root path prefix (empty for top level)
        """
        for key, value in mapping.items():
            current_path = f"{root}.{key}" if root else key

            # If it's an Agent or a leaf spec, add it
            if isinstance(value, Agent):
                self.add(current_path, value, replace=True)
            elif isinstance(value, dict):
                # Check if it's a leaf (has name/instructions or type)
                is_leaf = "name" in value or "instructions" in value or "type" in value

                if is_leaf:
                    self.add(current_path, value, replace=True)
                else:
                    # It's a group - recurse
                    self.ensure_group(current_path)
                    self.merge_config(value, root=current_path)
            elif isinstance(value, str):
                # String registry key
                self.add(current_path, value, replace=True)

    def __getitem__(self, path: str) -> Agent:
        """Dict-style access."""
        return self.get(path)

    def __contains__(self, path: str) -> bool:
        """Check if a path exists."""
        try:
            self.get(path)
            return True
        except KeyError:
            return False

    def keys(self) -> List[str]:
        """Return all agent paths (flattened)."""
        paths = []

        def _collect(node: Dict[str, Any], prefix: str = "") -> None:
            for key, value in node.items():
                current_path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, Agent):
                    paths.append(current_path)
                elif isinstance(value, dict):
                    _collect(value, current_path)

        _collect(self._tree)
        return paths

    def values(self) -> List[Agent]:
        """Return all agents."""
        agents = []

        def _collect(node: Dict[str, Any]) -> None:
            for value in node.values():
                if isinstance(value, Agent):
                    agents.append(value)
                elif isinstance(value, dict):
                    _collect(value)

        _collect(self._tree)
        return agents

    def items(self) -> List[tuple[str, Agent]]:
        """Return all (path, agent) pairs."""
        items = []

        def _collect(node: Dict[str, Any], prefix: str = "") -> None:
            for key, value in node.items():
                current_path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, Agent):
                    items.append((current_path, value))
                elif isinstance(value, dict):
                    _collect(value, current_path)

        _collect(self._tree)
        return items


# Global pipeline agent store reference (set by BasePipeline)
_current_pipeline_store: Optional[AgentStore] = None


def set_current_pipeline_store(store: Optional[AgentStore]) -> None:
    """Set the current pipeline's agent store for auto-registration."""
    global _current_pipeline_store
    _current_pipeline_store = store


def get_current_pipeline_store() -> Optional[AgentStore]:
    """Get the current pipeline's agent store."""
    return _current_pipeline_store


def create_agents(
    names_or_map: Union[str, List[str], Dict[str, Any]],
    config: Any,
    *,
    group: Optional[str] = None,
) -> Union[Any, Dict[str, Any]]:
    """Create one or more agents from agent name(s) or mapping.

    Automatically registers created agents in the current pipeline's AgentStore
    if one is active.

    Args:
        names_or_map: Can be:
            - str: Single agent name
            - List[str]: List of agent names
            - Dict[str, Any]: Mapping of name -> Agent/spec/registry_key
        config: Configuration (BaseConfig, LLMConfig, or path to resolve)
        group: Optional group name for automatic registration (e.g., "manager", "workers")

    Returns:
        - If names_or_map is a string: returns a single Agent instance
        - Otherwise: returns Dict mapping agent names to Agent instances

    Examples:
        # Create single agent
        observe_agent = create_agents("observe_agent", config)

        # Create multiple agents
        agents = create_agents(["observe_agent", "data_loader_agent"], config)

        # Create with group registration
        manager = create_agents("observe_agent", config, group="manager")
        # -> Auto-registered at "manager.observe_agent"

        # Create from mapping
        agents = create_agents({
            "custom": Agent(name="custom", instructions="..."),
            "loader": "data_loader_agent"  # registry key
        }, config, group="workers")
    """
    # Resolve config if it's a string/path
    from agentz.configuration.base import BaseConfig
    if isinstance(config, (str, Path)):
        from agentz.configuration.base import resolve_config
        config = resolve_config(config)
    # If it's LLMConfig, try to get the BaseConfig from current pipeline store
    elif hasattr(config, "full_config") and not isinstance(config, BaseConfig):
        # It's LLMConfig - try to get BaseConfig from pipeline store
        store = get_current_pipeline_store()
        if store and hasattr(store, "_config"):
            config = store._config

    store = get_current_pipeline_store()

    # Single name (string)
    if isinstance(names_or_map, str):
        agent_name = names_or_map
        agent = _sync_run(_build_one_async(agent_name, config))
        logger.info(f"Created agent: {agent_name}")

        # Auto-register if store available
        if store:
            path = f"{group}.{agent_name}" if group else agent_name
            store.add(path, agent, replace=True)
            logger.debug(f"Auto-registered agent at '{path}'")

        return agent

    # List of names
    if isinstance(names_or_map, list):
        async def _build_many():
            out: Dict[str, Any] = {}
            for name in names_or_map:
                out[name] = await _build_one_async(name, config)
                logger.info(f"Created agent: {name}")
            return out

        agents = _sync_run(_build_many())

        # Auto-register if store available
        if store:
            for name, agent in agents.items():
                path = f"{group}.{name}" if group else name
                store.add(path, agent, replace=True)
                logger.debug(f"Auto-registered agent at '{path}'")

        return agents

    # Dict mapping
    if isinstance(names_or_map, dict):
        result: Dict[str, Any] = {}

        for name, value in names_or_map.items():
            # Coerce each value to an agent
            if isinstance(value, Agent):
                agent = value
            elif isinstance(value, str):
                # Registry key
                agent = _sync_run(_build_one_async(value, config))
            elif isinstance(value, dict):
                # Spec dict - use coerce_to_agent
                agent = coerce_to_agent(name, value, config)
            else:
                raise TypeError(f"Unsupported agent value type for '{name}': {type(value)}")

            result[name] = agent
            logger.info(f"Created agent: {name}")

            # Auto-register if store available
            if store:
                path = f"{group}.{name}" if group else name
                store.add(path, agent, replace=True)
                logger.debug(f"Auto-registered agent at '{path}'")

        return result

    raise TypeError(f"names_or_map must be str, list, or dict; got {type(names_or_map)}")
