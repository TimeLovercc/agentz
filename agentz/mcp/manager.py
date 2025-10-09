from __future__ import annotations

from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Dict, Optional

from agents.mcp import MCPServer, MCPServerSse, MCPServerStdio


class MCPConfigurationError(ValueError):
    """Raised when an MCP server configuration is invalid."""


@dataclass(frozen=True)
class MCPServerSpec:
    """Lightweight specification describing how to build an MCP server."""

    type: str
    options: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.type:
            raise MCPConfigurationError("MCP server configuration requires a 'type'.")


class MCPRegistry:
    """Registry responsible for storing MCP server specifications."""

    def __init__(self, specs: Optional[Dict[str, MCPServerSpec]] = None) -> None:
        self._specs = specs or {}

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "MCPRegistry":
        """Create a registry from configuration mapping."""
        if config is None:
            return cls()

        servers_config = config.get("servers", {})
        if servers_config and not isinstance(servers_config, Mapping):
            raise MCPConfigurationError("'servers' must be a mapping of server definitions.")

        specs: Dict[str, MCPServerSpec] = {}
        for name, server_cfg in (servers_config or {}).items():
            if not isinstance(server_cfg, Mapping):
                raise MCPConfigurationError(f"MCP server '{name}' configuration must be a mapping.")

            server_type = str(server_cfg.get("type") or server_cfg.get("transport") or "").strip().lower()
            if not server_type:
                raise MCPConfigurationError(f"MCP server '{name}' must define a 'type' or 'transport'.")

            options = {k: v for k, v in server_cfg.items() if k not in {"type", "transport"}}
            specs[name] = MCPServerSpec(type=server_type, options=options)

        return cls(specs)

    def register(self, name: str, spec: MCPServerSpec) -> None:
        """Register (or overwrite) a spec by name."""
        self._specs[name] = spec

    def get(self, name: str) -> MCPServerSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise MCPConfigurationError(
                f"MCP server '{name}' is not defined; add it to the configuration."
            ) from exc

    def as_dict(self) -> Dict[str, MCPServerSpec]:
        return dict(self._specs)

    def contains(self, name: str) -> bool:
        return name in self._specs


SERVER_TYPE_MAP = {
    "stdio": MCPServerStdio,
    "sse": MCPServerSse,
}


class MCPManagerSession:
    """Async context manager that keeps MCP server connections alive."""

    def __init__(self, registry: MCPRegistry):
        self._registry = registry
        self._stack = AsyncExitStack()
        self._servers: Dict[str, MCPServer] = {}

    async def __aenter__(self) -> "MCPManagerSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._stack.aclose()
        self._servers.clear()

    async def get_server(self, name: str, overrides: Optional[Mapping[str, Any]] = None) -> MCPServer:
        """Return a connected MCP server instance by name, creating it on demand."""
        if name in self._servers:
            return self._servers[name]

        spec = self._registry.get(name)
        options = dict(spec.options)
        if overrides:
            options.update(overrides)

        try:
            server_cls = SERVER_TYPE_MAP[spec.type]
        except KeyError as exc:
            raise MCPConfigurationError(
                f"Unsupported MCP server type '{spec.type}' for '{name}'. "
                f"Supported types: {', '.join(SERVER_TYPE_MAP)}."
            ) from exc

        server_ctx = server_cls(**options)
        server = await self._stack.enter_async_context(server_ctx)
        self._servers[name] = server
        return server


class MCPManager:
    """Entry point that provides MCP manager sessions."""

    def __init__(self, registry: MCPRegistry):
        self._registry = registry

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "MCPManager":
        registry = MCPRegistry.from_config(config)
        return cls(registry)

    def ensure_server(self, name: str, spec: MCPServerSpec) -> None:
        """Add a default server if one isn't already configured."""
        if not self._registry.contains(name):
            self._registry.register(name, spec)

    def session(self) -> MCPManagerSession:
        """Create a new MCPManagerSession for a pipeline run."""
        return MCPManagerSession(self._registry)

    def list_servers(self) -> Dict[str, MCPServerSpec]:
        return self._registry.as_dict()
