"""
Tool registry and built-in tools for the LLM agent.
Ported from rpi4ai's tools/ package.
"""

from __future__ import annotations

import inspect
import platform
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional

ToolFunc = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: ToolFunc
    description: str
    signature: str


def _clean_docstring(doc: Optional[str]) -> str:
    if not doc:
        return ""
    return inspect.cleandoc(doc).strip()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(
        self,
        func: ToolFunc,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ToolFunc:
        tool_name = (name or func.__name__).strip()
        if not tool_name:
            raise ValueError("Tool name cannot be empty")
        if tool_name in self._tools:
            raise ValueError(f"Tool already registered: {tool_name}")

        sig = str(inspect.signature(func))
        doc = _clean_docstring(description or func.__doc__)
        spec = ToolSpec(
            name=tool_name,
            func=func,
            description=doc or "No description provided.",
            signature=f"{tool_name}{sig}",
        )
        self._tools[tool_name] = spec
        return func

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> Iterable[ToolSpec]:
        return (self._tools[k] for k in sorted(self._tools.keys()))


def tool(
    *,
    registry: ToolRegistry,
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """Decorator to register a function as an agent tool."""

    def _decorator(func: ToolFunc) -> ToolFunc:
        return registry.register(func, name=name, description=description)

    return _decorator


# ── Built-in tools ────────────────────────────────────────────────────────────


def _register_builtin_tools(registry: ToolRegistry) -> None:
    @tool(registry=registry)
    def get_time() -> str:
        """
        Get the current local date/time as an ISO-8601 string.

        Args: none
        Returns:
          ISO 8601 datetime string, e.g. "2026-03-09T12:34:56"
        """
        return datetime.now().replace(microsecond=0).isoformat()

    @tool(registry=registry)
    def get_platform_info() -> Dict[str, str]:
        """
        Get basic platform information (OS, machine, python version).

        Args: none
        Returns:
          Object with keys: system, release, version, machine, processor, python_version
        """
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    _register_builtin_tools(registry)
    return registry
