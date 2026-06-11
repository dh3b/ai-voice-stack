"""Shared test fixtures and helpers.

* Config is injected by constructing the project's dataclasses and passing them to constructors.
* The only thing patched is the network *warmup* side effect in the client
  constructors, which would otherwise POST to a server that doesn't exist.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest import mock

import pytest

from modules.client.llm_client import LLMClient
from modules.client.tts_client import TTSClient
from modules.tools.registry import registry

FIXTURES_DIR = Path(__file__).parent / "fixtures"

def _delta_event(content=None, tool_calls=None, finish_reason=None):
    delta = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = types.SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice])


def _tool_call_delta(index, *, id=None, name=None, arguments=None):
    function = types.SimpleNamespace(name=name, arguments=arguments)
    return types.SimpleNamespace(index=index, id=id, function=function)


class _FakeCompletions:
    def __init__(self, streams):
        self._streams = list(streams)
        self.calls: list[dict] = []  # kwargs of each create(), for assertions

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        events = self._streams.pop(0)  # IndexError => more iterations than expected

        async def _aiter():
            for event in events:
                yield event

        return _aiter()


class _FakeAsyncOpenAI:
    """Stands in for AsyncOpenAI: `await client.chat.completions.create(...)` returns
    an async iterator over the next seeded event-list."""

    def __init__(self, streams):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(streams))


@pytest.fixture
def openai_stub():
    """Builders for faking the streamed chat-completions boundary."""
    return types.SimpleNamespace(
        delta_event=_delta_event,
        tool_call_delta=_tool_call_delta,
        FakeAsyncOpenAI=_FakeAsyncOpenAI,
    )


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def clean_registry():
    """Snapshot the global ToolRegistry and restore it afterwards."""
    tools_snapshot = dict(registry._tools)
    handlers_snapshot = dict(registry._handlers)
    try:
        yield registry
    finally:
        registry._tools.clear()
        registry._tools.update(tools_snapshot)
        registry._handlers.clear()
        registry._handlers.update(handlers_snapshot)


@pytest.fixture
def make_llm_client():
    """Factory: build a real LLMClient from injected configs, skipping warmup."""

    def _make(client_config, tools_config=None) -> LLMClient:
        with mock.patch.object(LLMClient, "_warmup", lambda self: None):
            return LLMClient(client_config, tools_config)

    return _make


@pytest.fixture
def make_tts_client():
    """Factory: build a real TTSClient from an injected config, skipping warmup."""

    def _make(client_config) -> TTSClient:
        with mock.patch.object(TTSClient, "_warmup", lambda self: None):
            return TTSClient(client_config)

    return _make
