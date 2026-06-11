"""
Covers four behaviours with stubbed completions and a controlled clock (no sleeps):
  * the system message stays pinned at index 0,
  * pairs older than history_max_turns are dropped,
  * the history wipes back to [system] after history_idle_timeout_s,
  * in-turn tool messages never leak into the persisted history.
"""

import asyncio

import modules.client.llm_client as llm_mod
from config import LLMClientConfig, ToolsConfig


async def test_history_window_and_idle_wipe(make_llm_client, clean_registry, openai_stub, monkeypatch):
    registry = clean_registry
    de = openai_stub.delta_event
    tc = openai_stub.tool_call_delta

    clock = {"t": 1000.0}
    monkeypatch.setattr(llm_mod.time, "monotonic", lambda: clock["t"])

    @registry.register(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    )
    def get_weather(city):
        return f"sunny in {city}"

    client = make_llm_client(
        LLMClientConfig(
            mode="agent",
            history_enabled=True,
            history_max_turns=2,  # system + at most 2 (user, assistant) pairs
            history_idle_timeout_s=100.0,
        ),
        ToolsConfig(enabled_tool_modules=[]),
    )

    def content_stream(text):
        return [de(content=text), de(finish_reason="stop")]

    async def turn(user_text, streams):
        client._client = openai_stub.FakeAsyncOpenAI(streams)
        await client.run(user_text, asyncio.Queue())

    def contents(role):
        return [m["content"] for m in client._history if m["role"] == role]

    # Three plain turns fill and then overflow the window.
    await turn("u1", [content_stream("a1")])
    await turn("u2", [content_stream("a2")])
    assert client._history[0]["role"] == "system"
    assert [m["role"] for m in client._history] == ["system", "user", "assistant", "user", "assistant"]

    await turn("u3", [content_stream("a3")])
    # Oldest (u1/a1) pair dropped; system still pinned; newest two pairs kept.
    assert client._history[0]["role"] == "system"
    assert contents("user") == ["u2", "u3"]
    assert contents("assistant") == ["a2", "a3"]
    assert "a1" not in contents("assistant")

    # A turn that calls a tool: the tool message lands in the working copy only.
    tool_then_text = [
        [
            de(tool_calls=[tc(0, id="c1", name="get_weather", arguments='{"city": "X"}')]),
            de(finish_reason="tool_calls"),
        ],
        content_stream("a4"),
    ]
    await turn("u4", tool_then_text)
    assert all(m["role"] != "tool" for m in client._history), "tool message leaked into history"
    assert "tool_calls" not in {k for m in client._history for k in m}, "assistant tool_calls leaked into history"
    assert contents("assistant") == ["a3", "a4"]

    # Idle longer than the timeout wipes everything but the system message.
    clock["t"] += 200.0  # > history_idle_timeout_s
    await turn("u5", [content_stream("a5")])
    assert [m["role"] for m in client._history] == ["system", "user", "assistant"]
    assert contents("user") == ["u5"]
    assert client._history[0]["role"] == "system"
