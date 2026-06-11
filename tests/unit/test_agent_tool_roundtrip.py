"""
Drives LLMClient._run_agent with a faked streamed completion where one tool call is
split across deltas (id in one event, arguments fragmented over several), then a second
stream of plain content. This is the deterministic owner of the function-calling
contract (tier-2 deliberately does not assert tool calls against a tiny model).
"""

import asyncio

from config import LLMClientConfig, ToolsConfig


async def test_agent_tool_roundtrip(make_llm_client, clean_registry, openai_stub):
    registry = clean_registry
    invocations = []

    @registry.register(
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    )
    def get_weather(city):
        invocations.append(city)
        return f"It's sunny in {city}."

    de = openai_stub.delta_event
    tc = openai_stub.tool_call_delta

    stream_tool_call = [
        de(),  # leading null event the client skips
        de(tool_calls=[tc(0, id="call_abc", name="get_weather", arguments="")]),
        de(tool_calls=[tc(0, arguments='{"ci')]),
        de(tool_calls=[tc(0, arguments='ty": ')]),
        de(tool_calls=[tc(0, arguments='"Paris"}')]),
        de(finish_reason="tool_calls"),
    ]

    stream_content = [
        de(content="It is "),
        de(content="sunny in Paris."),
        de(finish_reason="stop"),
    ]
    fake = openai_stub.FakeAsyncOpenAI([stream_tool_call, stream_content])

    client = make_llm_client(
        LLMClientConfig(mode="agent", history_enabled=False),
        ToolsConfig(enabled_tool_modules=[]),
    )
    client._client = fake

    queue: asyncio.Queue = asyncio.Queue()
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "weather in Paris?"},
    ]
    final_text = await client._run_agent(messages, queue)

    # The handler ran exactly once, with the JSON arguments reassembled and parsed.
    assert invocations == ["Paris"]

    # The loop ran exactly two iterations (one create() per iteration).
    assert len(fake.chat.completions.calls) == 2

    # A role:"tool" message with the matching tool_call_id was appended, carrying the handler's return value.
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_abc"
    assert tool_messages[0]["content"] == "It's sunny in Paris."

    # The assistant tool-call turn was recorded before the tool result.
    assistant_tool_turn = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(assistant_tool_turn) == 1
    assert assistant_tool_turn[0]["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

    # The final spoken text reached the output queue and is returned.
    assert final_text == "It is sunny in Paris."
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert "".join(drained) == "It is sunny in Paris."
