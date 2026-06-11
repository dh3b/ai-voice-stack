"""
LLMClient._call_tool runs the tool in a worker thread under
asyncio.wait_for, so a hung tool is abandoned at the configured budget while the loop
keeps servicing other coroutines.
"""

import asyncio
import threading

from config import LLMClientConfig, ToolsConfig


async def test_tool_timeout_nonblocking(make_llm_client, clean_registry):
    registry = clean_registry
    release = threading.Event()

    @registry.register(
        {
            "type": "function",
            "function": {
                "name": "slow_tool",
                "description": "Blocks.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
    def slow_tool():
        release.wait(timeout=3.0)
        return "slow result"

    @registry.register(
        {
            "type": "function",
            "function": {
                "name": "fast_tool",
                "description": "Returns immediately.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
    def fast_tool():
        return "fast result"

    client = make_llm_client(
        LLMClientConfig(),
        ToolsConfig(enabled_tool_modules=[], tool_timeout=1.0),
    )

    loop = asyncio.get_running_loop()
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    try:
        start = loop.time()
        result = await client._call_tool("slow_tool", "{}")
        elapsed = loop.time() - start
    finally:
        release.set()

    # Returned the timeout error string at ~1s, not after the tool's 3s block.
    assert "timed out after 1.0 seconds" in result
    assert 0.9 <= elapsed < 2.5, f"elapsed={elapsed:.2f}s"
    assert ticks >= 20, f"only {ticks} ticks — loop was blocked"
    assert await client._call_tool("fast_tool", "{}") == "fast result"

    ticker_task.cancel()
    try:
        await ticker_task
    except asyncio.CancelledError:
        pass
