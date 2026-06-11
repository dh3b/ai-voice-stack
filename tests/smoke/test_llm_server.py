"""
Chatbot mode, temperature 0. We assert non-empty streamed text and a first token within a
generous deadline. We deliberately do NOT assert tool calls here — tiny models are flaky
at function calling, and test_agent_tool_roundtrip owns that contract deterministically.
"""

import asyncio

import pytest

from config import LLMClientConfig, ToolsConfig
from modules.client.llm_client import LLMClient


@pytest.mark.smoke
async def test_llm_server_streams(llm_server_config):
    cfg = llm_server_config
    client = LLMClient(
        LLMClientConfig(
            model_path=cfg.model_path,
            server_host=cfg.server_host,
            server_port=cfg.server_port,
            mode="chatbot",
            temperature=0.0,
            history_enabled=False,
        ),
        ToolsConfig(enabled_tool_modules=[]),
    )

    queue: asyncio.Queue = asyncio.Queue()
    run_task = asyncio.create_task(client.run("Say hello in exactly one short sentence.", queue))
    try:
        first_token = await asyncio.wait_for(queue.get(), timeout=60.0)  # first-token deadline
        assert isinstance(first_token, str)
        await asyncio.wait_for(run_task, timeout=120.0)
    finally:
        if not run_task.done():
            run_task.cancel()

    rest = []
    while not queue.empty():
        item = queue.get_nowait()
        if item is not None:
            rest.append(item)
    text = first_token + "".join(rest)
    assert text.strip(), "LLM produced no streamed text"
