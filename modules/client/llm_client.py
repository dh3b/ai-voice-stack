import asyncio
import sys
from pathlib import Path
import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # temporary
from config import LLMClientConfig, AppConfig
from modules.utility.tool_registry import registry
from modules.utility.latency import tracer, LLM_FIRST_TOKEN


class LLMClient:
    def __init__(self, llm_client_config: LLMClientConfig):
        self._config = llm_client_config
        self._base_url = f"http://{self._config.server_host}:{str(self._config.server_port)}/v1"
        self._client = AsyncOpenAI(base_url=self._base_url, api_key="none")
        if AppConfig().warmup_on_init:
            self._warmup()

    def _warmup(self) -> None:
        """Prime llama-server (blocking) with a 1-token completion so the first
        real turn isn't a cold prefill. Warms the tool-schema prefill path too when in agent mode.
        """
        try:
            payload = {
                "model": self._config.model_path,
                "messages": [
                    {"role": "system", "content": self._config.system_instructions},
                    {"role": "user", "content": "Hi"},
                ],
                "max_tokens": 1,
                "temperature": self._config.temperature,
            }
            if self._config.mode == "agent":
                payload["tools"] = registry.schemas()
            with httpx.Client() as client:
                resp = client.post(
                    f"{self._base_url}/chat/completions", json=payload, timeout=120.0
                )
                resp.raise_for_status()
            print("[llm] warmed up.")
        except Exception as e:
            print(f"[llm] warmup skipped ({e!r}); first turn may be cold.")

    async def run(self, user_message: str, queue: asyncio.Queue | None = None):
        messages = [
            {"role": "system", "content": self._config.system_instructions},
            {"role": "user", "content": user_message},
        ]

        if self._config.mode == "agent":
            await self._run_agent(messages, queue)
        elif self._config.mode == "chatbot":
            await self._run_chatbot(messages, queue)
        else:
            raise ValueError(f"Invalid mode: {self._config.mode}")

    async def _run_chatbot(self, messages: str, queue: asyncio.Queue | None = None):
        stream = await self._client.chat.completions.create(
            model=self._config.model_path,
            messages=messages,
            stream=True,
            temperature=self._config.temperature,
        )

        async for event in stream:
            chunk = event.choices[0].delta.content
            if isinstance(chunk, str):
                tracer.mark(LLM_FIRST_TOKEN)
                if queue:
                    await queue.put(chunk)
                print(chunk, end="", flush=True)

        print()

    async def _run_agent(self, messages: str, queue: asyncio.Queue | None = None):
        for it in range(self._config.max_iterations):
            stream = await self._client.chat.completions.create(
                model=self._config.model_path,
                messages=messages,
                tools=registry.schemas(),
                tool_choice="auto",
                temperature=self._config.temperature,
                stream=True,
            )

            response_content = ""
            pending_tool_calls: dict[int, dict] = {}
            finish_reason = None

            async for event in stream:
                choice = event.choices[0]
                delta = choice.delta

                if (
                    delta.content is None
                    and not delta.tool_calls
                    and not choice.finish_reason
                ):
                    continue  # first event is null

                if delta.content:
                    tracer.mark(LLM_FIRST_TOKEN)
                    response_content += delta.content
                    if queue:
                        await queue.put(delta.content)
                    print(delta.content, end="", flush=True)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index

                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": tc_delta.id,
                                "name": tc_delta.function.name,
                                "arguments": "",
                            }

                        if tc_delta.function and tc_delta.function.arguments:
                            pending_tool_calls[idx]["arguments"] += tc_delta.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                    break

            # Newline after any streamed content.
            if response_content:
                print()

            if not pending_tool_calls:
                break

            assistant_message = {
                "role": "assistant",
                "content": response_content if response_content else None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in pending_tool_calls.values()
                ],
            }
            messages.append(assistant_message)

            # Execute every requested tool call and append each result.
            for tc in pending_tool_calls.values():
                result_str = registry.call(tc["name"], tc["arguments"])
                print(f"[Tool call: {tc['name']}({tc['arguments']})] {result_str}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

            if finish_reason != "tool_calls":
                break

        else:
            print(f"[Agent] Reached max iterations ({self._config.max_iterations}) without completing.")


if __name__ == "__main__":
    llm_client = LLMClient(LLMClientConfig())
    asyncio.run(llm_client.run("What is the weather in New York and what time is it in Tokyo?"))