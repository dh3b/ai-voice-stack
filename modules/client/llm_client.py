import asyncio
import logging
import time
import sys
import httpx
from openai import AsyncOpenAI

from config import LLMClientConfig, ToolsConfig, AppConfig
from modules.tools import registry, load_modules
from modules.utility.latency import tracer, LLM_FIRST_TOKEN

logger = logging.getLogger("voice_stack.llm")


class LLMClient:
    def __init__(self, llm_client_config: LLMClientConfig, tools_config: ToolsConfig | None = None):
        self._config = llm_client_config
        self._tools_config = tools_config if tools_config is not None else ToolsConfig()
        load_modules(self._tools_config.enabled_tool_modules)
        # Memory's prompt guidance is only appended when its tools are actually loaded.
        self._system_instructions = self._config.system_instructions
        if "memory_tools" in self._tools_config.enabled_tool_modules:
            self._system_instructions = (
                f"{self._config.system_instructions} "
                f"{self._tools_config.memory_system_instructions}"
            )
        self._base_url = f"http://{self._config.server_host}:{str(self._config.server_port)}/v1"
        self._client = AsyncOpenAI(base_url=self._base_url, api_key="none")
        self._history: list[dict] = []
        self._last_turn_at: float = 0.0
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
                    {"role": "system", "content": self._system_instructions},
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
            logger.debug("warmed up.")
        except Exception as e:
            logger.warning(f"warmup skipped ({e!r}); first turn may be cold.")

    async def run(self, user_message: str, queue: asyncio.Queue | None = None):
        if not self._config.history_enabled:
            messages = [
                {"role": "system", "content": self._system_instructions},
                {"role": "user", "content": user_message},
            ]
            if self._config.mode == "agent":
                await self._run_agent(messages, queue)
            elif self._config.mode == "chatbot":
                await self._run_chatbot(messages, queue)
            else:
                raise ValueError(f"Invalid mode: {self._config.mode}")
            return

        # Wipe to [system] after idle, wakeword being the reset button
        if len(self._history) > 1 and (time.monotonic() - self._last_turn_at) > self._config.history_idle_timeout_s:
            self._history = [self._history[0]]

        # Seed system message once (keeps KV prefix stable)
        if not self._history:
            self._history.append({"role": "system", "content": self._system_instructions})

        # Working copy carries user + any in-turn tool messages; _history gets only spoken text
        working = list(self._history) + [{"role": "user", "content": user_message}]

        spoken_text = ""
        try:
            if self._config.mode == "agent":
                spoken_text = await self._run_agent(working, queue)
            elif self._config.mode == "chatbot":
                spoken_text = await self._run_chatbot(working, queue)
            else:
                raise ValueError(f"Invalid mode: {self._config.mode}")
        finally:
            if spoken_text:
                self._history.append({"role": "user", "content": user_message})
                self._history.append({"role": "assistant", "content": spoken_text})
                # Drop oldest pairs from index 1 onward. 0 is system
                max_entries = 1 + 2 * self._config.history_max_turns
                while len(self._history) > max_entries:
                    del self._history[1:3]
            self._last_turn_at = time.monotonic()

    async def _run_chatbot(self, messages: list[dict], queue: asyncio.Queue | None = None) -> str:
        spoken_text = []
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
                spoken_text.append(chunk)
                if queue:
                    await queue.put(chunk)
                sys.stdout.write(chunk)
                sys.stdout.flush()

        sys.stdout.write("\n")
        return "".join(spoken_text)

    async def _run_agent(self, messages: list[dict], queue: asyncio.Queue | None = None) -> str:
        total_spoken = []
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
                    total_spoken.append(delta.content)
                    if queue:
                        await queue.put(delta.content)
                    sys.stdout.write(delta.content)
                    sys.stdout.flush()

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

            if response_content:
                sys.stdout.write("\n")

            if not pending_tool_calls:
                break

            # Tool messages appended only to the working copy
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

            for tc in pending_tool_calls.values():
                result_str = registry.call(tc["name"], tc["arguments"])
                logger.info(f"tool call {tc['name']}({tc['arguments']}): {result_str}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

            if finish_reason != "tool_calls":
                break

        else:
            logger.warning(f"Reached max iterations ({self._config.max_iterations}) without completing.")

        return "".join(total_spoken)


if __name__ == "__main__":
    logging.basicConfig(level=AppConfig().logging_level, format=AppConfig().logging_format)
    llm_client = LLMClient(LLMClientConfig())
    asyncio.run(llm_client.run("What is the weather in New York and what time is it in Tokyo?"))
