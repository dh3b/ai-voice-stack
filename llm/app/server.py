"""
LLM service: llama-cpp-python + llama-cpp-agent.

Exposes two inference endpoints:
  POST /chat   — agent mode (tool calling via llama-cpp-agent), returns full JSON
  POST /stream — streaming mode (token-by-token SSE via llama-cpp-python)
  POST /reset  — clear conversation history
  GET  /health
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List, Optional

from aiohttp import web
from llama_cpp import Llama
from llama_cpp_agent import (
    FunctionCallingAgent,
    LlamaCppFunctionTool,
    MessagesFormatterType,
)
from llama_cpp_agent.providers import LlamaCppPythonProvider

from app.tools import default_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm")


class LLMService:
    def __init__(self):
        self.model_path = os.getenv("LLM_MODEL_PATH", "/models/llm/model.gguf")
        self.system_prompt = os.getenv(
            "LLM_SYSTEM_PROMPT",
            "You are a helpful AI voice assistant. Be concise and clear.",
        )
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.8"))
        self.n_ctx = int(os.getenv("LLM_N_CTX", "4096"))
        self.n_threads = int(os.getenv("LLM_N_THREADS", "6"))
        self.top_p = float(os.getenv("LLM_TOP_P", "0.95"))
        self.agent_enabled = os.getenv("AGENT_ENABLED", "false").lower() == "true"

        # Conversation history for streaming (non-agent) mode
        self.history: list[dict[str, str]] = []

        # Load model
        logger.info(
            "Loading LLM  path=%s  n_ctx=%d  threads=%d",
            self.model_path, self.n_ctx, self.n_threads,
        )
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            verbose=False,
        )
        logger.info("LLM ready")

        # Set up agent (if enabled)
        self._agent: Optional[FunctionCallingAgent] = None
        if self.agent_enabled:
            self._setup_agent()

    def _setup_agent(self) -> None:
        registry = default_registry()
        provider = LlamaCppPythonProvider(self.model)

        function_tools: List[LlamaCppFunctionTool] = [
            LlamaCppFunctionTool(spec.func) for spec in registry.all()
        ]

        def _send_message_callback(message: str, **_kwargs: Any) -> None:
            text = (message or "").strip()
            if text:
                logger.info("Agent(message) -> %s", text)

        self._agent = FunctionCallingAgent(
            provider,
            llama_cpp_function_tools=function_tools,
            system_prompt=self.system_prompt,
            allow_parallel_function_calling=True,
            send_message_to_user_callback=_send_message_callback,
            messages_formatter_type=MessagesFormatterType.LLAMA_3,
        )
        logger.info("Agent ready with %d tools", len(function_tools))

    # ── /chat endpoint (agent mode) ──────────────────────────────────────────

    async def chat(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            user_text = data.get("text", "").strip()
            if not user_text:
                return web.json_response({"error": "No text provided"}, status=400)

            logger.info("chat <- '%s'", user_text)

            if self._agent is not None:
                response_text = str(
                    self._agent.generate_response(user_text)
                ).strip()
            else:
                response_text = self._chat_plain(user_text)

            logger.info("chat -> '%s'", response_text)
            return web.json_response({"text": response_text})

        except Exception as e:
            logger.error("Chat error: %s", e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    def _chat_plain(self, user_text: str) -> str:
        """Non-streaming, non-agent chat via llama-cpp-python directly."""
        self.history.append({"role": "user", "content": user_text})

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.history,
        ]
        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        reply = response["choices"][0]["message"]["content"].strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    # ── /stream endpoint (token-by-token SSE) ────────────────────────────────

    async def stream(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        response.headers["Content-Type"] = "text/event-stream"
        response.headers["Cache-Control"] = "no-cache"
        await response.prepare(request)

        try:
            data = await request.json()
            user_text = data.get("text", "").strip()
            if not user_text:
                await response.write(b"data: ERROR:No text provided\n\n")
                await response.write_eof()
                return response

            logger.info("stream <- '%s'", user_text)

            self.history.append({"role": "user", "content": user_text})
            messages = [
                {"role": "system", "content": self.system_prompt},
                *self.history,
            ]

            full_reply: list[str] = []

            for chunk in self.model.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
            ):
                token = chunk["choices"][0]["delta"].get("content", "")
                if token:
                    full_reply.append(token)
                    await response.write(f"data: {token}\n\n".encode("utf-8"))

            reply = "".join(full_reply).strip()
            self.history.append({"role": "assistant", "content": reply})
            logger.info("stream -> '%s'", reply)

            await response.write(b"data: [DONE]\n\n")

        except Exception as e:
            logger.error("Stream error: %s", e, exc_info=True)
            await response.write(f"data: ERROR:{e}\n\n".encode("utf-8"))
        finally:
            await response.write_eof()

        return response

    # ── /reset endpoint ──────────────────────────────────────────────────────

    async def reset(self, _request: web.Request) -> web.Response:
        self.history.clear()
        logger.info("Conversation history cleared")
        return web.json_response({"status": "cleared"})

    # ── /health ──────────────────────────────────────────────────────────────

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "healthy"})


async def create_app() -> web.Application:
    app = web.Application()
    service = LLMService()

    app.router.add_post("/chat", service.chat)
    app.router.add_post("/stream", service.stream)
    app.router.add_post("/reset", service.reset)
    app.router.add_get("/health", service.health)

    return app


if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=8003)
