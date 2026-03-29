import os
import logging
import json
import asyncio
from aiohttp import web
from llama_cpp import Llama
from llama_cpp_agent import FunctionCallingAgent, LlamaCppFunctionTool, MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent.chat_history import BasicChatHistory
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.model_path = os.getenv("LLM_MODEL_PATH", "/models/llm/model.gguf")
        self.context_size = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        self.top_p = float(os.getenv("LLM_TOP_P", "0.9"))
        self.top_k = int(os.getenv("LLM_TOP_K", "40"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
        self.n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", "0"))
        self.n_threads = int(os.getenv("LLM_N_THREADS", "4"))
        self.agent_mode = os.getenv("LLM_AGENT_MODE", "true").lower() == "true"
        self.system_prompt = os.getenv("LLM_SYSTEM_PROMPT", "You are a helpful voice assistant.")
        
        logger.info(f"Loading LLM model: {self.model_path}")
        
        # Initialize llama-cpp model
        self.llama_model = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_gpu_layers=self.n_gpu_layers,
            n_threads=self.n_threads,
            verbose=False
        )
        
        # Create provider for llama-cpp-agent
        self.provider = LlamaCppPythonProvider(self.llama_model)
        
        logger.info("LLM model loaded successfully")
    
    def _create_tool_from_spec(self, tool_spec: Dict[str, Any]) -> LlamaCppFunctionTool:
        """Create a LlamaCppFunctionTool from a JSON spec"""
        name = tool_spec.get("name")
        description = tool_spec.get("description", "")
        parameters = tool_spec.get("parameters", {})
        
        # Create a dynamic function with proper annotations
        def tool_func(**kwargs) -> str:
            """Dynamic tool function stub that will be intercepted by the orchestrator"""
            return json.dumps({
                "tool_call": True,
                "name": name,
                "arguments": kwargs
            })
        
        tool_func.__name__ = name
        tool_func.__doc__ = description
        
        # Add parameter annotations
        annotations = {}
        for param_name, param_info in parameters.get("properties", {}).items():
            param_type = param_info.get("type", "string")
            if param_type == "integer":
                annotations[param_name] = int
            elif param_type == "number":
                annotations[param_name] = float
            elif param_type == "boolean":
                annotations[param_name] = bool
            else:
                annotations[param_name] = str
        
        annotations["return"] = str
        tool_func.__annotations__ = annotations
        
        return LlamaCppFunctionTool(tool_func)

    async def generate(self, request):
        """Generate LLM response using FunctionCallingAgent"""
        try:
            data = await request.json()
            user_text = data.get("text", "")
            tools_specs = data.get("tools", [])
            stream = data.get("stream", False)
            
            logger.info(f"Generating response for: {user_text}")
            
            # Prepare tools
            function_tools = [self._create_tool_from_spec(spec) for spec in tools_specs]
            
            # We create a new agent instance per request to handle statelessness if needed, 
            # or we could cache it. For voice, usually it's one turn at a time.
            agent = FunctionCallingAgent(
                self.provider,
                llama_cpp_function_tools=function_tools,
                system_prompt=self.system_prompt,
                allow_parallel_function_calling=True,
                messages_formatter_type=MessagesFormatterType.LLAMA_3,
            )
            
            if stream:
                return await self._stream_response(request, agent, user_text)
            else:
                chat_history = BasicChatHistory()
                response_text = agent.get_chat_response(user_text, chat_history=chat_history)
                return web.json_response({
                    "text": response_text.strip()
                })
                
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
    
    async def _stream_response(self, request, agent: FunctionCallingAgent, user_text: str):
        """Stream response tokens or tool calls"""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        await response.prepare(request)
        
        try:
            # Use get_chat_response for streaming support
            chat_history = BasicChatHistory()
            result = agent.get_chat_response(
                user_text,
                chat_history=chat_history,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            # The result is a generator yielding chunks or tool calls
            for chunk in result:
                if isinstance(chunk, str):
                    await response.write(f"data: {chunk}\n\n".encode('utf-8'))
                elif isinstance(chunk, dict):
                    # Tool call or metadata
                    await response.write(f"data: TOOL_CALL:{json.dumps(chunk)}\n\n".encode('utf-8'))
            
            await response.write(b"data: [DONE]\n\n")
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            await response.write(f"data: ERROR:{str(e)}\n\n".encode('utf-8'))
        finally:
            await response.write_eof()
        
        return response

    async def health(self, request):
        return web.json_response({"status": "healthy"})

async def create_app():
    app = web.Application()
    service = LLMService()
    
    app.router.add_post('/generate', service.generate)
    app.router.add_get('/health', service.health)
    
    return app

if __name__ == '__main__':
    web.run_app(create_app(), host='0.0.0.0', port=8003)
