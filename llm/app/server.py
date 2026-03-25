import os
import logging
import json
from aiohttp import web
from llama_cpp import Llama
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent.providers import LlamaCppPythonProvider
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
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
        
        # Chat history storage (per session - simplified for now)
        self.chat_histories: Dict[str, BasicChatHistory] = {}
        
        logger.info("LLM model loaded successfully")
    
    def _get_or_create_chat_history(self, session_id: str = "default") -> BasicChatHistory:
        """Get or create chat history for session"""
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = BasicChatHistory()
        return self.chat_histories[session_id]
    
    def _convert_tools_to_functions(self, tools: List[Dict[str, Any]]) -> List[callable]:
        """Convert tool definitions to callable functions for llama-cpp-agent"""
        functions = []
        
        for tool in tools:
            tool_name = tool.get("name")
            tool_description = tool.get("description", "")
            parameters = tool.get("parameters", {})
            
            # Create a dynamic function with proper annotations
            def create_tool_function(name: str, desc: str, params: Dict):
                def tool_func(**kwargs) -> str:
                    """Dynamic tool function"""
                    # Return tool call instruction for orchestrator to execute
                    return json.dumps({
                        "tool_call": True,
                        "name": name,
                        "arguments": kwargs
                    })
                
                # Set function metadata
                tool_func.__name__ = name
                tool_func.__doc__ = desc
                
                # Add parameter annotations
                annotations = {}
                for param_name, param_info in params.get("properties", {}).items():
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
                
                return tool_func
            
            functions.append(create_tool_function(tool_name, tool_description, parameters))
        
        return functions
    
    async def generate(self, request):
        """Generate LLM response"""
        try:
            data = await request.json()
            user_text = data.get("text", "")
            tools = data.get("tools", [])
            stream = data.get("stream", False)
            session_id = data.get("session_id", "default")
            
            # Get chat history
            chat_history = self._get_or_create_chat_history(session_id)
            
            # Add user message to history
            chat_history.add_message(Roles.user, user_text)
            
            # Convert tools to functions if in agent mode
            tool_functions = None
            if self.agent_mode and tools:
                tool_functions = self._convert_tools_to_functions(tools)
            
            # Create agent
            agent = LlamaCppAgent(
                self.provider,
                system_prompt=self.system_prompt,
                predefined_messages_formatter_type=None,  # Auto-detect from model
                debug_output=False
            )
            
            if stream:
                return await self._stream_response(request, agent, chat_history, user_text, tool_functions)
            else:
                return await self._complete_response(agent, chat_history, user_text, tool_functions)
                
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
    
    async def _stream_response(self, request, agent: LlamaCppAgent, chat_history: BasicChatHistory, 
                               user_text: str, tool_functions: Optional[List[callable]]):
        """Stream response tokens using llama-cpp-agent"""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/event-stream'
        response.headers['Cache-Control'] = 'no-cache'
        await response.prepare(request)
        
        try:
            # Generate with streaming
            if tool_functions:
                # Agent mode with tools
                result = agent.get_chat_response(
                    user_text,
                    chat_history=chat_history,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens,
                    tools=tool_functions,
                    stream=True
                )
            else:
                # Chat mode without tools
                result = agent.get_chat_response(
                    user_text,
                    chat_history=chat_history,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens,
                    stream=True
                )
            
            # Stream tokens
            full_response = ""
            for chunk in result:
                if isinstance(chunk, str):
                    full_response += chunk
                    await response.write(f"data: {chunk}\n\n".encode('utf-8'))
                elif isinstance(chunk, dict):
                    # Tool call result
                    await response.write(f"data: TOOL_CALL:{json.dumps(chunk)}\n\n".encode('utf-8'))
            
            # Add assistant response to history
            chat_history.add_message(Roles.assistant, full_response)
            
            await response.write(b"data: [DONE]\n\n")
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            await response.write(f"data: ERROR:{str(e)}\n\n".encode('utf-8'))
        finally:
            await response.write_eof()
        
        return response
    
    async def _complete_response(self, agent: LlamaCppAgent, chat_history: BasicChatHistory,
                                 user_text: str, tool_functions: Optional[List[callable]]):
        """Generate complete response using llama-cpp-agent"""
        try:
            # Generate response
            if tool_functions:
                # Agent mode with tools
                result = agent.get_chat_response(
                    user_text,
                    chat_history=chat_history,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens,
                    tools=tool_functions
                )
            else:
                # Chat mode without tools
                result = agent.get_chat_response(
                    user_text,
                    chat_history=chat_history,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_tokens=self.max_tokens
                )
            
            # Add assistant response to history
            chat_history.add_message(Roles.assistant, result)
            
            return web.json_response({
                "text": result,
                "session_id": "default"
            })
            
        except Exception as e:
            logger.error(f"Complete response error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)
    
    async def clear_history(self, request):
        """Clear chat history for a session"""
        try:
            data = await request.json()
            session_id = data.get("session_id", "default")
            
            if session_id in self.chat_histories:
                del self.chat_histories[session_id]
            
            return web.json_response({"status": "cleared", "session_id": session_id})
        except Exception as e:
            logger.error(f"Clear history error: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def health(self, request):
        return web.json_response({"status": "healthy"})

async def create_app():
    app = web.Application()
    service = LLMService()
    
    app.router.add_post('/generate', service.generate)
    app.router.add_post('/clear_history', service.clear_history)
    app.router.add_get('/health', service.health)
    
    return app

if __name__ == '__main__':
    web.run_app(create_app(), host='0.0.0.0', port=8003)
