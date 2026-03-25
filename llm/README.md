# LLM Service with llama-cpp-agent

This service uses `llama-cpp-agent` for proper system-level tool integration with LLMs.

## Architecture

### llama-cpp-agent Integration

The service uses `llama-cpp-agent` library which provides:

1. **System-level tool calling** - Tools are passed to the agent at the system level, not via prompt instructions
2. **Automatic function schema generation** - Converts tool definitions to proper function signatures
3. **Chat history management** - Maintains conversation context across turns
4. **Streaming support** - Real-time token generation
5. **Provider abstraction** - Clean interface over llama-cpp-python

### How It Works

#### Tool Registration Flow

```
Orchestrator Tool Registry
    ↓ (sends tool definitions)
LLM Service
    ↓ (converts to callable functions)
llama-cpp-agent
    ↓ (system-level integration)
LLM Model
```

#### Tool Conversion

Tools are converted from JSON schema to Python callables:

```python
# Input: Tool definition from orchestrator
{
    "name": "get_current_time",
    "description": "Get the current time",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}

# Output: Python function with annotations
def get_current_time() -> str:
    """Get the current time"""
    return json.dumps({
        "tool_call": True,
        "name": "get_current_time",
        "arguments": {}
    })
```

The function annotations allow llama-cpp-agent to:
- Generate proper tool schemas for the model
- Validate parameters at runtime
- Handle type conversions automatically

#### Response Flow

1. User message received
2. Agent created with system prompt and tools
3. llama-cpp-agent handles:
   - Formatting messages for the model
   - Injecting tool schemas at system level
   - Parsing model output for tool calls
   - Streaming tokens back
4. Tool calls returned as JSON for orchestrator to execute

### Agent vs Chat Mode

**Agent Mode** (`LLM_AGENT_MODE=true`):
- Tools passed to llama-cpp-agent
- Model can call functions
- Responses may include tool calls

**Chat Mode** (`LLM_AGENT_MODE=false`):
- No tools provided
- Pure conversational responses
- Faster generation

### Session Management

Each conversation has a session ID (default: "default"):
- Chat history maintained per session
- Context preserved across turns
- Can clear history via `/clear_history` endpoint

### Streaming

Tokens are streamed via Server-Sent Events (SSE):

```
data: Hello
data: ,
data:  how
data:  can
data:  I
data:  help
data: ?
data: [DONE]
```

Tool calls are marked:
```
data: TOOL_CALL:{"tool_call":true,"name":"get_time","arguments":{}}
```

## API Endpoints

### POST /generate

Generate LLM response with optional tool calling.

**Request:**
```json
{
    "text": "What time is it?",
    "tools": [
        {
            "name": "get_current_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ],
    "stream": true,
    "session_id": "default"
}
```

**Response (streaming):**
```
data: Let
data:  me
data:  check
data: TOOL_CALL:{"tool_call":true,"name":"get_current_time","arguments":{}}
data:  the
data:  time
data: [DONE]
```

**Response (non-streaming):**
```json
{
    "text": "The current time is 3:45 PM",
    "session_id": "default"
}
```

### POST /clear_history

Clear chat history for a session.

**Request:**
```json
{
    "session_id": "default"
}
```

**Response:**
```json
{
    "status": "cleared",
    "session_id": "default"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
    "status": "healthy"
}
```

## Configuration

Environment variables:

- `LLM_MODEL_PATH` - Path to GGUF model file
- `LLM_CONTEXT_SIZE` - Context window size (default: 4096)
- `LLM_TEMPERATURE` - Sampling temperature (default: 0.7)
- `LLM_TOP_P` - Nucleus sampling (default: 0.9)
- `LLM_TOP_K` - Top-k sampling (default: 40)
- `LLM_MAX_TOKENS` - Max tokens to generate (default: 512)
- `LLM_N_GPU_LAYERS` - GPU layers to offload (default: 0)
- `LLM_N_THREADS` - CPU threads (default: 4)
- `LLM_AGENT_MODE` - Enable agent mode (default: true)
- `LLM_SYSTEM_PROMPT` - System prompt (default: "You are a helpful voice assistant.")

## Dependencies

- `llama-cpp-python` - Core LLM inference
- `llama-cpp-agent` - Agent framework with tool support
- `aiohttp` - Async HTTP server

## Benefits of llama-cpp-agent

1. **Proper Tool Integration**: Tools are part of the model's system context, not prompt hacks
2. **Type Safety**: Function annotations ensure correct parameter types
3. **Automatic Schema Generation**: No manual JSON schema writing
4. **Chat History**: Built-in conversation management
5. **Model Agnostic**: Works with any llama-cpp-python compatible model
6. **Streaming Support**: Real-time token generation
7. **Clean Abstraction**: Separates concerns between model inference and tool execution

## Adding Custom Tools

Tools are defined in `orchestrator/app/tool_registry.py`:

```python
@registry.register(description="Get weather for a location")
def get_weather(location: str, units: str = "celsius") -> str:
    # Implementation
    return f"Weather in {location}: 22°{units[0].upper()}"
```

The LLM service automatically:
1. Receives tool definition from orchestrator
2. Converts to callable function with proper annotations
3. Passes to llama-cpp-agent
4. Returns tool call instructions to orchestrator
5. Orchestrator executes actual tool and returns result

This separation ensures:
- LLM service remains stateless
- Tool execution happens in orchestrator context
- Easy to add/modify tools without changing LLM service
