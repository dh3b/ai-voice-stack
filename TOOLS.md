# Tool System Documentation

## Overview

The voice assistant uses a tool registry system that allows the LLM agent to call functions. Tools are defined once in the orchestrator and automatically made available to the LLM via llama-cpp-agent.

## Architecture

```
┌─────────────────────┐
│   Tool Registry     │  ← Define tools here
│  (orchestrator)     │
└──────────┬──────────┘
           │ Tool definitions (JSON schema)
           ▼
┌─────────────────────┐
│    LLM Service      │  ← Converts to callables
│  (llama-cpp-agent)  │
└──────────┬──────────┘
           │ Tool call instructions
           ▼
┌─────────────────────┐
│   Orchestrator      │  ← Executes actual tools
│    (pipeline)       │
└─────────────────────┘
```

## How It Works

### 1. Tool Definition

Tools are defined in `orchestrator/app/tool_registry.py` using decorators:

```python
from app.tool_registry import registry

@registry.register(description="Get the current time")
def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
```

### 2. Automatic Schema Generation

The registry automatically extracts:
- Function name
- Description (from decorator or docstring)
- Parameters (from function signature)
- Parameter types (from type annotations)
- Required vs optional parameters (from defaults)

Generated schema:
```json
{
    "name": "get_current_time",
    "description": "Get the current time",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

### 3. LLM Integration

When the orchestrator calls the LLM:
1. Tool definitions sent in request
2. LLM service converts to Python callables
3. llama-cpp-agent integrates at system level
4. Model can now "see" and call tools

### 4. Tool Execution

When LLM wants to use a tool:
1. Model generates tool call
2. LLM service returns: `TOOL_CALL:{"tool_call":true,"name":"get_current_time","arguments":{}}`
3. Orchestrator parses and executes actual function
4. Result incorporated into response

## Adding New Tools

### Simple Tool

```python
@registry.register(description="Get current date")
def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
```

### Tool with Parameters

```python
@registry.register(description="Calculate sum of two numbers")
def add_numbers(a: int, b: int) -> int:
    return a + b
```

### Tool with Optional Parameters

```python
@registry.register(description="Greet a person")
def greet(name: str, formal: bool = False) -> str:
    if formal:
        return f"Good day, {name}."
    return f"Hey {name}!"
```

### Async Tool

```python
@registry.register(description="Fetch data from API")
async def fetch_data(url: str) -> str:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()
```

### Tool with Complex Return

```python
@registry.register(description="Get system information")
def get_system_info() -> str:
    import platform
    import json
    
    info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine()
    }
    
    # Return as formatted string for voice output
    return f"System: {info['system']}, Release: {info['release']}"
```

## Type Annotations

Supported parameter types:
- `str` → string
- `int` → integer
- `float` → number
- `bool` → boolean

Return type should always be `str` for voice output.

## Best Practices

### 1. Clear Descriptions

```python
# Good
@registry.register(description="Get weather forecast for a specific city")
def get_weather(city: str) -> str:
    pass

# Bad
@registry.register(description="Weather")
def get_weather(city: str) -> str:
    pass
```

### 2. Voice-Friendly Output

```python
# Good - Natural speech
def get_time() -> str:
    return "It's 3:45 PM"

# Bad - Not natural for TTS
def get_time() -> str:
    return "15:45:00"
```

### 3. Error Handling

```python
@registry.register(description="Divide two numbers")
def divide(a: float, b: float) -> str:
    try:
        result = a / b
        return f"The result is {result}"
    except ZeroDivisionError:
        return "I cannot divide by zero"
```

### 4. Parameter Validation

```python
@registry.register(description="Set timer for specified minutes")
def set_timer(minutes: int) -> str:
    if minutes <= 0:
        return "Timer duration must be positive"
    if minutes > 1440:  # 24 hours
        return "Timer duration too long, maximum is 24 hours"
    
    # Set timer logic
    return f"Timer set for {minutes} minutes"
```

## Example Tools

### Time & Date

```python
@registry.register(description="Get current time")
def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p")

@registry.register(description="Get current date")
def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%B %d, %Y")

@registry.register(description="Get day of week")
def get_day_of_week() -> str:
    from datetime import datetime
    return datetime.now().strftime("%A")
```

### System Control

```python
@registry.register(description="Get system uptime")
def get_uptime() -> str:
    import subprocess
    result = subprocess.run(['uptime', '-p'], capture_output=True, text=True)
    return result.stdout.strip()

@registry.register(description="Get CPU temperature")
def get_cpu_temp() -> str:
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read()) / 1000
        return f"CPU temperature is {temp:.1f} degrees Celsius"
    except:
        return "Unable to read CPU temperature"
```

### Calculations

```python
@registry.register(description="Calculate percentage")
def calculate_percentage(part: float, whole: float) -> str:
    if whole == 0:
        return "Cannot calculate percentage of zero"
    percentage = (part / whole) * 100
    return f"{part} is {percentage:.1f} percent of {whole}"

@registry.register(description="Convert temperature from Celsius to Fahrenheit")
def celsius_to_fahrenheit(celsius: float) -> str:
    fahrenheit = (celsius * 9/5) + 32
    return f"{celsius} degrees Celsius is {fahrenheit:.1f} degrees Fahrenheit"
```

### Information Retrieval

```python
@registry.register(description="Get definition of a word")
async def get_definition(word: str) -> str:
    # Implement dictionary lookup
    # This is a placeholder
    return f"Definition of {word}: [dictionary lookup would go here]"
```

## Testing Tools

Test tools directly:

```python
from app.tool_registry import registry
import asyncio

# Test sync tool
result = asyncio.run(registry.execute_tool("get_current_time", {}))
print(result)

# Test async tool
result = asyncio.run(registry.execute_tool("fetch_data", {"url": "https://example.com"}))
print(result)

# Test with parameters
result = asyncio.run(registry.execute_tool("add_numbers", {"a": 5, "b": 3}))
print(result)
```

## Debugging

Enable debug logging in `orchestrator/app/pipeline.py`:

```python
logger.setLevel(logging.DEBUG)
```

This will show:
- Tool definitions sent to LLM
- Tool calls received from LLM
- Tool execution results
- Any errors during execution

## Limitations

1. **Return Type**: Tools must return strings (for voice output)
2. **Execution Context**: Tools run in orchestrator, not LLM service
3. **Synchronous Execution**: Tool calls are sequential, not parallel
4. **No State**: Tools should be stateless or manage their own state
5. **Voice Output**: Results must be TTS-friendly

## Future Enhancements

- Tool categories/namespaces
- Parallel tool execution
- Tool result caching
- Permission system for sensitive tools
- Tool usage analytics
- Dynamic tool loading from plugins
