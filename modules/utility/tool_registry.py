import json

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, callable] = {}

    def register(self, schema: dict):
        """Decorator to register a function as a tool."""
        def decorator(fn):
            name = schema["function"]["name"]
            self._tools[name] = schema
            self._handlers[name] = fn
            return fn
        return decorator

    def schemas(self) -> list[dict]:
        """Return all tool schemas for the API call."""
        return list(self._tools.values())

    def call(self, name: str, arguments: str) -> str:
        fn = self._handlers.get(name)
        if not fn:
            return f"Unknown tool: {name}"
        try:
            return str(fn(**json.loads(arguments)))
        except json.JSONDecodeError as e:
            return f"Error parsing arguments for {name}: {e}"
        except Exception as e:
            return f"Error running {name}: {e}"

registry = ToolRegistry()

@registry.register({
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
})
def get_weather(city: str) -> str:
    # Replace with a real API call
    return f"It's 22°C and sunny in {city}."


@registry.register({
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "e.g. '2 + 2 * 10'"}
            },
            "required": ["expression"]
        }
    }
})
def calculate(expression: str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"