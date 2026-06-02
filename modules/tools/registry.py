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
