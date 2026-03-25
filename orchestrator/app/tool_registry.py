from typing import Callable, Dict, Any, List
import inspect
import json

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str = None, description: str = None):
        """Decorator to register a function as a tool"""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or "No description"
            
            # Extract parameters from function signature
            sig = inspect.signature(func)
            parameters = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                
                parameters[param_name] = {"type": param_type}
                
                if param.default == inspect.Parameter.empty:
                    required.append(param_name)
            
            self.tools[tool_name] = {
                "name": tool_name,
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required
                },
                "function": func
            }
            
            return func
        return decorator
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions for LLM"""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in self.tools.values()
        ]
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a registered tool"""
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        
        func = self.tools[name]["function"]
        
        if inspect.iscoroutinefunction(func):
            return await func(**arguments)
        else:
            return func(**arguments)

# Global registry instance
registry = ToolRegistry()

# Example tools
@registry.register(description="Get the current time")
def get_current_time() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

@registry.register(description="Get the current date")
def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")
