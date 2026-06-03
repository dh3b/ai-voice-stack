"""Persistent memory tools: let the model store and recall facts about the user
across sessions. Backed by a module-level MemoryStore singleton (SQLite).
"""
from config import ToolsConfig
from modules.tools.registry import registry
from modules.utility.memory import MemoryStore

memory_store = MemoryStore(ToolsConfig().memory_db_path)


@registry.register({
    "type": "function",
    "function": {
        "name": "store_memory",
        "description": (
            "Store a fact about the user for future reference. Use a short descriptive key "
            "like 'name', 'preferred_units', 'work_schedule'. Overwrites if the key already exists."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "Short topic slug identifying the fact, e.g. 'name', "
                        "'preferred_units', 'work_schedule'."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "The fact to remember, e.g. 'Dave' or 'Celsius'.",
                },
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
    },
})
def store_memory(key: str, value: str) -> str:
    return memory_store.store(key, value)


@registry.register({
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": (
            "Search stored memories about the user. Use when you need to recall a previously "
            "stored fact or check if you already know something. Search by topic keyword(s) such "
            "as 'language', 'name', or 'units' rather than a full sentence"
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Topic keyword(s) to search for, e.g. 'favorite sport' or 'name'. "
                        "Individual words are matched against stored keys and values."
                    ),
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
})
def recall_memory(query: str) -> str:
    return memory_store.recall(query)
