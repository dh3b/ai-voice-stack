import importlib
import importlib.util
import logging

from modules.tools.registry import registry

logger = logging.getLogger("voice_stack.tools")


def load_modules(names: list[str]) -> None:
    for name in names:
        if importlib.util.find_spec(f"{__name__}.{name}") is None:
            logger.warning("Unknown tool module %r in enabled_tool_modules; skipping.", name)
            continue
        importlib.import_module(f"{__name__}.{name}")


__all__ = ["registry", "load_modules"]
