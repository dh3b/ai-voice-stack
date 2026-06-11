import logging
import sys
import signal
import subprocess
import time
from pathlib import Path

from config import LLMServerConfig, AppConfig

logger = logging.getLogger("llm_server")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(path: str) -> str:
    """Absolute, OS-native path (Windows CreateProcess rejects relative '/' exe paths)."""
    p = Path(path)
    return str(p if p.is_absolute() else (_REPO_ROOT / p))


def start_server(config: LLMServerConfig) -> subprocess.Popen:
    cmd = [
        _resolve(config.executable_path),
        "-m", _resolve(config.model_path),
        "--host", config.server_host,
        "--port", str(config.server_port),
        "--jinja",
        "-c", str(config.context_window),
    ]
    if config.gpu_layers:
        cmd += ["-ngl", str(config.gpu_layers)]
    if config.flash_attn:
        cmd += ["-fa", "on"]

    logger.info(f"Starting:{' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

def main():
    logging.basicConfig(level=AppConfig().logging_level, format=AppConfig().logging_format)
    config = LLMServerConfig()
    proc = start_server(config)

    def _shutdown(sig, frame):
        logger.info("Shutting down...")
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(f"Listening on {config.server_host}:{config.server_port}")
    while True:
        ret = proc.poll()
        if ret is not None:
            logger.info(f"Process exited with code {ret}")
            sys.exit(ret)
        time.sleep(1)

if __name__ == "__main__":
    main()
