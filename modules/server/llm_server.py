import logging
from pathlib import Path
import sys
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import LLMServerConfig

logger = logging.getLogger("llm_server")

def start_server(config: LLMServerConfig) -> subprocess.Popen:
    cmd = [
        "llama_cpp_bin/llama-server.exe",
        "-m", config.model_path,
        "--host", config.server_host,
        "--port", str(config.server_port),
        "--jinja",
        "-c", str(config.context_window),
        # TODO: add "-ngl", "99" (and "-fa", "on");
    ]

    logger.info(f"[server] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

def main():
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    config = LLMServerConfig()
    proc = start_server(config)

    def _shutdown(sig, frame):
        logger.info("[server] Shutting down...")
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info(f"[server] Listening on {config.server_host}:{config.server_port}")
    while True:
        ret = proc.poll()
        if ret is not None:
            logger.info(f"[server] Process exited with code {ret}")
            sys.exit(ret)
        time.sleep(1)

if __name__ == "__main__":
    main()
