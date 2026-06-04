import logging
import sys
import signal
import subprocess
import time

from config import LLMServerConfig, AppConfig

logger = logging.getLogger("llm_server")

def start_server(config: LLMServerConfig) -> subprocess.Popen:
    cmd = [
        config.executable_path,
        "-m", config.model_path,
        "--host", config.server_host,
        "--port", str(config.server_port),
        "--jinja",
        "-c", str(config.context_window),
    ]

    if config.n_gpu_layers > 0:
        cmd += ["-ngl", str(config.n_gpu_layers)]
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
