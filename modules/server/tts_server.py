import logging
import sys
import signal
import subprocess
import time

from config import TTSServerConfig, AppConfig

logger = logging.getLogger("tts_server")


def start_server(config: TTSServerConfig) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "piper.http_server",
        "--model", config.model_path,
        "--host", config.server_host,
        "--port", str(config.server_port),
    ]
    logger.info(f"Starting:{' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def main():
    logging.basicConfig(level=AppConfig().logging_level, format=AppConfig().logging_format)
    config = TTSServerConfig()
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
