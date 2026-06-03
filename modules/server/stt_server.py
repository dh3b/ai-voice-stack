import logging
import sys
import signal
import subprocess
import time

from config import STTServerConfig, AppConfig

logger = logging.getLogger("stt_server")


def start_server(config: STTServerConfig) -> subprocess.Popen:
    cmd = [
        sys.executable, "simulstreaming_lib/simulstreaming_whisper_server.py",
        "--host", config.server_host,
        "--port", str(config.server_port),
        "--model_path", config.model_path,
        "--language", config.language,
        "--vac",                  # Silero VAD, fires is_final
        "--min-chunk-size", str(config.min_chunk_size),
    ]

    if AppConfig().warmup_on_init:
        cmd.append("--warmup-file")
        cmd.append(config.warmup_audio_path)

    logger.info(f"Starting:{' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def main():
    logging.basicConfig(level=AppConfig().logging_level, format=AppConfig().logging_format)
    config = STTServerConfig()
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
