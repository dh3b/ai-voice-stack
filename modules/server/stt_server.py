from pathlib import Path
import sys
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import STTServerConfig

def start_server(config: STTServerConfig) -> subprocess.Popen:
    cmd = [
        sys.executable, "simulstreaming_lib/simulstreaming_whisper_server.py",
        "--host", config.server_host,
        "--port", str(config.server_port),
        "--model_path", config.model_path,
        "--language", config.language,
        "--task", "transcribe",
        "--vac",                  # Silero VAD, fires is_final
        "--min-chunk-size", str(config.min_chunk_size),
    ]
    print(f"[server] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def main():
    config = STTServerConfig()
    proc = start_server(config)

    def _shutdown(sig, frame):
        print("\n[server] Shutting down...")
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[server] Listening on {config.server_host}:{config.server_port}")
    while True:
        ret = proc.poll()
        if ret is not None:
            print(f"[server] Process exited with code {ret}")
            sys.exit(ret)
        time.sleep(1)


if __name__ == "__main__":
    main()