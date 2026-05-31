from pathlib import Path
import sys
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TTSServerConfig


def start_server(config: TTSServerConfig) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "piper.http_server",
        "--model", config.model_path,
        "--host", config.server_host,
        "--port", str(config.server_port),
    ]
    print(f"[server] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def main():
    config = TTSServerConfig()
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
