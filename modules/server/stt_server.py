import signal
import subprocess
import sys
import time

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 43002
MODEL_PATH  = "C:/Users/user/Documents/Projects/python/ai-voice-stack/models/whisper-base.pt"
LANGUAGE    = "auto"

def start_server() -> subprocess.Popen:
    cmd = [
        sys.executable, "simulstreaming_lib/simulstreaming_whisper_server.py",
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--model_path", MODEL_PATH,
        "--language", LANGUAGE,
        "--task", "transcribe",
        "--vac",                  # Silero VAD, fires is_final
        "--min-chunk-size", "1",  # process every ~1s of audio
    ]
    print(f"[server] Starting: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def main():
    proc = start_server()

    def _shutdown(sig, frame):
        print("\n[server] Shutting down...")
        proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[server] Listening on {SERVER_HOST}:{SERVER_PORT}")
    while True:
        ret = proc.poll()
        if ret is not None:
            print(f"[server] Process exited with code {ret}")
            sys.exit(ret)
        time.sleep(1)


if __name__ == "__main__":
    main()