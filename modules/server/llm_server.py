import signal
import subprocess
import sys
import time

# Preferrably run this in a docker container

AGENT_MODEL = "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
CHATBOT_MODEL = "./models/gemma-4-E4B-it-Q4_K_M.gguf"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 43001

def start_server() -> subprocess.Popen:
    cmd = [
        "llama_cpp_bin/llama-server.exe",
        "-m", AGENT_MODEL,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--jinja",
        "-c", "4096",  # cap context; default loads the model's full ctx -> wasted KV on CPU
        # No NVIDIA GPU detected on this machine (nvidia-smi absent), so GPU offload
        # is omitted -- the ~1s time-to-first-token is CPU inference. With a
        # CUDA/Vulkan llama.cpp build + GPU, add "-ngl", "99" (and "-fa", "on");
        # that is the single biggest TTFT win. On CPU, prefer a smaller/faster
        # model or trimming prompt tokens (see P1-3 tool-schema gating).
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