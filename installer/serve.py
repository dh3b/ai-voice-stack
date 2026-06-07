"""`task run` supervisor: start the three backend servers (output to logs/<name>.log)
one at a time, then run pipeline.py on the console. Tears everything down on exit."""
from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from . import util

# (label, module), started in this order, each waited-on before the next.
_SERVERS = [
    ("llm", "modules.server.llm_server"),
    ("stt", "modules.server.stt_server"),
    ("tts", "modules.server.tts_server"),
]


@dataclass
class _Proc:
    label: str
    popen: subprocess.Popen
    log_path: Path | None = None
    log_file: IO | None = None


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500  # any HTTP response means the server is up
    except urllib.error.URLError:
        return False
    except OSError:
        return False


def _tail(path: Path | None, n: int = 20) -> str:
    if not path:
        return ""
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _mem() -> str:
    """Unified-memory snapshot (Linux/Jetson). GPU shares this pool, so it's the
    number that matters for CUDA allocations."""
    try:
        info = dict(
            line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line
        )
        avail = int(info["MemAvailable"].split()[0]) // 1024
        free = int(info["MemFree"].split()[0]) // 1024
        swap = int(info["SwapFree"].split()[0]) // 1024
        total = int(info["MemTotal"].split()[0]) // 1024
        return f"avail={avail}MB free={free}MB swapfree={swap}MB / {total}MB"
    except (OSError, KeyError, ValueError, IndexError):
        return "n/a"


def _log_env() -> None:
    """The CUDA-relevant env the child servers inherit (compare to your manual shell)."""
    for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "CUDA_HOME", "VIRTUAL_ENV"):
        util.logger.info("  env  %s=%s", k, os.environ.get(k, "(unset)"))


def _ready_checks() -> dict:
    import config as cfg  # repo-root module; importable once deps/venv are set up

    llm, stt, tts = cfg.LLMServerConfig(), cfg.STTServerConfig(), cfg.TTSServerConfig()
    return {
        "llm": lambda: _http_ok(f"http://{llm.server_host}:{llm.server_port}/health"),
        "stt": lambda: _port_open(stt.server_host, stt.server_port),
        "tts": lambda: _port_open(tts.server_host, tts.server_port),
    }


def _spawn_server(label: str, module: str) -> _Proc:
    util.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = util.LOGS_DIR / f"{label}_server.log"
    log_file = open(log_path, "w")
    popen = subprocess.Popen(
        [util.venv_python(), "-m", module],
        cwd=str(util.REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return _Proc(label, popen, log_path, log_file)


def _wait_one(proc: _Proc, check, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.popen.poll() is not None:
            util.logger.error("  !!   %s exited early (code %s). Last log lines:",
                              proc.label, proc.popen.returncode)
            util.logger.error("%s", _tail(proc.log_path))
            return False
        if check():
            return True
        time.sleep(1.0)
    util.logger.error("  !!   timed out waiting for %s", proc.label)
    return False


def _terminate_tree(p: subprocess.Popen) -> None:
    # On Windows .terminate() kills only the supervisor, orphaning its llama-server /
    # piper child (which keeps holding its port). taskkill /T takes the whole tree.
    # On POSIX the supervisors forward SIGTERM to their child themselves.
    if util.IS_WINDOWS:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
    else:
        p.terminate()


def _shutdown(procs: list[_Proc]) -> None:
    for proc in reversed(procs):
        if proc.popen.poll() is None:
            util.logger.info("  stop %s", proc.label)
            _terminate_tree(proc.popen)
    for proc in reversed(procs):
        try:
            proc.popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.popen.kill()
        if proc.log_file:
            proc.log_file.close()


def run(ready_timeout: float = 300.0) -> int:
    util.banner("Launch ai-voice-stack")

    from . import models
    gaps = models.missing()
    if gaps:
        util.fail(
            "missing model files",
            "Put each model at the path config.py expects, then re-run:\n"
            + "\n".join(f"  - {label:18s} {path}" for label, path in gaps),
            rerun="task run",
        )

    checks = _ready_checks()
    procs: list[_Proc] = []
    try:
        util.logger.info("  mem  %s", _mem())
        _log_env()
        for label, module in _SERVERS:
            util.logger.info("  start %-3s -> logs/%s_server.log", label, label)
            proc = _spawn_server(label, module)
            procs.append(proc)
            if not _wait_one(proc, checks[label], ready_timeout):
                util.fail(
                    f"{label} server did not come up",
                    f"See logs/{label}_server.log (tail above).\nmem at failure: {_mem()}",
                    rerun="task run",
                )
            util.logger.info("  ok   %-3s ready | mem %s", label, _mem())

        util.logger.info("  ok   all backends ready - starting assistant (Ctrl-C to stop)\n")
        pipeline = subprocess.Popen([util.venv_python(), "pipeline.py"], cwd=str(util.REPO_ROOT))
        procs.append(_Proc("pipeline", pipeline))
        return pipeline.wait()
    except KeyboardInterrupt:
        util.logger.info("\n  interrupted - shutting down")
        return 0
    finally:
        _shutdown(procs)
