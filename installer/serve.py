"""`task run` supervisor: start the three backend servers (output to logs/<name>.log),
wait until ready, then run pipeline.py on the console. Tears everything down on exit."""

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

# (label, module) for the three backend supervisors, started in this order.
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
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def _spawn_server(label: str, module: str) -> _Proc:
    util.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = util.LOGS_DIR / f"{label}_server.log"
    log_file = open(log_path, "w")
    env = {**os.environ, "TORCH_HUB_TRUST_REPO": "1"}
    popen = subprocess.Popen(
        [util.venv_python(), "-m", module],
        cwd=str(util.REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return _Proc(label, popen, log_path, log_file)


def _wait_ready(procs: list[_Proc], timeout: float) -> bool:
    import config as cfg  # repo-root module; importable once deps/venv are set up

    llm, stt, tts = cfg.LLMServerConfig(), cfg.STTServerConfig(), cfg.TTSServerConfig()
    checks = {
        "llm": lambda: _http_ok(f"http://{llm.server_host}:{llm.server_port}/health"),
        "stt": lambda: _port_open(stt.server_host, stt.server_port),
        "tts": lambda: _port_open(tts.server_host, tts.server_port),
    }
    by_label = {p.label: p for p in procs}
    ready: set[str] = set()
    deadline = time.time() + timeout
    while time.time() < deadline:
        for label, check in checks.items():
            if label in ready:
                continue
            proc = by_label.get(label)
            if proc is not None and proc.popen.poll() is not None:
                util.logger.error(
                    "  !!   %s server exited early (code %s). Last log lines:",
                    label,
                    proc.popen.returncode,
                )
                util.logger.error("%s", _tail(proc.log_path))
                return False
            if check():
                util.logger.info("  ok   %s ready", label)
                ready.add(label)
        if len(ready) == len(checks):
            return True
        time.sleep(1.0)
    util.logger.error(
        "  !!   timed out waiting for: %s", ", ".join(sorted(set(checks) - ready))
    )
    return False


def _terminate_tree(p: subprocess.Popen) -> None:
    # On Windows .terminate() kills only the supervisor, orphaning its llama-server /
    # piper child (which keeps holding its port). taskkill /T takes the whole tree.
    # On POSIX the supervisors forward SIGTERM to their child themselves.
    if util.IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True
        )
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
            "Fetch the examples with `task models`, or put each model at the path "
            "config.py expects, then re-run:\n"
            + "\n".join(f"  - {m.label:18s} {m.path}" for m in gaps),
            rerun="task run",
        )

    procs: list[_Proc] = []
    try:
        for label, module in _SERVERS:
            util.logger.info("  start %-3s -> logs/%s_server.log", label, label)
            procs.append(_spawn_server(label, module))

        if not _wait_ready(procs, ready_timeout):
            util.fail(
                "backend servers did not become ready",
                "See the logs/ directory for details, or run `task doctor` to confirm the "
                "models and llama-server binary are in place.",
                rerun="task run",
            )

        util.logger.info(
            "  ok   all backends ready - starting assistant (Ctrl-C to stop)\n"
        )
        pipeline = subprocess.Popen(
            [util.venv_python(), "pipeline.py"], cwd=str(util.REPO_ROOT)
        )
        procs.append(_Proc("pipeline", pipeline))
        return pipeline.wait()
    except KeyboardInterrupt:
        util.logger.info("\n  interrupted - shutting down")
        return 0
    finally:
        _shutdown(procs)
