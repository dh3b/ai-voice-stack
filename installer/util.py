"""Shared installer helpers: paths, logging, subprocess, downloads, fail-loud.

Everything here is stdlib-only so the installer can run before any dependency is
installed (it bootstraps the venv itself).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NoReturn, Sequence

# Repo layout
REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALLER_DIR = REPO_ROOT / "installer"
MODELS_DIR = REPO_ROOT / "models"
ASSETS_DIR = REPO_ROOT / "assets"
LLAMA_BIN_DIR = REPO_ROOT / "llama_cpp_bin"
SIMUL_DIR = REPO_ROOT / "simulstreaming_lib"
BUILD_DIR = REPO_ROOT / ".build"
LOGS_DIR = REPO_ROOT / "logs"  # per-server logs for `task run`
VENV_DIR = REPO_ROOT / ".venv"

logger = logging.getLogger("installer")

IS_WINDOWS = os.name == "nt"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )


def fail(
    title: str, body: str = "", *, rerun: str = "task setup", code: int = 1
) -> NoReturn:
    bar = "=" * 70
    lines = [f"\n{bar}", f"  SETUP STOPPED: {title}", "-" * 70]
    if body:
        lines.append(body.rstrip("\n"))
        lines.append("")
    lines.append(f"  Fix the above, then re-run:   {rerun}")
    lines.append(bar)
    print("\n".join(lines), file=sys.stderr, flush=True)
    raise SystemExit(code)


def banner(title: str) -> None:
    logger.info("\n=== %s ===", title)


def run(
    cmd: Sequence[str | os.PathLike],
    *,
    cwd: str | os.PathLike | None = None,
    extra_env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    """Run a command. Streams to the console unless ``capture`` is set.

    Raises CalledProcessError when ``check`` and the command fails.
    """
    argv = [str(c) for c in cmd]
    if not quiet:
        logger.info("$ %s", " ".join(argv))
    env = None
    if extra_env:
        env = {**os.environ, **extra_env}
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            capture_output=capture,
            check=check,
        )
    except FileNotFoundError as exc:
        if check:
            fail(f"command not found: {argv[0]}", str(exc))
        raise


def try_output(
    cmd: Sequence[str | os.PathLike], *, timeout: float = 15.0
) -> str | None:
    """Run a probe command, returning stripped stdout or None on any failure."""
    try:
        res = subprocess.run(
            [str(c) for c in cmd],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def which(name: str) -> str | None:
    return shutil.which(name)


def shallow_clone(dst: Path, repo: str, ref: str) -> None:
    """Shallow-fetch an exact pinned ref (tag or SHA) into *dst*."""
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "-q", str(dst)])
    run(["git", "-C", str(dst), "remote", "add", "origin", repo])
    run(["git", "-C", str(dst), "fetch", "--depth", "1", "origin", ref])
    run(["git", "-C", str(dst), "checkout", "-q", "FETCH_HEAD"])


def venv_bin(name: str) -> Path | None:
    sub = "Scripts" if IS_WINDOWS else "bin"
    for cand in (name, f"{name}.exe") if IS_WINDOWS else (name,):
        p = VENV_DIR / sub / cand
        if p.exists():
            return p
    return None


def venv_python() -> str:
    """Path to the venv's python, or the current interpreter as a fallback."""
    p = venv_bin("python")
    return str(p) if p else sys.executable


def uv() -> str:
    exe = which("uv")
    if not exe:
        fail(
            "uv is not installed",
            "uv manages the Python venv and dependencies. Install it:\n"
            '  Windows : powershell -c "irm https://astral.sh/uv/install.ps1 | iex"\n'
            "  Linux: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
            "  (or)    : pip install uv",
        )
    return exe


# downloads, skip if present
def sha256_file(path: str | os.PathLike, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def download(
    url: str,
    dest: str | os.PathLike,
    *,
    sha256: str | None = None,
    label: str | None = None,
) -> Path:
    """
    Download ``url`` to ``dest``, resuming partial downloads and verifying sha256.
    Skips the download entirely when ``dest`` already exists and (if given) its checksum matches.
    """
    dest = Path(dest)
    name = label or dest.name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if sha256 is None:
            logger.info("  ok   %s (present)", name)
            return dest
        logger.info("  ..   %s (verifying checksum)", name)
        if sha256_file(dest).lower() == sha256.lower():
            logger.info("  ok   %s (checksum verified)", name)
            return dest
        # Present but not the default file: assume the user supplied their own model.
        # Never destroy it; `--force` (which deletes first) is how you re-fetch the default.
        logger.warning(
            "  ok   %s present but not the default checksum - keeping it "
            "(assuming a user-provided model; use --force to replace)",
            name,
        )
        return dest

    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(
        url, headers={"User-Agent": "ai-voice-stack-installer"}
    )
    if have:
        req.add_header("Range", f"bytes={have}-")

    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:
            resp = None
        else:
            fail(f"download failed: {name}", f"{url}\nHTTP {exc.code}: {exc.reason}")
    except urllib.error.URLError as exc:
        fail(f"download failed: {name}", f"{url}\n{exc.reason}")

    if resp is not None:
        mode = "ab" if (have and resp.status == 206) else "wb"
        if mode == "wb":
            have = 0  # server ignored Range; restart
        total = have + int(resp.headers.get("Content-Length", 0) or 0)
        logger.info(
            "  get  %s  (%s)", name, _fmt_bytes(total) if total else "size unknown"
        )
        done = have
        next_tick = done + (32 << 20)
        with open(part, mode) as fh:
            while True:
                buf = resp.read(1 << 20)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if done >= next_tick:
                    pct = f" {done * 100 // total}%" if total else ""
                    logger.info("       %s%s", _fmt_bytes(done), pct)
                    next_tick = done + (32 << 20)
        resp.close()

    if sha256 is not None:
        actual = sha256_file(part)
        if actual.lower() != sha256.lower():
            part.unlink(missing_ok=True)
            fail(
                f"checksum mismatch: {name}",
                f"{url}\nexpected sha256 {sha256}\nactual   sha256 {actual}\n"
                "The source may have changed. Update installer/models.toml or retry.",
            )

    part.replace(dest)
    logger.info("  ok   %s", name)
    return dest
