"""Clone the SimulStreaming (Whisper) backend into simulstreaming_lib/ and install
its requirements. Run after torch so its unpinned torch req isn't reinstalled."""

from __future__ import annotations

import shutil

from . import util
from .detect import Profile

SIMUL_REPO = "https://github.com/dh3b/SimulStreaming-lite"
SIMUL_REF = "c281b2afba55befbc77d5f7f474a80796d41478a"  # pinned; edit to bump.

_ENTRY = "simulstreaming_whisper_server.py"


def _clone() -> None:
    tmp = util.BUILD_DIR / "SimulStreaming-lite"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    util.BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Shallow-fetch the exact pinned commit (works for both tags and SHAs).
    util.run(["git", "init", "-q", str(tmp)])
    util.run(["git", "-C", str(tmp), "remote", "add", "origin", SIMUL_REPO])
    util.run(["git", "-C", str(tmp), "fetch", "--depth", "1", "origin", SIMUL_REF])
    util.run(["git", "-C", str(tmp), "checkout", "-q", "FETCH_HEAD"])

    # Copy the tree into simulstreaming_lib/ (keeping the tracked .gitkeep there).
    util.SIMUL_DIR.mkdir(parents=True, exist_ok=True)
    for item in tmp.iterdir():
        if item.name == ".git":
            continue
        dest = util.SIMUL_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def _predownload_silero_vad() -> None:
    """Pre-download the silero VAD model into torch hub cache so the STT server
    doesn't hit an interactive trust prompt at startup."""
    import torch
    import os

    cache = os.path.expanduser("~/.cache/torch/hub/snakers4_silero-vad_master")
    if os.path.isdir(cache):
        util.logger.info("  ok   silero-vad model cached")
        return
    util.logger.info("  ..   downloading silero-vad VAD model")
    torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    util.logger.info("  ok   silero-vad model cached")


def install(profile: Profile, force: bool = False) -> None:
    util.banner("SimulStreaming (STT backend)")
    entry = util.SIMUL_DIR / _ENTRY

    if entry.exists() and not force:
        util.logger.info(
            "  ok   simulstreaming_lib present (skip clone; --force to refresh)"
        )
    else:
        util.logger.info("  get  %s @ %s", SIMUL_REPO, SIMUL_REF[:12])
        _clone()
        if not entry.exists():
            util.fail(
                "SimulStreaming clone did not produce the expected entry script",
                f"expected {entry}\nrepo {SIMUL_REPO} ref {SIMUL_REF}",
            )

    reqs = util.SIMUL_DIR / "requirements_whisper.txt"
    if not reqs.exists():
        util.fail(
            "requirements_whisper.txt missing in simulstreaming_lib",
            f"looked for {reqs}",
        )
    util.logger.info("  pip  installing SimulStreaming requirements (after torch)")
    reqs_cmd = [util.uv(), "pip", "install", "-r", str(reqs)]
    if profile.numpy_pin:
        reqs_cmd.append(profile.numpy_pin)  # keep numpy<2 through this resolve too
    util.run(reqs_cmd)
    _predownload_silero_vad()
    util.logger.info("  ok   SimulStreaming ready")
