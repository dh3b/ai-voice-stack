"""Ensure system audio/codec libs (portaudio, libsndfile, ffmpeg) on Linux/macOS.

Best-effort install, else instruct. Windows/macOS Python wheels bundle these.
"""
from __future__ import annotations

import ctypes.util

from . import tools, util
from .detect import Profile

_LINUX_MSG = (
    "Install the audio/codec libraries and re-run:\n"
    "  Debian/Ubuntu : sudo apt-get install -y {apt}\n"
    "  Fedora/RHEL   : sudo dnf install -y portaudio libsndfile ffmpeg\n"
    "  Arch          : sudo pacman -S --needed portaudio libsndfile ffmpeg"
)
_MAC_MSG = (
    "Install Homebrew (https://brew.sh), then:\n  brew install {brew}"
)


def _missing() -> list[str]:
    """Return human pkg hints for whatever audio/codec libs are absent."""
    missing = []
    if ctypes.util.find_library("portaudio") is None:
        missing.append("portaudio")
    if ctypes.util.find_library("sndfile") is None:
        missing.append("sndfile")
    if not util.which("ffmpeg"):
        missing.append("ffmpeg")
    return missing


def ensure(profile: Profile) -> None:
    util.banner("System audio/codec libraries")
    if profile.os == "windows":
        util.logger.info("  ok   bundled in Python wheels on Windows - nothing to do")
        return

    missing = _missing()
    if not missing:
        util.logger.info("  ok   portaudio / sndfile / ffmpeg present")
        return

    if profile.os == "darwin":
        brew_pkgs = ["portaudio" if m == "portaudio" else "libsndfile" if m == "sndfile" else m
                     for m in missing]
        if util.which("brew"):
            util.logger.info("  ..   brew install %s", " ".join(brew_pkgs))
            if util.run(["brew", "install", *brew_pkgs], check=False).returncode == 0:
                util.logger.info("  ok   installed via brew")
                return
        util.fail("missing system audio/codec libraries", _MAC_MSG.format(brew=" ".join(brew_pkgs)))

    # linux
    apt_pkgs = []
    if "portaudio" in missing:
        apt_pkgs.append("libportaudio2")
        if profile.arch == "arm64":
            apt_pkgs.append("portaudio19-dev")  # arm often source-builds sounddevice
    if "sndfile" in missing:
        apt_pkgs.append("libsndfile1")
    if "ffmpeg" in missing:
        apt_pkgs.append("ffmpeg")

    util.logger.info("  ..   attempting: apt-get install %s", " ".join(apt_pkgs))
    if tools.apt_install(apt_pkgs) and not _missing():
        util.logger.info("  ok   installed system libs")
        return
    util.fail("missing system audio/codec libraries", _LINUX_MSG.format(apt=" ".join(apt_pkgs)))
