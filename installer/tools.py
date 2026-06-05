"""Build-toolchain provisioning.

Per tool, tries the cleanest mechanism first: pip into the venv (cmake, ninja, and
experimental CUDA wheels), then the system package manager (C++ compiler, system
CUDA), then stops with an exact instruction.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import detect, util
from .detect import Profile


@dataclass
class BuildTools:
    cmake: str
    ninja: str | None = None
    nvcc: str | None = None
    extra_cmake: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


_WIN_MSVC_MSG = (
    "llama.cpp needs the MSVC C++ build tools. Install them, then re-run:\n\n"
    "  winget install --id Microsoft.VisualStudio.2022.BuildTools -e \\\n"
    "    --override \"--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended\"\n\n"
    "or download \"Build Tools for Visual Studio 2022\" and check the\n"
    "\"Desktop development with C++\" workload:\n"
    "  https://visualstudio.microsoft.com/visual-cpp-build-tools/"
)
_LINUX_CC_MSG = (
    "No C++ compiler found. Install the build toolchain, then re-run:\n\n"
    "  Debian/Ubuntu : sudo apt-get install -y build-essential\n"
    "  Fedora/RHEL   : sudo dnf groupinstall -y 'Development Tools'\n"
    "  Arch          : sudo pacman -S --needed base-devel"
)
_MAC_CC_MSG = (
    "No C++ compiler found. Install the Apple command-line tools, then re-run:\n\n"
    "  xcode-select --install\n\n"
    "(The installer opens a GUI dialog; finish it, then re-run setup.)"
)


def _venv_tool(name: str) -> str | None:
    p = util.venv_bin(name)
    return str(p) if p else None


def _pip_install(*pkgs: str) -> None:
    util.run([util.uv(), "pip", "install", *pkgs])


# cmake + ninja (pip tier)
def ensure_cmake_ninja(profile: Profile) -> tuple[str, str | None]:
    cmake = util.which("cmake") or _venv_tool("cmake")
    ninja = util.which("ninja") or _venv_tool("ninja")

    need: list[str] = []
    if not cmake:
        need.append("cmake")
    # Ninja is the generator on every OS (on Windows with the MSVC env from vcvars).
    if not ninja:
        need.append("ninja")
    if need:
        util.logger.info("  pip  installing build tools: %s", " ".join(need))
        _pip_install(*need)
        cmake = cmake or _venv_tool("cmake")
        ninja = ninja or _venv_tool("ninja")

    if not cmake:
        util.fail("cmake unavailable after install attempt", "Try manually: uv pip install cmake")
    util.logger.info("  ok   cmake: %s", cmake)
    if ninja:
        util.logger.info("  ok   ninja: %s", ninja)
    return cmake, ninja


def windows_msvc_env() -> dict[str, str]:
    """Environment vcvars64.bat sets (cl/INCLUDE/LIB on PATH), for Ninja + MSVC builds."""
    install = detect._vswhere("installationPath")
    if not install:
        util.fail("no C++ compiler (MSVC) found", _WIN_MSVC_MSG)
    vcvars = Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if not vcvars.exists():
        util.fail("MSVC vcvars64.bat not found", f"expected at {vcvars}\n\n{_WIN_MSVC_MSG}")

    # Capture vcvars' env via a temp .bat (inline `cmd /c call "spaced path"` mangles quoting).
    util.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    probe = util.BUILD_DIR / "_vcvars_probe.bat"
    probe.write_text(f'@echo off\r\ncall "{vcvars}"\r\nset\r\n')
    try:
        res = util.run(["cmd", "/c", str(probe)], capture=True, check=False, quiet=True)
    finally:
        probe.unlink(missing_ok=True)
    if res.returncode != 0:
        util.fail("could not initialize the MSVC environment", res.stderr or res.stdout or "")

    env: dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "=" in line and not line.startswith("("):
            key, val = line.split("=", 1)
            env[key] = val
    if not env.get("INCLUDE") or not env.get("LIB"):
        util.fail(
            "MSVC environment incomplete (no INCLUDE/LIB)",
            "vcvars64.bat ran but did not set up the C++ toolchain.\n" + _WIN_MSVC_MSG,
        )
    return env


# C++ compiler (system tier, else instruct)
def ensure_compiler(profile: Profile) -> None:
    if detect._has_compiler(profile.os):
        util.logger.info("  ok   C++ compiler present")
        return

    if profile.os == "windows":
        if util.which("winget"):
            util.logger.info("  ..   attempting winget install of VS 2022 Build Tools (C++)")
            util.run(
                [
                    "winget", "install", "--id", "Microsoft.VisualStudio.2022.BuildTools",
                    "-e", "--accept-package-agreements", "--accept-source-agreements",
                    "--override",
                    "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended",
                ],
                check=False,
            )
            if detect._has_compiler("windows"):
                util.logger.info("  ok   MSVC build tools installed")
                return
        util.fail("no C++ compiler (MSVC) found", _WIN_MSVC_MSG)

    if profile.os == "linux":
        if apt_install(["build-essential"]) and detect._has_compiler("linux"):
            util.logger.info("  ok   build-essential installed")
            return
        util.fail("no C++ compiler found", _LINUX_CC_MSG)

    # darwin: xcode-select --install is interactive, so we instruct rather than block.
    util.fail("no C++ compiler found", _MAC_CC_MSG)


# CUDA build components
def ensure_cuda_build(profile: Profile) -> BuildTools:
    """cmake/nvcc/env for a CUDA build: JetPack nvcc on Jetson, else system nvcc,
    else experimental pip CUDA wheels, else instruct."""
    cmake, ninja = ensure_cmake_ninja(profile)
    bt = BuildTools(cmake=cmake, ninja=ninja)

    if profile.is_jetson:
        nvcc = _find_jetson_nvcc()
        if not nvcc:
            util.fail(
                "Jetson CUDA toolkit (nvcc) not found",
                "Expected JetPack's nvcc at /usr/local/cuda/bin/nvcc.\n"
                "Install the CUDA toolkit from JetPack (sudo apt-get install -y cuda-toolkit-12-6)\n"
                "or set CUDACXX to your nvcc path, then re-run.",
            )
        bt.nvcc = nvcc
        bt.env["CUDACXX"] = nvcc
        util.logger.info("  ok   Jetson nvcc: %s", nvcc)
        return bt

    # x86_64 CUDA: prefer a real system toolkit.
    nvcc = util.which("nvcc")
    if nvcc:
        bt.nvcc = nvcc
        bt.env["CUDACXX"] = nvcc
        util.logger.info("  ok   system nvcc: %s", nvcc)
        return bt

    # Experimental: build against the pip CUDA wheels (no system toolkit).
    util.logger.info("  ..   no system nvcc; trying pip CUDA wheels (experimental)")
    wheel_bt = _try_pip_cuda(profile, bt)
    if wheel_bt is not None:
        return wheel_bt

    util.fail(
        "CUDA toolkit (nvcc) not found and the pip-wheel route failed",
        "Install the CUDA Toolkit matching your driver, then re-run:\n"
        "  Linux  : https://developer.nvidia.com/cuda-downloads (or distro 'cuda-toolkit' pkg)\n"
        "  Windows: https://developer.nvidia.com/cuda-downloads (choose the network installer)\n"
        f"Detected driver CUDA: {profile.cuda_version or 'unknown'}; install a matching toolkit.",
    )


def _find_jetson_nvcc() -> str | None:
    cand = util.which("nvcc")
    if cand:
        return cand
    for base in sorted(Path("/usr/local").glob("cuda*"), reverse=True):
        nvcc = base / "bin" / "nvcc"
        if nvcc.exists():
            return str(nvcc)
    return None


def _try_pip_cuda(profile: Profile, bt: BuildTools) -> BuildTools | None:
    """Install CUDA wheels and point cmake at the wheel nvcc. Experimental; returns
    None on any shortfall so the caller can instruct."""
    pkgs = [
        "nvidia-cuda-nvcc-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cuda-cccl-cu12",
        "nvidia-cublas-cu12",
    ]
    try:
        _pip_install(*pkgs)
    except Exception:
        return None

    nvidia_root = _venv_nvidia_root()
    if not nvidia_root:
        return None
    nvcc = nvidia_root / "cuda_nvcc" / "bin" / ("nvcc.exe" if profile.os == "windows" else "nvcc")
    if not nvcc.exists():
        return None

    include_dirs = [str(d) for d in nvidia_root.glob("*/include") if d.exists()]
    lib_glob = "*/lib/x64" if profile.os == "windows" else "*/lib"
    lib_dirs = [str(d) for d in nvidia_root.glob(lib_glob) if d.exists()]

    bt.nvcc = str(nvcc)
    bt.env["CUDACXX"] = str(nvcc)
    bt.extra_cmake += [
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCUDAToolkit_ROOT={nvidia_root / 'cuda_nvcc'}",
    ]
    if include_dirs:
        bt.extra_cmake.append("-DCMAKE_CUDA_FLAGS=" + " ".join(f'-I"{d}"' for d in include_dirs))
    # Make the wheel libs discoverable at link + runtime.
    path_sep = ";" if profile.os == "windows" else ":"
    if lib_dirs:
        if profile.os == "windows":
            bt.env["PATH"] = path_sep.join(lib_dirs + [os.environ.get("PATH", "")])
        else:
            bt.env["LD_LIBRARY_PATH"] = path_sep.join(
                lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")]
            )
    util.logger.info("  ok   pip CUDA wheels wired (experimental): nvcc=%s", nvcc)
    util.logger.warning(
        "  !!   building llama.cpp from pip CUDA wheels is unverified; if cmake's CUDA\n"
        "       detection fails, install a system CUDA toolkit and re-run."
    )
    return bt


def _venv_nvidia_root() -> Path | None:
    res = util.run(
        [util.venv_python(), "-c", "import nvidia, os; print(os.path.dirname(nvidia.__file__))"],
        check=False,
        capture=True,
        quiet=True,
    )
    if res.returncode != 0:
        return None
    root = Path(res.stdout.strip())
    return root if root.exists() else None


def apt_install(pkgs: list[str]) -> bool:
    """Best-effort, non-interactive apt install. Returns True only on success."""
    if not util.which("apt-get"):
        return False
    is_root = getattr(os, "geteuid", lambda: 1)() == 0
    sudo = [] if is_root else ["sudo", "-n"]
    util.run([*sudo, "apt-get", "update"], check=False)
    res = util.run(
        [*sudo, "apt-get", "install", "-y", *pkgs],
        check=False,
        extra_env={"DEBIAN_FRONTEND": "noninteractive"},
    )
    return res.returncode == 0
