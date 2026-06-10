"""Platform detection → a structured Profile that drives everything downstream.

The Profile decides the torch wheel index (uv) and the llama.cpp cmake flags, so
detection runs first and every other module just consumes its fields rather than
re-branching on os/arch/accelerator.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import util

logger = logging.getLogger("installer.detect")

# Jetson (Tegra/L4T) torch index for JetPack 6.x / CUDA 12.6.
JETSON_TORCH_INDEX = "https://pypi.jetson-ai-lab.io/jp6/cu126"

# Detected CUDA version -> highest pytorch wheel tag <= it.
_TORCH_CUDA_TAGS: list[tuple[tuple[int, int], str]] = [
    ((12, 6), "cu126"),
    ((12, 4), "cu124"),
    ((12, 1), "cu121"),
    ((11, 8), "cu118"),
]


@dataclass
class Profile:
    os: str  # windows | linux
    arch: str  # x86_64 | arm64
    accel: str  # cpu | cuda | metal
    is_jetson: bool = False
    cuda_version: str | None = None  # "12.6"
    gpu_compute_cap: str | None = None  # "89"
    gpu_name: str | None = None
    torch_index_url: str | None = None  # None => default PyPI wheels
    llama_cuda_arch: str | None = None  # "89" | "87"
    generator: str | None = None  # cmake -G value, or None for the default
    has: dict = field(default_factory=dict)  # cmake, ninja, compiler, nvcc, git

    @property
    def exe_suffix(self) -> str:
        return ".exe" if self.os == "windows" else ""

    @property
    def numpy_pin(self) -> str | None:
        """numpy<2 for the CUDA/Jetson torch stack."""
        return "numpy<2" if self.accel == "cuda" else None

    def llama_cmake_flags(self) -> list[str]:
        """cmake flags for this profile."""
        flags = ["-DCMAKE_BUILD_TYPE=Release"]
        if self.accel == "cuda":
            # FA_ALL_QUANTS omitted: very heavy to compile.
            flags += [
                "-DGGML_CUDA=ON",
                "-DGGML_CUDA_F16=ON",
                f"-DCMAKE_CUDA_ARCHITECTURES={self.llama_cuda_arch or '89'}",
            ]
        else:
            flags += ["-DGGML_NATIVE=OFF", "-DBUILD_SHARED_LIBS=OFF"]

        # Disable CPU features the host doesn't support (e.g. AVX masked by a
        # hypervisor).  Without this, the ggml cmake defaults to enabling all
        # x86-64 instruction sets, which causes SIGILL at runtime.
        flags += _llama_cpu_disable_flags()

        # libcurl off on Windows to avoid the extra dependency.
        flags.append("-DLLAMA_CURL=OFF" if self.os == "windows" else "-DLLAMA_CURL=ON")
        return flags

    def summary(self) -> str:
        bits = [
            f"os={self.os}",
            f"arch={self.arch}",
            f"accel={self.accel}",
        ]
        if self.is_jetson:
            bits.append("jetson=yes")
        if self.accel == "cuda":
            bits.append(f"cuda={self.cuda_version or '?'}")
            bits.append(f"cc={self.gpu_compute_cap or '?'}")
            bits.append(f"llama_arch={self.llama_cuda_arch}")
        return "  ".join(bits)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---------------------------------------------------------------------------
# Raw probes
# ---------------------------------------------------------------------------
def _detect_os() -> str:
    return "windows" if platform.system().lower() == "windows" else "linux"


def _normalize_arch(machine: str) -> str:
    m = machine.lower()
    if m in ("amd64", "x86_64", "x64"):
        return "x86_64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    return m or "unknown"


def _detect_jetson() -> bool:
    if Path("/etc/nv_tegra_release").exists():
        return True
    model = Path("/proc/device-tree/model")
    try:
        if model.exists():
            text = model.read_text(errors="ignore").lower()
            return "jetson" in text or "tegra" in text
    except OSError:
        pass
    return False


def _nvidia_compute_cap() -> tuple[str | None, str | None]:
    """Returns (compute_cap like '89', gpu_name) from nvidia-smi, or (None, None)."""
    out = util.try_output(
        ["nvidia-smi", "--query-gpu=compute_cap,name", "--format=csv,noheader"]
    )
    if not out:
        return None, None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    cap = (
        parts[0].replace(".", "") if parts and re.match(r"\d+\.\d+", parts[0]) else None
    )
    name = parts[1] if len(parts) > 1 else None
    return cap, name


def _cuda_version() -> str | None:
    # Prefer nvcc (the toolkit version we'd actually compile against).
    out = util.try_output(["nvcc", "--version"])
    if out:
        m = re.search(r"release (\d+\.\d+)", out)
        if m:
            return m.group(1)
    # Fall back to the driver's max supported CUDA from nvidia-smi.
    out = util.try_output(["nvidia-smi"])
    if out:
        m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
        if m:
            return m.group(1)
    return None


def _vswhere(prop: str) -> str | None:
    """Query vswhere for a property of the latest VS install that has the C++ toolset."""
    pf = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles", "")
    vswhere = Path(pf) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None
    return (
        util.try_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                prop,
            ]
        )
        or None
    )


def _vswhere_has_vctools() -> bool:
    """True if Visual Studio with the C++ toolset is installed (Windows)."""
    return bool(_vswhere("installationPath"))


def _has_compiler(osname: str) -> bool:
    if osname == "windows":
        return bool(shutil.which("cl")) or _vswhere_has_vctools()
    return bool(shutil.which("gcc") or shutil.which("g++") or shutil.which("clang"))


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------
def _torch_index(p: Profile) -> str | None:
    if p.accel == "cuda":
        if p.is_jetson:
            return JETSON_TORCH_INDEX
        return (
            f"https://download.pytorch.org/whl/{_cuda_tag(p.cuda_version) or 'cu124'}"
        )
    if p.arch == "arm64":
        return None  # Raspberry Pi etc. → default PyPI CPU wheels
    return "https://download.pytorch.org/whl/cpu"  # linux/windows x86 CPU


def _cuda_tag(version: str | None) -> str | None:
    if not version:
        return None
    m = re.match(r"(\d+)\.(\d+)", version)
    if not m:
        return None
    detected = (int(m.group(1)), int(m.group(2)))
    for ver, tag in _TORCH_CUDA_TAGS:
        if detected >= ver:
            return tag
    return None


_CPU_FLAGS_CACHE: frozenset | None = None
_VBOX_CACHE: bool | None = None


def _is_virtualbox() -> bool:
    """Whether the host is a VirtualBox VM (Windows-only; Linux has DMI)."""
    global _VBOX_CACHE
    if _VBOX_CACHE is not None:
        return _VBOX_CACHE
    try:
        bios = (
            util.try_output(
                ["wmic", "bios", "get", "SerialNumber,Manufacturer", "/format:value"],
                timeout=10.0,
            )
            or ""
        ).lower()
        _VBOX_CACHE = "virtualbox" in bios
    except Exception:
        _VBOX_CACHE = False
    return _VBOX_CACHE


def _cpu_flags() -> frozenset:
    """Return the set of CPU flags advertised by /proc/cpuinfo (Linux) or via inspection on Windows."""
    global _CPU_FLAGS_CACHE
    if _CPU_FLAGS_CACHE is not None:
        return _CPU_FLAGS_CACHE
    flags: set[str] = set()

    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.startswith("flags"):
                flags.update(line.split(":", 1)[1].strip().lower().split())
    except OSError:
        pass

    if not flags and platform.system().lower() == "windows":
        if _is_virtualbox():
            # VirtualBox may mask some CPU features. Be conservative.
            flags = {"sse", "sse2"}
        else:
            # Assume all modern x64 features (no cpuid from Python on Windows).
            flags = {"sse", "sse2", "sse4_2", "avx", "avx2", "fma", "f16c", "bmi2"}

    _CPU_FLAGS_CACHE = frozenset(flags)
    return _CPU_FLAGS_CACHE


def _llama_cpu_disable_flags() -> list[str]:
    """Return cmake -D flags to disable CPU features the host lacks."""
    have = _cpu_flags()
    if not have:
        return []

    # mapping: cmake GGML_FOO variable  ->  /proc/cpuinfo flag
    checks: list[tuple[str, str]] = [
        ("GGML_SSE42", "sse4_2"),
        ("GGML_AVX", "avx"),
        ("GGML_AVX2", "avx2"),
        ("GGML_FMA", "fma"),
        ("GGML_F16C", "f16c"),
        ("GGML_BMI2", "bmi2"),
        ("GGML_AVX512", "avx512f"),
        ("GGML_AVX512_VBMI", "avx512_vbmi"),
        ("GGML_AVX512_VNNI", "avx512_vnni"),
        ("GGML_AVX512_BF16", "avx512_bf16"),
    ]
    flags = [f"-D{var}=OFF" for var, flag in checks if flag not in have]

    # On VirtualBox disable OpenMP — the MSVC runtime dispatch can also SIGILL.
    if _is_virtualbox():
        flags.append("-DGGML_OPENMP=OFF")

    return flags


def _derive(p: Profile) -> None:
    if p.accel == "cuda" and not p.llama_cuda_arch:
        p.llama_cuda_arch = "87" if p.is_jetson else (p.gpu_compute_cap or "89")
    p.torch_index_url = _torch_index(p)
    if p.generator is None:
        # We build with Ninja on every OS (the installer pip-installs it). On Windows
        # it runs under the MSVC env from vcvars64.bat, which is more reliable than
        # cmake's VS generator with a Build Tools-only install.
        if p.has.get("ninja") or p.os == "windows":
            p.generator = "Ninja"
        else:
            p.generator = None


def detect() -> Profile:
    osname = _detect_os()
    is_jetson = _detect_jetson()
    cc, gpu_name = _nvidia_compute_cap()
    has_nvidia = cc is not None or bool(shutil.which("nvidia-smi"))

    if is_jetson or has_nvidia:
        accel = "cuda"
    else:
        accel = "cpu"

    prof = Profile(
        os=osname,
        arch=_normalize_arch(platform.machine()),
        accel=accel,
        is_jetson=is_jetson,
        cuda_version=_cuda_version() if accel == "cuda" else None,
        gpu_compute_cap=cc,
        gpu_name=gpu_name,
        has={
            "git": bool(shutil.which("git")),
            "cmake": bool(shutil.which("cmake")),
            "ninja": bool(shutil.which("ninja")),
            "nvcc": bool(shutil.which("nvcc")),
            "compiler": _has_compiler(osname),
        },
    )

    _derive(prof)
    return prof
