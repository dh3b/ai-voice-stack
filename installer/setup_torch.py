"""Install torch + torchaudio with the index the detected machine needs.

torch is deliberately kept out of pyproject because the right wheel depends on the
runtime hardware (CPU index vs cuXXX vs the Jetson/JetPack index). Runs before the
SimulStreaming requirements so they don't drag in a wrong-index torch.
"""
from __future__ import annotations

import re

from . import util
from .detect import Profile

_JETSON_CUDA_PKGS = {
    "libcudss": "nvidia-cudss-cu12",
    "libcusparseLt": "nvidia-cusparselt-cu12",
}


def _import_torch(profile: Profile) -> tuple[bool, str]:
    """(ok, error text). ok = torch+torchaudio import and satisfy the accelerator."""
    if profile.accel == "cuda":
        code = "import torch, torchaudio; assert torch.version.cuda is not None"
    elif profile.accel == "metal":
        code = "import torch, torchaudio; assert torch.backends.mps.is_built()"
    else:
        code = "import torch, torchaudio"
    res = util.run([util.venv_python(), "-c", code], check=False, capture=True, quiet=True)
    return res.returncode == 0, (res.stderr or res.stdout or "").strip()


def _probe(profile: Profile) -> bool:
    return _import_torch(profile)[0]


def _missing_lib(err: str) -> str | None:
    m = re.search(r"(lib[A-Za-z0-9_]+)\.so", err)
    return m.group(1) if m else None


def _link_into_torch(libname: str) -> None:
    # Copy the lib next to torch's own libs (torch/lib is on torch's $ORIGIN RPATH),
    # so it's found regardless of the wheel's layout.
    code = (
        "import glob, os, shutil, sysconfig\n"
        "sp = sysconfig.get_paths()['purelib']\n"
        "dst = os.path.join(sp, 'torch', 'lib')\n"
        f"libs = glob.glob(os.path.join(sp, 'nvidia', '**', '{libname}.so*'), recursive=True)\n"
        "if os.path.isdir(dst):\n"
        "    for s in libs: shutil.copy2(s, dst)\n"
    )
    util.run([util.venv_python(), "-c", code], check=False, quiet=True)


def _ensure_jetson_cuda_libs(profile: Profile) -> None:
    for _ in range(len(_JETSON_CUDA_PKGS) + 1):
        ok, err = _import_torch(profile)
        if ok:
            return
        lib = _missing_lib(err)
        pkg = _JETSON_CUDA_PKGS.get(lib) if lib else None
        if not pkg:
            return  # not an auto-fixable missing lib; install()'s check reports it
        util.logger.info("  dep  %s.so missing -> installing %s", lib, pkg)
        util.run([util.uv(), "pip", "install", pkg])
        _link_into_torch(lib)


def install(profile: Profile, force: bool = False) -> None:
    util.banner(f"PyTorch  (accel={profile.accel})")
    if not force and _probe(profile):
        util.logger.info("  ok   torch + torchaudio already present for accel=%s", profile.accel)
        return

    uv = util.uv()

    # numpy<2 before torch on CUDA/Jetson (NVIDIA wheels use the numpy 1.x ABI).
    if profile.numpy_pin:
        util.logger.info("  pin  %s (required for the CUDA/Jetson torch stack)", profile.numpy_pin)
        util.run([uv, "pip", "install", profile.numpy_pin])

    cmd = [uv, "pip", "install"]
    if profile.torch_index_url:
        cmd += ["--index-url", profile.torch_index_url]
        util.logger.info("  idx  %s", profile.torch_index_url)
    cmd += ["torch", "torchaudio"]
    util.run(cmd)

    if profile.is_jetson:
        _ensure_jetson_cuda_libs(profile)

    ok, err = _import_torch(profile)
    if not ok:
        tail = err.splitlines()[-1] if err else ""
        hint = ""
        if profile.is_jetson:
            hint = (
                "\nJetson: if a libXXX.so is missing, install the matching wheel "
                "(e.g. nvidia-<xxx>-cu12). Confirm the JetPack version matches the torch "
                "index (JetPack 6.x -> cu126; set JETSON_TORCH_INDEX in installer/detect.py)."
            )
        util.fail(
            "torch did not import correctly after install",
            f"{tail}\nindex: {profile.torch_index_url or 'default PyPI'}{hint}",
        )
    util.logger.info("  ok   torch + torchaudio installed")
