"""Install torch + torchaudio with the index the detected machine needs.

torch is kept out of pyproject because the right wheel depends on the runtime
hardware (CPU index vs cuXXX vs the Jetson index). Runs before the SimulStreaming
requirements so they don't pull a wrong-index torch.
"""

from __future__ import annotations

from . import util
from .detect import Profile

_JETSON_TORCH = "2.8.0"
_JETSON_TORCHAUDIO = "2.8.0"


def _import_torch(profile: Profile) -> tuple[bool, str]:
    """(ok, error text). ok = torch+torchaudio import and satisfy the accelerator."""
    if profile.accel == "cuda":
        code = "import torch, torchaudio; assert torch.version.cuda is not None"
    else:
        code = "import torch, torchaudio"
    res = util.run(
        [util.venv_python(), "-c", code], check=False, capture=True, quiet=True
    )
    return res.returncode == 0, (res.stderr or res.stdout or "").strip()


def _probe(profile: Profile) -> bool:
    return _import_torch(profile)[0]


def install(profile: Profile, force: bool = False) -> None:
    util.banner(f"PyTorch  (accel={profile.accel})")
    if not force and _probe(profile):
        util.logger.info(
            "  ok   torch + torchaudio already present for accel=%s", profile.accel
        )
        return

    uv = util.uv()

    # numpy<2 first to avoid torch version mismatch
    if profile.numpy_pin:
        util.logger.info("  pin  %s (CUDA/Jetson torch stack)", profile.numpy_pin)
        util.run([uv, "pip", "install", profile.numpy_pin])

    cmd = [uv, "pip", "install"]
    if profile.torch_index_url:
        cmd += ["--index-url", profile.torch_index_url]
        util.logger.info("  idx  %s", profile.torch_index_url)
    if profile.is_jetson:
        cmd += [f"torch=={_JETSON_TORCH}", f"torchaudio=={_JETSON_TORCHAUDIO}"]
    else:
        cmd += ["torch", "torchaudio"]
    util.run(cmd)

    ok, err = _import_torch(profile)
    if not ok:
        tail = err.splitlines()[-1] if err else ""
        hint = ""
        if profile.is_jetson:
            hint = (
                f"\nJetson: pinned torch=={_JETSON_TORCH}. If your JetPack/CUDA differs, adjust "
                "_JETSON_TORCH in installer/setup_torch.py and JETSON_TORCH_INDEX in installer/detect.py."
            )
        util.fail(
            "torch did not import correctly after install",
            f"{tail}\nindex: {profile.torch_index_url or 'default PyPI'}{hint}",
        )
    util.logger.info("  ok   torch + torchaudio installed")
