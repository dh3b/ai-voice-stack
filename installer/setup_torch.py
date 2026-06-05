"""Install torch + torchaudio with the index the detected machine needs.

torch is deliberately kept out of pyproject because the right wheel depends on the
runtime hardware (CPU index vs cuXXX vs the Jetson/JetPack index). Runs before the
SimulStreaming requirements so they don't drag in a wrong-index torch.
"""
from __future__ import annotations

from . import util
from .detect import Profile


def _probe(profile: Profile) -> bool:
    """True if torch+torchaudio already satisfy this profile (so we can skip)."""
    if profile.accel == "cuda":
        code = "import torch, torchaudio; assert torch.version.cuda is not None"
    elif profile.accel == "metal":
        code = "import torch, torchaudio; assert torch.backends.mps.is_built()"
    else:
        code = "import torch, torchaudio"
    res = util.run([util.venv_python(), "-c", code], check=False, capture=True, quiet=True)
    return res.returncode == 0


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
    cmd += ["torch", "torchaudio"]
    if profile.torch_index_url:
        util.logger.info("  idx  %s", profile.torch_index_url)
    util.run(cmd)

    if not _probe(profile):
        hint = ""
        if profile.is_jetson:
            hint = (
                "\nJetson: confirm the JetPack version matches the torch index "
                "(JetPack 6.x -> cu126). Override the index in installer/detect.py "
                "(JETSON_TORCH_INDEX) if your board differs."
            )
        util.fail(
            "torch did not import correctly after install",
            f"index: {profile.torch_index_url or 'default PyPI'}{hint}",
        )
    util.logger.info("  ok   torch + torchaudio installed")
