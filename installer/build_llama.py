"""Build llama.cpp's llama-server from source into llama_cpp_bin/."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import tools, util
from .detect import Profile

LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_REF = "b9528"  # bNNNN tag or commit SHA; edit to bump.


def server_binary_path(profile: Profile) -> Path:
    return util.LLAMA_BIN_DIR / f"llama-server{profile.exe_suffix}"


def _default_jobs(profile: Profile) -> int:
    n = os.cpu_count() or 4
    if profile.is_jetson:
        return max(1, min(4, n))  # Jetson: fewer jobs (power/thermal)
    return max(1, min(8, n))  # cap to keep compile RAM/heat sane


def build(profile: Profile, *, jobs: int | None = None, force: bool = False) -> Path:
    util.banner(f"llama.cpp  (accel={profile.accel}, ref={LLAMA_REF})")
    out = server_binary_path(profile)
    if out.exists() and not force:
        util.logger.info("  ok   %s present (skip; --force to rebuild)", out.name)
        return out

    if not util.which("git"):
        util.fail("git not found", "Install git, then re-run.")

    cmake, ninja = tools.ensure_cmake_ninja(profile)
    tools.ensure_compiler(profile)
    if profile.accel == "cuda":
        bt = tools.ensure_cuda_build(profile)  # adds nvcc + env (+ wheel flags)
    else:
        bt = tools.BuildTools(cmake=cmake, ninja=ninja)

    if profile.os == "windows":
        # Build with Ninja under the MSVC env (vcvars). bt's own keys (e.g. CUDACXX) win.
        bt.env = {**tools.windows_msvc_env(), **bt.env}

    src = _clone(profile)
    builddir = src / "build"
    _configure(profile, bt, src, builddir)
    _compile(profile, bt, builddir, jobs)
    _install_artifacts(profile, builddir, out)

    if profile.accel == "cuda":
        util.logger.info(
            "  note GPU build: set gpu_layers=99 (and flash_attn=True) in "
            "LLMServerConfig in config.py to use the GPU."
        )
    return out


def _clone(profile: Profile) -> Path:
    src = util.BUILD_DIR / "llama.cpp"
    if (src / "CMakeLists.txt").exists():
        util.logger.info("  ok   llama.cpp source present (%s)", src)
        return src
    util.logger.info("  get  %s @ %s", LLAMA_REPO, LLAMA_REF)
    util.shallow_clone(src, LLAMA_REPO, LLAMA_REF)
    return src


def _configure(
    profile: Profile, bt: tools.BuildTools, src: Path, builddir: Path
) -> None:
    cmd = [bt.cmake, "-S", str(src), "-B", str(builddir)]
    if bt.ninja:
        # Ninja everywhere (single-config: CMAKE_BUILD_TYPE picks Release). On Windows
        # cl is provided via bt.env (vcvars); elsewhere it's the system gcc/clang.
        cmd += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={bt.ninja}"]
    cmd += profile.llama_cmake_flags()
    cmd += bt.extra_cmake

    # On Linux, override the build rpath with $ORIGIN so the binary
    # finds its shared libs (e.g. libllama-server-impl.so) next to
    # itself after we copy it to llama_cpp_bin/.
    if profile.os == "linux":
        cmd += ["-DCMAKE_BUILD_RPATH=$ORIGIN"]

    res = util.run(cmd, cwd=src, extra_env=bt.env, check=False)
    if res.returncode != 0 and builddir.exists():
        # Most often a stale CMakeCache (e.g. the generator changed between runs).
        # Wipe and retry once; a valid cache (the Jetson resume case) never gets here.
        util.logger.info("  ..   clearing stale build dir and reconfiguring")
        shutil.rmtree(builddir, ignore_errors=True)
        res = util.run(cmd, cwd=src, extra_env=bt.env, check=False)
    if res.returncode != 0:
        _stop_configure(profile)


def _compile(
    profile: Profile, bt: tools.BuildTools, builddir: Path, jobs: int | None
) -> None:
    jobs = jobs or _default_jobs(profile)
    if profile.is_jetson:
        util.logger.warning(
            "  !!   Jetson: building llama-server with -j%d. Watch power/thermal -\n"
            "       run `tegrastats` in another shell. If the board resets, lower\n"
            "       --jobs and/or set a lower-power nvpmodel, then re-run.",
            jobs,
        )
    cmd = [
        bt.cmake,
        "--build",
        str(builddir),
        "--target",
        "llama-server",
        "-j",
        str(jobs),
    ]
    res = util.run(cmd, extra_env=bt.env, check=False)
    if res.returncode != 0:
        _stop_compile(profile)


def _find_built_server(profile: Profile, builddir: Path) -> Path | None:
    name = f"llama-server{profile.exe_suffix}"
    for cand in (builddir / "bin" / name, builddir / "bin" / "Release" / name):
        if cand.exists():
            return cand
    hits = list(builddir.rglob(name))
    return hits[0] if hits else None


def _install_artifacts(profile: Profile, builddir: Path, out: Path) -> None:
    server = _find_built_server(profile, builddir)
    if not server:
        util.fail(
            "build finished but llama-server was not produced",
            f"searched under {builddir}/bin",
        )
    util.LLAMA_BIN_DIR.mkdir(parents=True, exist_ok=True)

    # CUDA/shared builds emit ggml backends as sibling shared libs; copy them too.
    libext = ".dll" if profile.os == "windows" else ".so"
    copied = 0
    for item in server.parent.iterdir():
        if item == server:
            continue
        if item.suffix == libext or (libext == ".so" and ".so" in item.suffixes):
            shutil.copy2(item, util.LLAMA_BIN_DIR / item.name)
            copied += 1

    shutil.copy2(server, out)
    util.logger.info(
        "  ok   installed %s (+%d shared libs) -> %s",
        out.name,
        copied,
        util.LLAMA_BIN_DIR,
    )


def _stop_configure(profile: Profile) -> None:
    if profile.accel == "cuda":
        util.fail(
            "cmake configure failed for the CUDA build",
            "cmake could not configure the CUDA toolchain. Most often this means the\n"
            "CUDA toolkit isn't fully discoverable. Install a system CUDA toolkit that\n"
            "matches your driver and ensure nvcc is on PATH (or CUDACXX is set), then re-run.\n"
            "See the cmake output above for the specific missing component.",
        )
    if profile.os == "windows":
        util.fail(
            "cmake configure failed",
            "cmake could not configure the MSVC toolchain. Ensure the Visual Studio\n"
            "C++ build tools are installed (Desktop development with C++), then re-run.\n"
            "See the cmake output above.",
        )
    util.fail(
        "cmake configure failed", "See the cmake output above for the failing check."
    )


def _stop_compile(profile: Profile) -> None:
    extra = ""
    if profile.is_jetson:
        extra = (
            "\nOn Jetson a mid-compile failure or board reset is usually power/thermal,\n"
            "not a code bug: lower --jobs, set a higher-power-budget nvpmodel with active\n"
            "cooling, watch `tegrastats`, then re-run (the build resumes)."
        )
    util.fail("llama-server compile failed", f"See the build output above.{extra}")
