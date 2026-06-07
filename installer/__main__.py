"""Installer CLI. Runnable standalone (so the Taskfile is a convenience, not a
hard dependency):

    uv run python -m installer doctor
    uv run python -m installer setup [--force] [--jobs N]
    uv run python -m installer <torch|stt|llama|toolchain|run|clean>
"""
from __future__ import annotations

import argparse
import shutil
import sys

from . import (
    build_llama,
    detect,
    models,
    runtime_libs,
    serve,
    setup_stt,
    setup_torch,
    tools,
    util,
)


def _ensure_toolchain(p) -> None:
    util.banner("Build toolchain")
    tools.ensure_cmake_ninja(p)
    tools.ensure_compiler(p)
    if p.accel == "cuda":
        tools.ensure_cuda_build(p)


def _uv_sync() -> None:
    util.banner("Python dependencies (uv sync)")
    # --inexact so syncing never strips the provisioner-installed torch / STT deps.
    util.run([util.uv(), "sync", "--inexact"])


def cmd_detect(args) -> int:
    print(detect.detect().to_json())
    return 0


def cmd_doctor(args) -> int:
    p = detect.detect()
    log = util.logger.info
    log("ai-voice-stack - doctor\n")
    log("  platform : %s", p.summary())
    log("  torch idx: %s", p.torch_index_url or "default PyPI wheels")
    log("  cmake    : %s", " ".join(p.llama_cmake_flags()))
    if p.generator:
        log("  generator: %s", p.generator)

    log("\n  build toolchain:")
    for k in ("git", "cmake", "ninja", "compiler", "nvcc"):
        log("    [%s] %s", "x" if p.has.get(k) else " ", k)

    log("\n  models (provide these yourself):")
    for label, path in models.REQUIRED:
        present = models.resolve(path).exists()
        log("    [%s] %-44s %s", "x" if present else " ", path, "" if present else "missing")

    log("\n  artifacts:")
    server = build_llama.server_binary_path(p)
    stt_entry = util.SIMUL_DIR / "simulstreaming_whisper_server.py"
    torch_ok = setup_torch._probe(p)
    log("    [%s] %s", "x" if server.exists() else " ", server.name + " (llama_cpp_bin/)")
    log("    [%s] simulstreaming_lib (STT backend)", "x" if stt_entry.exists() else " ")
    log("    [%s] torch+torchaudio for accel=%s", "x" if torch_ok else " ", p.accel)

    gaps = (
        not all(p.has.get(k) for k in ("git", "cmake", "compiler"))
        or not server.exists()
        or not torch_ok
        or bool(models.missing())
    )
    log("\n  %s", "Gaps found - run `task setup` (and place any missing models), then `task run`."
        if gaps else "All set. Run `task run`.")
    return 0


def cmd_setup(args) -> int:
    p = detect.detect()
    util.logger.info("Profile: %s\n", p.summary())
    runtime_libs.ensure(p)
    _uv_sync()
    _ensure_toolchain(p)  # fail-fast on a missing compiler before the build
    setup_torch.install(p, force=args.force)
    setup_stt.install(p, force=args.force)
    build_llama.build(p, jobs=args.jobs, force=args.force)

    util.logger.info("\n%s\nSetup complete.", "=" * 70)
    gaps = models.missing()
    if gaps:
        util.logger.info("Place these models where config.py expects them, then `task run`:")
        for label, path in gaps:
            util.logger.info("  - %-18s %s", label, path)
    else:
        util.logger.info("Launch the stack with:  task run")
    return 0


def cmd_toolchain(args) -> int:
    _ensure_toolchain(detect.detect())
    return 0


def cmd_torch(args) -> int:
    setup_torch.install(detect.detect(), force=args.force)
    return 0


def cmd_stt(args) -> int:
    setup_stt.install(detect.detect(), force=args.force)
    return 0


def cmd_llama(args) -> int:
    build_llama.build(detect.detect(), jobs=args.jobs, force=args.force)
    return 0


def cmd_run(args) -> int:
    return serve.run()


def cmd_clean(args) -> int:
    util.banner("Clean")
    if util.BUILD_DIR.exists():
        shutil.rmtree(util.BUILD_DIR, ignore_errors=True)
        util.logger.info("  ok   removed %s", util.BUILD_DIR)
    else:
        util.logger.info("  ok   nothing to clean (%s absent)", util.BUILD_DIR.name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="installer", description="ai-voice-stack provisioner")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--verbose", action="store_true", help="debug logging")
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, func, *, force=False, jobs=False, help=""):
        sp = sub.add_parser(name, parents=[common], help=help)
        if force:
            sp.add_argument("--force", action="store_true", help="rebuild/redownload even if present")
        if jobs:
            sp.add_argument("--jobs", type=int, default=None, help="parallel build jobs")
        sp.set_defaults(func=func, force=False, jobs=None)
        return sp

    add("doctor", cmd_doctor, help="report the machine profile + what's missing")
    add("detect", cmd_detect, help="print the detection profile as JSON")
    add("setup", cmd_setup, force=True, jobs=True, help="install everything (idempotent)")
    add("toolchain", cmd_toolchain, help="ensure cmake/ninja/compiler/CUDA")
    add("torch", cmd_torch, force=True, help="install torch for the detected accelerator")
    add("stt", cmd_stt, force=True, help="clone SimulStreaming + install requirements")
    add("llama", cmd_llama, force=True, jobs=True, help="build llama.cpp llama-server")
    add("run", cmd_run, help="launch the stack (servers + pipeline)")
    add("clean", cmd_clean, help="remove build scratch (.build/)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    util.setup_logging(args.verbose)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        util.logger.info("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
