import subprocess, os

os.chdir(os.environ["USERPROFILE"] + "/ai-voice-stack")

# Clean build dir first
subprocess.run(
    ["cmd", "/c", "rmdir", "/s", "/q", ".build\\llama.cpp\\build"], capture_output=True
)

# Reconfigure with minimal CPU features
cmd = [
    ".venv/Scripts/cmake.exe",
    "-S",
    ".build/llama.cpp",
    "-B",
    ".build/llama.cpp/build",
    "-G",
    "Ninja",
    "-DCMAKE_MAKE_PROGRAM=.venv/Scripts/ninja.EXE",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_NATIVE=OFF",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DLLAMA_CURL=OFF",
    "-DGGML_AVX=OFF",
    "-DGGML_AVX2=OFF",
    "-DGGML_FMA=OFF",
    "-DGGML_F16C=OFF",
    "-DGGML_BMI2=OFF",
    "-DGGML_SSE42=OFF",
    "-DGGML_OPENMP=OFF",
]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print(r.stdout[-1000:])
print(r.stderr[-500:])
if r.returncode == 0:
    print("CONFIGURE OK")
    # Build
    r2 = subprocess.run(
        [
            ".venv/Scripts/cmake.exe",
            "--build",
            ".build/llama.cpp/build",
            "--target",
            "llama-server",
            "-j",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if r2.returncode == 0:
        print("BUILD OK")
    else:
        print("BUILD FAIL", r2.returncode)
        print(r2.stdout[-500:])
        print(r2.stderr[-500:])
else:
    print("CONFIGURE FAIL", r.returncode)
