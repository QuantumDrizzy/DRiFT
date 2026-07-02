@echo off
REM Build the GPU Ising parallel-tempering engine.
REM
REM Run this from the "x64 Native Tools Command Prompt for VS 2022" so cl.exe is on PATH
REM (nvcc needs the MSVC host compiler). Alternatively, uncomment the vcvars line and adjust
REM the path to your Visual Studio install.
REM
REM call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM sm_120 = Blackwell (RTX 50-series, e.g. RTX 5060 Ti). Change -arch for other GPUs.
nvcc -O3 -arch=sm_120 ising_pt.cu -o ising_pt.exe
if %ERRORLEVEL% neq 0 (
    echo.
    echo BUILD FAILED. Common causes: not in the x64 Native Tools prompt (cl.exe missing),
    echo wrong -arch for your GPU, or CUDA toolkit not on PATH.
    exit /b %ERRORLEVEL%
)
echo.
echo Built ising_pt.exe. Validate against the exact engine with:
echo     python -m experiments.phase14_gpu
