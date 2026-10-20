@echo off
REM Regenerates cmake-intellisense\compile_commands.json, the IntelliSense
REM database for VS Code's C/C++ extension.
REM
REM The real build uses the "Visual Studio 17 2022" generator, which cannot
REM export a compilation database at all. So this is a second, Ninja-configured
REM build tree that exists purely to produce compile_commands.json. Nothing is
REM ever compiled from it -- `cmake --build build --config Release` remains the
REM real build. Re-run this only when CMakeLists.txt changes include paths or
REM defines.
REM
REM Note: control flow deliberately uses goto rather than `if ... ( ... )`
REM blocks. A parenthesised block containing %VCVARS% would break, because cmd
REM expands the variable while parsing the block, and the ")" in "(x86)" would
REM terminate the block early.
setlocal

set "VCVARS=B:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
set "NINJA=D:\vcpkg\downloads\tools\ninja-1.13.2-windows\ninja.exe"
set "REPO=%~dp0.."

REM CMake's Ninja generator does not locate MSVC on its own; it needs a compiler
REM environment, which vcvarsall provides.
if exist "%VCVARS%" goto have_vcvars
echo [configure-intellisense] vcvarsall.bat not found at:
echo   %VCVARS%
echo Update the VCVARS path in this script.
exit /b 1

:have_vcvars
call "%VCVARS%" x64 >nul 2>&1
if not errorlevel 1 goto have_env
echo [configure-intellisense] vcvarsall.bat failed.
exit /b 1

:have_env
cmake -S "%REPO%" -B "%REPO%\cmake-intellisense" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ^
    -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake
exit /b %ERRORLEVEL%
