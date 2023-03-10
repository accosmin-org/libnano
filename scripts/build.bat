@echo off
setlocal enabledelayedexpansion

set script=%0
set basedir=%~dp0..\
set installdir=%basedir%\install
set libnanodir=%basedir%\build\libnano
set exampledir=%basedir%\build\example

set build_type="Debug"

REM set VCPKG_INSTALLATION_ROOT=C:\Users\accos\vcpkg
REM set Eigen3_DIR=%VCPKG_INSTALLATION_ROOT%\installed\x64-windows\share\eigen3

set cmake_options=-DCMAKE_TOOLCHAIN_FILE=%VCPKG_INSTALLATION_ROOT%\scripts\buildsystems\vcpkg.cmake ^
    -DCMAKE_CXX_FLAGS="/W4 /WX /bigobj /EHsc /wd4251 /wd4702 /MP" ^
    -DCMAKE_CONFIGURATION_TYPES="Debug;Release"

cd %basedir%

REM ---------------------------------------------------------------------------------
:cmd_loop
REM ---------------------------------------------------------------------------------
if "%~1"=="" goto cmd_done

if "%~1"=="-h" (
	goto print_help
)
if "%~1"=="--help" (
	goto print_help
)

if "%~1"=="--build-type" (
	set build_type=%~2
)

if "%~1"=="--suffix" (
    set installdir=%basedir%\install\%~2
    set libnanodir=%basedir%\build\libnano\%~2
    set exampledir=%basedir%\build\example\%~2
)

set gg=%~1
if "%gg:~0,2%"=="-D" (
    set cmake_options=%cmake_options% %gg%
)
if "%gg:~0,2%"=="-G" (
    set cmake_options=%cmake_options% %gg%
)

if "%~1"=="--config" (
    call:config || exit /B 1
)

if "%~1"=="--build" (
    call:build || exit /B 1
)

if "%~1"=="--test" (
    call:tests || exit /B 1
)

if "%~1"=="--install" (
    call:install || exit /B 1
)

if "%~1"=="--build-example" (
    call:build_example || exit /B 1
)

shift
goto cmd_loop
:cmd_done

REM echo build type: %build_type%
REM echo installdir: %installdir%
REM echo libnanodir: %libnanodir%
REM echo exampledir: %exampledir%
REM echo cmake_options: %cmake_options%

cd %basedir%

@echo on

exit /B %errorlevel%

REM ---------------------------------------------------------------------------------
:config
REM ---------------------------------------------------------------------------------
cd %basedir%
cmake -H%basedir% -B%libnanodir% %cmake_options% ^
    -DCMAKE_INSTALL_RPATH=%installdir%\lib ^
    -DCMAKE_INSTALL_PREFIX=%installdir% || exit /B 1
exit /B 0

REM ---------------------------------------------------------------------------------
:build
REM ---------------------------------------------------------------------------------
cd %libnanodir%
cmake --build %libnanodir% --config %build_type% -- -m:%NUMBER_OF_PROCESSORS% || exit /B 1
exit /B 0

REM ---------------------------------------------------------------------------------
:tests
REM ---------------------------------------------------------------------------------
cd %libnanodir%
ctest --output-on-failure --build-config %build_type% -j %NUMBER_OF_PROCESSORS% || exit /B 1
exit /B 0

REM ---------------------------------------------------------------------------------
:install
REM ---------------------------------------------------------------------------------
cd %libnanodir%
cmake --build %libnanodir% --config %build_type% --target install || exit /B 1
exit /B 0

REM ---------------------------------------------------------------------------------
:build_example
REM ---------------------------------------------------------------------------------
set NANO_DIR=%installdir%
cd %basedir%
cmake -Hexample -B%exampledir% %cmake_options% || exit /B 1
cd %exampledir%
cmake --build %exampledir% --config %build_type% -- /m:%NUMBER_OF_PROCESSORS% || exit /B 1
ctest --output-on-failure --build-config %build_type% -j %NUMBER_OF_PROCESSORS% || exit /B 1
exit /B 0

REM ---------------------------------------------------------------------------------
:print_help
REM ---------------------------------------------------------------------------------
echo usage: %script% [OPTIONS]
echo.
echo options:
echo    -h,--help
echo        print usage
echo    --build-type [string]
echo        select the build type from Debug, Release, MinSizeRel or RelWithDebInfo
echo    --suffix [string]
echo        suffix for the build and installation directories
echo    --config
echo        generate build using CMake
echo    --build
echo        compile the library, the unit tests and the command line applications
echo    --test
echo        run the unit tests
echo    --install
echo        install the library and the command line applications
echo    --build-example
echo        build example project
echo    -D[option]
echo        options to pass directly to cmake build (e.g. -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON)
echo    -G[option]
echo        options to pass directly to cmake build (e.g. -GNinja)
echo.
exit /B 1
