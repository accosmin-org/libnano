#!/bin/bash

basedir=$(pwd)
installdir=${basedir}/install
libnanodir=${basedir}/build/libnano
exampledir=${basedir}/build/example
clang_tidy_suffix=""
build_type="Debug"

generator=""
build_shared="ON"

cores=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu || echo "$NUMBER_OF_PROCESSORS")
threads=$((cores+1))

export PATH="${PATH}:${installdir}"
export CXXFLAGS="${CXXFLAGS} -Wshadow -Werror"

function asan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=address -fno-omit-frame-pointer"
}

function lsan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=leak -fno-omit-frame-pointer"
}

function usan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=undefined -fno-omit-frame-pointer"
}

function msan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=memory -fno-omit-frame-pointer"
}

function tsan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=thread -fno-omit-frame-pointer"
}

function gold {
    export CXXFLAGS="${CXXFLAGS} -fuse-ld=gold"
}

function native {
    export CXXFLAGS="${CXXFLAGS} -mtune=native -march=native"
}

function libcpp {
    export CXXFLAGS="${CXXFLAGS} -stdlib=libc++"
    export LDFLAGS="${LDFLAGS} -lc++abi"
}

function coverage {
    export CXXFLAGS="${CXXFLAGS} -fno-inline -fno-omit-frame-pointer -fno-inline-small-functions -fno-default-inline"
    export CXXFLAGS="${CXXFLAGS} -coverage -O0"
}

function suffix {
    installdir=${basedir}/install/$1
    libnanodir=${basedir}/build/libnano/$1
    exampledir=${basedir}/build/example/$1

    export PATH="${PATH}:${installdir}"
}

function config {
    cd ${basedir}
    cmake ${generator} -Hlibnano -B${libnanodir} \
        -DCMAKE_BUILD_TYPE=${build_type} \
        -DBUILD_SHARED_LIBS=${build_shared} \
        -DCMAKE_INSTALL_RPATH=${installdir}/lib \
        -DCMAKE_INSTALL_PREFIX=${installdir} || return 1
}

function build {
    cd ${libnanodir}
    cmake --build ${libnanodir} -- -j ${threads} || return 1
}

function tests {
    cd ${libnanodir}
    ctest --output-on-failure -j ${threads} || return 1
}

function install {
    cd ${libnanodir}
    cmake --build ${libnanodir} --target install || return 1
}

function build_example {
    cd ${basedir}
    cmake ${generator} -Hexample -B${exampledir} -DCMAKE_BUILD_TYPE=Debug || return 1
    cd ${exampledir}
    cmake --build ${exampledir} -- -j ${threads} || return 1
}

function cppcheck {
    cd ${libnanodir}

    version=1.87
    rm -rf cppcheck-${version}

    wget -N https://github.com/danmar/cppcheck/archive/${version}.tar.gz || return 1
    tar -xf ${version}.tar.gz > /dev/null || return 1

    OLD_CXXFLAGS=${CXXFLAGS}
    export CXXFLAGS=""
    cd cppcheck-${version} && mkdir build && cd build
    cmake .. ${generator} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/cppcheck > config.log || return 1
    cmake --build . -- -j ${threads} > build.log || return 1
    cmake --build . --target install || return 1
    cd ../../
    export CXXFLAGS="${OLD_CXXFLAGS}"

    /tmp/cppcheck/bin/cppcheck --version || return 1

    # NB: the warnings are not fatal (exitcode=0) as they are usually false alarms!
    #--suppress=shadowFunction
    #--suppress=shadowVar
    #--suppress=unusedFunction
    /tmp/cppcheck/bin/cppcheck \
        --project=compile_commands.json \
        --enable=all --quiet --std=c++14 --error-exitcode=0 --inline-suppr --force \
        --template='{file}:{line},{severity},{id},{message}' \
        --suppress=unknownMacro \
        --suppress=unmatchedSuppression || return 1
}

function install_json {
    bash ${basedir}/scripts/install_json.sh || return 1
}

function codecov {
    cd ${basedir}

    bash <(curl -s https://codecov.io/bash) -R ${basedir} -f '!*test_*' || return 1
}

function coveralls {
    cd ${basedir}

    coveralls --root ${basedir} --gcov-options '\-lp' || return 1
}

function memcheck {
    cd ${libnanodir}

    version=3.14.0
    wget -N http://www.valgrind.org/downloads/valgrind-${version}.tar.bz2 || return 1
    tar -xf valgrind-${version}.tar.bz2 > /dev/null || return 1

    OLD_CXXFLAGS=${CXXFLAGS}
    export CXXFLAGS=""
    cd valgrind-${version}
    ./autogen.sh > autogen.log || return 1
    ./configure --prefix=/tmp/valgrind > config.log || return 1
    make -j > build.log || return 1
    make install > install.log || return 1
    cd ..
    export CXXFLAGS="${OLD_CXXFLAGS}"

    returncode=0
    /tmp/valgrind/bin/valgrind --version || return 1

    # NB: not using ctest directly because I cannot pass options to memcheck!
    #ctest --output-on-failure -T memcheck

    utests=$(ls test/test_* | grep -v .log)
    for utest in ${utests}
    do
        printf "Running %s ...\n" ${utest}
        log=${utest/test\//}.log
        /tmp/valgrind/bin/valgrind --tool=memcheck \
            --leak-check=yes --show-reachable=yes --num-callers=50 --error-exitcode=1 \
            --log-file=${log} ${utest} || return 1

        if [[ $? -gt 0 ]]
        then
            cat ${log}
            returncode=1
        fi
        printf "\n"
    done

    return ${returncode}
}

function spinner()
{
    local pid=$!
    local delay=0.75
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]
    do
        local temp=${spinstr#?}
        printf " [%c] " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
            printf "\r"
    done
    printf "      \r"
}

function clang_tidy {
    cd ${libnanodir}

    check=$1

    # misc
    # cert
    # hicpp
    # bugprone
    # modernize
    # performance
    # portability
    # readability
    # clang-analyzer
    # cppcoreguidelines

    printf "running $check ...\n"
    log=clang_tidy_${check}.log
    run-clang-tidy${clang_tidy_suffix}.py -clang-tidy-binary clang-tidy${clang_tidy_suffix} \
        -header-filter=.* -checks=-*,${check}* -quiet > $log&
    spinner

    cat $log | grep warning: | grep -oE "[^ ]+$" | sort | uniq -c
    printf "\n"

    # show log only if any warning is detected
    warnings=$(cat $log | grep warning: | sort -u | wc -l)
    #grep warning: $log
    if [[ $warnings -gt 0 ]]
    then
            cat $log
    fi

    # decide if should exit with failure
    if [[ $warnings -gt 0 ]]
    then
            printf "failed with $warnings warnings!\n\n"
            return 1
    else
            printf "passed.\n"
            return 0
    fi
}

function usage {
	cat <<EOF
usage: $0 [OPTIONS]

options:
	-h,--help
		print usage
    --asan
        setup compiler and linker flags to use the address sanitizer
    --lsan
        setup compiler and linker flags to use the leak sanitizer
    --usan
        setup compiler and linker flags to use the undefined behaviour sanitizer
    --tsan
        setup compiler and linker flags to use the thread sanitizer
    --msan
        setup compiler and linker flags to use the memory sanitizer
    --gold
        setup compiler and linker flags to use the gold linker
    --native
        setup compiler flags to optimize for the native platform
    --libcpp
        setup compiler and linker flags to use libc++
    --coverage
        setup compiler and linker flags to setup code coverage
    --suffix <string>
        suffix for the build and installation directories
    --build-type [Debug,Release,RelWithDebInfo,MinSizeRel]
        build type as defined by CMake
    --config
        generate build using CMake
    --build
        compile the library, the unit tests and the command line applications
    --test
        run the unit tests
    --install
        install the library and the command line applications
    --cppcheck
        run cppcheck (static code analyzer)
    --codecov
        upload code coverage results to codecov.io
    --coveralls
        upload code coverage results to coveralls.io
    --memcheck
        run the unit tests through memcheck
    --clang-tidy-check <check name>
        run a particular clang-tidy check (e.g. misc, cert)
    --clang-tidy-suffix <string>
        suffix for the clang-tidy binaries (e.g. -6.0)
    --build-example
        build example project
    --generator
        overwrite the default build generator (e.g. --generator Ninja to use Ninja as the build system)
    --shared
        build libnano as a shared library (default)
    --static
        build libnano as a static library
    --install-json
        install json dependency system-wide
EOF
	exit 1
}

if [ "$1" == "" ]; then
	usage
fi

while [ "$1" != "" ]; do
	case $1 in
		-h | --help)        usage
                            ;;
        --asan)             asan
                            ;;
        --lsan)             lsan
                            ;;
        --usan)             usan
                            ;;
        --tsan)             tsan
                            ;;
        --msan)             msan
                            ;;
        --gold)             gold
                            ;;
        --native)           native
                            ;;
        --libcpp)           libcpp
                            ;;
        --coverage)         coverage
                            ;;
        --suffix)           shift
                            suffix $1
                            ;;
        --build-type)       shift
                            build_type=$1
                            ;;
        --config)           config || exit 1
                            ;;
        --build)            build || exit 1
                            ;;
        --test)             tests || exit 1
                            ;;
        --install)          install || exit 1
                            ;;
        --cppcheck)         cppcheck || exit 1
                            ;;
        --codecov)          codecov || exit 1
                            ;;
        --coveralls)        coveralls || exit 1
                            ;;
        --memcheck)         memcheck || exit 1
                            ;;
        --clang-tidy-check) shift
                            clang_tidy $1 || exit 1
                            ;;
        --clang-tidy-suffix) shift
                            clang_tidy_suffix=$1
                            ;;
        --build-example)    build_example || exit 1
                            ;;
        --generator)        shift
                            generator="-G$1"
                            ;;
        --shared)           build_shared="ON"
                            ;;
        --static)           build_shared="OFF"
                            ;;
        --install-json)     install_json || exit 1
                            ;;
		*)                  echo "unrecognized option $1"
					        echo
					        usage
					        ;;
	esac
	shift
done

exit 0
