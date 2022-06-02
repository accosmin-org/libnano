#!/usr/bin/env bash

function setup {
    sudo apt update -qq
    sudo apt install -y git vim cmake lcov cppcheck valgrind
    sudo apt install -y libomp-dev libeigen3-dev ninja-build
    sudo apt install -y gcc g++ clang clang-tidy clang-tools python3-pretty-yaml libc++-dev libc++abi-dev
}

function setup_gcc {
    local gcc=$1

    sudo apt update -qq
    sudo apt install -y gcc-${gcc} g++-${gcc}
}

function setup_llvm {
    local llvm=$1

    sudo apt update -qq
    sudo apt install -y clang-${llvm} clang-tidy-${llvm} clang-tools-${llvm}
}

function usage {
    cat <<EOF
usage: $0 [OPTIONS]

options:
    -h,--help
        print usage
    --default
        install builtin gcc & clang versions
    --gcc [version]
        install additional supported gcc versions (9, 10, 11, 12)
    --llvm [version]
        install additional supported llvm versions (11, 12, 13, 14)
EOF
    exit 1
}

if [ "$1" == "" ]; then
    usage
fi

while [ "$1" != "" ]; do
    case $1 in
        -h | --help)                    usage;;
        --default)                      setup || exit 1;;
        --gcc)                          shift; setup_gcc $1 || exit 1;;
        --llvm)                         shift; setup_llvm $1 || exit 1;;
        *)                              echo "unrecognized option $1"; echo; usage;;
    esac
    shift
done

exit 0
