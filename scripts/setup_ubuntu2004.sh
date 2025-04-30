#!/usr/bin/env bash

function setup {
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt update -qq
    sudo apt install -y git vim cmake lcov cppcheck valgrind ninja-build
    sudo apt install -y libomp-dev libeigen3-dev
    sudo apt install -y gcc g++
    sudo apt install -y clang clang-format clang-tidy clang-tools python-yaml libc++-dev libc++abi-dev llvm-dev lld
}

function usage {
    cat <<EOF
usage: $0 [OPTIONS]

options:
    -h,--help
        print usage
    --default
        install builtin gcc & clang versions
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
        *)                              echo "unrecognized option $1"; echo; usage;;
    esac
    shift
done

exit 0
