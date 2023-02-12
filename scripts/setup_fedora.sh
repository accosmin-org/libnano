#!/usr/bin/env bash

function setup {
    sudo dnf update -qq
    sudo dnf install -y git vim cmake lcov cppcheck valgrind ninja-build
    sudo dnf install -y eigen3-devel
    sudo dnf install -y gcc g++
    sudo dnf install -y clang clang-tools-extra libcxx libcxx-devel libcxxabi llvm
}

function setup_gcc {
    local gcc=$1

    # FIXME: find supported versions!
    exit 1

    sudo dnf update -qq
    sudo dnf install -y gcc-${gcc} g++-${gcc}
}

function setup_llvm {
    local llvm=$1

    # FIXME: find supported versions!
    exit 1

    sudo dnf update -qq
    sudo dnf install -y clang-${llvm} clang-tidy-${llvm} clang-tools-${llvm} clang-format-${llvm}
    # libc++-${llvm}-dev libc++abi-${llvm}-dev
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
        install additional supported llvm versions (11, 12, 13, 14, 15)
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
