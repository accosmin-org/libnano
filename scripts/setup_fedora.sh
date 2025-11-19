#!/usr/bin/env bash

function setup {
    sudo dnf update -qq
    sudo dnf install -y git vim cmake lcov cppcheck valgrind ninja-build ccache shellcheck
    sudo dnf install -y eigen3-devel
    sudo dnf install -y gcc g++ libubsan liblsan libasan libtsan
    sudo dnf install -y clang clang-tools-extra libcxx libcxx-devel libcxxabi llvm lld
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
