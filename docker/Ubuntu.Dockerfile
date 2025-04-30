ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION}

RUN apt-get update -qq
RUN apt-get -y upgrade
RUN apt-get install -y git vim cmake lcov cppcheck valgrind ninja-build gdb
RUN apt-get install -y libomp-dev libeigen3-dev
RUN apt-get install -y gcc g++
RUN apt-get install -y clang clang-format clang-tidy clang-tools python3-pretty-yaml libc++-dev libc++abi-dev llvm-dev lld

CMD bash
