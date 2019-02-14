#!/bin/bash

basedir=$(pwd)

#TODO: check exit codes

# build and test library
cd ${basedir}
cmake -GNinja -Hlibnano -Bbuild/libnano -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF
#cmake --build build/libnano --target install
cd build/libnano && ninja -k1000 && ctest --output-on-failure && sudo ninja install

# build examples
cd ${basedir}
cmake -GNinja -Hexample -Bbuild/example -DCMAKE_BUILD_TYPE=Debug
#cmake --build build/example
cd build/example && ninja -k1000
