#!/bin/bash

basedir=$(pwd)

cmake -GNinja -Hlibnano -Bbuild/libnano -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF
#cmake --build build/libnano --target install
cd build/libnano && ninja -k1000 && sudo ninja install && cd ${basedir}

cmake -GNinja -Hexample -Bbuild/example -DCMAKE_BUILD_TYPE=Debug
#cmake --build build/example
cd build/example && ninja -k1000 && cd ${basedir}
