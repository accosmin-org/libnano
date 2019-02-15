#!/bin/bash

basedir=$(pwd)

# build and test library
cd ${basedir}
cmake -GNinja -Hlibnano -Bbuild/libnano -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=OFF || exit 1
#cmake --build build/libnano --target install || exit 1
cd build/libnano
ninja -k1000 || exit 1
ctest --output-on-failure || exit 1
sudo ninja install || exit 1

# build examples
cd ${basedir}
cmake -GNinja -Hexample -Bbuild/example -DCMAKE_BUILD_TYPE=Debug || exit 1
#cmake --build build/example || exit 1
cd build/example
ninja -k1000 || exit 1
