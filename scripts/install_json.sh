#!/bin/bash

cd /tmp/
wget https://github.com/nlohmann/json/archive/v3.6.1.tar.gz || exit 1
tar -xf v3.6.1.tar.gz || exit 1
cd json-3.6.1/ && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF || exit 1
make || exit 1
sudo make install || exit 1

cd ../../ && rm -rf v3.6.1.tar.gz json-3.6.1
