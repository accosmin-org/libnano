git clone https://github.com/llvm/llvm-project.git --depth 1

cd llvm-project
base_dir=$(pwd)

export CC=clang
export CXX=clang++

cd ${base_dir}
mkdir -p build_msan && cd build_msan
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_SANITIZER=Memory \
    -DCMAKE_INSTALL_PREFIX=/tmp/llvm/msan ../llvm
ninja
ninja install

cd ${base_dir}
mkdir -p build_tsan && cd build_tsan
cmake -G Ninja -DLLVM_ENABLE_PROJECTS="libcxx;libcxxabi" -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_SANITIZER=Thread \
    -DCMAKE_INSTALL_PREFIX=/tmp/llvm/tsan ../llvm
ninja
ninja install
