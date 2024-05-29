#!/usr/bin/env bash

basepath=`readlink -f "$0" || greadlink -f "$0"`
basedir=`dirname "${basepath}"`
basedir=`dirname "${basedir}"`

installdir=${basedir}/install
libnanodir=${basedir}/build/libnano
exampledir=${basedir}/build/example

clang_suffix=""
cmake_options="-GNinja -DBUILD_SHARED_LIBS=ON"

cores=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu || echo "$NUMBER_OF_PROCESSORS")
threads=$((cores+1))

export PATH="${PATH}:${installdir}"
export CXXFLAGS="${CXXFLAGS} -Werror -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -pedantic -pthread"

function lto {
    export CXXFLAGS="${CXXFLAGS} -flto"
}

function thinlto {
    export CXXFLAGS="${CXXFLAGS} -flto=thin"
}

function asan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=address -fsanitize=pointer-compare -fsanitize=pointer-subtract"
    export CXXFLAGS="${CXXFLAGS} -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function lsan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=leak -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function usan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function msan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=memory -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function tsan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=thread -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function gold {
    export CXXFLAGS="${CXXFLAGS} -fuse-ld=gold"
}

function lld {
    export CXXFLAGS="${CXXFLAGS} -fuse-ld=lld"
}

function native {
    export CXXFLAGS="${CXXFLAGS} -mtune=native -march=native"
}

function libcpp {
    export CXXFLAGS="${CXXFLAGS} -stdlib=libc++"
    export LDFLAGS="${LDFLAGS} -lc++abi"
}

function coverage {
    export CXXFLAGS="${CXXFLAGS} -fno-inline -fno-omit-frame-pointer -fno-inline-small-functions -fno-default-inline"
    export CXXFLAGS="${CXXFLAGS} -coverage -O0"
}

function llvm_coverage {
    export CXXFLAGS="${CXXFLAGS} -fprofile-instr-generate -fcoverage-mapping"
}

function suffix {
    installdir=${basedir}/install/$1
    libnanodir=${basedir}/build/libnano/$1
    exampledir=${basedir}/build/example/$1

    export PATH="${PATH}:${installdir}"
}

function config {
    cd ${basedir}
    cmake -H${basedir} -B${libnanodir} ${cmake_options} \
        -DCMAKE_INSTALL_PREFIX=${installdir} || return 1
}

function build {
    cd ${libnanodir}
    echo "-- Using ${threads} threads to build"
    cmake --build ${libnanodir} -- -j ${threads} || return 1
}

function tests {
    cd ${libnanodir}
    echo "-- Using ${threads} threads to test"
    ctest --output-on-failure -j ${threads} || return 1
}

function install {
    cd ${libnanodir}
    cmake --install ${libnanodir} --strip || return 1
}

function build_example {
    cd ${basedir}
    cmake -Hexample -B${exampledir} ${cmake_options} \
        -DCMAKE_PREFIX_PATH=${installdir} || return 1
    cd ${exampledir}
    echo "-- Using ${threads} threads to build"
    cmake --build ${exampledir} -- -j ${threads} || return 1
    echo "-- Using ${threads} threads to test"
    ctest --output-on-failure -j ${threads} || return 1
}

function call_cppcheck {
    cd ${libnanodir}

    cppcheck_version=$(cppcheck --version)
    echo "-- Using cppcheck ${cppcheck_version/* /}"

    # NB: the warnings are not fatal (exitcode=0) as they are usually false alarms!
    #--suppress=shadowVar
    cppcheck -j ${threads} \
        --project=compile_commands.json \
        --enable=all --quiet --std=c++17 --error-exitcode=0 --inline-suppr --force \
        --template='{file}:{line},{severity},{id},{message}' \
        --suppress=unknownMacro \
        --suppress=shadowFunction \
        --suppress=unusedFunction \
        --suppress=missingIncludeSystem \
        --suppress=unmatchedSuppression
}

function lcov_coverage_init {
    cd ${basedir}

    local base_output=${basedir}/lcov_base.info

    options=""
    options="${options} --rc lcov_branch_coverage=1 --rc lcov_function_coverage=0"
    options="${options} --rc genhtml_branch_coverage=1 --rc genhtml_function_coverage=0"

    lcov ${options} -d ${libnanodir} -i -c -o ${base_output} || return 1
    lcov ${options} -r ${base_output} '/usr/*' "${HOME}"'/.cache/*' '*/test/*' '*/app/*' '*/build/*' -o ${base_output} || return 1
}

function lcov_coverage {
    cd ${basedir}

    local output=${basedir}/lcov.info
    local base_output=${basedir}/lcov_base.info
    local test_output=${basedir}/lcov_test.info

    options=""
    options="${options} --rc lcov_branch_coverage=1 --rc lcov_function_coverage=0"
    options="${options} --rc genhtml_branch_coverage=1 --rc genhtml_function_coverage=0"

    lcov ${options} -d ${libnanodir} --gcov-tool ${GCOV} -c -o ${test_output} || return 1
    lcov ${options} -r ${test_output} '/usr/*' "${HOME}"'/.cache/*' '*/test/*' '*/app/*' '*/build/*' -o ${test_output} || return 1

    lcov ${options} -a ${base_output} -a ${test_output} -o ${output} || return 1
    rm -f ${base_output} ${test_output}

    lcov ${options} --list ${output} || return 1
    genhtml ${options} --output lcovhtml ${output} || return 1
}

function codecov {
    cd ${basedir}

    local output=${basedir}/lcov.info

    bash <(curl -s https://codecov.io/bash) -f ${output} || return 1
}

function llvm_cov_coverage {
    cd ${basedir}

    local output=${basedir}/llvmcov.info

    tests=$(find ${libnanodir}/test/test_* | grep -v profraw | grep -v profdata)
    objects=""
    for object in $(find ${libnanodir}/src/lib*.so); do
        objects="${objects} -object ${object}"
    done
    for utest in ${tests}; do
        objects="${objects} -object ${utest}"
    done

    llvm-profdata merge -sparse $(find ${libnanodir}/test/*.profraw) -o ${output}

    llvm-cov show \
        -instr-profile=${output} \
        -ignore-filename-regex=test\/ \
        -format=html -Xdemangler=c++filt -tab-size=4 \
        -show-line-counts --show-branches=count --show-expansions --show-instantiation-summary \
        -output-dir llvmcovhtml \
        ${objects}

    llvm-cov report \
        -instr-profile=${output} \
        -ignore-filename-regex=test\/ \
        -show-branch-summary --show-instantiation-summary \
        ${objects}

    llvm-cov show \
        -instr-profile=${output} \
        -ignore-filename-regex=test\/ \
        -format=text -Xdemangler=c++filt -tab-size=4 \
        -show-line-counts --show-branches=count --show-expansions --show-instantiation-summary \
        ${objects} > ${basedir}/llvmcov.text
}

function memcheck {
    cd ${libnanodir}
    echo "-- Using ${threads} threads to test"
    ctest -T memcheck --output-on-failure -j ${threads} || return 1
}

function helgrind {
    cd ${libnanodir}

    returncode=0
    utests="test/test_core_parallel"
    for utest in ${utests}
    do
        printf "Running helgrind@%s ...\n" ${utest}
        log=helgrind_${utest/test\//}.log
        valgrind --tool=helgrind \
            --error-exitcode=1 \
            --log-file=${log} ${utest}

        if [[ $? -ne 0 ]]
        then
            cat ${log}
            # NB: ignore for now the warnings reported by helgrind!
            returncode=1
        fi
        printf "\n"
    done

    return ${returncode}
}

function clang_tidy {
    cd ${libnanodir}

    check=$1

    echo "-- Running clang-tidy-$check"
    log=clang_tidy_${check//\**/}.log
    echo "-- Logging to ${log}"

    wrapper=run-clang-tidy${clang_suffix}
    wrapper=$(which ${wrapper} || which ${wrapper}.py || which /usr/share/clang/${wrapper}.py)
    echo "-- Using wrapper ${wrapper}"
    ${wrapper} -p ${libnanodir} -clang-tidy-binary clang-tidy${clang_suffix} \
        -header-filter=.* -checks=-*,${check} -quiet -j ${threads} > $log 2>&1

    if [[ $? -ne 0 ]]; then
        cat ${log}
        return 1
    fi

    cat $log | grep -E "warning:|error:" | grep -oE "[^ ]+$" | sort | uniq -c

    # show log only if any warning or error is detected
    warnings=$(cat $log | grep -E "warning:|error:" | sort -u | grep -v /usr/include | wc -l)
    if [[ $warnings -gt 0 ]]
    then
        grep -E ".*warning:|error:" $log | sort -u
    fi

    # decide if should exit with failure
    if [[ $warnings -gt 0 ]]
    then
        echo "!! Failed with $warnings warnings and errors!"
        return 1
    else
        echo "-- Check done"
        return 0
    fi
}

function clang_tidy_concurrency {
    clang_tidy "concurrency*"
}

function clang_tidy_misc {
    checks="misc*"
    checks="${checks},-misc-non-private-member-variables-in-classes,-misc-include-cleaner"
    clang_tidy ${checks}
}

function clang_tidy_cert {
    clang_tidy "cert*"
}

function clang_tidy_hicpp {
    checks="hicpp*"
    checks="${checks},-hicpp-avoid-c-arrays"
    checks="${checks},-hicpp-no-array-decay"
    checks="${checks},-hicpp-signed-bitwise"
    checks="${checks},-hicpp-named-parameter"
    clang_tidy ${checks}
}

function clang_tidy_bugprone {
    checks="bugprone*"
    checks="${checks},-bugprone-easily-swappable-parameters"
    clang_tidy ${checks}
}

function clang_tidy_modernize {
    checks="modernize*"
    checks="${checks},-modernize-avoid-c-arrays"
    checks="${checks},-modernize-use-trailing-return-type"
    checks="${checks},-modernize-use-nodiscard"
    clang_tidy ${checks}
}

function clang_tidy_performance {
    clang_tidy "performance*"
}

function clang_tidy_portability {
    clang_tidy "portability*"
}

function clang_tidy_readability {
    checks="readability*"
    checks="${checks},-readability-magic-numbers"
    checks="${checks},-readability-named-parameter"
    checks="${checks},-readability-isolate-declaration"
    checks="${checks},-readability-else-after-return"
    checks="${checks},-readability-function-cognitive-complexity"
    checks="${checks},-readability-identifier-length"
    checks="${checks},-readability-avoid-nested-conditional-operator"
    clang_tidy ${checks}
}

function clang_tidy_clang_analyzer {
    clang_tidy "clang-analyzer*"
}

function clang_tidy_cppcoreguidelines {
    checks="cppcoreguidelines*"
    checks="${checks},-cppcoreguidelines-avoid-c-arrays"
    checks="${checks},-cppcoreguidelines-avoid-magic-numbers"
    checks="${checks},-cppcoreguidelines-pro-bounds-pointer-arithmetic"
    checks="${checks},-cppcoreguidelines-pro-bounds-array-to-pointer-decay"
    checks="${checks},-cppcoreguidelines-avoid-const-or-ref-data-members"
    clang_tidy ${checks}
}

function clang_tidy_all {
    clang_tidy_misc || return 1
    clang_tidy_cert || return 1
    clang_tidy_hicpp || return 1
    clang_tidy_bugprone || return 1
    clang_tidy_modernize || return 1
    #clang_tidy_concurrency || return 1
    clang_tidy_performance || return 1
    clang_tidy_portability || return 1
    clang_tidy_readability || return 1
    clang_tidy_clang_analyzer || return 1
    clang_tidy_cppcoreguidelines || return 1
}

function clang_format {
    files=$(find \
        ${basedir}/src \
        ${basedir}/test \
        ${basedir}/example \
        ${basedir}/include \
        -type f \( -name "*.h" -o -name "*.cpp" \))

    cmd=clang-format${clang_suffix}
    echo "-- Using ${cmd}..."

    log=${basedir}/clang_format.log
    rm -f ${log}

    for file in ${files}; do
        ${cmd} --dry-run ${file} >> ${log} 2>&1
    done

    cat ${log}

    changes=$(cat ${log} | wc -l)
    rm -f ${log}

    for file in ${files}; do
        ${cmd} -i ${file}
    done

    if [[ ${changes} -gt 0 ]]; then
        return 1
    else
        return 0
    fi
}

function sonar {
    cd ${basedir}

    export SONAR_SCANNER_VERSION=5.0.1.3006
    export SONAR_SCANNER_HOME=$HOME/.sonar/sonar-scanner-$SONAR_SCANNER_VERSION-linux
    curl --create-dirs -sSLo $HOME/.sonar/sonar-scanner.zip \
        https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-$SONAR_SCANNER_VERSION-linux.zip
    unzip -o $HOME/.sonar/sonar-scanner.zip -d $HOME/.sonar/
    export PATH=$SONAR_SCANNER_HOME/bin:$PATH
    export SONAR_SCANNER_OPTS="-server"

    curl --create-dirs -sSLo $HOME/.sonar/build-wrapper-linux-x86.zip \
        https://sonarcloud.io/static/cpp/build-wrapper-linux-x86.zip
    unzip -o $HOME/.sonar/build-wrapper-linux-x86.zip -d $HOME/.sonar/
    export PATH=$HOME/.sonar/build-wrapper-linux-x86:$PATH

    sonar-scanner \
        -Dsonar.organization=accosmin \
        -Dsonar.projectKey=libnano \
        -Dsonar.sources=${basedir}/src,${basedir}/include/nano \
        -Dsonar.projectVersion=0.0.1 \
        -Dsonar.python.version=3 \
        -Dsonar.cfamily.compile-commands=${libnanodir}/compile_commands.json \
        -Dsonar.cfamily.llvm-cov.reportPath=${basedir}/llvmcov.text \
        -Dsonar.sourceEncoding=UTF-8 \
        -Dsonar.host.url=https://sonarcloud.io
}

function check_source_files {
    cd ${basedir}

    returncode=0

    filenames=`find include/nano -type f -name "*.h"`
    for filename in ${filenames}; do
        check=`grep ${filename} ${basedir}/src/CMakeLists.txt`
        if [ -z "${check}" ]; then
            echo "-- Error: unreferenced header file ${filename}!"
            returncode=1
        fi
    done

    filenames=`find src -type f -name "*.cpp"`
    for filename in ${filenames}; do
        filename=${filename/src\//}
        check=`grep ${filename} ${basedir}/src/CMakeLists.txt`
        if [ -z "${check}" ]; then
            echo "-- Error: unreferenced source file ${filename}!"
            returncode=1
        fi
    done

    filenames=`find test -type f -name "*.cpp"`
    for filename in ${filenames}; do
        filename=${filename/test\//}
        filename=${filename/\.cpp/}
        check=`grep ${filename} ${basedir}/test/CMakeLists.txt`
        if [ -z "${check}" ]; then
            echo "-- Error: unreferenced test file ${filename}!"
            returncode=1
        fi
    done

    return ${returncode}
}

function check_markdown_docs {
    cd ${basedir}

    returncode=0

    docfiles=`find ${basedir} -type f -name "*.md"`
    for docfile in ${docfiles}; do
        echo "-- Checking documentation file: ${docfile}"

        # check local links [linkname](filename) that point to existing files
        lines=`grep -E "\[.+\]\(.+\)" ${docfile} | grep -v "http"`
        while read -r line; do
            for token in ${line}; do
                filename=`echo ${token} | grep -oP "\]\(.+\)"`
                if [ -n "${filename}" ]; then
                    filename=${filename//\]/}
                    filename=${filename//\(/}
                    filename=${filename//\)/}
                    filename=`dirname ${docfile}`/${filename}
                    if [ ! -d ${filename} ] && [ ! -f ${filename} ]; then
                        echo "-- Error: invalid reference path ${filename}!"
                        returncode=1
                    fi
                fi
            done
        done <<< "${lines}"

        # check example codes with C++ includes that point to existing files
        lines=`grep -E "#include <.+>" ${docfile}`
        while read -r line; do
            for token in ${line}; do
                filename=`echo ${token} | grep -oP "\<.+\>" | grep -v "<iostream>"`
                if [ -n "${filename}" ]; then
                    filename=${filename//\</}
                    filename=${filename//\>/}
                    filename=include/${filename}
                    if [ ! -f ${filename} ]; then
                        echo "-- Error: invalid C++ include ${filename}!"
                        returncode=1
                    fi
                fi
            done
        done <<< "${lines}"
    done

    return ${returncode}
}

function usage {
    cat <<EOF
usage: $0 [OPTIONS]

options:
    -h,--help
        print usage
    --lld
        setup compiler and linker flags to enable the llvm linker
    --lto
        setup compiler and linker flags to enable link-time optimization
    --thinlto
        setup compiler and linker flags to enable link-time optimization with parallelization
    --asan
        setup compiler and linker flags to enable the address sanitizer
    --lsan
        setup compiler and linker flags to enable the leak sanitizer
    --usan
        setup compiler and linker flags to enable the undefined behaviour sanitizer
    --tsan
        setup compiler and linker flags to enable the thread sanitizer
    --msan
        setup compiler and linker flags to enable the memory sanitizer
    --gold
        setup compiler and linker flags to enable the gold linker
    --native
        setup compiler flags to optimize for the native platform
    --libcpp
        setup compiler and linker flags to use libc++
    --coverage
        setup compiler and linker flags to setup code coverage using gcov (gcc and clang)
    --llvm-coverage
        setup compiler and linker flags to setup source-based code coverage (clang)
    --suffix <string>
        suffix for the build and installation directories
    --config
        generate build using CMake
    --build
        compile the library, the unit tests and the command line applications
    --test
        run the unit tests
    --install
        install the library and the command line applications
    --build-example
        build example project
    --cppcheck
        run cppcheck (static code analyzer)
    --lcov
        generate the code coverage report using lcov and genhtml
    --lcov-init
        generate the initial code coverage report using lcov and genhtml
    --llvm-cov
        generate the code coverage report usign llvm's source-based code coverage
    --memcheck
        run the unit tests through memcheck (e.g. detects unitialized variales, memory leaks, invalid memory accesses)
    --helgrind
        run the unit tests through helgrind (e.g. detects data races)
    --clang-suffix <string>
        suffix for the llvm tools like clang-tidy or clang-format (e.g. -6.0)
    --clang-tidy-check <check name>
        run a particular clang-tidy check (e.g. misc, cert)
    --clang-tidy-all
    --clang-tidy-misc
    --clang-tidy-cert
    --clang-tidy-hicpp
    --clang-tidy-bugprone
    --clang-tidy-modernize
    --clang-tidy-concurrency
    --clang-tidy-performance
    --clang-tidy-portability
    --clang-tidy-readability
    --clang-tidy-clang-analyzer
    --clang-tidy-cppcoreguidelines
    --clang-format
        check formatting with clang-format (the code will be modified in-place)
    --check-markdown-docs
        check the markdown documentation (e.g. invalid C++ includes, invalid local links)
    --check-source-files
        check the source files are used properly (e.g. unreferenced files in CMake scripts)
    -D[option]
        options to pass directly to cmake build (e.g. -DCMAKE_BUILD_TYPE=Debug -DBUILD_SHARED_LIBS=ON)
    -G[option]
        options to pass directly to cmake build (e.g. -GNinja)
EOF
    exit 1
}

if [ "$1" == "" ]; then
    usage
fi

while [ "$1" != "" ]; do
    case $1 in
        -h | --help)                    usage;;
        --lld)                          lld;;
        --lto)                          lto;;
        --thinlto)                      thinlto;;
        --asan)                         asan;;
        --lsan)                         lsan;;
        --usan)                         usan;;
        --tsan)                         tsan;;
        --msan)                         msan;;
        --gold)                         gold;;
        --native)                       native;;
        --libcpp)                       libcpp;;
        --coverage)                     coverage;;
        --llvm-coverage)                llvm_coverage;;
        --suffix)                       shift; suffix $1;;
        --config)                       config || exit 1;;
        --build)                        build || exit 1;;
        --test)                         tests || exit 1;;
        --install)                      install || exit 1;;
        --cppcheck)                     call_cppcheck || exit 1;;
        --lcov)                         lcov_coverage || exit 1;;
        --lcov-init)                    lcov_coverage_init || exit 1;;
        --llvm-cov)                     llvm_cov_coverage || exit 1;;
        --memcheck)                     memcheck || exit 1;;
        --helgrind)                     helgrind || exit 1;;
        --clang-suffix)                 shift; clang_suffix=$1;;
        --clang-tidy-check)             shift; clang_tidy $1 || exit 1;;
        --clang-tidy-all)               clang_tidy_all || exit 1;;
        --clang-tidy-misc)              clang_tidy_misc || exit 1;;
        --clang-tidy-cert)              clang_tidy_cert || exit 1;;
        --clang-tidy-hicpp)             clang_tidy_hicpp || exit 1;;
        --clang-tidy-bugprone)          clang_tidy_bugprone || exit 1;;
        --clang-tidy-modernize)         clang_tidy_modernize || exit 1;;
        --clang-tidy-concurrency)       clang_tidy_concurrency || exit 1;;
        --clang-tidy-performance)       clang_tidy_performance || exit 1;;
        --clang-tidy-portability)       clang_tidy_portability || exit 1;;
        --clang-tidy-readability)       clang_tidy_readability || exit 1;;
        --clang-tidy-clang-analyzer)    clang_tidy_clang_analyzer || exit 1;;
        --clang-tidy-cppcoreguidelines) clang_tidy_cppcoreguidelines || exit 1;;
        --build-example)                build_example || exit 1;;
        --clang-format)                 clang_format || exit 1;;
        --sonar)                        sonar || exit 1;;
        --codecov)                      codecov || exit 1;;
        --check-markdown-docs)          check_markdown_docs || exit 1;;
        --check-source-files)           check_source_files || exit 1;;
        -D*)                            cmake_options="${cmake_options} $1";;
        -G*)                            cmake_options="${cmake_options} $1";;
        *)                              echo "unrecognized option $1"; echo; usage;;
    esac
    shift
done

exit 0
