#!/usr/bin/env bash

basepath=$(readlink -f "$0" || greadlink -f "$0")
basedir=$(dirname "$(dirname "${basepath}")")

installdir="${basedir}"/install
libnanodir="${basedir}"/build/libnano
exampledir="${basedir}"/build/example

clang_suffix=""
cmake_options=(-GNinja -DBUILD_SHARED_LIBS=ON)

cores=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu || echo "$NUMBER_OF_PROCESSORS")
threads=$((cores+1))

export PATH="${PATH}:${installdir}"
export CXXFLAGS="${CXXFLAGS} -Werror -Wall -Wextra -Wconversion -Wsign-conversion -Wshadow"
export CXXFLAGS="${CXXFLAGS} -pedantic -pthread"

function setup_gcc {
    export CXX=g++
    export CXXFLAGS="${CXXFLAGS} -Wno-maybe-uninitialized"
}

function setup_clang {
    export CXX=clang++
}

function setup_lto {
    export CXXFLAGS="${CXXFLAGS} -flto"
}

function setup_thinlto {
    export CXXFLAGS="${CXXFLAGS} -flto=thin"
}

function setup_asan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=address -fsanitize=pointer-compare -fsanitize=pointer-subtract"
    export CXXFLAGS="${CXXFLAGS} -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function setup_lsan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=leak -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function setup_usan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=undefined"
    export CXXFLAGS="${CXXFLAGS} -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
    # export CXXFLAGS="${CXXFLAGS} -fno-sanitize-merge"
    export UBSAN_OPTIONS=print_stacktrace=1
}

function setup_msan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=memory -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function setup_tsan {
    export CXXFLAGS="${CXXFLAGS} -fsanitize=thread -fno-omit-frame-pointer -fno-optimize-sibling-calls -O1 -g"
    export CXXFLAGS="${CXXFLAGS} -fno-sanitize-recover=all"
}

function setup_gold {
    export CXXFLAGS="${CXXFLAGS} -fuse-ld=gold"
}

function setup_lld {
    export CXXFLAGS="${CXXFLAGS} -fuse-ld=lld"
}

function setup_no_werror {
    export CXXFLAGS="${CXXFLAGS} -Wno-error"
}

function setup_native {
    export CXXFLAGS="${CXXFLAGS} -mtune=native -march=native"
}

function setup_libcpp {
    export CXXFLAGS="${CXXFLAGS} -stdlib=libc++"
    export LDFLAGS="${LDFLAGS} -lc++abi"
}

function setup_coverage {
    export CXXFLAGS="${CXXFLAGS} -coverage -fno-omit-frame-pointer -Og"
    export LDFLAGS="${LDFLAGS} -coverage"
}

function setup_llvm_coverage {
    export CXXFLAGS="${CXXFLAGS} -fprofile-instr-generate -fcoverage-mapping -Og"
}

function setup_suffix {
    installdir="${basedir}"/install/$1
    libnanodir="${basedir}"/build/libnano/$1
    exampledir="${basedir}"/build/example/$1

    export PATH="${PATH}:${installdir}"
}

function call_config {
    cd "${basedir}" || return 1
    cmake -H"${basedir}" -B"${libnanodir}" "${cmake_options[@]}" \
        -DCMAKE_INSTALL_PREFIX="${installdir}" || return 1
}

function call_build {
    cd "${libnanodir}" || return 1
    echo "-- Using ${threads} threads to build"
    cmake --build "${libnanodir}" -- -j ${threads} || return 1
}

function call_test {
    cd "${libnanodir}" || return 1
    echo "-- Using ${threads} threads to test"
    ctest --output-on-failure -j ${threads} || return 1
}

function call_install {
    cd "${libnanodir}" || return 1
    cmake --install "${libnanodir}" --strip || return 1
}

function call_example {
    cd "${basedir}" || return 1
    cmake -Hexample -B"${exampledir}" "${cmake_options[@]}" \
        -DCMAKE_PREFIX_PATH="${installdir}" || return 1
    cd "${exampledir}" || return 1
    echo "-- Using ${threads} threads to build"
    cmake --build "${exampledir}" -- -j ${threads} || return 1
    echo "-- Using ${threads} threads to test"
    ctest --output-on-failure -j ${threads} || return 1
}

function call_cppcheck {
    cd "${libnanodir}" || return 1

    cppcheck_version=$(cppcheck --version)
    echo "-- Using cppcheck ${cppcheck_version/* /}"

    # NB: the warnings are not fatal (exitcode=0) as they are usually false alarms!
    #--suppress=shadowVar
    cppcheck -j ${threads} \
        --project=compile_commands.json \
        --enable=all --quiet --std=c++17 --error-exitcode=0 --inline-suppr --force \
        --template='{file}:{line},{severity},{id},{message}' \
        --check-level=exhaustive \
        --suppress=unknownMacro \
        --suppress=shadowFunction \
        --suppress=unusedFunction \
        --suppress=missingIncludeSystem \
        --suppress=unmatchedSuppression
}

lcov_options=(
    --ignore-errors unused
    --ignore-errors mismatch
    --rc lcov_branch_coverage=1
    --rc genhtml_dark_mode=1
    --rc genhtml_branch_coverage=1)

function call_lcov_init {
    cd "${basedir}" || return 1

    local base_output="${basedir}"/lcov_base.info

    lcov "${lcov_options[@]}" --no-external --capture --initial --directory "${basedir}" \
        --output-file "${base_output}" || return 1
}

function call_lcov {
    cd "${basedir}" || return 1

    local output="${basedir}"/lcov.info
    local html_output="${basedir}"/lcovhtml
    local comb_output="${basedir}"/lcov_comb.info
    local base_output="${basedir}"/lcov_base.info
    local test_output="${basedir}"/lcov_test.info

    lcov "${lcov_options[@]}" --no-external --capture --directory "${basedir}" \
        --output-file "${test_output}" || return 1

    lcov "${lcov_options[@]}" --add-tracefile "${base_output}" --add-tracefile "${test_output}" \
        --output-file "${comb_output}" || return 1

    lcov "${lcov_options[@]}" --remove "${comb_output}" '/usr/*' '*/test/*' '*/app/*' '*/build/*' \
        --output-file "${output}" || return 1

    rm -f "${base_output}" "${test_output}" "${comb_output}"

    lcov "${lcov_options[@]}" --list "${output}" || return 1

    genhtml "${lcov_options[@]}" --prefix "${basedir}" --ignore-errors source "${output}" --legend --title "libnano" \
        --output-directory="${html_output}" || return 1
}

function call_codecov {
    cd "${basedir}" || return 1

    local output="${basedir}"/lcov.info

    bash <(curl -s https://codecov.io/bash) -f "${output}" || return 1
}

function call_llvm_cov {
    cd "${basedir}" || return 1

    local output="${basedir}"/llvmcov.info

    libs=$(find "${libnanodir}"/src/lib*.so)
    tests=$(find "${libnanodir}"/test/test_* | grep -v profraw | grep -v profdata)

    objects=()
    for object in ${libs}; do
        objects+=(-object "${object}")
    done
    for utest in ${tests}; do
        objects+=(-object "${utest}")
    done

    readarray -d '' profraws < <(find "${libnanodir}"/test/*.profraw -print0)

    llvm-profdata merge -sparse "${profraws[@]}" -o "${output}"

    llvm-cov show \
        -instr-profile="${output}" \
        -ignore-filename-regex=test/ \
        -format=html -Xdemangler=c++filt -tab-size=4 \
        -show-line-counts --show-branches=count --show-expansions --show-instantiation-summary \
        -output-dir llvmcovhtml \
        "${objects[@]}"

    llvm-cov report \
        -instr-profile="${output}" \
        -ignore-filename-regex=test/ \
        -show-branch-summary --show-instantiation-summary \
        "${objects[@]}"

    llvm-cov show \
        -instr-profile="${output}" \
        -ignore-filename-regex=test/ \
        -format=text -Xdemangler=c++filt -tab-size=4 \
        -show-line-counts --show-branches=count --show-expansions --show-instantiation-summary \
        "${objects[@]}" > "${basedir}"/llvmcov.text
}

function call_memcheck {
    cd "${libnanodir}" || return 1
    echo "-- Using ${threads} threads to test"
    ctest -T memcheck --output-on-failure -j ${threads} || return 1
}

function call_helgrind {
    cd "${libnanodir}" || return 1

    returncode=0
    utests="test/test_core_parallel"
    for utest in ${utests}
    do
        printf "Running helgrind@%s ...\n" "${utest}"
        log=helgrind_${utest/test\//}.log
        if ! valgrind --tool=helgrind --error-exitcode=1 --log-file="${log}" "${utest}"; then
            cat "${log}"
            # NB: ignore for now the warnings reported by helgrind!
            returncode=1
        fi
        printf "\n"
    done

    return ${returncode}
}

function call_clang_tidy {
    cd "${libnanodir}" || return 1

    check=$1

    echo "-- Running clang-tidy-$check"
    log=clang_tidy_${check//\**/}.log
    echo "-- Logging to ${log}"

    wrapper=run-clang-tidy${clang_suffix}
    wrapper=$(which "${wrapper}" || which "${wrapper}".py || which /usr/share/clang/"${wrapper}".py)

    echo "-- Using wrapper ${wrapper}"
    if ! ${wrapper} -p "${libnanodir}" -clang-tidy-binary clang-tidy"${clang_suffix}" \
        -header-filter=.* -checks=-*,"${check}" -quiet -j ${threads} > "$log" 2>&1; then
        cat "${log}"
        return 1
    fi

    grep -E "warning:|error:" < "$log" | grep -oE "[^ ]+$" | sort | uniq -c

    # show log only if any warning or error is detected
    warnings=$(grep -E "warning:|error:" < "$log" | sort -u | grep -c -v /usr/include)
    if [[ $warnings -gt 0 ]]
    then
        grep -E ".*warning:|error:" "$log" | sort -u
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

function call_clang_tidy_concurrency {
    call_clang_tidy "concurrency*"
}

function call_clang_tidy_misc {
    checks="misc*"
    checks="${checks},-misc-non-private-member-variables-in-classes,-misc-include-cleaner"
    call_clang_tidy "${checks}"
}

function call_clang_tidy_cert {
    call_clang_tidy "cert*"
}

function call_clang_tidy_hicpp {
    checks="hicpp*"
    checks="${checks},-hicpp-avoid-c-arrays"
    checks="${checks},-hicpp-no-array-decay"
    checks="${checks},-hicpp-signed-bitwise"
    checks="${checks},-hicpp-named-parameter"
    call_clang_tidy "${checks}"
}

function call_clang_tidy_bugprone {
    checks="bugprone*"
    checks="${checks},-bugprone-easily-swappable-parameters"
    call_clang_tidy "${checks}"
}

function call_clang_tidy_modernize {
    checks="modernize*"
    checks="${checks},-modernize-avoid-c-arrays"
    checks="${checks},-modernize-use-trailing-return-type"
    checks="${checks},-modernize-use-nodiscard"
    call_clang_tidy "${checks}"
}

function call_clang_tidy_performance {
    call_clang_tidy "performance*"
}

function call_clang_tidy_portability {
    call_clang_tidy "portability*"
}

function call_clang_tidy_readability {
    checks="readability*"
    checks="${checks},-readability-magic-numbers"
    checks="${checks},-readability-named-parameter"
    checks="${checks},-readability-isolate-declaration"
    checks="${checks},-readability-else-after-return"
    checks="${checks},-readability-function-cognitive-complexity"
    checks="${checks},-readability-identifier-length"
    checks="${checks},-readability-redundant-member-init"
    checks="${checks},-readability-avoid-nested-conditional-operator"
    call_clang_tidy "${checks}"
}

function call_clang_tidy_clang_analyzer {
    call_clang_tidy "clang-analyzer*"
}

function call_clang_tidy_cppcoreguidelines {
    checks="cppcoreguidelines*"
    checks="${checks},-cppcoreguidelines-avoid-c-arrays"
    checks="${checks},-cppcoreguidelines-avoid-magic-numbers"
    checks="${checks},-cppcoreguidelines-pro-bounds-pointer-arithmetic"
    checks="${checks},-cppcoreguidelines-pro-bounds-array-to-pointer-decay"
    checks="${checks},-cppcoreguidelines-avoid-const-or-ref-data-members"
    call_clang_tidy "${checks}"
}

function call_clang_tidy_all {
    call_clang_tidy_misc || return 1
    call_clang_tidy_cert || return 1
    call_clang_tidy_hicpp || return 1
    call_clang_tidy_bugprone || return 1
    call_clang_tidy_modernize || return 1
    #call_clang_tidy_concurrency || return 1
    call_clang_tidy_performance || return 1
    call_clang_tidy_portability || return 1
    call_clang_tidy_readability || return 1
    call_clang_tidy_clang_analyzer || return 1
    call_clang_tidy_cppcoreguidelines || return 1
}

function call_clang_format {
    files=$(find \
        "${basedir}"/app \
        "${basedir}"/src \
        "${basedir}"/test \
        "${basedir}"/example \
        "${basedir}"/include \
        -type f \( -name "*.h" -o -name "*.cpp" \))

    cmd=clang-format${clang_suffix}
    echo "-- Using ${cmd}..."

    log="${basedir}"/clang_format.log
    rm -f "${log}"

    for file in ${files}; do
        ${cmd} --dry-run "${file}" >> "${log}" 2>&1
    done

    cat "${log}"

    changes=$(wc -l < "$log")
    rm -f "${log}"

    for file in ${files}; do
        "${cmd}" -i "${file}"
    done

    if [[ ${changes} -gt 0 ]]; then
        return 1
    else
        return 0
    fi
}

function call_sonar {
    cd "${basedir}" || return 1

    export SONAR_SCANNER_VERSION=7.3.0.5189
    export SONAR_SCANNER_HOME="$HOME"/.sonar/sonar-scanner-$SONAR_SCANNER_VERSION-linux-x64
    curl --create-dirs -sSLo "$HOME"/.sonar/sonar-scanner.zip \
        https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-$SONAR_SCANNER_VERSION-linux-x64.zip
    unzip -o "$HOME"/.sonar/sonar-scanner.zip -d "$HOME"/.sonar/
    export PATH=$SONAR_SCANNER_HOME/bin:$PATH
    export SONAR_SCANNER_OPTS="-server"

    curl --create-dirs -sSLo "$HOME"/.sonar/build-wrapper-linux-x86.zip \
        https://sonarcloud.io/static/cpp/build-wrapper-linux-x86.zip
    unzip -o "$HOME"/.sonar/build-wrapper-linux-x86.zip -d "$HOME"/.sonar/
    export PATH="$HOME"/.sonar/build-wrapper-linux-x86:$PATH

    sonar-scanner \
        -Dsonar.organization=accosmin \
        -Dsonar.projectKey=libnano \
        -Dsonar.sources="${basedir}"/src,"${basedir}"/include/nano,"${basedir}"/app,"${basedir}"/example \
        -Dsonar.projectVersion=0.0.1 \
        -Dsonar.python.version=3 \
        -Dsonar.cfamily.compile-commands="${libnanodir}"/compile_commands.json \
        -Dsonar.cfamily.llvm-cov.reportPath="${basedir}"/llvmcov.text \
        -Dsonar.sourceEncoding=UTF-8 \
        -Dsonar.host.url=https://sonarcloud.io
}

function call_shellcheck {
    cd "${basedir}" || return 1

    scripts=$(find . -type f -name "*.sh")
    for script in ${scripts}; do
        echo "-- Checking bash script: ${script}"
        shellcheck "${script}" || return 1
    done
}

function check_source_files {
    cd "${basedir}" || return 1

    returncode=0

    # NB: the headers in the public interface should include only files
    # from external libraries and own public interface.
    filenames=$(find include/nano -type f -name "*.h")
    for filename in ${filenames}; do
        includes=$(grep "#include <nano" "${filename}" | cut -d ' ' -f 2)
        for include in ${includes}; do
            include=${include/</}
            include=${include/>/}
            incpath=include/${include}
            if [ ! "${include}" = "nano/version.h" ] && [ ! -f "${incpath}" ]; then
                echo -n "-- Error: found include '${include}' from the library interface '${filename}' "
                echo "which is not in 'include/nano'!"
                returncode=1
            fi
        done
    done

    # NB: the interface header files should be included exactly once in the CMakeLists.txt!
    filenames=$(find include -type f -name "*.cpp" -o -name "*.h")
    for filename in ${filenames}; do
        count=$(find src -type f -name CMakeLists.txt -exec grep "${filename/\./\\\.}" {} \; | wc -l)

        if [ ! "${count}" = 1 ]; then
            echo -n "-- Error: found library interface '${filename}' "
            echo "which should be referenced exactly once in 'src/CMakeLists.txt', got ${count} times instead!"
            returncode=1
        fi
    done

    # NB: the implementation files should be included exactly once in the CMakeLists.txt!
    filenames=$(find src -type f -name "*.cpp" -o -name "*.h")
    for filename in ${filenames}; do
        count=$(grep -c "$(basename "${filename}")" "$(dirname "${filename}")"/CMakeLists.txt)

        if [ "${count}" = 0 ]; then
            echo -n "-- Error: found source '${filename}' "
            echo "which should be referenced exactly once in 'src/*/CMakeLists.txt', got ${count} times instead!"
            returncode=1
        fi
    done

    # NB: the test source files should be included exactly once in the CMakeLists.txt!
    filenames=$(find test -type f -name "*.cpp")
    for filename in ${filenames}; do
        filename=${filename/test\//}
        filename=${filename/\.cpp/}
        count=$(grep -c make_test\("${filename}"\ NANO "${basedir}"/test/CMakeLists.txt)
        if [ ! "${count}" = 1 ]; then
            echo -n "-- Error: found test '${filename}' "
            echo "which should be referenced exactly once in 'test/CMakeLists.txt', got ${count} times instead!"
            returncode=1
        fi
    done

    return ${returncode}
}

function check_markdown_docs {
    cd "${basedir}" || return 1

    returncode=0

    docfiles=$(find "${basedir}" -type f -name "*.md")
    for docfile in ${docfiles}; do
        echo "-- Checking documentation file: ${docfile}"

        # check local links [linkname](filename) that point to existing files
        lines=$(grep -E "\[.+\]\(.+\)" "${docfile}" | grep -v "http")
        while read -r line; do
            for token in ${line}; do
                filename=$(echo "${token}" | grep -oP "\]\(.+\)")
                if [ -n "${filename}" ]; then
                    filename=${filename//\]/}
                    filename=${filename//\(/}
                    filename=${filename//\)/}
                    filename="$(dirname "${docfile}")"/"${filename}"
                    if [ ! -d "${filename}" ] && [ ! -f "${filename}" ]; then
                        echo "-- Error: invalid reference path '${filename}'!"
                        returncode=1
                    fi
                fi
            done
        done <<< "${lines}"

        # check example codes with C++ includes that point to existing files
        lines=$(grep -E "#include <.+>" "${docfile}")
        while read -r line; do
            for token in ${line}; do
                filename=$(echo "${token}" | grep -oP "\<.+\>" | grep -v "<iostream>")
                if [ -n "${filename}" ]; then
                    filename=${filename//\</}
                    filename=${filename//\>/}
                    filename=include/${filename}
                    if [ ! -f "${filename}" ]; then
                        echo "-- Error: invalid C++ include '${filename}'!"
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
    --gcc
        setup g++ compiler
    --clang
        setup clang++ compiler
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
    --no-werror
        disable treating compilation warnings as errors
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
    --shellcheck
        check bash scripts with shellcheck
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
        --gcc)                          setup_gcc;;
        --clang)                        setup_clang;;
        --lld)                          setup_lld;;
        --lto)                          setup_lto;;
        --thinlto)                      setup_thinlto;;
        --asan)                         setup_asan;;
        --lsan)                         setup_lsan;;
        --usan)                         setup_usan;;
        --tsan)                         setup_tsan;;
        --msan)                         setup_msan;;
        --gold)                         setup_gold;;
        --no-werror)                    setup_no_werror;;
        --native)                       setup_native;;
        --libcpp)                       setup_libcpp;;
        --coverage)                     setup_coverage;;
        --llvm-coverage)                setup_llvm_coverage;;
        --suffix)                       shift; setup_suffix "$1";;
        --config)                       call_config || exit 1;;
        --build)                        call_build || exit 1;;
        --test)                         call_test || exit 1;;
        --install)                      call_install || exit 1;;
        --cppcheck)                     call_cppcheck || exit 1;;
        --lcov)                         call_lcov || exit 1;;
        --lcov-init)                    call_lcov_init || exit 1;;
        --llvm-cov)                     call_llvm_cov || exit 1;;
        --memcheck)                     call_memcheck || exit 1;;
        --helgrind)                     call_helgrind || exit 1;;
        --clang-suffix)                 shift; clang_suffix="$1";;
        --clang-tidy-check)             shift; call_clang_tidy "$1" || exit 1;;
        --clang-tidy-all)               call_clang_tidy_all || exit 1;;
        --clang-tidy-misc)              call_clang_tidy_misc || exit 1;;
        --clang-tidy-cert)              call_clang_tidy_cert || exit 1;;
        --clang-tidy-hicpp)             call_clang_tidy_hicpp || exit 1;;
        --clang-tidy-bugprone)          call_clang_tidy_bugprone || exit 1;;
        --clang-tidy-modernize)         call_clang_tidy_modernize || exit 1;;
        --clang-tidy-concurrency)       call_clang_tidy_concurrency || exit 1;;
        --clang-tidy-performance)       call_clang_tidy_performance || exit 1;;
        --clang-tidy-portability)       call_clang_tidy_portability || exit 1;;
        --clang-tidy-readability)       call_clang_tidy_readability || exit 1;;
        --clang-tidy-clang-analyzer)    call_clang_tidy_clang_analyzer || exit 1;;
        --clang-tidy-cppcoreguidelines) call_clang_tidy_cppcoreguidelines || exit 1;;
        --build-example)                call_example || exit 1;;
        --clang-format)                 call_clang_format || exit 1;;
        --sonar)                        call_sonar || exit 1;;
        --codecov)                      call_codecov || exit 1;;
        --shellcheck)                   call_shellcheck || exit 1;;
        --check-markdown-docs)          check_markdown_docs || exit 1;;
        --check-source-files)           check_source_files || exit 1;;
        -D*)                            cmake_options+=("$1");;
        -G*)                            cmake_options+=("$1");;
        *)                              echo "unrecognized option $1"; echo; usage;;
    esac
    shift
done

exit 0
