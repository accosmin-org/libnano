#!/usr/bin/env bash

echo "$@"

export HOME=/home/cosmin/

perf record -F 100 -a -g "$@"

perf script | ./stackcollapse-perf.pl > out.perf-folded
./flamegraph.pl out.perf-folded > perf.svg
