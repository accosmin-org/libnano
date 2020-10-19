#!/usr/bin/env bash

bash scripts/build.sh --suffix debug --build-type Debug \
    --generator Ninja --config --build --test --install --build-example

bash scripts/build.sh --suffix release --build-type Release --native \
    --generator Ninja --config --build --test --install --build-example
