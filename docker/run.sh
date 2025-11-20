#!/usr/bin/env bash

# ALPINE_VERSION=3.21
UBUNTU_VERSION=24.04

# podman run \
#    --userns=keep-id --security-opt label=disable \
#    -v $(pwd):/code -w /code \
#    libnano-dev:alpine-${ALPINE_VERSION} \
#    "$@"

podman run \
    --userns=keep-id --security-opt label=disable \
    -v "$(pwd)":/code -w /code \
    libnano-dev:ubuntu-${UBUNTU_VERSION} \
    "$@"
