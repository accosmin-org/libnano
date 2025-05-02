#!/usr/bin/env bash

ALPINE_VERSION=3.21
UBUNTU_VERSION=24.04

podman build \
    -f docker/Alpine.Dockerfile \
    --build-arg ALPINE_VERSION=${ALPINE_VERSION} \
    -t libnano-dev:alpine-${ALPINE_VERSION} \
    .

podman build \
    -f docker/Ubuntu.Dockerfile \
    --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
    -t libnano-dev:ubuntu-${UBUNTU_VERSION} \
    .
