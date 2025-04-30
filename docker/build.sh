#!/usr/bin/env bash

ALPINE_VERSION=3.21

podman build \
    -f docker/Alpine.Dockerfile \
    --build-arg ALPINE_VERSION=${ALPINE_VERSION} \
    -t libnano-dev:alpine-${ALPINE_VERSION} \
    .
