#!/usr/bin/env bash

podman build -f docker/Alpine.Dockerfile \
    -t libnano-alpine:latest
