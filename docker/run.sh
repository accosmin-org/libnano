#!/usr/bin/env bash

podman run \
    --userns=keep-id --security-opt label=disable \
    -v $(pwd):/code -w /code libnano-alpine \
    "$@"
