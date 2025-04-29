#!/usr/bin/env bash

# TODO: run the given arguments

podman run --userns=keep-id \
    -v $(pwd):/code libnano-alpine \
    ls -la code
