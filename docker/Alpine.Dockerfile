ARG ALPINE_VERSION=3.21

FROM alpine:${ALPINE_VERSION}

RUN apk update

RUN apk add --no-cache bash cmake git eigen-dev grep

RUN apk add --no-cache ccache samurai g++ clang clang-extra-tools valgrind cppcheck

CMD bash
