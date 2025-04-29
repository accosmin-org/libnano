ARG ALPINE_VERSION=3.21

FROM alpine:${ALPINE_VERSION}

RUN apk add --no-cache bash gcc

CMD bash
