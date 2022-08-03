#!/usr/bin/env bash

brew update
#brew upgrade
brew ls --versions cmake || brew install -f cmake
brew ls --versions eigen || brew install -f eigen
brew ls --versions coreutils || brew install -f coreutils
