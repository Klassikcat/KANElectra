#!/bin/bash

set -e -u -o pipefail

apt-get update && apt-get install -y $(cat .devcontainer/packages.txt)
pip install uv && uv sync

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
    architecture="amd64"
elif [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
    architecture="arm64"
else
    echo "Your CPU architecture is not supported. exit."
    exit 1
fi

git config --global core.editor 'vim'
