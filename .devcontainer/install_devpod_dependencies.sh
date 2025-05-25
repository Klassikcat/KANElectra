#!/bin/bash

set -e -u -o pipefail

arch=$(uname -m)
if [ "$arch" = "x86_64" ]; then
    architecture="amd64"
elif [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then
    architecture="arm64"
else
    echo "Your CPU architecture is not supported. exit."
    exit 1
fi

curl "https://awscli.amazonaws.com/awscli-exe-linux-$architecture.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install
rm awscliv2.zip
rm -rf aws

git config --global core.editor 'vim'
echo 'export EDITOR=vim' >> ~/.bashrc
echo 'export VISUAL=vim' >> ~/.bashrc
. $HOME/.bashrc

nvidia-smi
