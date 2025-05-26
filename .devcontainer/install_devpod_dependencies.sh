#!/bin/bash

set -e -u -o pipefail

pip install uv && uv sync

# Install kubernetes client
curl -fsSLo /etc/apt/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | tee /etc/apt/sources.list.d/kubernetes.list
apt-get install -y kubectl

curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
chmod 700 get_helm.sh && \
./get_helm.sh && \
rm get_helm.sh

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

