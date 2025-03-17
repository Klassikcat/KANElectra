#!/bin/bash

set -e -u -o pipefail

install_apt_requirements() {
    local packages_path=$1
    if ! command -v sudo > /dev/null 2>&1; then
        echo "sudo is not available. Install mock sudo command"
        apt-get install sudo
    fi

    sudo apt-get update
    sudo apt-get install -y $(cat ./.devcontainer/$1)
}

install_python_requirements() {
    local requirements_path=${1:-"requirements.txt"}
    local package_manager=${2:-"uv"}
    if [ "$package_manager" == "uv" ]; then
        pip install uv
        package_manager="uv pip"
        /bin/bash -c "$package_manager install -r $requirements_path --system"
    else
        /bin/bash -c "$package_manager install -r $requirements_path"
    fi
}

main() {
    install_apt_requirements "packages.txt"
    install_python_requirements "requirements.txt" "uv"
    nvidia-smi
}
