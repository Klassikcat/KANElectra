ARG VERSION=dev-3.13-bookworm
FROM mcr.microsoft.com/devcontainers/python:${VERSION}

# install packages
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        git \
        gh \
        vim \
        apt-transport-https \
        ca-certificates \
        curl \
        g++
RUN pip install uv && uv sync
