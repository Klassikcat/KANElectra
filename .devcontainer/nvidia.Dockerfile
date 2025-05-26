FROM glcr.b-data.ch/jupyterlab/cuda/python/base:3-devtools-docker-root

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

# Install kubernetes client
RUN curl -fsSLo /etc/apt/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg && \
        echo "deb [signed-by=/etc/apt/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main" | \
        tee /etc/apt/sources.list.d/kubernetes.list && \
        apt-get install -y kubectl

RUN curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 && \
    chmod 700 get_helm.sh && \
    ./get_helm.sh && \
    rm get_helm.sh
