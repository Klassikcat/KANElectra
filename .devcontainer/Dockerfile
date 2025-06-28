ARG IMAGE=
FROM ${IMAGE}

COPY .devcontainer/packages.txt /tmp/packages.txt

RUN (type -p wget >/dev/null || (apt update && apt install wget -y)) \
        && mkdir -p -m 755 /etc/apt/keyrings \
            && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
            && cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
        && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
        && apt update \
        && apt-get install -y $(cat /tmp/packages.txt)

RUN usermod -l vscode ubuntu && \
    usermod -d /home/vscode -m vscode && \
    usermod -aG sudo vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY --chown=vscode:vscode .devcontainer/init /usr/local/bin/init
RUN chmod +x /usr/local/bin/init

USER vscode
