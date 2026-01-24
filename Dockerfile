FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 as builder
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install uv && \
    uv sync --no-dev && \
    uv pip install -e .

FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY ./scripts /scripts

RUN chmod +x /scripts/pretraining/train.py
# uncomment this to run the script without "uv python /scripts/pretraining/train.py" option
# CMD ["/scripts/pretraining/train.py"]