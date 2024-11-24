FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04
WORKDIR ./install
COPY ./ ./
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install ./

WORKDIR scripts
copy ./scripts/pretraining ./
