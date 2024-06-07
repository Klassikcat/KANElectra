# ElectraModel based on Kolmogorov–Arnold Network

## Introduction

Recently, Kolmogorov–Arnold Networks (KANs) have been introduced as a replacement for Fully Connected Layers.
According to the authors of the paper, KANs have a significant advantage in terms of speed and performance compared to traditional FC layers.

They are currently being actively applied in vision-based models and were recently applied to Transformer decoder models as well.
It has been reported that this has resulted in a substantial improvement in performance.

Despite the verification of KANs' performance in various areas, I have not yet found examples of their application to the Transformer encoder part. 
In this repository, we aim to verify whether KANs can be applied to Transformer encoder models and to evaluate their performance.

## Download link

TODO

## About KAN Electra

### Hyperparameters

### Vocab

I use Google's `google/electra-small-discriminator` for English model, `monologg/koelectra-base-v3-discriminator` for korean model. to train KANElectra model for pretraining.

### Data

### Pretraining Details

## Requirements

```text
python 3.8 or higher
torch >= 2.2.0
torch-tensorRT
hydra
lightning
transformers
```

## Benchmark

### SQuAD 2.0

TODO

### GLUE

## CLI

### Train

ElectraKAN uses Hydra from Meta Inc to simplify configuring parameters and Easy-to-plug parameters in CLI. You can train ElectraKAN with your own code by using Hydra's own syntax here ss an example of training Electra Language model using generator and discriminator.

```shell
python scripts/train.py \
    
```

### Test

```shell
python scripts/test.py \

```

## TensorRT Convert

There are two ways to convert your model into TensorRT engine. first one is using TensorRT CLI using trtexec shell command. You need to compile your model into onnx engine first.

You can replace "512" in minShape, optShape, maxShape into max_length you specified.
```shell
trtexec \
    --onnx=${your_onnx_engine_path} \
    --saveEngine=${engine_save_path} \
    --minShape=1x512,1x512,1x512 \
    --optShape=${opt_batch_size}x512,${opt_batch_size}x512,${opt_batch_size}x512 \
    --maxShape=${max_batch_size}x512,${max_batch_size}x512,${max_batch_size}x512 
```

The second one is specifying "TensorRTCompiler" on your hydra's configuration file. Here are two examples of how to specify them.

#### 1. Specify on your yaml file.

```yaml
```

#### 2. Using Hydra CLI

```shell
```
