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

### Data

### Pretraining Details

## Requirements

```text
python 3.8 or higher
torch >= 2.2.0
torch-tensorRT
hydra
lightning
```

## Benchmark

### SQuAD 2.0

TODO

### GLUE

## CLI

### Train

```shell
```

### Test

## TensorRT Convert

```shell
```

## Appendix

**original paper**: arXiv:2404.19756v1