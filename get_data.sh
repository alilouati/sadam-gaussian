#!/usr/bin/env bash
# Downloads the two public datasets into ./data (about 250 MB).
set -e
cd "$(dirname "$0")"; mkdir -p data; cd data
[ -d nndl ] || git clone --depth 1 https://github.com/mnielsen/neural-networks-and-deep-learning.git nndl   # MNIST: nndl/data/mnist.pkl.gz
[ -d cifar_imgs ] || git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images.git cifar_imgs           # CIFAR-10 as JPEG
echo "Data ready in $(pwd)"
