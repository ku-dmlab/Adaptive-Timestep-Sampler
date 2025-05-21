# Adaptive Non-Uniform Timestep Sampling for Diffusion Model Training
This repository contains the official implementation of [(CVPR 2025) Adaptive Non-Uniform Timestep Sampling for Diffusion Model Training](https://arxiv.org/abs/2411.09998) by Myunsoo Kim*, Donghyeon Ki*, Seong-Woong Shim, and Byung-Jun Lee.

# How to run code
### Train
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset cifar10 --num-gpus 2 --distributed --rigid-launch --eval
```
We recommend using 2 GPUs to reproduce the results reported in the paper, 

While training with more GPUs is also possible, the optimal hyperparameters may differ from those reported in the paper.

### Sample
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python generate.py --dataset cifar10 --chkpt-path ./chkpts/cifar10/cifar10_2040.pt --suffix _ddpm --num-gpus 4
```
### Evaluate FID
```
python eval.py --dataset cifar10 --sample-folder ./images/eval/cifar10/cifar10_2040
```

# Acknowledgements
Our code implementation is largely borrowed from [ddpm-torch](https://github.com/tqch/ddpm-torch).



---
