# CUDA simulated quantum annealing algorithm

Used OS: Ubuntu 20.04.3 LTS

kernel: Linux 5.13.0-51-generic

Used image version: nvidia/cuda:10.0-devel-ubuntu18.04

command used to create container: 
```
docker run -it --gpus all nvidia/cuda:10.0-devel-ubuntu18.04 bash
```

The implementation of the algorithm is not yet finished.

## simulatedQA_binary.cu

the posible working code, still under implementation.

## simulatedQA_binary.c

protytype of the .cu file

## simulatedQA.cu

sqa for spin. energy function still need to be fixed.

# concerns
1. how to initialize trotter?
2. what if previous and/or following trotter is out-of-bound?
3. which trotter to select as final result?
4. what value should the hyperparameter "K" be?