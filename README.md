# resamcupy

100% copy of [resampy](https://github.com/bmcfee/resampy), but faster.

This package implements resampy on cuda, so it can leverage the GPU power and run even faster, especially on very large data.


# Documentation

Resamcupy provide the same interface as resampy, but it can accept not only `numpy.ndarray` but `cupy.ndarray`.

Other option please refer to [resampy documentation](https://resampy.readthedocs.io/en/master/).

# Installation

1. Make sure you have a [NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) and [CUDA Toolkit](https://developer.nvidia.com/cuda-zone) installed.

2. Install [CuPy](https://docs-cupy.chainer.org/en/stable/install.html):

```
pip install cupy
```

3. Clone this repo and install:
```
pip install git+https://github.com/yoyololicon/resamcupy
```