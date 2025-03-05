# DistanceTransforms

[![Documentation][docs-img]][docs-url]
[![CI Stable][ci-img]][ci-url]
[![Build status][buildkite-img]][buildkite-url]
[![Coverage][cov-img]][cov-url]

DistanceTransforms.jl is a Julia package that provides efficient distance transform operations on arrays (CPU & GPU).

## Installation

```julia
using Pkg
Pkg.add("DistanceTransforms")
```

## Quick Start

```julia
using DistanceTransforms

# Create a simple binary array
arr = [
    0 1 1 0
    0 0 0 0
    1 1 0 0
]

# Apply distance transform
result = transform(boolean_indicator(arr))
```

## Features

- **Fast CPU implementation** using the Felzenszwalb algorithm
- **Multi-threading support** for enhanced CPU performance
- **GPU acceleration** for NVIDIA (CUDA), AMD (ROCm), and Apple (Metal)
- **Simple API** with unified transform interface
- **Multi-dimensional support** for 1D, 2D, and 3D arrays

## Advanced Usage

### Multi-threading

```julia
# Compare single vs multi-threaded performance
result = transform(arr; threaded = true)  # default
result = transform(arr; threaded = false) # single-threaded
```

### GPU Acceleration

```julia
using CUDA
# Automatically uses GPU implementation for CUDA arrays
x_gpu = CUDA.CuArray(boolean_indicator(rand([0, 1], 100, 100)))
gpu_result = transform(x_gpu)
```

## Python Support

For Python users, check out [distance_transforms](https://github.com/MolloiLab/py-distance-transforms), a Python wrapper providing the same functionality.

```python
import numpy as np
import distance_transforms as dts

arr = np.random.choice([0, 1], size=(10, 10)).astype(np.float32)
result = dts.transform(arr)
```

## Documentation

For comprehensive documentation and examples, visit our [documentation site](https://molloilab.github.io/DistanceTransforms.jl/).

[docs-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-url]: https://molloilab.github.io/DistanceTransforms.jl/

[ci-img]: https://github.com/MolloiLab/DistanceTransforms.jl/actions/workflows/CI.yml/badge.svg?branch=master
[ci-url]: https://github.com/MolloiLab/DistanceTransforms.jl/actions/workflows/CI.yml

[buildkite-img]: https://badge.buildkite.com/1509baa1122772e8ec377463a6c188753d35b8fcec300a658e.svg?branch=master
[buildkite-url]: https://buildkite.com/julialang/distancetransforms-dot-jl

[cov-img]: https://codecov.io/gh/MolloiLab/DistanceTransforms.jl/branch/master/graph/badge.svg
[cov-url]: https://codecov.io/gh/MolloiLab/DistanceTransforms.jl