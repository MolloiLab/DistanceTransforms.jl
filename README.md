# DistanceTransforms

[![Glass Notebook](https://img.shields.io/badge/Docs-Glass%20Notebook-aquamarine.svg)](https://glassnotebook.io/r/DxnIPJnIqpEqiQnJgqiBP/index.jl)
[![CI Stable](https://github.com/MolloiLab/DistanceTransforms.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/MolloiLab/DistanceTransforms.jl/actions/workflows/CI.yml)
[![Build status][buildkite-img]][buildkite-url]
[![CI Nightly](https://github.com/MolloiLab/DistanceTransforms.jl/actions/workflows/Nightly.yml/badge.svg?branch=master)](https://github.com/MolloiLab/DistanceTransforms.jl/actions/workflows/Nightly.yml)
[![Coverage](https://codecov.io/gh/MolloiLab/DistanceTransforms.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MolloiLab/DistanceTransforms.jl)

DistanceTransforms.jl is a Julia package that provides efficient distance transform operations on arrays.

## Table of Contents

- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Transforms](#transforms)
- [Advanced Usage](#advanced-usage)
- [Python](#python)

## Getting Started

To get started with DistanceTransforms.jl, you'll first need to import the package:

```julia
using DistanceTransforms
```

The most up-to-date version of DistanceTransforms.jl can be found on the main/master branch of the [GitHub repository](https://github.com/Dale-Black/DistanceTransforms.jl). If you're using an unregistered version, you may need to add the package explicitly.

For detailed documentation and tutorials, you can refer to the [official notebook](https://glassnotebook.io/r/DxnIPJnIqpEqiQnJgqiBP/index.jl).

## Quick Start

Distance transforms are essential for many computer vision-related tasks. With DistanceTransforms.jl, you can easily apply efficient distance transform operations on arrays in Julia.

For example, to use the quintessential distance transform operation:

```julia
using DistanceTransforms

arr = [
    0 1 1 0
    0 0 0 0
    1 1 0 0
]

result = transform(boolean_indicator(arr))
```

## Transforms

The library is built around a common `transform` interface, allowing users to apply various distance transform algorithms to arrays using a unified approach.

## Advanced Usage

DistanceTransforms.jl offers advanced features such as multi-threading and GPU acceleration. These capabilities significantly enhance performance, especially for large data sets and high-resolution images.

### Multi-threading

DistanceTransforms.jl efficiently utilizes multi-threading, particularly in its Felzenszwalb distance transform algorithm. This parallelization improves performance for large data sets and high-resolution images.

```julia
x = boolean_indicator(rand([0f0, 1f0], 100, 100))
single_threaded = @benchmark transform($x; threaded = false)
multi_threaded = @benchmark transform($x; threaded = true)
```

### GPU Acceleration

DistanceTransforms.jl extends its performance capabilities by embracing GPU acceleration. The same `transform` function used for CPU computations automatically adapts to leverage GPU resources when available.

```julia
x_gpu = CUDA.CuArray(boolean_indicator(rand([0, 1], 100, 100)))
gpu_transformed = transform(x_gpu)
```

For benchmarks and more details on advanced usage, refer to the [advanced usage notebook](https://glassnotebook.io/r/DxnIPJnIqpEqiQnJgqiBP/index.jl).

## Python

Check out the corresponding Python (wrapper) package: [py-distance-transforms](https://github.com/MolloiLab/py-distance-transforms)

[buildkite-img]: https://badge.buildkite.com/1509baa1122772e8ec377463a6c188753d35b8fcec300a658e.svg?branch=master
[buildkite-url]: https://buildkite.com/julialang/distancetransforms-dot-jl