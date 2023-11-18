# DistanceTransforms

[![Glass Notebook](https://img.shields.io/badge/Docs-Glass%20Notebook-aquamarine.svg)](https://glassnotebook.io/r/DxnIPJnIqpEqiQnJgqiBP/index.jl)
[![CI Stable](https://github.com/Dale-Black/DistanceTransforms.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/Dale-Black/DistanceTransforms.jl/actions/workflows/CI.yml)
[![CI Nightly](https://github.com/Dale-Black/DistanceTransforms.jl/actions/workflows/Nightly.yml/badge.svg?branch=master)](https://github.com/Dale-Black/DistanceTransforms.jl/actions/workflows/Nightly.yml)
[![Coverage](https://codecov.io/gh/Dale-Black/DistanceTransforms.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Dale-Black/DistanceTransforms.jl)


DistanceTransforms.jl is a Julia package that provides efficient distance transform operations on arrays.

## Table of Contents

- [Getting Started](#getting-started)
- [Quick Start](#quick-start)
- [Transforms](#transforms)
- [Python](#python)

## Getting Started

To get started with DistanceTransforms.jl, you'll first need to import the package:

```julia
using DistanceTransforms
```

The most up-to-date version of DistanceTransforms.jl can be found on the main/master branch of the [GitHub repository](https://github.com/Dale-Black/DistanceTransforms.jl). If you're using an unregistered version, you may need to add the package explicitly.

For detailed documentation and tutorials, you can refer to the [official notebook](#).

## Quick Start

Distance transforms are essential for many computer vision-related tasks. With DistanceTransforms.jl, you can easily apply efficient distance transform operations on arrays in Julia.

For example, to use the quintessential distance transform operation:

```julia
using DistanceTransforms

array1 = [
    0 1 1 0
    0 0 0 0
    1 1 0 0
]

result = transform(array1, Maurer())
```

## Transforms

The library is built around a common `transform` interface, allowing users to apply various distance transform algorithms to arrays using a unified approach.

## Python

Check out the Colab notebook to see how to utilize this distance transform in Python
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-CDqQgrBHoxNqs2IbMebMRxsp0m21jSa?usp=sharing]

