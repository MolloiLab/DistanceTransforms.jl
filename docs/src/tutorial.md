# Tutorial

## Vanilla Squared Euclidean Distance Transform
The `SquaredEuclidean(x)` struct requires a few key components. First one must determine whether the 
array `x` that is being passed in, is a boolean indicator or not. This means, is `x` ones and zeros, and
do these values correspond to foreground (one) and background (zero). If so, `x` must first be run through
the `boolean_indicator` function.

After this, all one needs to do is instantiate the `SquaredEuclidean` function by passing the array `x` 
through it, and then call `transform(x, tfm)`

<!-- ```jldoctest
julia> using DistanceTransforms
julia>  x = [
            0 1 1 1 0
            1 1 1 1 1
            1 0 0 0 1
            1 0 0 0 1
            1 0 0 0 1
            1 1 1 1 1
            0 1 1 1 0
            ]
julia>  x = boolean_indicator(x)
julia>  tfm = SquaredEuclidean(x)
julia>  transform(x, tfm)
``` -->