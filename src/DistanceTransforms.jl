module DistanceTransforms

using ImageMorphology
using CUDA
using FLoops
using FoldsCUDA

include("./chamfer.jl")
include("./euclidean_distance_transform.jl")
include("./squared_euclidean.jl")
include("./utils.jl")

export
    # Export chamfer.jl functions
    Chamfer,
    transform,

    # Export euclidean_distance_transform.jl functions
    euclidean_distance_transform,

    # Export squared_euclidean.jl functions
    transform,
    transform!
    SquaredEuclidean,

    # Export utils.jl functions
    boolean_indicator
end
