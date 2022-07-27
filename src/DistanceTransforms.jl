module DistanceTransforms

using ImageMorphology
using CUDA
using FLoops
using FoldsCUDA

include("./chamfer.jl")
include("./euclidean.jl")
include("./squared_euclidean.jl")
include("./utils.jl")
include("./wenbo.jl")

export transform,
    transform!,

    # Export chamfer.jl functions
    Chamfer,

    # Export euclidean.jl functions
    euclidean,

    # Export squared_euclidean.jl functions
    SquaredEuclidean,

    # Export utils.jl functions
    boolean_indicator,

    # Export utils.jl functions
    Wenbo
end
