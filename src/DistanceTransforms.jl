module DistanceTransforms

using ImageMorphology
using CUDA
using FLoops
using FoldsCUDA

include("./chamfer.jl")
include("./maurer.jl")
include("./felzenszwalb.jl")
include("./utils.jl")
include("./wenbo.jl")

export DistanceTransform,
    transform,
    transform!,

    # Export chamfer.jl functions
    Chamfer,

    # Export maurer.jl functions
    Maurer,

    # Export felzenszwalb.jl functions
    Felzenszwalb,

    # Export utils.jl functions
    boolean_indicator,

    # Export wenbo.jl functions
    Wenbo
end
