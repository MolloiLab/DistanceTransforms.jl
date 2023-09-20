module DistanceTransforms

abstract type DistanceTransform end

include("./felzenszwalb.jl")
include("./maurer.jl")
include("./utils.jl")

export DistanceTransform,
    transform,
    transform!,

    # Export maurer.jl functions
    Maurer,

    # Export felzenszwalb.jl functions
    Felzenszwalb,

    # Export utils.jl functions
    boolean_indicator
end
