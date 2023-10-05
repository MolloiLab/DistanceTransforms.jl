module DistanceTransforms

abstract type DistanceTransform end

include("./felzenszwalb.jl")
include("./maurer.jl")
include("./utils.jl")

export transform

end
