module DistanceTransforms

abstract type DistanceTransform end

export DistanceTransform

include("./felzenszwalb.jl")
include("./maurer.jl")
include("./utils.jl")

end
