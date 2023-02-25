module DistanceTransforms
using PythonCall

const scipy = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(scipy, pyimport("scipy"))
end

export scipy

using ImageMorphology
using CUDA
using FLoops
using FoldsCUDA

include("./borgefors.jl")
include("./felzenszwalb.jl")
include("./maurer.jl")
include("./scipy.jl")
include("./utils.jl")
include("./wenbo.jl")

export DistanceTransform,
    transform,
    transform!,

    # Export chamfer.jl functions
    Borgefors,

    # Export maurer.jl functions
    Maurer,

    # Export felzenszwalb.jl functions
    Felzenszwalb,

    # Export Scipy.jl functions
    Scipy,

    # Export utils.jl functions
    boolean_indicator,

    # Export wenbo.jl functions
    Wenbo
end
