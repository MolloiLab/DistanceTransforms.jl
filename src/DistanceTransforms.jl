module DistanceTransforms

using ImageEdgeDetection
using ImageMorphology
using Statistics
using StatsBase
using Distances
using CUDA
using FLoops
using FoldsCUDA

include("./chamfer.jl")
include("./euclidean_distance_transform.jl")
include("./losses.jl")
include("./metrics.jl")
include("./squared_euclidean.jl")
include("./utils.jl")

export
    # Export chamfer.jl functions
    Chamfer,
    transform,

    # Export euclidean_distance_transform.jl functions
    euclidean_distance_transform,

    # Export losses.jl functions
    dice_loss,
    hd_loss,
    # dice_lossP,
    # hd_lossP,

    # Export metrics.jl functions
    dice_metric,
    mean_hausdorff,
    percentile_hausdorff,
    mean_hausdorff_2D,

    # Export squared_euclidean.jl functions
    transform,
    transform!
    SquaredEuclidean,

    # Export utils.jl functions
    euc,
    find_edges,
    detect_edges_3D,
    boolean_indicator
end
