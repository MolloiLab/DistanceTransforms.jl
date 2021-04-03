module DistanceTransforms

using ImageEdgeDetection
using ImageMorphology
using Tullio
using Statistics
using Distances

include("./losses.jl")
include("./metrics.jl")
include("./utils.jl")

export 
    # Export losses.jl functions
    dice_loss,
    hd_loss,
    dice_lossP,
    hd_lossP,

    # Export metrics.jl functions
    dice_metric,
    mean_hausdorff,
    mean_hausdorff_2D,

    # Export utils.jl functions
    euc,
    find_edges,
    detect_edges_3D,
    compute_dtm

end
