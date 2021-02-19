module DistanceTransforms

using ImageEdgeDetection
using Images
using Distances
using Statistics
using Tullio

include("./losses.jl")
include("./metrics.jl")
include("./utils.jl")

export 
    # Export losses.jl functions
    dice_loss,
    hd_loss,

    # Export metrics.jl functions
    mean_hausdorff_2D,

    # Export utils.jl functions
    detect_edges_3D,
    compute_dtm,

end
