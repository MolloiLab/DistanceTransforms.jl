using DistanceTransforms
using Test
using TestSetExtensions
using CUDA

using DistanceTransforms:
    dice_loss,
    hd_loss,
    dice_metric,
    mean_hausdorff,
    percentile_hausdorff,
    mean_hausdorff_2D,
    euc,
    find_edges,
    detect_edges_3D,
    boolean_indicator,
    SquaredEuclidean
