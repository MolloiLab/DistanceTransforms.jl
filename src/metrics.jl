"
    mean_hausdorff_2D(u, v, d, f)

Extract the edges of two images and then find the average Hausdorff distance along those edges.
Both arrays u and v are required to be the same size and they must be binary or boolean images.
The argument d corresponds to the distance metric used for computing. Typically, Euclidean() is
a common metric for use in Hausdorff distance computation. More options can be found in
Distances.jl. The argument `f` is the algorithm used for edge detection.  More options can be 
found in ImageEdgeDetection.jl. 
"

function mean_hausdorff_2D(u, v, d, f)
    edges_1 = detect_edges(u, f)
    edges_2 = detect_edges(v, f)

    sz_1 = size(edges_1, 1)
    sz_2 = size(edges_2, 1)

    # Loop through every point on `edge_1` and find its corresponding closest point to `edge_2`
    min_u = minimum.([map(x -> evaluate(d, edges_1[i, :], edges_2[x, :]), 1:sz_2) for i in 1:sz_1])

    # Loop through every point on `edge_2` and find its corresponding closest point to `edge_1`
    min_v = minimum.([map(x -> evaluate(d, edges_2[i, :], edges_1[x, :]), 1:sz_1) for i in 1:sz_2])

    # Take the average of each of these points to return the average Hausdorff distance 
    return mean([mean(min_u), mean(min_v)])
end

## TODO implement mean_hausdorff_ND
## TODO implement dice metric
## TODO implement mean_dice metric