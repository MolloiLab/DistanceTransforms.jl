"
    mean_hausdorff(u, v, d)

Extract the edges of two images and then find the average Hausdorff distance along those edges.
Both arrays u and v are required to be the same size and they must be binary or boolean images.
The argument d corresponds to the distance metric used for computing. Typically, Euclidean() is
a common metric for use in Hausdorff distance computation. More options can be found in
Distances.jl
"

function mean_hausdorff(u, v, d)
    edges_1 = find_edges(u)
    edges_2 = find_edges(v)

    sz_1 = size(edges_1, 1)
    sz_2 = size(edges_2, 1)

    # Loop through every point on `edge_1` and find its corresponding closest point to `edge_2`
    min_u = minimum.([map(x -> evaluate(d, edges_1[i, :], edges_2[x, :]), 1:sz_2) for i in 1:sz_1])

    # Loop through every point on `edge_2` and find its corresponding closest point to `edge_1`
    min_v = minimum.([map(x -> evaluate(d, edges_2[i, :], edges_1[x, :]), 1:sz_1) for i in 1:sz_2])

    # Take the average of each of these points to return the average Hausdorff distance 
    return mean([mean(min_u), mean(min_v)])
end

## TODO implement mean_hausdorff that works with n-dimensional images

function mean_dice(score, target)
    smooth = 1e-5
    intersect = sum(score .* target)
    y_sum = sum(target .* target)
    z_sum = sum(score .* score)
    metric = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return metric
end
