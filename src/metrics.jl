function mean_hausdorff(u, v)
    "
    This function extracts the edges of two images and then finds the average Hausdorff distance
    Both arrays u and v are required to be the same size and they must be binary or boolean images.
    "
    d = Euclidean()

    # Find the coordinates of the pixels that are edges in the images
    edges_1 = find_edges(u)
    edges_2 = find_edges(v)

    # Turn the array of CartesianIndices into an array of integers corresponding to the CartesianIndices
    edges_1 = (transpose ∘ reshape)(reinterpret(Int, edges_1), 2, :)
    edges_2 = (transpose ∘ reshape)(reinterpret(Int, edges_2), 2, :)


    min_euc_list_u = []
    min_euc_list_v = []

    # Loop through every edge point on `edge_1` and find its corresponding closest point to `edge_2`
    for i in 1:size(edges_1, 1)
        euc_list_1 = []
        for j in 1:size(edges_2, 1)
            euc = evaluate(d, edges_1[i, :], edges_2[j, :])
            append!(euc_list_1 , euc)
        end
        append!(min_euc_list_u, minimum(euc_list_1))
    end

    # Loop through every edge point on `edge_2` and find its corresponding closest point to `edge_1`
    for i in 1:size(edges_2, 1)
        euc_list_1 = []
        for j in 1:size(edges_1, 1)
            euc = evaluate(d, edges_2[i, :], edges_1[j, :])
            append!(euc_list_1 , euc)
        end
        append!(min_euc_list_v, minimum(euc_list_1))
    end

    # Take the average of each of these points to return the average Hausdorff distance 
    return mean([mean(min_euc_list_u), mean(min_euc_list_v)])
end

function mean_dice(score, target)
    smooth = 1e-5
    intersect = sum(score .* target)
    y_sum = sum(target .* target)
    z_sum = sum(score .* score)
    metric = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    return metric
end

