
function find_edges(A)
    "
    This function finds the edges by finding the distance to the nearest background (0)
    pixel and keeping only the pixels that are adjacent to the nearest background pixel
    "

    # Set every pixel that is currently background (0) equal to 1, as these are the pixels we are interested in
    A = A .== 0

    # Find the indeces of these background pixels (now set equal to 1)
    A = feature_transform(A)

    # Find the distance away from the nearest background pixel (now set to 1)
    A = distance_transform(A)

    # Return the indeces where the distance is equal to 1 as this means this pixel is adjacent to a background pixel
    return Tuple.(findall(x->x==1, A))
end