"
    find_edges(A)

Find the edges by finding the distance to the nearest background pixel and 
keep only the pixels that are adjacent to the nearest background pixel
"

function find_edges(A)
    # Set every pixel that is currently background (0) equal to 1, as these are the pixels we are interested in
    A = A .== 0

    # Find the indeces of these background pixels (now set equal to 1)
    A = feature_transform(A)

    # Find the distance away from the nearest background pixel (now set to 1)
    A = distance_transform(A)

    # Extract the indeces where the distance is equal to 1 as this means this pixel is adjacent to a background pixel
    edge = Tuple.(findall(x->x==1, A))

    # Turn the array of CartesianIndices into an array of integers corresponding to the CartesianIndice
    edge = (transpose ∘ reshape)(reinterpret(Int, edge), 2, :)
end

## TODO function detect_edges(A) using ImageEdgeDection.jl