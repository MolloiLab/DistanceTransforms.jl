"""
	euc(u::CartesianIndex{3}, v::CartesianIndex{3})

Compute the euclidean distance given two 3 dimensional CartesianIndex arrays
"""
euc(u::CartesianIndex{3}, v::CartesianIndex{3}) = √((u[1] - v[1])^2 + (u[2] - v[2])^2 + (u[3] - v[3])^2)

"""
	find_edges(A)

Find the edges of an array by computing the distance to the nearest background pixel `0` and keeping
only the values that have a distance of 1. This corresponds to an array indice being adjacent to the
background
"""
function find_edges(A)
    # Set every pixel that is currently background (0) equal to 1, as these are the pixels we are interested in
    A = A .== 0

    # Find the indeces of these background pixels (now set equal to 1)
    A = ImageMorphology.feature_transform(A)

    # Find the distance away from the nearest background pixel (now set to 1)
    A = ImageMorphology.distance_transform(A)

    # Return the indeces where the distance is equal to 1 as this means this pixel is adjacent to a background pixel
    return (findall(x->x==1, A))
	
	# return (transpose ∘ reshape)(reinterpret(Int, A), 2, :)
end 

"""
    detect_edges_3D(img, f)

Modifies `detect_edges` to work with 3D images.
"""

function detect_edges_3D(img, f)
	container = Array{Int64}(undef, size(img))
	for k in 1:(size(img)[3])
		container[:, :, k] = detect_edges(img[:, :, k], f)
	end
	return container
end

function compute_dtm(img)
    f = feature_transform(.!(Bool.(img)))
    foreground_dtm = distance_transform(f)
end