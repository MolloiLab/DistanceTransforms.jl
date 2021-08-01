"""
    transform(f::Vector{T}, tfm::SquaredEuclidean)
    transform(f, tfm::SquaredEuclidean, dt, v, z)
    transform(img::Matrix{T}, tfm::SquaredEuclidean)
    transform!(img::Matrix{T}, tfm::SquaredEuclidean, nthreads)

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.

# Arguments
- f/img: 1D, 2D, or 3D to be transformed based on location 
    to the nearest background (0) pixel
- tfm: `SquaredEuclidean` type
- dt: Empty array that is the size of `f` or `img`. Will be filled
    with the distance transform values of each element in `f` or `img`
- v: `ones(Int64, length(f))` or 
    `ones(Int64, size(img))`
- z: `zeros(Float32, length(f) + 1)` or 
    `zeros(Float32, size(img) .+ 1)`
- nthreads: The number of threads on the computer `Threads.nthreads()`. 
    Allows you to use a parallelized `transform`
    function if you have access to multiple threads.

# Citation
'Distance Transforms of Sampled Functions' [Felzenszwalb and
Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)
"""
struct SquaredEuclidean{T1 <: AbstractArray, T2 <: AbstractArray} <: DistanceTransform 
	dt::T1
	v::T2
	z::T1
end

function SquaredEuclidean(
		f::AbstractArray, 
		dt = zeros(Float32, size(f)),
		v = ones(Int64, size(f)),
		z = zeros(Float32, size(f) .+ 1)
	)

	SquaredEuclidean(dt, v, z)
end

function transform(f::Vector{T}, tfm::SquaredEuclidean) where {T}
	dt = tfm.dt
	v = tfm.v
	z = tfm.z
	n = length(f)
	k = 1
	z[1] = -Inf32
	z[2] = Inf32
	
	# Lower envelope operation
	for q in 2:n
		while true
			s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2 * q - 2 * v[k])
			if s ≤ z[k]
				k -= 1
			else
				k += 1
				v[k] = q
				z[k] = s
				z[k + 1] = Inf32
				break
			end
		end
	end

    # Distance transform operation
    k = 1
    for q in 1:n
        while z[k + 1] < q
            k = k + 1
        end
        dt[q] = (q - v[k])^2 + f[v[k]]
    end
    return dt
end

# This is called to for multi-dimensional transforms
function transform(f, tfm::SquaredEuclidean, dt, v, z) where {T}
	n = length(f)
	k = 1
	z[1] = -Inf32
	z[2] = Inf32
	
	# Lower envelope operation
	for q in 2:n
		while true
			s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2 * q - 2 * v[k])
			if s ≤ z[k]
				k -= 1
			else
				k += 1
				v[k] = q
				z[k] = s
				z[k + 1] = Inf32
				break
			end
		end
	end

    # Distance transform operation
    k = 1
    for q in 1:n
        while z[k + 1] < q
            k = k + 1
        end
        dt[q] = (q - v[k])^2 + f[v[k]]
    end
    return dt
end

function transform(img::Matrix{T}, tfm::SquaredEuclidean) where {T}
    rows, columns = size(img)
	dt = tfm.dt
	v = tfm.v
	z = tfm.z
    for x in 1:rows
        dt[x, :] = transform(img[x, :], tfm, dt[x, :], v[x, :], z[x, :])
    end

    for y in 1:columns
        dt[:, y] = transform(img[:, y], tfm, dt[:, y], v[:, y], z[:, y])
    end

    return dt
end

function transform!(img::Matrix{T}, tfm::SquaredEuclidean, nthreads) where {T}
    if nthreads ≤ 1
        transform(img, tfm)
    else
		dt = tfm.dt
		v = tfm.v
		z = tfm.z
        rows, columns = size(img)
        Threads.@threads for x in 1:rows
            @views transform(img[x, :], tfm, dt[x, :], v[x, :], z[x, :])
        end
    
        Threads.@threads for y in 1:columns
            @views transform(img[:, y], tfm, dt[:, y], v[:, y], z[:, y])
        end
        return dt
    end
end

# function squared_euclidean_distance_transform(img::CuArray{T,2}, dt, v, z) where {T}
#     rows, columns = size(img)
#     @floop CUDAEx() for x in 1:rows
#         @views squared_euclidean_distance_transform(img[x, :], dt[x, :], v[x, :], z[x, :])
#     end

#     @floop CUDAEx() for y in 1:columns
#         @views squared_euclidean_distance_transform(img[:, y], dt[:, y], v[:, y], z[:, y])
#     end
#     return dt
# end