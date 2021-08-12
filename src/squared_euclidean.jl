"""
    squared_euclidean(f::Array{T,1}, dt, v, z)
    squared_euclidean(img::Array{T,2}, dt)
    squared_euclidean(img::AbstractArray{T,2}, dt, threads)

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.

# Arguments
- img: 1D, 2D, or 3D to be transformed based on location 
    to the nearest background (0) pixel
- dt: Empty array that is the size of `f` or `img`. Will be filled
    with the distance transform values of each element in `f` or `img`
- v: `ones(Int64, size(img))`
- z: `zeros(Float32, size(img) .+ 1)`
- threads: The number of threads on the computer `Threads.nthreads()`. 
    Allows you to use a parallelized `squared_euclidean`
    function if you have access to multiple threads.

# Citation
'Distance Transforms of Sampled Functions' [Felzenszwalb and
Huttenlocher](DOI: 10.4086/toc.2012.v008a019)
"""
function squared_euclidean(f::AbstractArray{T,1}, dt, v, z) where {T}
	n = length(f)
	k = 1
	z[1] = -1.0f12
	z[2] = 1.0f12
	
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
				z[k + 1] = 1.0f12
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

function squared_euclidean(img::AbstractArray{T,2}, dt, v, z) where {T}
    rows, columns = size(img)
    for x in 1:rows
        dt[x, :] = squared_euclidean(img[x, :], dt[x, :], v[x, :], z[x, :])
    end

    for y in 1:columns
        dt[:, y] = squared_euclidean(img[:, y], dt[:, y], v[:, y], z[:, y])
    end

    return dt
end

function squared_euclidean!(img::AbstractMatrix{T}, dt, v, z, nthreads) where {T}
    @assert nthreads > 1
    rows, columns = size(img)
    Threads.@threads for x in 1:rows
        @views squared_euclidean(img[x, :], dt[x, :], fill!(v[x, :], 1), fill!(z[x, :], 0))
    end
	Threads.@threads for y in 1:columns
		@views squared_euclidean(img[:, y], dt[:, y], fill!(v[:, y], 1), fill!(z[:, y], 0))
	end
    return dt
end

function squared_euclidean!(img::CuArray{T,2}, dt, v, z) where {T}
    rows, columns = size(img)
    @floop CUDAEx() for x in 1:rows
        @views squared_euclidean(img[x, :], dt[x, :], fill!(v[x, :], 1), fill!(z[x, :], 0))
    end

    @floop CUDAEx() for y in 1:columns
        @views squared_euclidean(img[:, y], dt[:, y], fill!(v[:, y], 1), fill!(z[:, y], 0))
    end
    return dt
end