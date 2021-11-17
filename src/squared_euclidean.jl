struct SquaredEuclidean{T1<:AbstractArray,T2<:AbstractArray} <: DistanceTransform
    dt::T1
    v::T2
    z::T1
end

"""
	SquaredEuclidean(
		f::AbstractArray, 
		dt = zeros(Float32, size(f)),
		v = ones(Int64, size(f)),
		z = zeros(Float32, size(f) .+ 1)
	)

Prepares an array to be `transform`ed using the squared euclidean algorithm
laid out in 'Distance Transforms of Sampled Functions' [Felzenszwalb and
Huttenlocher] (DOI: 10.4086/toc.2012.v008a019)

# Arguments
- f/img: 1D, 2D, or 3D to be transformed based on location 
    to the nearest background (0) pixel
- dt: Empty array that is the size of `f` or `img`. Will be filled
    with the distance transform values of each element in `f` or `img`
- v: `ones(Int64, length(f))` or 
    `ones(Int64, size(img))`
- z: `zeros(Float32, length(f) + 1)` or 
    `zeros(Float32, size(img) .+ 1)`
"""
function SquaredEuclidean(
    f::AbstractArray,
    dt=zeros(Float32, size(f)),
    v=ones(Int64, size(f)),
    z=zeros(Float32, size(f) .+ 1),
)
    return SquaredEuclidean(dt, v, z)
end

"""
	transform(f::AbstractVector{T}, tfm::SquaredEuclidean)
    transform(img::AbstractMatrix{T}, tfm::SquaredEuclidean)
    transform!(img::AbstractMatrix{T}, tfm::SquaredEuclidean, nthreads)
	transform!(img::CuArray{T,2}, tfm::SquaredEuclidean)

Applies a squared euclidean distance transform to an input image.
Returns an array with spatial information embedded in the array 
elements.

# Arguments
- f/img: 1D, 2D, or 3D to be transformed based on location 
    to the nearest background (0) pixel
- tfm: `SquaredEuclidean` type
- nthreads: The number of threads on the computer `Threads.nthreads()`. 
    Allows you to use a parallelized `transform`
    function if you have access to multiple threads.

# Citation
'Distance Transforms of Sampled Functions' [Felzenszwalb and
Huttenlocher](DOI: 10.4086/toc.2012.v008a019)
"""
function transform(f::AbstractVector{T}, tfm::SquaredEuclidean) where {T}
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

# This function is called to for multi-dimensional transforms
function _transform(f, tfm::SquaredEuclidean, dt, v, z) where {T}
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

# This function is called for the GPU version
function _transform(f::AbstractArray{T,1}, dt, v, z) where {T}
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

function transform(img::AbstractMatrix{T}, tfm::SquaredEuclidean) where {T}
    rows, columns = size(img)
    dt = tfm.dt
    v = tfm.v
    z = tfm.z
    for x in 1:rows
        dt[x, :] = _transform(img[x, :], tfm, dt[x, :], v[x, :], z[x, :])
    end

    for y in 1:columns
        dt[:, y] = _transform(img[:, y], tfm, dt[:, y], v[:, y], z[:, y])
    end

    return dt
end

function transform!(img::AbstractMatrix{T}, tfm::SquaredEuclidean, nthreads) where {T}
    @assert nthreads > 1
    dt = tfm.dt
    v = tfm.v
    z = tfm.z
    rows, columns = size(img)
    Threads.@threads for x in 1:rows
        @views _transform(img[x, :], tfm, dt[x, :], fill!(v[x, :], 1), fill!(z[x, :], 0))
    end
    Threads.@threads for y in 1:columns
        @views _transform(img[:, y], tfm, dt[:, y], fill!(v[:, y], 1), fill!(z[:, y], 0))
    end
    return dt
end

function transform!(img::CuArray{T,2}, tfm::SquaredEuclidean) where {T}
    dt = tfm.dt
    v = tfm.v
    z = tfm.z
    rows, columns = size(img)
    @floop CUDAEx() for x in 1:rows
        @views _transform(img[x, :], dt[x, :], fill!(v[x, :], 1), fill!(z[x, :], 0))
    end

    @floop CUDAEx() for y in 1:columns
        @views _transform(img[:, y], dt[:, y], fill!(v[:, y], 1), fill!(z[:, y], 0))
    end
    return dt
end
