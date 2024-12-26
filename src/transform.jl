import KernelAbstractions as KA
import AcceleratedKernels as AK
using GPUArraysCore: AbstractGPUVector, AbstractGPUMatrix, AbstractGPUArray

export transform!, transform

"""
## `transform!`

```julia
transform!(f::AbstractVector, output, v, z)
transform!(img::AbstractMatrix, output, v, z; threaded=true)
transform!(vol::AbstractArray{<:Real,3}, output, v, z, temp; threaded=true)
transform!(img::AbstractGPUMatrix, output, v, z)
```
In-place squared Euclidean distance transforms. These functions apply the transform to the input data and store the result in the `output` argument.

- The first function operates on vectors.
- The second function operates on matrices with optional threading.
- The third function operates on 3D arrays with optional threading.
- The fourth function is specialized for GPU matrices.

The intermediate arrays `v` and `z` (and `temp` for 3D arrays) are used for computation. The `threaded` parameter enables parallel computation on the CPU.

#### Arguments
- `f`: Input vector, matrix, or 3D array.
- `output`: Preallocated array to store the result.
- `v`: Preallocated array for indices, matching the dimensions of `f`.
- `z`: Preallocated array for intermediate values, one element larger than `f`.
- `temp`: Preallocated array for intermediate values when transforming 3D arrays, matching the dimensions of `output`.

#### Examples
```julia
f = rand([0f0, 1f0], 10)
f_bool = boolean_indicator(f)
output = similar(f)
v = ones(Int32, size(f))
z = ones(eltype(f), size(f) .+ 1)
transform!(f_bool, output, v, z)
```
"""
function transform!(f::AbstractVector, output, v, z)
    z[1] = -Inf32
    z[2] = Inf32

    k = 1
    @inbounds for q in 2:length(f)
        s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2 * q - 2 * v[k])
        while s ≤ z[k]
            k -= 1
            s = ((f[q] + q^2) - (f[v[k]] + v[k]^2)) / (2 * q - 2 * v[k])
        end
        k += 1
        v[k] = q
        z[k] = s
        z[k+1] = Inf32
    end

    k = 1
    @inbounds for q in 1:length(f)
        while z[k+1] < q
            k += 1
        end
        output[q] = (q - v[k])^2 + f[v[k]]
    end
end

# 2D
function transform!(img::AbstractMatrix, output, v, z; threaded=true)
    if threaded
        Threads.@threads for i in eachindex(@view(img[:, 1]))
            @views transform!(img[i, :], output[i, :], v[i, :], z[i, :])
        end

        copyto!(img, output)

        Threads.@threads for j in eachindex(@view(img[1, :]))
            @views transform!(
                img[:, j], output[:, j], fill!(v[:, j], 1), fill!(z[:, j], 1)
            )
        end
    else
        for i in eachindex(@view(img[:, 1]))
            @views transform!(img[i, :], output[i, :], v[i, :], z[i, :])
        end

        copyto!(img, output)
		
        for j in eachindex(@view(img[1, :]))
            @views transform!(
                img[:, j], output[:, j], fill!(v[:, j], 1), fill!(z[:, j], 1)
            )
        end
    end
end

function transform!(vol::AbstractArray{<:Real,3}, output, v, z; threaded=true)
    if threaded
        # X dimension
        Threads.@threads for i in axes(vol, 1)
            for j in axes(vol, 2)
                @views transform!(vol[i, j, :], output[i, j, :], v[i, j, :], z[i, j, :])
            end
        end

        copyto!(vol, output)

        # Y dimension 
        Threads.@threads for i in axes(vol, 1)
            for k in axes(vol, 3)
                @views transform!(vol[i, :, k], output[i, :, k], fill!(v[i, :, k], 1), fill!(z[i, :, k], 1))
            end
        end

        copyto!(vol, output)

        # Z dimension
        Threads.@threads for j in axes(vol, 2)
            for k in axes(vol, 3)
                @views transform!(vol[:, j, k], output[:, j, k], fill!(v[:, j, k], 1), fill!(z[:, j, k], 1))
            end
        end
    else
        # X dimension
        for i in axes(vol, 1)
            for j in axes(vol, 2)
                @views transform!(vol[i, j, :], output[i, j, :], v[i, j, :], z[i, j, :])
            end
        end

        copyto!(vol, output)

        # Y dimension
        for i in axes(vol, 1)
            for k in axes(vol, 3)
                @views transform!(vol[i, :, k], output[i, :, k], fill!(v[i, :, k], 1), fill!(z[i, :, k], 1))
            end
        end

        copyto!(vol, output)

        # Z dimension
        for j in axes(vol, 2)
            for k in axes(vol, 3)
                @views transform!(vol[:, j, k], output[:, j, k], fill!(v[:, j, k], 1), fill!(z[:, j, k], 1))
            end
        end
    end
end

function transform!(img::AbstractGPUMatrix, output, v, z)
    AK.foreachindex(@view(img[:, 1])) do i
        @views transform!(img[i, :], output[i, :], v[i, :], z[i, :])
    end

    copyto!(img, output)

    AK.foreachindex(@view(img[1, :])) do j
        @views transform!(img[:, j], output[:, j], fill!(v[:, j], 1), fill!(z[:, j], 1))
    end
end

function transform!(vol::AbstractGPUArray{T,3}, output, v, z) where {T}
    # X dimension
    AK.foreachindex(@view(vol[:, 1, 1])) do i
        for j in axes(vol, 2)
            @views transform!(vol[i, j, :], output[i, j, :], v[i, j, :], z[i, j, :])
        end
    end

    copyto!(vol, output)

    # Y dimension
    AK.foreachindex(@view(vol[:, 1, 1])) do i
        for k in axes(vol, 3)
            @views transform!(vol[i, :, k], output[i, :, k], fill!(v[i, :, k], 1), fill!(z[i, :, k], 1))
        end
    end

    copyto!(vol, output)

    # Z dimension 
    AK.foreachindex(@view(vol[1, :, 1])) do j
        for k in axes(vol, 3)
            @views transform!(vol[:, j, k], output[:, j, k], fill!(v[:, j, k], 1), fill!(z[:, j, k], 1))
        end
    end
end

"""
## `transform`

```julia
transform(f::AbstractVector)
transform(img::AbstractMatrix; threaded=true)
transform(vol::AbstractArray{<:Real,3}; threaded=true)
transform(img::AbstractGPUMatrix)
```

Non-in-place squared Euclidean distance transforms that return a new array with the result. They allocate necessary intermediate arrays internally.

- The first function operates on vectors.
- The second function operates on matrices with optional threading.
- The third function operates on 3D arrays with optional threading.
- The fourth function is specialized for GPU matrices.

The `threaded` parameter can be used to enable or disable parallel computation on the CPU.

#### Arguments
- `f/img/vol`: Input vector, matrix, or 3D array to be transformed.

#### Examples
```julia
f = rand([0f0, 1f0], 10)
f_bool = boolean_indicator(f)
f_tfm = transform(f_bool)
```
"""
function transform(f::AbstractVector)
    output = similar(f, eltype(f))
    v = ones(Int32, length(f))
    z = ones(eltype(f), length(f) + 1)

    transform!(f, output, v, z)
    return output
end

# 2D
function transform(img::AbstractMatrix; threaded=true)
    output = similar(img, eltype(img))
    v = ones(Int32, size(img))
    z = ones(eltype(img), size(img) .+ 1)

    transform!(img, output, v, z; threaded=threaded)
    return output
end

# 3D
function transform(vol::AbstractArray{<:Real,3}; threaded=true)
    output = similar(vol, eltype(vol))
    v = ones(Int32, size(vol))
    z = ones(eltype(vol), size(vol) .+ 1)

    transform!(vol, output, v, z; threaded=threaded)
    return output
end

# GPU (2D)
function transform(img::AbstractGPUMatrix)
    backend = KA.get_backend(img)

    output = similar(img, Float32)
    v = KA.ones(backend, Int32, size(img))
    z = KA.ones(backend, eltype(img), size(img) .+ 1)

    transform!(img, output, v, z)
    return output
end

# GPU (3D)
function transform(vol::AbstractGPUArray{T,3}) where {T}
    backend = KA.get_backend(vol)

    output = similar(vol, Float32)
    v = KA.ones(backend, Int32, size(vol))
    z = KA.ones(backend, eltype(vol), size(vol) .+ 1)

    transform!(vol, output, v, z)
    return output
end
